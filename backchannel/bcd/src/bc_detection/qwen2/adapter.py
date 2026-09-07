"""
Qwen2-Audio 모델 어댑터

Qwen2-Audio-7B / Qwen2-Audio-7B-Instruct 로딩, 추론, LoRA 적용을 담당한다.
기존 프로젝트의 model backend adapter 역할을 한다.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .constants import MODEL_REGISTRY

logger = logging.getLogger(__name__)


def _resolve_device_map(device_map: Optional[Union[str, dict]]) -> Union[str, dict]:
    """
    DDP 환경을 감지하여 적절한 device_map을 반환한다.

    - DDP(torchrun) 환경: 각 프로세스가 자신의 LOCAL_RANK에 해당하는 GPU에만 모델을 올려야 함
      → "cuda:{LOCAL_RANK}"
    - 단일 프로세스: 사용자가 지정한 값 그대로 사용
      (기본값 "auto"는 model-parallel 시도이므로, 단일 GPU 전용 학습에는 "cuda:0" 권장)
    """
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        resolved = f"cuda:{local_rank}"
        logger.info(f"DDP 감지 (LOCAL_RANK={local_rank}) → device_map={resolved}")
        return resolved

    if device_map is None:
        return "cuda:0"
    return device_map


class Qwen2AudioAdapter:
    """
    Qwen2-Audio 모델/프로세서 로딩 및 추론을 담당한다.

    - base / instruct 모델 선택
    - processor를 통한 입력 구성
    - batch generation
    - LoRA 적용 모델 로딩
    """

    def __init__(
        self,
        model_type: str = "instruct",
        model_id: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: str = "float16",
        max_new_tokens: int = 32,
        do_sample: bool = False,
        attn_implementation: str = "sdpa",
    ):
        """
        Args:
            model_type: "base" 또는 "instruct"
            model_id: 직접 모델 ID 지정 (None이면 model_type으로 결정)
            device_map: 디바이스 배치 전략
            torch_dtype: 모델 dtype ("float16", "bfloat16", "float32")
            max_new_tokens: 최대 생성 토큰 수 (백채널 레이블은 짧으므로 32 충분)
            do_sample: 샘플링 사용 여부 (분류 태스크이므로 기본 False)
            attn_implementation: attention 구현 ("sdpa", "eager", "flash_attention_2")
        """
        self.model_type = model_type
        self.model_id = model_id or MODEL_REGISTRY.get(model_type)
        if self.model_id is None:
            raise ValueError(f"model_type은 {list(MODEL_REGISTRY.keys())} 중 하나여야 합니다: {model_type}")

        # DDP 호환: LOCAL_RANK가 있으면 해당 GPU로, 없으면 사용자 지정값 사용
        self.device_map = _resolve_device_map(device_map)
        self.torch_dtype = getattr(torch, torch_dtype, torch.float16)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.attn_implementation = attn_implementation

        self.model = None
        self.processor = None

        logger.info(
            f"Qwen2AudioAdapter 초기화: model_id={self.model_id}, "
            f"dtype={torch_dtype}, attn={attn_implementation}"
        )

    def load_model(self):
        """모델과 프로세서를 로드한다."""
        import os
        import torch.distributed as dist
        from transformers import (
            Qwen2AudioForConditionalGeneration,
            Qwen2AudioProcessor,
        )

        logger.info(f"모델 로딩 시작: {self.model_id}")

        is_dist = dist.is_available() and dist.is_initialized()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if is_dist and local_rank != 0:
            dist.barrier()

        self.processor = Qwen2AudioProcessor.from_pretrained(self.model_id)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            dtype=self.torch_dtype,
            attn_implementation=self.attn_implementation,
        )
        self.model.eval()

        if is_dist and local_rank == 0:
            dist.barrier()

        logger.info(f"모델 로딩 완료: {self.model_id} (device_map={self.device_map})")

    def apply_lora(self, lora_config_dict: dict):
        """
        모델에 LoRA를 적용한다.

        Args:
            lora_config_dict: QwenLoraConfigBuilder.build()의 반환값
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError("peft 패키지가 필요합니다: pip install peft")

        if self.model is None:
            raise RuntimeError("load_model()을 먼저 호출하세요")

        lora_config = LoraConfig(**lora_config_dict)
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        logger.info("LoRA 적용 완료")

    def load_lora_weights(self, lora_path: Union[str, Path]):
        """저장된 LoRA 가중치를 로드한다."""
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("peft 패키지가 필요합니다: pip install peft")

        if self.model is None:
            raise RuntimeError("load_model()을 먼저 호출하세요")

        self.model = PeftModel.from_pretrained(self.model, str(lora_path))
        self.model.eval()

        logger.info(f"LoRA 가중치 로드 완료: {lora_path}")

    def generate(
        self,
        audios: List[np.ndarray],
        prompts: List,
        use_chat_template: bool = True,
    ) -> List[str]:
        """
        배치 단위로 generation을 수행한다.

        Args:
            audios: numpy 오디오 배열 리스트
            prompts: 프롬프트 리스트
                     - use_chat_template=True: conversation dict 리스트의 리스트
                     - use_chat_template=False: raw prompt 문자열 리스트
            use_chat_template: chat template 사용 여부

        Returns:
            생성된 텍스트 리스트
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("load_model()을 먼저 호출하세요")

        # 프롬프트 텍스트 준비
        if use_chat_template:
            text_inputs = [
                self.processor.apply_chat_template(
                    conv, add_generation_prompt=True, tokenize=False
                )
                for conv in prompts
            ]
        else:
            text_inputs = prompts

        # processor로 입력 구성
        inputs = self.processor(
            text=text_inputs,
            audio=audios,
            return_tensors="pt",
            padding=True,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
        )
        inputs = inputs.to(self.model.device)

        # generation
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=1,
                do_sample=self.do_sample,
            )

        # 입력 토큰 제거 후 디코딩
        generated_ids = output_ids[:, inputs.input_ids.size(1):]
        outputs = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return [o.strip() for o in outputs]

    def generate_single(
        self,
        audio: np.ndarray,
        prompt,
        use_chat_template: bool = True,
    ) -> str:
        """단일 샘플에 대한 generation을 수행한다."""
        results = self.generate([audio], [prompt], use_chat_template)
        return results[0]

    def generate_multi_audio_single(
        self,
        audios: List[np.ndarray],
        prompt,
        use_chat_template: bool = True,
    ) -> str:
        """
        한 샘플에 여러 audio item이 들어가는 generation을 수행한다.

        Qwen2-Audio processor는 한 텍스트 안의 여러 <|AUDIO|> 토큰과
        flat audio 리스트를 순서대로 매칭한다. separated speaker/listener
        smoke test에서는 batch_size=1로 이 경로를 사용한다.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("load_model()을 먼저 호출하세요")

        if use_chat_template:
            text_inputs = [
                self.processor.apply_chat_template(
                    prompt, add_generation_prompt=True, tokenize=False
                )
            ]
        else:
            text_inputs = [prompt]

        inputs = self.processor(
            text=text_inputs,
            audio=audios,
            return_tensors="pt",
            padding=True,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=1,
                do_sample=self.do_sample,
            )

        generated_ids = output_ids[:, inputs.input_ids.size(1):]
        outputs = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return outputs[0].strip()

    def get_metadata(self) -> dict:
        """모델 설정 메타데이터를 반환한다."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "torch_dtype": str(self.torch_dtype),
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
