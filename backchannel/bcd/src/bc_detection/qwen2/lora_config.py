"""
Qwen2-Audio용 LoRA 설정 빌더

Audio Encoder / QwenLM / Both 조합을 관리하며,
base / instruct 모델별 target module 차이를 대응한다.
"""

import logging
from typing import List, Optional

from .constants import LORA_TARGET_PRESETS

logger = logging.getLogger(__name__)


class QwenLoraConfigBuilder:
    """
    Qwen2-Audio용 LoRA 설정을 생성한다.

    실험 변형:
    1. Audio Encoder에만 LoRA 적용
    2. QwenLM에만 LoRA 적용
    3. Audio Encoder + QwenLM 모두에 LoRA 적용
    """

    def __init__(
        self,
        target_preset: str = "llm_only",
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        custom_target_modules: Optional[List[str]] = None,
    ):
        """
        Args:
            target_preset: "audio_only", "llm_only", "both" 중 하나
            rank: LoRA rank
            alpha: LoRA alpha (scaling factor)
            dropout: LoRA dropout
            custom_target_modules: 프리셋 대신 직접 target module 지정
        """
        if custom_target_modules:
            self.target_modules = custom_target_modules
        elif target_preset in LORA_TARGET_PRESETS:
            self.target_modules = LORA_TARGET_PRESETS[target_preset]
        else:
            raise ValueError(
                f"target_preset은 {list(LORA_TARGET_PRESETS.keys())} 중 하나여야 합니다: {target_preset}"
            )

        self.target_preset = target_preset
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

        logger.info(
            f"QwenLoraConfigBuilder 초기화: preset={target_preset}, "
            f"rank={rank}, alpha={alpha}, dropout={dropout}"
        )

    def build(self) -> dict:
        """
        peft.LoraConfig에 전달할 설정 dict를 반환한다.

        Returns:
            dict with r, lora_alpha, target_modules, lora_dropout, bias, task_type
        """
        return {
            "r": self.rank,
            "lora_alpha": self.alpha,
            "target_modules": self.target_modules,
            "lora_dropout": self.dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

    def build_peft_config(self):
        """
        peft.LoraConfig 객체를 직접 생성하여 반환한다.

        Returns:
            peft.LoraConfig 인스턴스

        Raises:
            ImportError: peft가 설치되지 않은 경우
        """
        try:
            from peft import LoraConfig
        except ImportError:
            raise ImportError(
                "peft 패키지가 필요합니다: pip install peft"
            )

        config_dict = self.build()
        return LoraConfig(**config_dict)

    def get_metadata(self) -> dict:
        """실험 추적용 메타데이터를 반환한다."""
        return {
            "lora_target_preset": self.target_preset,
            "lora_rank": self.rank,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "lora_target_modules": self.target_modules,
        }
