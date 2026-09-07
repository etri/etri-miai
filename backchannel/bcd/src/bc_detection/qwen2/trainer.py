"""
Qwen2-Audio LoRA 학습 모듈

Qwen2-Audio 모델에 LoRA를 적용하여 백채널 분류 태스크로 fine-tuning한다.
학습 목표: 오디오 + 프롬프트 → 레이블 텍스트 생성 (next-token prediction)

HuggingFace Trainer 기반으로 구현한다.
"""

import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, get_worker_info
from torch.utils.data.sampler import Sampler
from transformers import Trainer, TrainingArguments

from .constants import AUDIO_SAMPLE_RATE, IGNORE_INDEX
from ..utils.sampling import ReseedableSampler, create_balanced_sampler

logger = logging.getLogger(__name__)


def _enabled(config: Optional[dict]) -> bool:
    """Return True when an optional config block is enabled."""
    return bool(config and config.get("enabled", False))


class Qwen2AudioTrainDataset(Dataset):
    """
    Qwen2-Audio LoRA 학습용 데이터셋

    각 샘플을 (audios, prompt, label_text) 형태로 반환한다.
    mixed mode에서는 audios 길이가 1이고, separated mode에서는 길이가 2이다.
    실제 토크나이징은 collator에서 수행한다.
    """

    def __init__(
        self,
        samples: list,
        prompt_builder,
        max_samples: Optional[int] = None,
        fewshot_selector=None,
        variants_per_target: int = 1,
        audio_cache_config: Optional[dict] = None,
        timing_profile_config: Optional[dict] = None,
        crop_speaker_sec: Optional[float] = None,
        crop_listener_sec: Optional[float] = None,
        silence_speaker: bool = False,
        silence_listener: bool = False,
    ):
        """
        Args:
            samples: Qwen2AudioDataset.samples 리스트
            prompt_builder: QwenPromptBuilder 인스턴스
            max_samples: 최대 샘플 수 (디버깅용)
            fewshot_selector: dynamic few-shot support selector (None이면 비활성)
            variants_per_target: target당 support sampling variant 수
            audio_cache_config: raw audio LRU cache 설정
            timing_profile_config: dataloader timing 로그 설정
            crop_speaker_sec / crop_listener_sec: 끝에서 N초만 사용 (None=전체)
            silence_speaker / silence_listener: True면 0 array로 치환
        """
        target_samples = samples
        if max_samples is not None:
            target_samples = target_samples[:max_samples]

        self.target_samples = list(target_samples)
        self.prompt_builder = prompt_builder
        self.fewshot_selector = fewshot_selector
        self.crop_speaker_sec = crop_speaker_sec
        self.crop_listener_sec = crop_listener_sec
        self.silence_speaker = silence_speaker
        self.silence_listener = silence_listener
        self.variants_per_target = (
            max(1, int(variants_per_target)) if fewshot_selector is not None else 1
        )
        self.audio_cache_config = audio_cache_config or {}
        self.timing_profile_config = timing_profile_config or {}
        self.audio_cache_enabled = _enabled(self.audio_cache_config)
        self.timing_profile_enabled = (
            _enabled(self.timing_profile_config) or self.audio_cache_enabled
        )
        self.audio_cache_max_items = max(
            0,
            int(self.audio_cache_config.get("raw_max_items", 0) or 0),
        )
        self.timing_log_every_n = max(
            0,
            int(
                self.timing_profile_config.get(
                    "log_every_n",
                    self.audio_cache_config.get("log_every_n", 0),
                )
                or 0
            ),
        )
        self._audio_cache = OrderedDict()
        self._audio_cache_hits = 0
        self._audio_cache_misses = 0
        self._audio_load_requests = 0
        self._audio_disk_reads = 0
        self._audio_load_time_sec = 0.0

        # Balanced sampler utilities inspect dataset.samples[i]["label"] directly.
        # Repeat target samples for virtual support variants so sampler weights stay
        # target-label based and support labels never affect batch sampling.
        self.samples = [
            sample
            for sample in self.target_samples
            for _ in range(self.variants_per_target)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def get_class_labels(self) -> List[str]:
        """샘플에서 실제 등장한 클래스 레이블을 정렬된 리스트로 반환한다."""
        return sorted({str(s["label"]) for s in self.samples})

    def _get_audio_paths(self, sample: dict) -> List[str]:
        """prompt_builder의 audio_input_mode에 맞는 오디오 경로를 반환한다."""
        audio_input_mode = getattr(self.prompt_builder, "audio_input_mode", "mixed")
        if audio_input_mode == "separated":
            return [sample["speaker_audio_path"], sample["listener_audio_path"]]
        return [sample["mixed_audio_path"]]

    def _get_audio_path_role(self, sample: dict, audio_path: str) -> str:
        if audio_path == sample.get("speaker_audio_path"):
            return "speaker"
        if audio_path == sample.get("listener_audio_path"):
            return "listener"
        return "mixed"

    def _crop_tail(self, audio: np.ndarray, target_sec: float) -> np.ndarray:
        target_n = int(round(target_sec * AUDIO_SAMPLE_RATE))
        if target_n <= 0:
            return np.zeros(0, dtype=np.float32)
        if len(audio) >= target_n:
            return audio[-target_n:]
        pad = target_n - len(audio)
        return np.concatenate([np.zeros(pad, dtype=audio.dtype), audio])

    def _apply_role_policy(self, audio: np.ndarray, role: str) -> np.ndarray:
        if role == "speaker":
            if self.silence_speaker:
                n_sec = self.crop_speaker_sec if self.crop_speaker_sec else 1.5
                return np.zeros(int(round(n_sec * AUDIO_SAMPLE_RATE)), dtype=np.float32)
            if self.crop_speaker_sec is not None:
                return self._crop_tail(audio, self.crop_speaker_sec)
        elif role == "listener":
            if self.silence_listener:
                n_sec = self.crop_listener_sec if self.crop_listener_sec else 1.5
                return np.zeros(int(round(n_sec * AUDIO_SAMPLE_RATE)), dtype=np.float32)
            if self.crop_listener_sec is not None:
                return self._crop_tail(audio, self.crop_listener_sec)
        return audio

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """16kHz mono wav를 로드한다."""
        self._audio_load_requests += 1

        if self.audio_cache_enabled and audio_path in self._audio_cache:
            self._audio_cache_hits += 1
            audio = self._audio_cache.pop(audio_path)
            self._audio_cache[audio_path] = audio
            self._maybe_log_audio_stats()
            return audio

        started_at = time.perf_counter()
        audio, sr = sf.read(audio_path, dtype="float32")
        elapsed = time.perf_counter() - started_at
        self._audio_disk_reads += 1
        if self.timing_profile_enabled:
            self._audio_load_time_sec += elapsed

        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)
        if sr != AUDIO_SAMPLE_RATE:
            raise ValueError(
                f"샘플링 레이트 불일치: {audio_path} "
                f"(expected {AUDIO_SAMPLE_RATE}, got {sr})"
            )

        if self.audio_cache_enabled and self.audio_cache_max_items > 0:
            self._audio_cache_misses += 1
            self._audio_cache[audio_path] = audio
            while len(self._audio_cache) > self.audio_cache_max_items:
                self._audio_cache.popitem(last=False)
        elif self.audio_cache_enabled:
            self._audio_cache_misses += 1

        self._maybe_log_audio_stats()
        return audio

    def _maybe_log_audio_stats(self) -> None:
        """Periodically log per-worker audio load/cache stats."""
        if not self.timing_profile_enabled or self.timing_log_every_n <= 0:
            return
        if self._audio_load_requests % self.timing_log_every_n != 0:
            return

        total = self._audio_load_requests
        hits = self._audio_cache_hits
        misses = self._audio_cache_misses
        hit_rate = hits / total if total else 0.0
        avg_read_ms = (
            self._audio_load_time_sec / max(self._audio_disk_reads, 1) * 1000.0
            if self.timing_profile_enabled
            else 0.0
        )
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else "main"

        logger.info(
            "[audio-load] worker=%s requests=%d cache_items=%d "
            "hits=%d misses=%d disk_reads=%d hit_rate=%.3f read_avg_ms=%.2f",
            worker_id,
            total,
            len(self._audio_cache),
            hits,
            misses,
            self._audio_disk_reads,
            hit_rate,
            avg_read_ms,
        )

    def _extract_audio_paths_from_prompt(self, prompt, fallback_paths: List[str]) -> List[str]:
        """chat template prompt에 포함된 audio URL을 순서대로 추출한다."""
        if not isinstance(prompt, list):
            return fallback_paths

        audio_paths = []
        for message in prompt:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "audio":
                        audio_paths.append(item["audio_url"])

        return audio_paths or fallback_paths

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        target_index = idx // self.variants_per_target
        variant_id = idx % self.variants_per_target
        audio_paths = self._get_audio_paths(sample)

        support_examples = []
        if self.fewshot_selector is not None:
            support_examples = self.fewshot_selector.select(
                target_sample=sample,
                target_index=target_index,
                variant_id=variant_id,
            )

        # 프롬프트 생성
        prompt = self.prompt_builder.build(
            audio_paths=audio_paths,
            speaker_text=sample.get("speaker_text"),
            listener_text=sample.get("listener_text"),
            support_examples=support_examples,
        )
        prompt_audio_paths = self._extract_audio_paths_from_prompt(prompt, audio_paths)

        # 오디오 로드 (soundfile이 librosa보다 ~150배 빠름)
        # 데이터는 이미 16kHz mono이므로 resample 불필요
        audios = []
        for path in prompt_audio_paths:
            audio = self._load_audio(path)
            role = self._get_audio_path_role(sample, path)
            audios.append(self._apply_role_policy(audio, role))

        return {
            "audio": audios[0],
            "audios": audios,
            "audio_paths": prompt_audio_paths,
            "prompt": prompt,
            "label": sample["label"],
            "sample_id": sample["sample_id"],
            "support_sample_ids": [
                support.get("sample_id") for support in support_examples
            ],
            "support_labels": [
                support.get("label") for support in support_examples
            ],
        }


class Qwen2AudioTrainCollator:
    """
    Qwen2-Audio 학습용 collator

    전략:
    1. label 뒤에 EOS 토큰(<|im_end|>)을 포함시켜 모델이 생성 종료를 학습
       - use_chat_template=True: apply_chat_template이 assistant role에 EOS 자동 추가
       - use_chat_template=False: 수동으로 tokenizer.eos_token 추가
       - base/instruct 모두 EOS 토큰은 <|im_end|> (id=151645)로 동일
    2. tokenizer 직접 호출 (processor의 audio token 확장 없음)
       → input_ids에 <|AUDIO|> 토큰을 audio 개수만큼 유지
       → 모델 forward의 legacy 경로가 자동으로 audio 위치를 확장
    3. Left-padding (Qwen2Audio는 left-padding이 기본값)
       → _merge_input_ids_with_audio_features의 padding 방향 감지와 일치

    핵심: loss는 label + EOS 부분에만 계산 (prompt는 IGNORE_INDEX).
    """

    def __init__(
        self,
        processor,
        use_chat_template: bool = True,
        timing_profile_config: Optional[dict] = None,
    ):
        """
        Args:
            processor: Qwen2AudioProcessor 인스턴스
            use_chat_template: chat template 사용 여부 (instruct/base 모두 지원)
        """
        self.processor = processor
        self.use_chat_template = use_chat_template
        self.timing_profile_config = timing_profile_config or {}
        self.timing_profile_enabled = _enabled(self.timing_profile_config)
        self.timing_log_every_n = max(
            0,
            int(self.timing_profile_config.get("log_every_n", 0) or 0),
        )
        self._collate_calls = 0
        self._timing_totals: Dict[str, float] = {
            "template_sec": 0.0,
            "tokenize_sec": 0.0,
            "feature_sec": 0.0,
            "pad_sec": 0.0,
            "collate_sec": 0.0,
        }
        self._timing_audio_count = 0

    def _process_single(self, item: dict) -> dict:
        """
        단일 샘플을 legacy 경로용 입력으로 변환한다.

        - tokenizer로만 텍스트 토큰화 (processor의 audio token 확장 없음)
        - feature_extractor로 audio list만 별도 처리
        - label 뒤에 EOS 토큰 추가 (생성 종료 학습)
        """
        audios = item.get("audios")
        if audios is None:
            audios = [item["audio"]]
        label_text = item["label"]
        tokenizer = self.processor.tokenizer
        feature_extractor = self.processor.feature_extractor
        timing = {
            "template_sec": 0.0,
            "tokenize_sec": 0.0,
            "feature_sec": 0.0,
            "audio_count": len(audios),
        }

        # 1. 텍스트 생성
        started_at = time.perf_counter()
        if self.use_chat_template:
            user_conv = item["prompt"]
            # prompt: user role + generation prompt까지
            prompt_text = self.processor.apply_chat_template(
                user_conv,
                add_generation_prompt=True,
                tokenize=False,
            )
            # full: assistant role에 label 포함 → EOS(<|im_end|>)가 자동 추가됨
            full_conv = list(user_conv) + [
                {"role": "assistant", "content": label_text}
            ]
            full_text = self.processor.apply_chat_template(
                full_conv,
                add_generation_prompt=False,
                tokenize=False,
            )
        else:
            # raw prompt 모드: 수동으로 EOS 추가
            # base/instruct 모두 eos_token = '<|im_end|>' (id=151645)
            prompt_text = item["prompt"]
            eos = tokenizer.eos_token or ""
            full_text = prompt_text + label_text + eos
        if self.timing_profile_enabled:
            timing["template_sec"] = time.perf_counter() - started_at

        # 2. tokenizer 직접 호출 (audio token 확장 없음)
        started_at = time.perf_counter()
        full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
        if self.timing_profile_enabled:
            timing["tokenize_sec"] = time.perf_counter() - started_at

        # 3. labels: prompt 부분은 IGNORE_INDEX, 나머지(label + EOS)만 학습
        prompt_len = prompt_ids.shape[0]
        labels = full_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX

        # 4. attention_mask (모든 실제 토큰 = 1)
        attention_mask = torch.ones_like(full_ids)

        # 5. audio features (별도 처리)
        started_at = time.perf_counter()
        audio_inputs = feature_extractor(
            audios,
            sampling_rate=feature_extractor.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if self.timing_profile_enabled:
            timing["feature_sec"] = time.perf_counter() - started_at

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "input_features": audio_inputs.input_features,
            "feature_attention_mask": audio_inputs.attention_mask,
            "labels": labels,
            "_timing": timing,
        }

    def __call__(self, batch: List[dict]) -> dict:
        collate_started_at = time.perf_counter()
        per_sample = [self._process_single(item) for item in batch]

        pad_token_id = self.processor.tokenizer.pad_token_id

        # Qwen2Audio는 left-padding이 기본값이다.
        # _merge_input_ids_with_audio_features가 padding 방향을 attention_mask로 감지하므로,
        # 반드시 left-padding을 적용해야 legacy 경로의 확장이 정상 작동한다.
        max_len = max(s["input_ids"].shape[0] for s in per_sample)
        pad_started_at = time.perf_counter()

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for s in per_sample:
            cur_len = s["input_ids"].shape[0]
            pad_len = max_len - cur_len

            padded_input_ids.append(
                torch.cat([
                    torch.full((pad_len,), pad_token_id, dtype=s["input_ids"].dtype),
                    s["input_ids"],
                ])
            )
            padded_attention_mask.append(
                torch.cat([
                    torch.zeros(pad_len, dtype=s["attention_mask"].dtype),
                    s["attention_mask"],
                ])
            )
            padded_labels.append(
                torch.cat([
                    torch.full((pad_len,), IGNORE_INDEX, dtype=s["labels"].dtype),
                    s["labels"],
                ])
            )

        input_ids = torch.stack(padded_input_ids)
        attention_mask = torch.stack(padded_attention_mask)
        labels = torch.stack(padded_labels)

        # audio features는 processor와 동일하게 batch 전체 audio를 flat하게 둔다.
        # mixed:     B개 audio -> (B, 128, 3000)
        # separated: 2B개 audio -> (2B, 128, 3000)
        input_features = torch.cat([s["input_features"] for s in per_sample], dim=0)
        feature_attention_mask = torch.cat(
            [s["feature_attention_mask"] for s in per_sample],
            dim=0,
        )

        audio_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        num_audio_tokens = int((input_ids == audio_token_id).sum().item())
        if num_audio_tokens != input_features.shape[0]:
            raise ValueError(
                "audio token 수와 input_features 수가 일치하지 않습니다: "
                f"tokens={num_audio_tokens}, features={input_features.shape[0]}"
            )

        if self.timing_profile_enabled:
            self._record_timing(
                per_sample=per_sample,
                pad_sec=time.perf_counter() - pad_started_at,
                collate_sec=time.perf_counter() - collate_started_at,
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "labels": labels,
        }

    def _record_timing(
        self,
        per_sample: List[dict],
        pad_sec: float,
        collate_sec: float,
    ) -> None:
        """Accumulate and periodically log collator timing."""
        self._collate_calls += 1
        for sample in per_sample:
            timing = sample.get("_timing", {})
            self._timing_totals["template_sec"] += float(timing.get("template_sec", 0.0))
            self._timing_totals["tokenize_sec"] += float(timing.get("tokenize_sec", 0.0))
            self._timing_totals["feature_sec"] += float(timing.get("feature_sec", 0.0))
            self._timing_audio_count += int(timing.get("audio_count", 0))
        self._timing_totals["pad_sec"] += pad_sec
        self._timing_totals["collate_sec"] += collate_sec

        if self.timing_log_every_n <= 0:
            return
        if self._collate_calls % self.timing_log_every_n != 0:
            return

        calls = max(self._collate_calls, 1)
        audios = max(self._timing_audio_count, 1)
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else "main"
        logger.info(
            "[collator] worker=%s calls=%d audios=%d "
            "template_avg_ms=%.2f tokenize_avg_ms=%.2f "
            "feature_avg_ms_per_audio=%.2f pad_avg_ms=%.2f collate_avg_ms=%.2f",
            worker_id,
            self._collate_calls,
            self._timing_audio_count,
            self._timing_totals["template_sec"] / calls * 1000.0,
            self._timing_totals["tokenize_sec"] / calls * 1000.0,
            self._timing_totals["feature_sec"] / audios * 1000.0,
            self._timing_totals["pad_sec"] / calls * 1000.0,
            self._timing_totals["collate_sec"] / calls * 1000.0,
        )


class BalancedQwen2Trainer(Trainer):
    """
    `WeightedRandomSampler`로 클래스 균형 샘플링을 수행하는 Trainer.

    HuBERT+BERT 파이프라인의 `create_balanced_sampler`와 **동일한 함수**를
    `label_key="label"`로 호출해 Qwen2AudioTrainDataset에 적용한다.

    HF Trainer + accelerate 조합에서 DDP 환경이면 accelerate가
    `BatchSamplerShard`로 sampler 시퀀스를 rank 단위로 자동 분할한다.
    따라서 여기서는 rank 슬라이싱을 하지 않고, 모든 rank가 동일 seed로
    동일한 full 시퀀스를 만들도록만 보장한다. 단일 디바이스에서도 같은 경로로
    동작한다. `set_epoch`은 `DataLoaderShard.set_epoch`을 통해 전파된다.
    """

    def _get_train_sampler(self, train_dataset=None) -> Optional[Sampler]:
        train_dataset = train_dataset if train_dataset is not None else self.train_dataset
        if train_dataset is None:
            return None

        base_seed = int(getattr(self.args, "seed", 42) or 42)
        dataset_size = len(train_dataset)

        # balanced_num_samples_per_epoch가 지정되면 그 값을 epoch당 sample 수로 사용.
        # SWBD처럼 train이 2.7M으로 너무 크면 자연 크기 그대로 가면 학습이 며칠 걸림.
        # cap만 두고 epoch 수는 유지 (다양한 epoch reseed로 다른 ALONE subset 노출).
        cap = getattr(self, "_balanced_num_samples_per_epoch", None)
        if cap is None or cap <= 0:
            total_size = dataset_size
        else:
            total_size = min(int(cap), dataset_size)

        def make_sampler(seed: int, epoch: int):
            return create_balanced_sampler(
                dataset=train_dataset,
                replacement=True,
                num_samples=total_size,
                seed=seed,
                epoch=epoch,
                label_key="label",
            )

        sampler = ReseedableSampler(
            make_sampler=make_sampler,
            total_size=total_size,
            base_seed=base_seed,
        )
        logger.info(
            f"[BalancedQwen2Trainer] balanced sampler 생성 "
            f"(epoch_samples={total_size}, dataset_size={dataset_size}, "
            f"base_seed={base_seed}) - accelerate가 rank 분할 자동 처리"
        )
        return sampler


def setup_training(
    adapter,
    dataset,
    val_dataset,
    prompt_builder,
    lora_config_dict: dict,
    training_config: dict,
    fewshot_selector=None,
    train_variants_per_target: int = 1,
    eval_variants_per_target: int = 1,
):
    """
    LoRA 학습 환경을 구성한다.

    Args:
        adapter: Qwen2AudioAdapter (model + processor 로드 완료 상태)
        dataset: Qwen2AudioDataset (train)
        val_dataset: Qwen2AudioDataset (valid), None이면 평가 건너뜀
        prompt_builder: QwenPromptBuilder
        lora_config_dict: QwenLoraConfigBuilder.build() 반환값
        training_config: 학습 설정 dict
        fewshot_selector: dynamic few-shot support selector (None이면 비활성)
        train_variants_per_target: 학습 target당 support variant 수
        eval_variants_per_target: 검증 target당 support variant 수

    Returns:
        HuggingFace Trainer 인스턴스
    """
    # 학습 시 KV cache 비활성화 (gradient_checkpointing과 호환 + 메모리 절약)
    adapter.model.config.use_cache = False

    # gradient checkpointing 활성화 (메모리 대폭 절약, 큰 batch 가능)
    if training_config.get("gradient_checkpointing", True):
        adapter.model.gradient_checkpointing_enable()
        # LoRA는 base model parameter가 frozen이므로 input에 requires_grad 명시 필요
        if hasattr(adapter.model, "enable_input_require_grads"):
            adapter.model.enable_input_require_grads()
        logger.info("gradient_checkpointing 활성화")

    # LoRA 적용
    adapter.apply_lora(lora_config_dict)

    # 학습용 데이터셋
    max_train = training_config.get("max_train_samples")
    max_val = training_config.get("max_val_samples")
    audio_cache_config = training_config.get("audio_cache", {})
    timing_profile_config = training_config.get("timing_profile", {})

    crop_silence_kwargs = dict(
        crop_speaker_sec=getattr(dataset, "crop_speaker_sec", None),
        crop_listener_sec=getattr(dataset, "crop_listener_sec", None),
        silence_speaker=getattr(dataset, "silence_speaker", False),
        silence_listener=getattr(dataset, "silence_listener", False),
    )

    train_dataset = Qwen2AudioTrainDataset(
        samples=dataset.samples,
        prompt_builder=prompt_builder,
        max_samples=max_train,
        fewshot_selector=fewshot_selector,
        variants_per_target=train_variants_per_target,
        audio_cache_config=audio_cache_config,
        timing_profile_config=timing_profile_config,
        **crop_silence_kwargs,
    )

    val_torch_dataset = None
    if val_dataset is not None:
        val_torch_dataset = Qwen2AudioTrainDataset(
            samples=val_dataset.samples,
            prompt_builder=prompt_builder,
            max_samples=max_val,
            fewshot_selector=fewshot_selector,
            variants_per_target=eval_variants_per_target,
            audio_cache_config=audio_cache_config,
            timing_profile_config=timing_profile_config,
            **crop_silence_kwargs,
        )

    # collator
    use_chat_template = training_config.get("use_chat_template", True)
    collator = Qwen2AudioTrainCollator(
        processor=adapter.processor,
        use_chat_template=use_chat_template,
        timing_profile_config=timing_profile_config,
    )

    # 학습 인자
    output_dir = training_config.get("output_dir", "checkpoints/qwen2_audio_lora")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.get("num_epochs", 3),
        per_device_train_batch_size=training_config.get("batch_size", 2),
        per_device_eval_batch_size=training_config.get("eval_batch_size", 2),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        learning_rate=training_config.get("learning_rate", 2e-4),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        lr_scheduler_type=training_config.get("lr_scheduler", "cosine"),
        fp16=training_config.get("fp16", True),
        bf16=training_config.get("bf16", False),
        logging_steps=training_config.get("logging_steps", 10),
        save_strategy=training_config.get("save_strategy", "epoch"),
        eval_strategy=training_config.get("eval_strategy", "epoch") if val_torch_dataset else "no",
        save_total_limit=training_config.get("save_total_limit", 3),
        load_best_model_at_end=bool(val_torch_dataset),
        metric_for_best_model="eval_loss" if val_torch_dataset else None,
        greater_is_better=False if val_torch_dataset else None,
        report_to=training_config.get("report_to", "none"),
        dataloader_num_workers=training_config.get("num_workers", 4),
        dataloader_pin_memory=training_config.get("pin_memory", True),
        dataloader_persistent_workers=training_config.get("persistent_workers", True),
        dataloader_prefetch_factor=training_config.get("prefetch_factor", 2),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # DDP 관련
        ddp_find_unused_parameters=training_config.get("ddp_find_unused_parameters", False),
        ddp_bucket_cap_mb=training_config.get("ddp_bucket_cap_mb", 25),
        seed=training_config.get("seed", 42),
        remove_unused_columns=False,
    )

    trainer_cls = (
        BalancedQwen2Trainer
        if training_config.get("balanced_sampling", False)
        else Trainer
    )
    if trainer_cls is BalancedQwen2Trainer:
        logger.info("balanced_sampling 활성화 → BalancedQwen2Trainer 사용")

    trainer = trainer_cls(
        model=adapter.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_torch_dataset,
        data_collator=collator,
    )

    # SWBD 등 dataset이 매우 큰 경우 epoch당 sample 수 cap (balanced sampler에만 적용)
    if trainer_cls is BalancedQwen2Trainer:
        cap = training_config.get("balanced_num_samples_per_epoch")
        if cap:
            trainer._balanced_num_samples_per_epoch = int(cap)
            logger.info(
                f"balanced_num_samples_per_epoch={cap} 설정 — epoch당 sample 수 capped"
            )

    return trainer


def save_lora_weights(adapter, output_dir: Union[str, Path]):
    """LoRA 가중치만 저장한다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter.model.save_pretrained(str(output_dir))
    adapter.processor.save_pretrained(str(output_dir))

    logger.info(f"LoRA 가중치 저장 완료: {output_dir}")
