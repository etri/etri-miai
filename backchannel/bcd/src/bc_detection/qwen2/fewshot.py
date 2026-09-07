"""
Dynamic few-shot support selection for Qwen2-Audio experiments.

This module is deliberately independent from dataloader worker random state.
Support examples are selected from a stable seed derived from the target sample
and variant id, so multi-worker and persistent-worker dataloaders remain
reproducible.
"""

import hashlib
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from .constants import VALID_LABELS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FewShotConfig:
    """Validated dynamic few-shot configuration."""

    enabled: bool = False
    support_source: str = "train"
    strategy: str = "class_balanced"
    seed: int = 42
    order: str = "shuffle"
    k_per_class: Optional[int] = None
    total_support: Optional[int] = None
    train_variants_per_target: int = 1
    eval_variants_per_target: int = 1
    exclude_self: bool = True

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "FewShotConfig":
        """Build and validate config from a YAML dictionary."""
        if not cfg:
            return cls(enabled=False)

        config = cls(
            enabled=bool(cfg.get("enabled", False)),
            support_source=cfg.get("support_source", "train"),
            strategy=cfg.get("strategy", "class_balanced"),
            seed=int(cfg.get("seed", 42)),
            order=cfg.get("order", "shuffle"),
            k_per_class=cfg.get("k_per_class"),
            total_support=cfg.get("total_support"),
            train_variants_per_target=int(cfg.get("train_variants_per_target", 1)),
            eval_variants_per_target=int(cfg.get("eval_variants_per_target", 1)),
            exclude_self=bool(cfg.get("exclude_self", True)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate config values."""
        if not self.enabled:
            return

        if self.support_source != "train":
            raise ValueError(
                "dynamic_fewshot.support_source는 현재 'train'만 지원합니다: "
                f"{self.support_source}"
            )

        if self.strategy not in ("class_balanced", "random_slots"):
            raise ValueError(
                "dynamic_fewshot.strategy는 'class_balanced' 또는 "
                f"'random_slots'여야 합니다: {self.strategy}"
            )

        if self.order not in ("shuffle", "fixed"):
            raise ValueError(
                "dynamic_fewshot.order는 'shuffle' 또는 'fixed'여야 합니다: "
                f"{self.order}"
            )

        if self.strategy == "class_balanced":
            if self.k_per_class is None or int(self.k_per_class) <= 0:
                raise ValueError(
                    "class_balanced few-shot은 k_per_class > 0이 필요합니다"
                )

        if self.strategy == "random_slots":
            if self.total_support is None or int(self.total_support) <= 0:
                raise ValueError(
                    "random_slots few-shot은 total_support > 0이 필요합니다"
                )

        if self.train_variants_per_target <= 0:
            raise ValueError("train_variants_per_target는 1 이상이어야 합니다")
        if self.eval_variants_per_target <= 0:
            raise ValueError("eval_variants_per_target는 1 이상이어야 합니다")

    def to_dict(self) -> dict:
        """Return JSON/YAML serializable metadata."""
        return {
            "enabled": self.enabled,
            "support_source": self.support_source,
            "strategy": self.strategy,
            "seed": self.seed,
            "order": self.order,
            "k_per_class": self.k_per_class,
            "total_support": self.total_support,
            "train_variants_per_target": self.train_variants_per_target,
            "eval_variants_per_target": self.eval_variants_per_target,
            "exclude_self": self.exclude_self,
        }


class FewShotSelector:
    """Select dynamic few-shot support examples from a support sample pool."""

    def __init__(self, support_samples: List[dict], config: FewShotConfig):
        if not config.enabled:
            raise ValueError("FewShotSelector는 enabled=True config가 필요합니다")

        self.support_samples = list(support_samples)
        self.config = config
        self.by_label: Dict[str, List[dict]] = {label: [] for label in VALID_LABELS}
        for sample in self.support_samples:
            label = str(sample.get("label"))
            self.by_label.setdefault(label, []).append(sample)

        logger.info(
            "FewShotSelector 초기화: strategy=%s, support=%d, labels=%s",
            config.strategy,
            len(self.support_samples),
            {label: len(samples) for label, samples in self.by_label.items()},
        )

    def select(
        self,
        target_sample: dict,
        target_index: int,
        variant_id: int = 0,
    ) -> List[dict]:
        """Select support examples for one target sample."""
        if self.config.strategy == "class_balanced":
            support = self._select_class_balanced(
                target_sample=target_sample,
                target_index=target_index,
                variant_id=variant_id,
            )
        elif self.config.strategy == "random_slots":
            support = self._select_random_slots(
                target_sample=target_sample,
                target_index=target_index,
                variant_id=variant_id,
            )
        else:
            raise ValueError(f"지원하지 않는 few-shot strategy: {self.config.strategy}")

        if self.config.order == "shuffle" and len(support) > 1:
            rng = self._rng(target_sample, target_index, variant_id, "order")
            support = list(support)
            rng.shuffle(support)

        return [self._to_prompt_example(sample) for sample in support]

    def _select_class_balanced(
        self,
        target_sample: dict,
        target_index: int,
        variant_id: int,
    ) -> List[dict]:
        support = []
        k = int(self.config.k_per_class or 0)
        for label in VALID_LABELS:
            candidates = self._filtered_candidates(
                self.by_label.get(label, []),
                target_sample,
            )
            if not candidates:
                continue
            rng = self._rng(target_sample, target_index, variant_id, label)
            support.extend(self._sample_without_replacement(candidates, k, rng))
        return support

    def _select_random_slots(
        self,
        target_sample: dict,
        target_index: int,
        variant_id: int,
    ) -> List[dict]:
        candidates = self._filtered_candidates(self.support_samples, target_sample)
        if not candidates:
            return []
        rng = self._rng(target_sample, target_index, variant_id, "random_slots")
        return self._sample_without_replacement(
            candidates,
            int(self.config.total_support or 0),
            rng,
        )

    def _filtered_candidates(
        self,
        candidates: List[dict],
        target_sample: dict,
    ) -> List[dict]:
        if not self.config.exclude_self:
            return list(candidates)

        target_id = target_sample.get("sample_id")
        return [
            sample
            for sample in candidates
            if sample.get("sample_id") != target_id
        ]

    @staticmethod
    def _sample_without_replacement(
        candidates: List[dict],
        k: int,
        rng: random.Random,
    ) -> List[dict]:
        if k <= 0:
            return []
        if len(candidates) <= k:
            selected = list(candidates)
            rng.shuffle(selected)
            return selected
        return rng.sample(candidates, k)

    def _rng(
        self,
        target_sample: dict,
        target_index: int,
        variant_id: int,
        salt: str,
    ) -> random.Random:
        target_id = str(target_sample.get("sample_id", target_index))
        payload = (
            f"{self.config.seed}|{self.config.strategy}|{target_id}|"
            f"{target_index}|{variant_id}|{salt}"
        )
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(digest, byteorder="big", signed=False)
        return random.Random(seed)

    @staticmethod
    def _to_prompt_example(sample: dict) -> dict:
        return {
            "sample_id": sample.get("sample_id"),
            "audio_paths": [
                sample["speaker_audio_path"],
                sample["listener_audio_path"],
            ]
            if "speaker_audio_path" in sample and "listener_audio_path" in sample
            else [sample["mixed_audio_path"]],
            "speaker_text": sample.get("speaker_text"),
            "label": sample["label"],
        }
