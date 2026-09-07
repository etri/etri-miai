"""
백채널 검출을 위한 샘플링 유틸리티 모듈

이 모듈은 클래스 불균형 문제를 해결하기 위한
가중치 계산과 샘플러 구성을 담당합니다.
"""

import logging
from typing import Callable, Dict, Iterator, List, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler

logger = logging.getLogger(__name__)


class ReseedableSampler(Sampler[int]):
    """
    매 epoch마다 내부 샘플러를 재생성해 epoch별로 다른 시퀀스를 내는 래퍼.

    HF `Trainer` + accelerate 조합에서 DDP 환경이면 accelerate가
    `BatchSamplerShard`로 sampler 시퀀스를 rank 단위로 자동 분할한다. 따라서
    우리는 rank 슬라이싱을 하지 않고, 모든 rank가 `seed + epoch`로 동일한
    full 시퀀스를 만들도록만 보장하면 된다.

    `DataLoaderShard.set_epoch`이 (1) `batch_sampler.sampler.set_epoch` 또는
    (2) `batch_sampler.batch_sampler.sampler.set_epoch`으로 전파되므로
    이 클래스의 `set_epoch`은 HF Trainer 루프에서 자동으로 호출된다.

    Args:
        make_sampler: `(seed: int, epoch: int) -> Sampler[int]` 팩토리.
            매 epoch 새로운 underlying sampler를 빌드할 때 호출된다.
        total_size: 내부 샘플러가 한 번에 뽑는 인덱스 개수 (보통 `len(dataset)`).
        base_seed: 기본 시드. 매 epoch의 실제 시드 = `base_seed + epoch`.
    """

    def __init__(
        self,
        make_sampler: Callable[[int, int], Sampler],
        total_size: int,
        base_seed: int = 0,
    ) -> None:
        super().__init__(data_source=None)
        self.make_sampler = make_sampler
        self.total_size = int(total_size)
        self.base_seed = int(base_seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        underlying = self.make_sampler(self.base_seed, self.epoch)
        return iter(underlying)

    def __len__(self) -> int:
        return self.total_size


def compute_class_weights(
    freqs: Union[Dict[str, int], List[int], np.ndarray],
    method: str = "balanced",
    beta: float = 0.9999,
    min_weight: float = 0.1,
    max_weight: float = 10.0,
) -> Dict[str, float]:
    """
    클래스 빈도에 기반하여 가중치를 계산합니다.

    Args:
        freqs: 클래스별 빈도 (딕셔너리, 리스트, 또는 numpy 배열)
        method: 가중치 계산 방법
            - "balanced": sklearn의 balanced 가중치
            - "inverse": 역수 기반 가중치
            - "focal": focal loss 스타일 가중치
            - "effective_samples": effective samples 기반 가중치
        beta: effective samples 계산에서 사용되는 베타 값
        min_weight: 최소 가중치 제한
        max_weight: 최대 가중치 제한

    Returns:
        클래스별 가중치 딕셔너리

    Raises:
        ValueError: 지원하지 않는 method 또는 잘못된 입력
    """
    # 입력 정규화
    if isinstance(freqs, dict):
        class_names = list(freqs.keys())
        frequencies = np.array(list(freqs.values()))
    elif isinstance(freqs, (list, np.ndarray)):
        frequencies = np.array(freqs)
        class_names = [f"class_{i}" for i in range(len(frequencies))]
    else:
        raise ValueError(f"지원하지 않는 freqs 형식: {type(freqs)}")

    if len(frequencies) == 0:
        raise ValueError("빈도 배열이 비어있습니다")

    if np.any(frequencies < 0):
        raise ValueError("빈도는 음수일 수 없습니다")

    # 가중치 계산
    if method == "balanced":
        weights = _compute_balanced_weights(frequencies)
    elif method == "inverse":
        weights = _compute_inverse_weights(frequencies)
    elif method == "focal":
        weights = _compute_focal_weights(frequencies, beta)
    elif method == "effective_samples":
        weights = _compute_effective_samples_weights(frequencies, beta)
    else:
        raise ValueError(f"지원하지 않는 method: {method}")

    # 가중치 범위 제한
    weights = np.clip(weights, min_weight, max_weight)

    # 딕셔너리로 변환
    weight_dict = {class_names[i]: float(weights[i]) for i in range(len(class_names))}

    logger.info(f"클래스 가중치 계산 완료 (method: {method})")
    for class_name, weight in weight_dict.items():
        logger.debug(f"  {class_name}: {weight:.4f}")

    return weight_dict


def _compute_balanced_weights(frequencies: np.ndarray) -> np.ndarray:
    """sklearn의 balanced 가중치 계산"""
    total_samples = np.sum(frequencies)
    weights = total_samples / (len(frequencies) * frequencies)
    return weights


def _compute_inverse_weights(frequencies: np.ndarray) -> np.ndarray:
    """역수 기반 가중치 계산"""
    # 0으로 나누기 방지
    safe_freqs = np.maximum(frequencies, 1)
    weights = 1.0 / safe_freqs
    return weights


def _compute_focal_weights(frequencies: np.ndarray, beta: float) -> np.ndarray:
    """focal loss 스타일 가중치 계산"""
    total_samples = np.sum(frequencies)
    p = frequencies / total_samples
    weights = (1 - p) ** beta
    return weights


def _compute_effective_samples_weights(
    frequencies: np.ndarray, beta: float
) -> np.ndarray:
    """effective samples 기반 가중치 계산"""
    total_samples = np.sum(frequencies)
    effective_samples = (1 - beta**frequencies) / (1 - beta)
    weights = total_samples / effective_samples
    return weights


def build_weighted_sampler(
    dataset: Dataset,
    weights: Union[Dict[str, float], List[float], np.ndarray],
    class_labels: Optional[List[str]] = None,
    replacement: bool = True,
    num_samples: Optional[int] = None,
    seed: Optional[int] = None,
    epoch: Optional[int] = None,
    label_key: str = "mapped_label",
) -> WeightedRandomSampler:
    """
    가중치 기반 랜덤 샘플러를 구성합니다.

    Args:
        dataset: 데이터셋
        weights: 클래스별 가중치
        class_labels: 데이터셋의 클래스 레이블 리스트 (None인 경우 자동 추정)
        replacement: 복원 추출 여부
        num_samples: 샘플 수 (None인 경우 데이터셋 크기와 동일)
        label_key: `dataset.samples[i]`에서 클래스 레이블을 읽을 키 이름
            (기본 `"mapped_label"`; Qwen2 계열처럼 다른 키를 쓰는 dataset에서는 `"label"` 등)

    Returns:
        가중치 기반 랜덤 샘플러

    Raises:
        ValueError: 가중치와 클래스 레이블 불일치
        RuntimeError: 데이터셋에서 클래스 레이블 추출 실패
    """
    # 클래스 레이블 추출
    if class_labels is None:
        class_labels = _extract_class_labels(dataset)

    if len(class_labels) != len(weights):
        raise ValueError(
            f"가중치 수({len(weights)})와 클래스 수({len(class_labels)})가 일치하지 않습니다"
        )

    # 샘플별 가중치 생성
    sample_weights = _create_sample_weights(dataset, weights, class_labels, label_key)

    # 샘플 수 결정
    if num_samples is None:
        num_samples = len(dataset)

    # generator 생성
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed + epoch)

    # WeightedRandomSampler 생성
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=replacement,
        generator=g,
    )

    logger.info(
        f"가중치 기반 샘플러 생성 완료 (samples: {num_samples}, replacement: {replacement})"
    )

    return sampler


def _extract_class_labels(dataset: Dataset) -> List[str]:
    """데이터셋에서 클래스 레이블을 추출합니다."""
    try:
        # 데이터셋에 get_class_labels 메서드가 있는 경우
        if hasattr(dataset, "get_class_labels"):
            return dataset.get_class_labels()

        raise RuntimeError("데이터셋에서 클래스 레이블을 추출할 수 없습니다")

    except Exception as e:
        raise RuntimeError(f"클래스 레이블 추출 실패: {e}")


def _create_sample_weights(
    dataset: Dataset,
    weights: Union[Dict[str, float], List[float], np.ndarray],
    class_labels: List[str],
    label_key: str = "mapped_label",
) -> List[float]:
    """샘플별 가중치를 생성합니다."""
    # 가중치를 리스트로 변환
    if isinstance(weights, dict):
        weight_list = [weights.get(cls, 1.0) for cls in class_labels]
    else:
        weight_list = list(weights)

    # 클래스별 가중치 매핑
    class_to_weight = {cls: weight for cls, weight in zip(class_labels, weight_list)}

    # 샘플별 가중치 생성
    sample_weights = []
    for i in range(len(dataset)):
        try:
            sample = dataset.samples[i]
            label_str = str(sample[label_key])

            # 레이블에 해당하는 가중치 찾기
            weight = class_to_weight.get(label_str, 1.0)
            sample_weights.append(weight)

        except Exception as e:
            logger.warning(f"샘플 {i}의 가중치 계산 실패, 기본값 1.0 사용: {e}")
            sample_weights.append(1.0)

    return sample_weights


def compute_class_frequencies(
    dataset: Dataset,
    class_labels: Optional[List[str]] = None,
    label_key: str = "mapped_label",
) -> Dict[str, int]:
    """
    데이터셋의 클래스별 빈도를 계산합니다.

    Args:
        dataset: 데이터셋
        class_labels: 클래스 레이블 리스트 (None인 경우 자동 추정)
        label_key: `dataset.samples[i]`에서 클래스 레이블을 읽을 키 이름
            (기본 `"mapped_label"`)

    Returns:
        클래스별 빈도 딕셔너리
    """
    if class_labels is None:
        class_labels = _extract_class_labels(dataset)

    frequencies = {cls: 0 for cls in class_labels}

    for i in range(len(dataset)):
        try:
            sample = dataset.samples[i]
            label_str = str(sample[label_key])

            if label_str in frequencies:
                frequencies[label_str] += 1
            else:
                frequencies[label_str] = 1

        except Exception as e:
            logger.warning(f"샘플 {i}의 레이블 추출 실패: {e}")

    # 빈도가 0인 클래스는 1로 최소 1로 보정
    frequencies = {cls: max(freq, 1) for cls, freq in frequencies.items()}

    logger.info("클래스 빈도 계산 완료:")
    for cls, freq in frequencies.items():
        logger.info(f"  {cls}: {freq}")

    return frequencies


def create_balanced_sampler(
    dataset: Dataset,
    class_labels: Optional[List[str]] = None,
    replacement: bool = True,
    num_samples: Optional[int] = None,
    seed: Optional[int] = None,
    epoch: Optional[int] = None,
    label_key: str = "mapped_label",
) -> WeightedRandomSampler:
    """
    균형 잡힌 샘플러를 생성합니다 (편의 함수).

    Args:
        dataset: 데이터셋
        class_labels: 클래스 레이블 리스트
        replacement: 복원 추출 여부
        num_samples: 샘플 수
        label_key: `dataset.samples[i]`에서 클래스 레이블을 읽을 키 이름
            (기본 `"mapped_label"`)

    Returns:
        균형 잡힌 가중치 샘플러
    """
    frequencies = compute_class_frequencies(dataset, class_labels, label_key=label_key)
    weights = compute_class_weights(frequencies, method="balanced")

    return build_weighted_sampler(
        dataset=dataset,
        weights=weights,
        class_labels=class_labels,
        replacement=replacement,
        num_samples=num_samples,
        seed=seed,
        epoch=epoch,
        label_key=label_key,
    )
