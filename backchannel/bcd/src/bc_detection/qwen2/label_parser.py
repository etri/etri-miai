"""
Qwen2-Audio 생성 출력을 백채널 레이블로 변환하는 파서

raw generation 문자열을 정리하고, 4개 클래스 중 하나로 매핑한다.
QwenGenerationPostprocessor의 책임까지 포함한다.
"""

import logging
import re
from typing import Iterable, Optional, Tuple

from .constants import VALID_LABELS as _VALID_LABELS_LIST

logger = logging.getLogger(__name__)

# Default (BCKO 4-class). 다른 corpus는 QwenLabelParser(valid_labels=...)로 override.
VALID_LABELS = frozenset(_VALID_LABELS_LIST)

# 제거 대상 prefix 키워드
_PREFIX_KEYWORDS = re.compile(
    r"(답변|분류|결과|유형|응답|출력|백채널|반응)\s*[:：]\s*", re.IGNORECASE
)

# 특수 토큰 패턴
_SPECIAL_TOKEN_PATTERN = re.compile(r"<\|.*?\|>")


def _compile_label_pattern(labels: Iterable[str]) -> re.Pattern:
    """주어진 라벨 집합에 맞는 word-boundary regex를 컴파일."""
    sorted_labels = sorted({str(l).upper() for l in labels}, key=len, reverse=True)
    body = "|".join(re.escape(l) for l in sorted_labels)
    return re.compile(rf"\b({body})\b", re.IGNORECASE)


class QwenLabelParser:
    """
    Qwen2-Audio의 생성 출력을 백채널 레이블로 변환한다.

    처리 순서:
    1. 특수 토큰 제거
    2. 불필요한 prefix/suffix 제거
    3. 따옴표, 줄바꿈, 설명문 제거
    4. canonical label로 매핑
    5. invalid output 처리
    """

    def __init__(
        self,
        fallback_label: Optional[str] = None,
        valid_labels: Optional[Iterable[str]] = None,
    ):
        """
        Args:
            fallback_label: 파싱 실패 시 사용할 기본 레이블.
                            None이면 파싱 실패를 그대로 반환.
            valid_labels: corpus 별 레이블 집합 (예: SWBD ["NOBC","CONTINUER","ASSESSMENT"]).
                          None이면 BCKO default (constants.VALID_LABELS).
        """
        labels = list(valid_labels) if valid_labels is not None else list(VALID_LABELS)
        self.valid_labels = frozenset(str(l).upper() for l in labels)
        self._label_pattern = _compile_label_pattern(self.valid_labels)
        if fallback_label is not None and fallback_label.upper() not in self.valid_labels:
            raise ValueError(
                f"fallback_label은 {self.valid_labels} 중 하나여야 합니다: {fallback_label}"
            )
        self.fallback_label = fallback_label

    def parse(self, raw_output: str) -> Tuple[str, str, bool]:
        """
        raw generation 출력을 파싱한다.

        Args:
            raw_output: 모델이 생성한 원본 텍스트

        Returns:
            (cleaned_output, parsed_label, valid_parse)
            - cleaned_output: 정리된 출력 문자열
            - parsed_label: 최종 레이블 (파싱 실패 시 fallback 또는 "INVALID")
            - valid_parse: 파싱 성공 여부
        """
        cleaned = self._clean(raw_output)
        label = self._extract_label(cleaned)

        if label is not None:
            return cleaned, label, True

        # cleaned에서 못 찾으면 원본에서 재시도
        label = self._extract_label(raw_output)
        if label is not None:
            return cleaned, label, True

        # 파싱 실패
        fallback = self.fallback_label if self.fallback_label else "INVALID"
        logger.warning(f"레이블 파싱 실패: raw='{raw_output}' → cleaned='{cleaned}' → {fallback}")
        return cleaned, fallback, False

    def _clean(self, text: str) -> str:
        """생성 출력을 정리한다."""
        # 1. 특수 토큰 제거
        text = _SPECIAL_TOKEN_PATTERN.sub("", text)

        # 2. 공백 정규화
        text = re.sub(r"\s+", " ", text).strip()

        # 3. prefix 키워드 제거 (예: "답변: NOBC" → "NOBC")
        text = _PREFIX_KEYWORDS.sub("", text).strip()

        # 4. 따옴표 제거
        if len(text) >= 2 and text[0] in ('"', "'", "「", "『") and text[-1] in ('"', "'", "」", "』"):
            text = text[1:-1].strip()

        # 5. 줄바꿈이 있으면 첫 번째 줄만 사용
        if "\n" in text:
            text = text.split("\n")[0].strip()

        return text

    def _extract_label(self, text: str) -> Optional[str]:
        """텍스트에서 유효한 레이블을 추출한다."""
        # 정확히 레이블 하나만 있는 경우
        upper = text.strip().upper()
        if upper in self.valid_labels:
            return upper

        # 정규식으로 탐색 (corpus 별 valid_labels 기준)
        match = self._label_pattern.search(text)
        if match:
            return match.group(1).upper()

        return None

    def parse_batch(self, raw_outputs: list) -> list:
        """
        여러 생성 출력을 일괄 파싱한다.

        Returns:
            list of (cleaned_output, parsed_label, valid_parse) tuples
        """
        return [self.parse(raw) for raw in raw_outputs]

    def get_stats(self, results: list) -> dict:
        """
        파싱 결과 통계를 계산한다.

        Args:
            results: parse_batch의 반환값

        Returns:
            dict with num_total, num_valid, num_invalid, invalid_rate, label_distribution
        """
        num_total = len(results)
        num_valid = sum(1 for _, _, valid in results if valid)
        num_invalid = num_total - num_valid

        label_dist = {}
        for _, label, _ in results:
            label_dist[label] = label_dist.get(label, 0) + 1

        return {
            "num_total": num_total,
            "num_valid": num_valid,
            "num_invalid": num_invalid,
            "invalid_rate": num_invalid / num_total if num_total > 0 else 0.0,
            "label_distribution": label_dist,
        }
