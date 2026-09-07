"""
Qwen2-Audio 생성형 분류 결과 평가 모듈

텍스트 레이블 기반의 생성형 출력을 기존 프로젝트의 metric 체계(정수 인덱스 기반)로
변환하여 평가한다. 기존 utils/metrics.py의 compute_metrics를 재사용한다.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .constants import CLASS_NAMES

logger = logging.getLogger(__name__)


class Qwen2AudioEvaluator:
    """
    Qwen2-Audio 생성형 분류 결과를 평가한다.

    - 텍스트 레이블 → 정수 인덱스 변환
    - 기존 compute_metrics 호출
    - confusion matrix 시각화
    - classification report 생성
    - invalid parse 결과 분석
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        exclude_invalid: bool = True,
    ):
        """
        Args:
            class_names: 클래스 이름 리스트 (None이면 기본값 사용)
            exclude_invalid: True면 파싱 실패 샘플을 metric 계산에서 제외
        """
        self.class_names = class_names or CLASS_NAMES
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        self.exclude_invalid = exclude_invalid

    def evaluate(self, results: List[dict]) -> dict:
        """
        추론 결과 리스트를 평가한다.

        Args:
            results: inference script의 결과 리스트
                     각 항목: {sample_id, gold_label, raw_output, cleaned_output,
                              parsed_label, valid_parse}

        Returns:
            evaluation dict with metrics, per_class_metrics, confusion_matrix,
            invalid_analysis, classification_report
        """
        from bc_detection.utils.metrics import compute_metrics, plot_confusion_matrix

        # valid/invalid 분리
        valid_results = [r for r in results if r["valid_parse"]]
        invalid_results = [r for r in results if not r["valid_parse"]]

        eval_results = valid_results if self.exclude_invalid else results

        if not eval_results:
            logger.warning("평가할 유효 결과가 없습니다")
            return self._empty_metrics(len(results), len(invalid_results))

        # 텍스트 레이블 → 정수 인덱스 변환
        gold_indices, pred_indices, skipped = self._labels_to_indices(eval_results)

        if len(gold_indices) == 0:
            logger.warning("인덱스 변환 후 유효 샘플이 없습니다")
            return self._empty_metrics(len(results), len(invalid_results))

        gold_array = np.array(gold_indices)
        pred_array = np.array(pred_indices)

        # 기존 metric 체계 호출
        metrics = compute_metrics(
            predictions=pred_array,
            targets=gold_array,
            class_names=self.class_names,
        )

        # invalid 분석
        invalid_analysis = self._analyze_invalid(invalid_results)

        # 전체 통계
        evaluation = {
            "num_total": len(results),
            "num_valid": len(valid_results),
            "num_invalid": len(invalid_results),
            "num_evaluated": len(gold_indices),
            "num_skipped_unknown_label": skipped,
            "invalid_rate": len(invalid_results) / len(results) if results else 0.0,
            "exclude_invalid": self.exclude_invalid,
            # 기존 metric 결과
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["f1"],
            "weighted_f1": metrics["wf1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "per_class_f1": metrics.get("per_class_f1", []),
            "per_class_precision": metrics.get("per_class_precision", []),
            "per_class_recall": metrics.get("per_class_recall", []),
            "per_class_metrics": metrics.get("per_class_metrics", {}),
            "confusion_matrix": metrics["confusion_matrix"],
            "class_names": self.class_names,
            # invalid 분석
            "invalid_analysis": invalid_analysis,
        }

        return evaluation

    def _labels_to_indices(
        self, results: List[dict]
    ) -> Tuple[List[int], List[int], int]:
        """텍스트 레이블을 정수 인덱스로 변환한다."""
        gold_indices = []
        pred_indices = []
        skipped = 0

        for r in results:
            gold = r["gold_label"]
            pred = r["parsed_label"]

            if gold not in self.label_to_idx:
                logger.warning(f"알 수 없는 gold_label 건너뜀: {gold}")
                skipped += 1
                continue

            if pred not in self.label_to_idx:
                # invalid parse인데 exclude_invalid=False인 경우
                skipped += 1
                continue

            gold_indices.append(self.label_to_idx[gold])
            pred_indices.append(self.label_to_idx[pred])

        return gold_indices, pred_indices, skipped

    def _analyze_invalid(self, invalid_results: List[dict]) -> dict:
        """파싱 실패 결과를 분석한다."""
        if not invalid_results:
            return {"count": 0, "examples": []}

        # raw output 패턴 분석
        examples = []
        for r in invalid_results[:10]:  # 최대 10개 예시
            examples.append({
                "sample_id": r["sample_id"],
                "gold_label": r["gold_label"],
                "raw_output": r["raw_output"][:200],  # 너무 길면 자름
                "cleaned_output": r["cleaned_output"][:200],
            })

        return {
            "count": len(invalid_results),
            "examples": examples,
        }

    def _empty_metrics(self, num_total: int, num_invalid: int) -> dict:
        """평가 불가 시 빈 metric을 반환한다."""
        return {
            "num_total": num_total,
            "num_valid": num_total - num_invalid,
            "num_invalid": num_invalid,
            "num_evaluated": 0,
            "num_skipped_unknown_label": 0,
            "invalid_rate": num_invalid / num_total if num_total > 0 else 0.0,
            "exclude_invalid": self.exclude_invalid,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "per_class_f1": [],
            "per_class_precision": [],
            "per_class_recall": [],
            "per_class_metrics": {},
            "confusion_matrix": [],
            "class_names": self.class_names,
            "invalid_analysis": self._analyze_invalid([]),
        }

    def save_evaluation(
        self,
        evaluation: dict,
        output_dir: Union[str, Path],
        split: str = "test",
    ):
        """
        평가 결과를 파일로 저장한다.

        저장되는 파일:
        - evaluation_{split}.json: 전체 평가 결과
        - confusion_matrix_{split}.png: 혼동 행렬 시각화
        - classification_report_{split}.txt: 분류 보고서
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. JSON 저장
        eval_path = output_dir / f"evaluation_{split}.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2)
        logger.info(f"평가 결과 저장: {eval_path}")

        # 2. Confusion matrix 시각화
        cm = evaluation.get("confusion_matrix")
        if cm and len(cm) > 0:
            try:
                from bc_detection.utils.metrics import plot_confusion_matrix

                cm_array = np.array(cm)
                cm_path = str(output_dir / f"confusion_matrix_{split}.png")
                plot_confusion_matrix(
                    cm_array,
                    self.class_names,
                    cm_path,
                    title=f"Qwen2-Audio Backchannel Detection ({split})",
                )
                logger.info(f"혼동 행렬 저장: {cm_path}")
            except Exception as e:
                logger.warning(f"혼동 행렬 시각화 실패: {e}")

        # 3. Classification report 텍스트
        report_path = output_dir / f"classification_report_{split}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self._format_report(evaluation, split))
        logger.info(f"분류 보고서 저장: {report_path}")

    def _format_report(self, evaluation: dict, split: str) -> str:
        """사람이 읽을 수 있는 분류 보고서를 생성한다."""
        lines = [
            "=" * 60,
            f"Qwen2-Audio 백채널 검출 평가 결과 ({split})",
            "=" * 60,
            "",
            f"총 샘플: {evaluation['num_total']}",
            f"유효 파싱: {evaluation['num_valid']}",
            f"파싱 실패: {evaluation['num_invalid']} "
            f"({evaluation['invalid_rate']:.2%})",
            f"평가 대상: {evaluation['num_evaluated']}",
            "",
            "--- 전체 성능 ---",
            f"Accuracy:  {evaluation['accuracy']:.4f}",
            f"Macro F1:  {evaluation['macro_f1']:.4f}",
            f"Precision: {evaluation['precision']:.4f}",
            f"Recall:    {evaluation['recall']:.4f}",
            "",
        ]

        # 클래스별 성능
        per_class = evaluation.get("per_class_metrics", {})
        if per_class:
            lines.append("--- 클래스별 성능 ---")
            lines.append(f"{'Class':<16} {'Precision':>10} {'Recall':>10} {'F1':>10}")
            lines.append("-" * 48)
            for cls_name in self.class_names:
                if cls_name in per_class:
                    m = per_class[cls_name]
                    lines.append(
                        f"{cls_name:<16} {m['precision']:>10.4f} "
                        f"{m['recall']:>10.4f} {m['f1']:>10.4f}"
                    )
            lines.append("")

        # Confusion matrix
        cm = evaluation.get("confusion_matrix")
        if cm and len(cm) > 0:
            lines.append("--- 혼동 행렬 ---")
            header = f"{'':>16}" + "".join(f"{c:>14}" for c in self.class_names)
            lines.append(header)
            for i, row in enumerate(cm):
                row_str = f"{self.class_names[i]:>16}" + "".join(
                    f"{v:>14}" for v in row
                )
                lines.append(row_str)
            lines.append("")

        # Invalid 분석
        inv = evaluation.get("invalid_analysis", {})
        if inv.get("count", 0) > 0:
            lines.append(f"--- 파싱 실패 분석 (상위 {min(10, inv['count'])}개) ---")
            for ex in inv.get("examples", []):
                lines.append(f"  [{ex['sample_id']}]")
                lines.append(f"    gold: {ex['gold_label']}")
                lines.append(f"    raw:  {ex['raw_output']}")
                lines.append(f"    cleaned: {ex['cleaned_output']}")
                lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def print_summary(self, evaluation: dict, split: str = "test"):
        """평가 결과 요약을 로거로 출력한다."""
        logger.info("=" * 60)
        logger.info(f"평가 결과 ({split})")
        logger.info(f"  총 샘플: {evaluation['num_total']}")
        logger.info(f"  유효 파싱: {evaluation['num_valid']} "
                     f"(invalid: {evaluation['invalid_rate']:.2%})")
        logger.info(f"  Accuracy:  {evaluation['accuracy']:.4f}")
        logger.info(f"  Weighted F1: {evaluation['weighted_f1']}")
        logger.info(f"  Macro F1:  {evaluation['macro_f1']:.4f}")
        logger.info(f"  Precision: {evaluation['precision']:.4f}")
        logger.info(f"  Recall:    {evaluation['recall']:.4f}")

        per_class = evaluation.get("per_class_metrics", {})
        if per_class:
            for cls_name in self.class_names:
                if cls_name in per_class:
                    m = per_class[cls_name]
                    logger.info(
                        f"  {cls_name}: P={m['precision']:.4f} "
                        f"R={m['recall']:.4f} F1={m['f1']:.4f}"
                    )

        logger.info("=" * 60)
