"""
백채널 검출을 위한 평가 메트릭 모듈

이 모듈은 분류 모델의 성능을 평가하기 위한
다양한 메트릭을 제공합니다.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
    roc_auc_score
)

logger = logging.getLogger(__name__)


def compute_accuracy(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    average: str = "micro"
) -> float:
    """
    정확도를 계산합니다.
    
    Args:
        predictions: 예측값 (클래스 인덱스 또는 확률)
        targets: 실제값 (클래스 인덱스)
        average: 평균 방법 ("micro", "macro", "weighted")
        
    Returns:
        정확도 값
    """
    # TODO: 구현 필요
    logger.warning("compute_accuracy 함수가 아직 구현되지 않았습니다")
    return 0.0


def compute_precision(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    average: str = "macro",
    zero_division: str = "warn"
) -> Union[float, np.ndarray]:
    """
    정밀도를 계산합니다.
    
    Args:
        predictions: 예측값 (클래스 인덱스 또는 확률)
        targets: 실제값 (클래스 인덱스)
        average: 평균 방법 ("micro", "macro", "weighted", "none")
        zero_division: 0으로 나누기 처리 방법 ("warn", "0", "1")
        
    Returns:
        정밀도 값 또는 클래스별 정밀도 배열
    """
    # TODO: 구현 필요
    logger.warning("compute_precision 함수가 아직 구현되지 않았습니다")
    return 0.0


def compute_recall(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    average: str = "macro",
    zero_division: str = "warn"
) -> Union[float, np.ndarray]:
    """
    재현율을 계산합니다.
    
    Args:
        predictions: 예측값 (클래스 인덱스 또는 확률)
        targets: 실제값 (클래스 인덱스)
        average: 평균 방법 ("micro", "macro", "weighted", "none")
        zero_division: 0으로 나누기 처리 방법 ("warn", "0", "1")
        
    Returns:
        재현율 값 또는 클래스별 재현율 배열
    """
    # TODO: 구현 필요
    logger.warning("compute_recall 함수가 아직 구현되지 않았습니다")
    return 0.0


def compute_f1_score(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    average: str = "macro",
    zero_division: str = "warn"
) -> Union[float, np.ndarray]:
    """
    F1 점수를 계산합니다.
    
    Args:
        predictions: 예측값 (클래스 인덱스 또는 확률)
        targets: 실제값 (클래스 인덱스)
        average: 평균 방법 ("micro", "macro", "weighted", "none")
        zero_division: 0으로 나누기 처리 방법 ("warn", "0", "1")
        
    Returns:
        F1 점수 또는 클래스별 F1 점수 배열
    """
    # TODO: 구현 필요
    logger.warning("compute_f1_score 함수가 아직 구현되지 않았습니다")
    return 0.0


def compute_macro_f1(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    zero_division: str = "warn"
) -> float:
    """
    Macro F1 점수를 계산합니다.
    
    Args:
        predictions: 예측값 (클래스 인덱스 또는 확률)
        targets: 실제값 (클래스 인덱스)
        zero_division: 0으로 나누기 처리 방법
        
    Returns:
        Macro F1 점수
    """
    return compute_f1_score(predictions, targets, average="macro", zero_division=zero_division)


def compute_micro_f1(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    zero_division: str = "warn"
) -> float:
    """
    Micro F1 점수를 계산합니다.
    
    Args:
        predictions: 예측값 (클래스 인덱스 또는 확률)
        targets: 실제값 (클래스 인덱스)
        zero_division: 0으로 나누기 처리 방법
        
    Returns:
        Micro F1 점수
    """
    return compute_f1_score(predictions, targets, average="micro", zero_division=zero_division)


def compute_per_class_f1(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    class_names: Optional[List[str]] = None,
    zero_division: str = "warn"
) -> Dict[str, float]:
    """
    클래스별 F1 점수를 계산합니다.
    
    Args:
        predictions: 예측값 (클래스 인덱스 또는 확률)
        targets: 실제값 (클래스 인덱스)
        class_names: 클래스 이름 리스트
        zero_division: 0으로 나누기 처리 방법
        
    Returns:
        클래스별 F1 점수 딕셔너리
    """
    # TODO: 구현 필요
    logger.warning("compute_per_class_f1 함수가 아직 구현되지 않았습니다")
    return {}


def compute_auprc(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    average: str = "macro"
) -> Union[float, np.ndarray]:
    """
    Average Precision (AUPRC)를 계산합니다.
    
    Args:
        predictions: 예측 확률 (N, C) 또는 클래스 인덱스
        targets: 실제값 (클래스 인덱스)
        average: 평균 방법 ("micro", "macro", "weighted", "none")
        
    Returns:
        AUPRC 값 또는 클래스별 AUPRC 배열
    """
    # TODO: 구현 필요
    logger.warning("compute_auprc 함수가 아직 구현되지 않았습니다")
    return 0.0


def compute_auc(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    average: str = "macro",
    multi_class: str = "ovr"
) -> Union[float, np.ndarray]:
    """
    ROC AUC를 계산합니다.
    
    Args:
        predictions: 예측 확률 (N, C) 또는 클래스 인덱스
        targets: 실제값 (클래스 인덱스)
        average: 평균 방법 ("micro", "macro", "weighted", "none")
        multi_class: 다중 클래스 처리 방법 ("ovr", "ovo")
        
    Returns:
        AUC 값 또는 클래스별 AUC 배열
    """
    # TODO: 구현 필요
    logger.warning("compute_auc 함수가 아직 구현되지 않았습니다")
    return 0.0


def compute_confusion_matrix(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    num_classes: Optional[int] = None,
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    혼동 행렬을 계산합니다.
    
    Args:
        predictions: 예측값 (클래스 인덱스)
        targets: 실제값 (클래스 인덱스)
        num_classes: 클래스 수
        normalize: 정규화 방법 ("true", "pred", "all", None)
        
    Returns:
        혼동 행렬 (C, C)
    """
    # TODO: 구현 필요
    logger.warning("compute_confusion_matrix 함수가 아직 구현되지 않았습니다")
    return np.array([])


def compute_classification_report(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    class_names: Optional[List[str]] = None,
    output_dict: bool = False
) -> Union[str, Dict]:
    """
    분류 성능 보고서를 생성합니다.
    
    Args:
        predictions: 예측값 (클래스 인덱스)
        targets: 실제값 (클래스 인덱스)
        class_names: 클래스 이름 리스트
        output_dict: 딕셔너리 형태로 출력할지 여부
        
    Returns:
        분류 보고서 (문자열 또는 딕셔너리)
    """
    # TODO: 구현 필요
    logger.warning("compute_classification_report 함수가 아직 구현되지 않았습니다")
    return ""


def compute_all_metrics(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    class_names: Optional[List[str]] = None,
    include_auc: bool = True,
    include_auprc: bool = True
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    모든 주요 메트릭을 계산합니다.
    
    Args:
        predictions: 예측값 (클래스 인덱스 또는 확률)
        targets: 실제값 (클래스 인덱스)
        class_names: 클래스 이름 리스트
        include_auc: AUC 계산 포함 여부
        include_auprc: AUPRC 계산 포함 여부
        
    Returns:
        모든 메트릭을 포함한 딕셔너리
    """
    metrics = {}
    
    # 기본 메트릭
    metrics["accuracy"] = compute_accuracy(predictions, targets)
    metrics["macro_f1"] = compute_macro_f1(predictions, targets)
    metrics["micro_f1"] = compute_micro_f1(predictions, targets)
    metrics["per_class_f1"] = compute_per_class_f1(predictions, targets, class_names)
    
    # 고급 메트릭
    if include_auc:
        metrics["auc"] = compute_auc(predictions, targets)
    
    if include_auprc:
        metrics["auprc"] = compute_auprc(predictions, targets)
    
    # 혼동 행렬
    metrics["confusion_matrix"] = compute_confusion_matrix(predictions, targets)
    
    return metrics


def convert_predictions_to_indices(
    predictions: Union[Tensor, np.ndarray]
) -> np.ndarray:
    """
    예측 확률을 클래스 인덱스로 변환합니다.
    
    Args:
        predictions: 예측 확률 (N, C) 또는 이미 클래스 인덱스
        
    Returns:
        클래스 인덱스 배열
    """
    if isinstance(predictions, Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    # 이미 인덱스인 경우
    if predictions.ndim == 1:
        return predictions
    
    # 확률인 경우 argmax 적용
    if predictions.ndim == 2:
        return np.argmax(predictions, axis=1)
    
    raise ValueError(f"지원하지 않는 예측 형식: {predictions.shape}")


def validate_metric_inputs(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    메트릭 계산을 위한 입력값을 검증하고 정규화합니다.
    
    Args:
        predictions: 예측값
        targets: 실제값
        
    Returns:
        정규화된 예측값과 실제값
        
    Raises:
        ValueError: 입력값 검증 실패
    """
    # 텐서를 numpy로 변환
    if isinstance(predictions, Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, Tensor):
        targets = targets.detach().cpu().numpy()
    
    # 1차원으로 변환
    if predictions.ndim > 1:
        # 2차원 이상인 경우 첫 번째 차원을 유지하고 나머지는 평탄화
        predictions = predictions.reshape(predictions.shape[0], -1)
        if predictions.shape[1] > 1:
            # 클래스 확률인 경우 argmax로 변환
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions.flatten()
    else:
        predictions = predictions.flatten()
    
    targets = targets.flatten()
    
    # 길이 검증
    if len(predictions) != len(targets):
        raise ValueError(f"예측값과 실제값의 길이가 일치하지 않습니다: {len(predictions)} != {len(targets)}")
    
    # 타입 검증
    if not np.issubdtype(predictions.dtype, np.integer) and not np.issubdtype(predictions.dtype, np.floating):
        raise ValueError(f"예측값의 데이터 타입이 잘못되었습니다: {predictions.dtype}")
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError(f"실제값은 정수형이어야 합니다: {targets.dtype}")
    
    return predictions, targets


def compute_metrics(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    class_names: Optional[List[str]] = None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    모든 주요 메트릭을 계산합니다.
    
    Args:
        predictions: 예측값 (클래스 인덱스 또는 확률)
        targets: 실제값 (클래스 인덱스)
        class_names: 클래스 이름 리스트
        
    Returns:
        모든 메트릭을 포함한 딕셔너리
    """
    # Convert to numpy arrays
    if isinstance(predictions, Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Convert predictions to indices if they are probabilities
    if predictions.ndim == 2:
        pred_indices = np.argmax(predictions, axis=1)
    else:
        pred_indices = predictions
    
    # Compute basic metrics
    accuracy = accuracy_score(targets, pred_indices)
    precision = precision_score(targets, pred_indices, average='macro', zero_division=0)
    recall = recall_score(targets, pred_indices, average='macro', zero_division=0)
    f1 = f1_score(targets, pred_indices, average='macro', zero_division=0)
    wf1 = f1_score(targets, pred_indices, average='weighted', zero_division=0)
    
    # Per-class metrics
    per_class_precision = precision_score(targets, pred_indices, average=None, zero_division=0)
    per_class_recall = recall_score(targets, pred_indices, average=None, zero_division=0)
    per_class_f1 = f1_score(targets, pred_indices, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(targets, pred_indices)
    
    # Compile results
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "wf1": float(wf1),
        "per_class_precision": per_class_precision.tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": cm.tolist()
    }
    
    # Add class names if provided
    if class_names:
        metrics["class_names"] = class_names
        metrics["per_class_metrics"] = {}
        for i, class_name in enumerate(class_names):
            if i < len(per_class_precision):
                metrics["per_class_metrics"][class_name] = {
                    "precision": float(per_class_precision[i]),
                    "recall": float(per_class_recall[i]),
                    "f1": float(per_class_f1[i])
                }
    
    return metrics


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix"
) -> None:
    """
    혼동 행렬을 시각화합니다.
    
    Args:
        confusion_matrix: 혼동 행렬
        class_names: 클래스 이름 리스트
        output_path: 출력 파일 경로
        title: 그래프 제목
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'}
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_curves(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Precision-Recall Curves"
) -> None:
    """
    PR 곡선을 시각화합니다.
    
    Args:
        predictions: 예측 확률 (N, C)
        targets: 실제 레이블 (N,)
        class_names: 클래스 이름 리스트
        output_path: 출력 파일 경로
        title: 그래프 제목
    """
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        # Binary classification for this class
        y_true_binary = (targets == i).astype(int)
        y_prob = predictions[:, i]
        
        if len(np.unique(y_true_binary)) > 1:  # Check if class exists in data
            precision, recall, _ = precision_recall_curve(y_true_binary, y_prob)
            auc_score = average_precision_score(y_true_binary, y_prob)
            
            plt.plot(
                recall,
                precision,
                linewidth=2,
                label=f"{class_name} (AUC = {auc_score:.3f})"
            )
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Add diagonal line for random classifier
    plt.plot([0, 1], [0.5, 0.5], 'k--', alpha=0.5, label='Random Classifier')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
