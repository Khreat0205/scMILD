"""
Evaluation metrics for scMILD.

분류 성능 평가를 위한 메트릭 함수들입니다.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve
)


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    AUC 계산.

    Args:
        y_true: True labels
        y_pred: Predicted probabilities for positive class

    Returns:
        AUC score
    """
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # All samples are from one class
        return 0.5


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    다양한 분류 메트릭을 계산합니다.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        'auc': compute_auc(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    method: str = "youden"
) -> Tuple[float, Dict[str, float]]:
    """
    최적의 분류 임계값을 찾습니다.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        method: Method for finding threshold
            - "youden": Maximize Youden's J statistic (sensitivity + specificity - 1)
            - "f1": Maximize F1 score

    Returns:
        optimal_threshold: Best threshold value
        metrics_at_threshold: Metrics computed at this threshold
    """
    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        # Youden's J = sensitivity + specificity - 1 = tpr - fpr
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[best_idx]

    elif method == "f1":
        best_f1 = 0
        optimal_threshold = 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_pred_proba >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                optimal_threshold = thresh

    else:
        raise ValueError(f"Unknown method: {method}")

    metrics_at_threshold = compute_metrics(y_true, y_pred_proba, optimal_threshold)

    return optimal_threshold, metrics_at_threshold


def aggregate_fold_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    여러 fold의 메트릭을 집계합니다.

    Args:
        fold_metrics: List of metric dictionaries from each fold

    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    if not fold_metrics:
        return {}

    # Get all metric names
    metric_names = fold_metrics[0].keys()

    aggregated = {}
    for name in metric_names:
        values = [m[name] for m in fold_metrics if name in m]
        if values:
            aggregated[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }

    return aggregated


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """메트릭을 보기 좋게 출력합니다."""
    print(f"\n{title}")
    print("-" * 40)
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")
    print("-" * 40)


def print_aggregated_metrics(aggregated: Dict[str, Dict[str, float]], title: str = "Aggregated Metrics"):
    """집계된 메트릭을 출력합니다."""
    print(f"\n{title}")
    print("=" * 50)
    for name, stats in aggregated.items():
        if 'mean' in stats:
            print(f"  {name}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                  f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})")
    print("=" * 50)


class MetricsLogger:
    """
    학습 중 메트릭을 로깅하는 클래스.

    CSV 파일로 결과를 저장합니다.
    """

    def __init__(self, log_path: str):
        """
        Args:
            log_path: Path to save CSV log
        """
        self.log_path = log_path
        self.records = []

    def log(self, fold: int, epoch: int, metrics: Dict[str, float], phase: str = "train"):
        """
        메트릭을 기록합니다.

        Args:
            fold: Fold index
            epoch: Epoch number
            metrics: Dictionary of metrics
            phase: "train" or "test"
        """
        record = {
            'fold': fold,
            'epoch': epoch,
            'phase': phase,
            **metrics
        }
        self.records.append(record)

    def log_fold_result(self, fold: int, metrics: Dict[str, float], sample_name: Optional[str] = None):
        """
        Fold 결과를 기록합니다.

        Args:
            fold: Fold index
            metrics: Dictionary of metrics
            sample_name: Name of test sample (for LOOCV)
        """
        record = {
            'fold': fold,
            'test_sample': sample_name,
            **metrics
        }
        self.records.append(record)

    def save(self):
        """로그를 CSV로 저장합니다."""
        import pandas as pd
        df = pd.DataFrame(self.records)
        df.to_csv(self.log_path, index=False)
        print(f"Metrics saved to {self.log_path}")

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """전체 결과 요약을 반환합니다."""
        import pandas as pd
        df = pd.DataFrame(self.records)

        # Filter to final results (not epoch-level)
        final_df = df[df['epoch'].isna()] if 'epoch' in df.columns else df

        summary = {}
        for col in final_df.select_dtypes(include=[np.number]).columns:
            if col not in ['fold', 'epoch']:
                summary[col] = {
                    'mean': final_df[col].mean(),
                    'std': final_df[col].std(),
                    'min': final_df[col].min(),
                    'max': final_df[col].max()
                }

        return summary
