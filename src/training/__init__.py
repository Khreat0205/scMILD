"""
scMILD Training module

학습 및 평가를 위한 모듈입니다:
- MILTrainer: MIL 모델 학습기
- AETrainer: 오토인코더 학습기
- Loss functions
"""

from .trainer import MILTrainer, FoldResult, TrainResult
from .trainer_ae import AETrainer, negative_binomial_loss
from .metrics import (
    compute_metrics,
    compute_auc,
    find_optimal_threshold,
    aggregate_fold_metrics,
    print_metrics,
    print_aggregated_metrics,
    MetricsLogger,
)
from .disease_ratio import (
    calculate_disease_ratio_from_dataloader,
    calculate_disease_ratio_from_adata,
    compute_ratio_regularization_loss,
    print_disease_ratio_summary,
)

__all__ = [
    "MILTrainer",
    "FoldResult",
    "TrainResult",
    "AETrainer",
    "negative_binomial_loss",
    "compute_metrics",
    "compute_auc",
    "find_optimal_threshold",
    "aggregate_fold_metrics",
    "print_metrics",
    "print_aggregated_metrics",
    "MetricsLogger",
    # Disease ratio regularization
    "calculate_disease_ratio_from_dataloader",
    "calculate_disease_ratio_from_adata",
    "compute_ratio_regularization_loss",
    "print_disease_ratio_summary",
]
