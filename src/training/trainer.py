"""
MIL Trainer for scMILD.

Teacher-Student 구조의 Multiple Instance Learning 학습을 담당합니다.
"""

import os
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .metrics import compute_metrics, find_optimal_threshold


@dataclass
class TrainResult:
    """Single epoch training result."""
    loss: float
    metrics: Optional[Dict[str, float]] = None


@dataclass
class FoldResult:
    """Result from training a single fold."""
    fold_idx: int
    test_sample: Optional[str]
    metrics: Dict[str, float]
    best_threshold: float
    model_teacher_state: dict
    model_student_state: dict
    model_encoder_state: dict
    # For overall AUROC calculation (LOOCV)
    y_true: Optional[np.ndarray] = None
    y_pred_proba: Optional[np.ndarray] = None


class MILTrainer:
    """
    MIL Trainer for scMILD.

    Teacher-Student 구조로 샘플 레벨 분류와 세포 레벨 점수를 학습합니다.

    Args:
        model_teacher: Teacher branch model
        model_student: Student branch model
        model_encoder: Encoder wrapper model
        device: Device for training
        use_conditional_ae: Whether encoder is conditional
        student_optimize_period: How often to optimize student (epochs)
        student_loss_weight_neg: Weight for negative class in student loss
        disease_ratio: VQ 코드별 질병 비율 (disease ratio regularization용)
        ratio_reg_lambda: Disease ratio regularization 가중치 (0이면 비활성화)
    """

    def __init__(
        self,
        model_teacher: nn.Module,
        model_student: nn.Module,
        model_encoder: nn.Module,
        device: torch.device,
        use_conditional_ae: bool = True,
        student_optimize_period: int = 3,
        student_loss_weight_neg: float = 0.3,
        disease_ratio: Optional[Dict[int, float]] = None,
        ratio_reg_lambda: float = 0.0,
    ):
        self.model_teacher = model_teacher
        self.model_student = model_student
        self.model_encoder = model_encoder
        self.device = device
        self.use_conditional_ae = use_conditional_ae
        self.student_optimize_period = student_optimize_period
        self.student_loss_weight_neg = student_loss_weight_neg

        # Disease ratio regularization
        self.disease_ratio = disease_ratio
        self.ratio_reg_lambda = ratio_reg_lambda

        # For normalizing attention scores
        self.attn_score_min = 0.0
        self.attn_score_max = 1.0

    def train_fold(
        self,
        train_bag_dl: DataLoader,
        train_instance_dl: DataLoader,
        test_bag_dl: DataLoader,
        n_epochs: int,
        learning_rate: float,
        encoder_learning_rate: float,
        use_early_stopping: bool = False,
        patience: int = 15,
        fold_idx: int = 0,
        test_sample_name: Optional[str] = None,
        skip_fold_metrics: bool = False,
    ) -> FoldResult:
        """
        Train a single fold.

        Args:
            train_bag_dl: Training dataloader (bag/sample level)
            train_instance_dl: Training dataloader (instance/cell level)
            test_bag_dl: Test dataloader (bag/sample level)
            n_epochs: Number of training epochs
            learning_rate: Learning rate for MIL branches
            encoder_learning_rate: Learning rate for encoder projection
            use_early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            fold_idx: Index of current fold
            test_sample_name: Name of test sample (for LOOCV)
            skip_fold_metrics: Skip per-fold metrics calculation (for LOOCV)

        Returns:
            FoldResult with trained models and metrics
        """
        # Setup optimizers
        optimizer_teacher = torch.optim.Adam(
            self.model_teacher.parameters(), lr=learning_rate
        )
        optimizer_student = torch.optim.Adam(
            self.model_student.parameters(), lr=learning_rate
        )

        # Encoder optimizer (only for projection layer if frozen)
        encoder_params = list(self.model_encoder.get_trainable_parameters())
        optimizer_encoder = torch.optim.Adam(
            encoder_params, lr=encoder_learning_rate
        ) if encoder_params else None

        # Best model tracking
        best_loss = float('inf')
        best_threshold = 0.5
        best_model_wts_teacher = None
        best_model_wts_student = None
        best_model_wts_encoder = None
        no_improvement = 0

        # Training loop
        for epoch in range(n_epochs):
            # Train teacher
            train_loss = self._train_teacher_epoch(
                train_bag_dl, optimizer_teacher, optimizer_encoder
            )

            # Train student periodically
            if epoch % self.student_optimize_period == 0:
                self._train_student_epoch(
                    train_instance_dl, optimizer_student, optimizer_encoder
                )

            # Evaluate (LOOCV: no validation, use training loss for monitoring)
            if use_early_stopping:
                eval_loss, eval_auc, threshold = self._evaluate(train_bag_dl)

                if eval_loss < best_loss:
                    best_loss = eval_loss
                    best_threshold = threshold
                    best_model_wts_teacher = copy.deepcopy(self.model_teacher.state_dict())
                    best_model_wts_student = copy.deepcopy(self.model_student.state_dict())
                    best_model_wts_encoder = copy.deepcopy(self.model_encoder.state_dict())
                    no_improvement = 0
                else:
                    no_improvement += 1
                    if no_improvement >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            else:
                # No early stopping - just save last epoch
                best_model_wts_teacher = copy.deepcopy(self.model_teacher.state_dict())
                best_model_wts_student = copy.deepcopy(self.model_student.state_dict())
                best_model_wts_encoder = copy.deepcopy(self.model_encoder.state_dict())

        # Load best models
        if best_model_wts_teacher is not None:
            self.model_teacher.load_state_dict(best_model_wts_teacher)
            self.model_student.load_state_dict(best_model_wts_student)
            self.model_encoder.load_state_dict(best_model_wts_encoder)

        # Get predictions for this fold
        y_true, y_pred_proba = self._get_predictions(test_bag_dl)

        # For LOOCV (skip_fold_metrics=True): skip per-fold metrics calculation
        # Metrics will be calculated on concatenated predictions later
        if skip_fold_metrics:
            metrics = {}  # Empty metrics for LOOCV
            threshold = 0.5  # Default threshold
        else:
            # Full evaluation for non-LOOCV cases
            test_loss, test_auc, threshold = self._evaluate(test_bag_dl)
            metrics = compute_metrics(y_true, y_pred_proba, threshold)

        return FoldResult(
            fold_idx=fold_idx,
            test_sample=test_sample_name,
            metrics=metrics,
            best_threshold=threshold,
            model_teacher_state=best_model_wts_teacher,
            model_student_state=best_model_wts_student,
            model_encoder_state=best_model_wts_encoder,
            y_true=y_true,
            y_pred_proba=y_pred_proba,
        )

    def _train_teacher_epoch(
        self,
        dataloader: DataLoader,
        optimizer_teacher: torch.optim.Optimizer,
        optimizer_encoder: Optional[torch.optim.Optimizer]
    ) -> float:
        """Train teacher for one epoch."""
        self.model_teacher.train()
        self.model_student.eval()

        if optimizer_encoder is not None:
            self.model_encoder.train()
        else:
            self.model_encoder.eval()

        total_loss = 0.0
        total_samples = 0
        all_attn_scores = []

        for batch in dataloader:
            # Unpack batch
            t_data = batch[0].to(self.device)
            t_bagids = batch[1].to(self.device)
            t_labels = batch[2].to(self.device)
            t_study_ids = batch[3].to(self.device) if len(batch) == 4 else None

            # Encode
            if self.use_conditional_ae and t_study_ids is not None:
                feat = self.model_encoder(t_data, t_study_ids)
            else:
                feat = self.model_encoder(t_data)

            # Get unique bags
            inner_ids = t_bagids[-1]
            unique_bags = torch.unique(inner_ids)

            batch_predictions = []
            batch_attn_scores = []

            for bag in unique_bags:
                bag_mask = inner_ids == bag
                bag_instances = feat[bag_mask]

                # Forward through teacher
                bag_pred = self.model_teacher(bag_instances)
                attn_score = self.model_teacher.attention_module(bag_instances)

                batch_predictions.append(bag_pred.unsqueeze(0))
                batch_attn_scores.append(attn_score.squeeze())

            # Stack predictions
            batch_predictions = torch.cat(batch_predictions, dim=0)
            bag_probs = torch.softmax(batch_predictions, dim=1)

            # Binary cross entropy loss
            loss = -1.0 * (
                t_labels * torch.log(bag_probs[:, 1] + 1e-5) +
                (1.0 - t_labels) * torch.log(1.0 - bag_probs[:, 1] + 1e-5)
            )

            total_loss += loss.sum().item()
            total_samples += loss.size(0)
            loss = loss.mean()

            # Disease ratio regularization
            if self.disease_ratio is not None and self.ratio_reg_lambda > 0:
                ratio_reg_loss = self._compute_ratio_regularization(
                    t_data, t_study_ids, batch_attn_scores
                )
                if ratio_reg_loss is not None:
                    loss = loss + self.ratio_reg_lambda * ratio_reg_loss

            # Backward
            optimizer_teacher.zero_grad()
            if optimizer_encoder is not None:
                optimizer_encoder.zero_grad()

            loss.backward()

            optimizer_teacher.step()
            if optimizer_encoder is not None:
                optimizer_encoder.step()

            # Collect attention scores
            all_attn_scores.extend([s.detach() for s in batch_attn_scores])

        # Update attention score normalization parameters
        if all_attn_scores:
            all_scores = torch.cat(all_attn_scores)
            self.attn_score_min = all_scores.min().item()
            self.attn_score_max = all_scores.max().item()

        return total_loss / total_samples

    def _train_student_epoch(
        self,
        dataloader: DataLoader,
        optimizer_student: torch.optim.Optimizer,
        optimizer_encoder: Optional[torch.optim.Optimizer]
    ):
        """Train student for one epoch."""
        self.model_teacher.eval()
        self.model_student.train()

        if optimizer_encoder is not None:
            self.model_encoder.train()
        else:
            self.model_encoder.eval()

        for batch in dataloader:
            # Unpack - InstanceDatasetWithBagLabel format
            if len(batch) == 5:
                t_data, t_bag_ids, t_instance_labels, t_bag_labels, t_study_ids = batch
                t_study_ids = t_study_ids.to(self.device)
            elif len(batch) == 4:
                t_data, t_bag_ids, t_instance_labels, t_bag_labels = batch
                t_study_ids = None
            else:
                continue  # Skip if no bag labels

            t_data = t_data.to(self.device)
            t_instance_labels = t_instance_labels.to(self.device)

            # Encode
            if self.use_conditional_ae and t_study_ids is not None:
                feat = self.model_encoder(t_data, t_study_ids)
            else:
                feat = self.model_encoder(t_data)

            # Get pseudo labels from teacher attention
            with torch.no_grad():
                attn_scores = self.model_teacher.attention_module(feat)
                pseudo_labels = self._normalize_attention(attn_scores).squeeze()
                pseudo_labels = pseudo_labels.clamp(1e-5, 1 - 1e-5)
                # Zero out control cells
                pseudo_labels[t_instance_labels == 0] = 0

            # Student prediction
            instance_preds = self.model_student(feat)
            instance_probs = torch.softmax(instance_preds, dim=1)

            # Weighted BCE loss
            loss = -1.0 * torch.mean(
                self.student_loss_weight_neg * (1 - pseudo_labels) * torch.log(instance_probs[:, 0] + 1e-5) +
                (1 - self.student_loss_weight_neg) * pseudo_labels * torch.log(instance_probs[:, 1] + 1e-5)
            )

            # Backward
            optimizer_student.zero_grad()
            if optimizer_encoder is not None:
                optimizer_encoder.zero_grad()

            loss.backward()

            optimizer_student.step()
            if optimizer_encoder is not None:
                optimizer_encoder.step()

    def _normalize_attention(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """Normalize attention scores to [0, 1]."""
        range_val = self.attn_score_max - self.attn_score_min
        if range_val < 1e-8:
            return torch.zeros_like(attn_scores)
        return (attn_scores - self.attn_score_min) / range_val

    def _compute_ratio_regularization(
        self,
        t_data: torch.Tensor,
        t_study_ids: Optional[torch.Tensor],
        batch_attn_scores: List[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Disease ratio regularization loss 계산.

        VQ 코드북의 각 코드별 질병 비율을 타겟으로 하여
        attention score가 이 비율과 유사해지도록 유도합니다.

        Args:
            t_data: Input data tensor
            t_study_ids: Study IDs (conditional encoder용)
            batch_attn_scores: Batch의 attention scores

        Returns:
            Regularization loss (MSE) 또는 None
        """
        # Get codebook indices
        if not hasattr(self.model_encoder, 'vq_model'):
            return None

        try:
            with torch.no_grad():
                if self.use_conditional_ae and t_study_ids is not None:
                    code_indices = self.model_encoder.vq_model.get_codebook_indices(t_data, t_study_ids)
                else:
                    code_indices = self.model_encoder.vq_model.get_codebook_indices(t_data)
        except Exception:
            return None

        if code_indices is None:
            return None

        # Concatenate all attention scores
        all_attn_scores = torch.cat(batch_attn_scores)

        # Get target ratios for each cell
        target_ratios = torch.tensor(
            [self.disease_ratio.get(int(c), 0.5) for c in code_indices.cpu().numpy()],
            dtype=torch.float32, device=self.device
        )

        # Normalize attention scores to [0, 1]
        attn_min = all_attn_scores.min()
        attn_max = all_attn_scores.max()
        attn_norm = (all_attn_scores - attn_min) / (attn_max - attn_min + 1e-8)

        # MSE loss between normalized attention and target disease ratio
        ratio_reg_loss = F.mse_loss(attn_norm, target_ratios)

        return ratio_reg_loss

    def set_disease_ratio(self, disease_ratio: Dict[int, float], ratio_reg_lambda: float = 0.1):
        """
        Disease ratio 설정 (학습 중 동적 업데이트용).

        Args:
            disease_ratio: {code_id: disease_ratio} mapping
            ratio_reg_lambda: Regularization weight
        """
        self.disease_ratio = disease_ratio
        self.ratio_reg_lambda = ratio_reg_lambda

    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """
        Evaluate model on dataloader.

        Returns:
            loss: Average loss
            auc: AUC score
            threshold: Optimal threshold
        """
        self.model_encoder.eval()
        self.model_teacher.eval()

        y_true, y_pred_proba = self._get_predictions(dataloader)

        # Compute loss
        y_true_t = torch.tensor(y_true, dtype=torch.float32)
        y_pred_t = torch.tensor(y_pred_proba, dtype=torch.float32)
        loss = -1.0 * (
            y_true_t * torch.log(y_pred_t + 1e-5) +
            (1.0 - y_true_t) * torch.log(1.0 - y_pred_t + 1e-5)
        ).mean().item()

        # Compute AUC and threshold
        threshold, metrics = find_optimal_threshold(y_true, y_pred_proba)
        auc = metrics['auc']

        return loss, auc, threshold

    @torch.no_grad()
    def _get_predictions(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions for all samples in dataloader."""
        self.model_encoder.eval()
        self.model_teacher.eval()

        all_labels = []
        all_probs = []

        for batch in dataloader:
            t_data = batch[0].to(self.device)
            t_bagids = batch[1].to(self.device)
            t_labels = batch[2].to(self.device)
            t_study_ids = batch[3].to(self.device) if len(batch) == 4 else None

            # Encode
            if self.use_conditional_ae and t_study_ids is not None:
                feat = self.model_encoder(t_data, t_study_ids)
            else:
                feat = self.model_encoder(t_data)

            # Get unique bags
            inner_ids = t_bagids[-1]
            unique_bags = torch.unique(inner_ids)

            for bag in unique_bags:
                bag_mask = inner_ids == bag
                bag_instances = feat[bag_mask]

                bag_pred = self.model_teacher(bag_instances)
                bag_prob = torch.softmax(bag_pred, dim=0)[1].item()

                all_probs.append(bag_prob)

            all_labels.extend(t_labels.cpu().numpy())

        return np.array(all_labels), np.array(all_probs)

    @torch.no_grad()
    def get_cell_scores(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get cell-level attention scores.

        Returns:
            attention_scores: Attention weights for each cell
            sample_ids: Sample ID for each cell
            predictions: Cell-level predictions from student
        """
        self.model_encoder.eval()
        self.model_teacher.eval()
        self.model_student.eval()

        all_attn = []
        all_sample_ids = []
        all_preds = []

        for batch in dataloader:
            t_data = batch[0].to(self.device)
            t_bagids = batch[1].to(self.device)
            t_study_ids = batch[3].to(self.device) if len(batch) == 4 else None

            # Encode
            if self.use_conditional_ae and t_study_ids is not None:
                feat = self.model_encoder(t_data, t_study_ids)
            else:
                feat = self.model_encoder(t_data)

            # Attention scores
            attn = self.model_teacher.attention_module(feat).squeeze()
            attn_norm = self._normalize_attention(attn)

            # Student predictions
            preds = torch.softmax(self.model_student(feat), dim=1)[:, 1]

            all_attn.append(attn_norm.cpu().numpy())
            all_sample_ids.append(t_bagids[-1].cpu().numpy())
            all_preds.append(preds.cpu().numpy())

        return (
            np.concatenate(all_attn),
            np.concatenate(all_sample_ids),
            np.concatenate(all_preds)
        )

    def save_models(self, save_dir: str, fold_idx: int):
        """Save models to directory."""
        os.makedirs(save_dir, exist_ok=True)

        torch.save(
            self.model_teacher.state_dict(),
            f"{save_dir}/model_teacher_fold{fold_idx}.pt"
        )
        torch.save(
            self.model_student.state_dict(),
            f"{save_dir}/model_student_fold{fold_idx}.pt"
        )
        torch.save(
            self.model_encoder.state_dict(),
            f"{save_dir}/model_encoder_fold{fold_idx}.pt"
        )

    def load_models(self, save_dir: str, fold_idx: int):
        """Load models from directory."""
        self.model_teacher.load_state_dict(
            torch.load(f"{save_dir}/model_teacher_fold{fold_idx}.pt", map_location=self.device)
        )
        self.model_student.load_state_dict(
            torch.load(f"{save_dir}/model_student_fold{fold_idx}.pt", map_location=self.device)
        )
        self.model_encoder.load_state_dict(
            torch.load(f"{save_dir}/model_encoder_fold{fold_idx}.pt", map_location=self.device)
        )
