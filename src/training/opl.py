"""
Orthogonal Projection Loss (OPL) for scMILD.

Codebook-level OPL: VQ 코드의 projection embedding을
attention score 기반 GMM 클러스터링 결과에 따라 직교하게 만듭니다.

Reference: https://github.com/kahnchana/opl/blob/master/loss.py
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
import warnings
from sklearn.exceptions import ConvergenceWarning


class OrthogonalProjectionLoss(nn.Module):
    """
    같은 클러스터의 feature는 가깝게, 다른 클러스터의 feature는 직교하게.

    Args:
        gamma: Weight for negative (cross-cluster) term
    """

    def __init__(self, gamma: float = 0.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (N, D) normalized features
            labels: (N,) cluster labels (0 or 1)

        Returns:
            loss: scalar
        """
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # (N, 1)
        mask = torch.eq(labels, labels.t()).bool()
        eye = torch.eye(mask.shape[0], device=mask.device).bool()

        mask_pos = mask.masked_fill(eye, False).float()
        mask_neg = (~mask).float()

        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean
        return loss


def compute_code_opl(
    model_encoder: nn.Module,
    model_teacher: nn.Module,
    device: torch.device,
    opl_criterion: OrthogonalProjectionLoss,
    n_gmm_components: int = 2,
) -> torch.Tensor:
    """
    Codebook-level OPL 계산.

    1. Codebook embeddings → projection layer → projected code embeddings
    2. Projected embeddings → attention module → code-level attention scores
    3. GMM 2-cluster on attention scores
    4. OPL loss on projected embeddings with cluster labels

    Args:
        model_encoder: VQEncoderWrapper (with projection layer)
        model_teacher: Teacher branch (with attention_module)
        device: torch device
        opl_criterion: OrthogonalProjectionLoss instance
        n_gmm_components: Number of GMM clusters (default 2)

    Returns:
        opl_loss: scalar tensor (0 if GMM fails or insufficient codes)
    """
    # 1. Get codebook embeddings (detached — codebook is EMA-updated, not grad-updated)
    codebook = model_encoder.vq_model.quantize.codebook.weight.detach().to(device)  # (num_codes, latent_dim)

    # 2. Project through projection layer (gradient flows through projection)
    if model_encoder.projection is not None:
        projected = model_encoder.projection(codebook)  # (num_codes, proj_dim)
    else:
        projected = codebook

    # 3. Get attention scores for each code (detached — labels only, no grad through attention)
    with torch.no_grad():
        attn_scores = model_teacher.attention_module(projected.detach())  # (1, num_codes)
        attn_scores = attn_scores.squeeze(0)  # (num_codes,)

    # 4. GMM clustering on attention scores
    scores_np = attn_scores.cpu().detach().numpy().reshape(-1, 1)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gmm = GaussianMixture(n_components=n_gmm_components, random_state=42)
        gmm.fit(scores_np)

        if len(w) > 0 and any(issubclass(wi.category, ConvergenceWarning) for wi in w):
            return torch.tensor(0.0, device=device, requires_grad=False)

    # Order components by mean attention (low=0, high=1)
    component_order = np.argsort(gmm.means_.flatten())
    raw_labels = gmm.predict(scores_np)
    ordered_labels = np.array([component_order.tolist().index(l) for l in raw_labels])

    # Need at least 2 codes in each cluster
    unique, counts = np.unique(ordered_labels, return_counts=True)
    if len(unique) < 2 or min(counts) < 2:
        return torch.tensor(0.0, device=device, requires_grad=False)

    labels = torch.tensor(ordered_labels, dtype=torch.float, device=device)

    # 5. OPL loss on projected embeddings
    opl_loss = opl_criterion(projected, labels)

    return opl_loss
