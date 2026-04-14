"""
Vector Quantization module for VQ-VAE.

Codebook 기반 이산 잠재 표현을 위한 양자화 모듈입니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Quantizer(nn.Module):
    """
    Vector Quantization module for VQ-VAE.

    Cosine similarity 기반의 코드북 매칭과 Straight-through estimator를 사용합니다.

    Args:
        num_codes: Number of codes in the codebook
        code_dim: Dimension of each code vector
        decay: Decay rate for exponential moving average of code usage
        commitment_weight: Weight for commitment loss
        ema_update: If True, update codebook weights in-place via EMA
            statistics (van den Oord 2017, Appendix A.1) instead of leaving
            them frozen after k-means init. Commitment loss is unchanged.
        ema_decay: EMA decay for cluster size / embedding sums.
        ema_eps: Laplace smoothing epsilon for cluster sizes.
    """

    def __init__(
        self,
        num_codes: int = 256,
        code_dim: int = 128,
        decay: float = 0.9,
        commitment_weight: float = 0.25,
        ema_update: bool = False,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
    ):
        super().__init__()

        self.num_codes = num_codes
        self.code_dim = code_dim
        self.decay = decay
        self.commitment_weight = commitment_weight

        # Codebook embedding
        self.codebook = nn.Embedding(self.num_codes, self.code_dim)

        # Track codebook usage
        self.register_buffer("code_usage", torch.zeros(self.num_codes))

        # EMA codebook update (Oord 2017 §Appendix A.1). When enabled, the
        # codebook weights are updated in-place each training step from the
        # running cluster statistics instead of remaining frozen after
        # k-means init — without an EMA path, straight-through reassignment
        # (z_q = z + (z_q - z).detach()) discards the only gradient that
        # could reach codebook.weight, so codes stay at their init values.
        self.ema_update = ema_update
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        if ema_update:
            self.register_buffer("ema_cluster_size", torch.zeros(self.num_codes))
            self.register_buffer(
                "ema_embedding_sum", torch.zeros(self.num_codes, self.code_dim)
            )
            self.register_buffer("ema_initialized", torch.tensor(False))

        # Initialize codebook
        self.codebook.weight.data.uniform_(-1.0 / self.num_codes, 1.0 / self.num_codes)

    def init_codebook(self, data: torch.Tensor, method: str = "kmeans"):
        """
        Initialize codebook with different methods.

        Args:
            data: Input data for initialization (N x D)
            method: Initialization method ("random", "kmeans", "uniform")
        """
        if method == "random":
            # Random initialization from data
            if data.shape[0] >= self.num_codes:
                indices = torch.randperm(data.shape[0])[:self.num_codes]
                centers_t = data[indices].detach().cpu().float()
                self.codebook.weight.data.copy_(centers_t)
                if self.ema_update:
                    self._prime_ema_from_assignments(
                        raw_data_t=data.detach().cpu().float(),
                        centers_t=centers_t,
                    )
            else:
                self.codebook.weight.data.uniform_(
                    -1.0 / self.num_codes, 1.0 / self.num_codes
                )

        elif method == "kmeans":
            # K-means initialization (requires faiss)
            try:
                import faiss
                if isinstance(data, torch.Tensor):
                    data_np = data.detach().cpu().numpy().astype(np.float32)
                else:
                    data_np = np.asarray(data, dtype=np.float32)

                d = data_np.shape[1]
                # Use spherical k-means for normalized embeddings
                kmeans = faiss.Kmeans(
                    d, k=self.num_codes, spherical=True, verbose=False, gpu=False
                )
                kmeans.train(data_np)

                # Get cluster centers (unit-norm due to spherical=True)
                centers = kmeans.centroids

                if self.ema_update:
                    # EMA path: overwrite codebook with raw-scale cluster
                    # means so codebook entries live at the natural scale of
                    # z. Matching is still cosine via F.normalize at lookup,
                    # so direction alone matters — but z_q must be at the
                    # same scale as z for commitment loss to stay sane.
                    # Unit-norming EMA sums would shrink the codebook to
                    # 1/cluster_size and blow commit loss up as the encoder
                    # drifts.
                    raw_data_t = torch.from_numpy(data_np).float()
                    centers_t = torch.from_numpy(centers).float()
                    self._prime_ema_from_assignments(
                        raw_data_t=raw_data_t, centers_t=centers_t
                    )
                else:
                    # Legacy path: direct copy of unit-norm centers.
                    self.codebook.weight.data.copy_(torch.from_numpy(centers))
            except ImportError:
                print("Warning: faiss not installed. Using random initialization instead.")
                self.init_codebook(data, method="random")

        elif method == "uniform":
            # Uniform initialization
            self.codebook.weight.data.uniform_(-1.0 / self.num_codes, 1.0 / self.num_codes)

    def _prime_ema_from_assignments(
        self,
        raw_data_t: torch.Tensor,
        centers_t: torch.Tensor,
    ) -> None:
        """
        Prime EMA buffers so the first training-step EMA update does not
        snap the codebook away from its freshly computed centers.

        Assigns each raw sample to its nearest center under cosine
        similarity (same metric as forward()), then seeds codebook.weight
        AND ema_embedding_sum / ema_cluster_size from the raw-scale
        per-cluster sums. Empty clusters fall back to the input centers.
        """
        cb_n = F.normalize(centers_t, dim=1)
        raw_n = F.normalize(raw_data_t, dim=1)
        sims = raw_n @ cb_n.t()
        idx = sims.argmax(dim=1)

        one_hot = torch.zeros(idx.shape[0], self.num_codes)
        one_hot.scatter_(1, idx.unsqueeze(1), 1.0)
        cluster_size = one_hot.sum(dim=0)
        embedding_sum_raw = one_hot.t() @ raw_data_t  # (num_codes, D) raw scale

        safe_size = cluster_size.clone().clamp_min_(1.0).unsqueeze(1)
        centers_raw = embedding_sum_raw / safe_size
        empty_mask = cluster_size == 0
        if empty_mask.any():
            centers_raw[empty_mask] = centers_t[empty_mask]

        self.codebook.weight.data.copy_(centers_raw)
        with torch.no_grad():
            self.ema_cluster_size.copy_(cluster_size.to(self.ema_cluster_size.dtype))
            self.ema_embedding_sum.copy_(
                embedding_sum_raw.to(self.ema_embedding_sum.dtype)
            )
            self.ema_initialized.fill_(True)
        print(
            f"[Quantizer] EMA primed: {int((cluster_size > 0).sum())}/"
            f"{self.num_codes} codes have >=1 sample, "
            f"total={int(cluster_size.sum())}"
        )

    def forward(self, z: torch.Tensor, return_info: bool = False):
        """
        Quantize input tensor.

        Args:
            z: Input tensor to quantize (B x D)
            return_info: Whether to return additional information

        Returns:
            z_q: Quantized tensor
            loss: Commitment loss
            info: Additional information (if return_info=True)
        """
        # Compute distances using cosine similarity
        if self.training:
            z_norm = F.normalize(z, dim=1).detach()
        else:
            z_norm = F.normalize(z, dim=1)

        codebook_norm = F.normalize(self.codebook.weight, dim=1)

        # Cosine similarity (B x num_codes)
        similarity = torch.einsum("bd,nd->bn", z_norm, codebook_norm)

        # Find nearest code
        indices = torch.argmax(similarity, dim=1)  # (B,)

        # One-hot encoding
        one_hot = torch.zeros(indices.shape[0], self.num_codes, device=z.device)
        one_hot.scatter_(1, indices.unsqueeze(1), 1)  # (B, num_codes)

        # Quantize: get corresponding codes
        z_q = torch.matmul(one_hot, self.codebook.weight)  # (B, D)

        # Commitment loss (encoder should commit to codebook entries)
        commitment_loss = self.commitment_weight * torch.mean((z_q.detach() - z) ** 2)

        # Update code usage statistics during training
        if self.training:
            # Straight-through estimator for gradient
            z_q = z + (z_q - z).detach()
            avg_probs = torch.mean(one_hot, dim=0)
            self.code_usage.mul_(self.decay).add_(avg_probs, alpha=1 - self.decay)

            # EMA codebook update (before revive, so freshly-moved codes
            # are not overwritten by EMA smoothing in the same step).
            if self.ema_update:
                with torch.no_grad():
                    # Accumulate RAW z (not unit-normed even for cosine).
                    # Matching above re-normalizes via F.normalize, so
                    # codebook direction is what matters — but z_q must
                    # live at the same scale as z for commitment loss to
                    # behave. Unit-norming EMA sums would shrink the
                    # codebook to 1/cluster_size and make commit loss
                    # explode as the encoder drifts.
                    z_for_ema = z.detach()

                    cluster_size_batch = one_hot.sum(dim=0)
                    embedding_sum_batch = one_hot.t() @ z_for_ema

                    d = self.ema_decay
                    self.ema_cluster_size.mul_(d).add_(
                        cluster_size_batch, alpha=1.0 - d
                    )
                    self.ema_embedding_sum.mul_(d).add_(
                        embedding_sum_batch, alpha=1.0 - d
                    )

                    # Laplace smoothing keeps near-dead codes from exploding.
                    n = self.ema_cluster_size.sum()
                    smoothed = (
                        (self.ema_cluster_size + self.ema_eps)
                        / (n + self.num_codes * self.ema_eps)
                        * n
                    )
                    self.codebook.weight.data.copy_(
                        self.ema_embedding_sum / smoothed.unsqueeze(1)
                    )

            # Deal with dead codes (low usage)
            self._revive_dead_codes(z, similarity)

        if return_info:
            info = {
                'indices': indices,
                'one_hot': one_hot,
                'similarity': similarity,
                'code_usage': self.code_usage.clone(),
                'perplexity': self._compute_perplexity(one_hot)
            }
            return z_q, commitment_loss, info

        return z_q, commitment_loss

    def _revive_dead_codes(self, z: torch.Tensor, similarity: torch.Tensor):
        """
        Revive dead codes by reinitializing them with training samples.

        Args:
            z: Input embeddings
            similarity: Similarity matrix between embeddings and codes
        """
        # Identify dead codes (usage below threshold)
        dead_codes = self.code_usage < (1e-3 / self.num_codes)

        if dead_codes.sum() > 0 and z.shape[0] > 0:
            num_dead = int(dead_codes.sum().item())

            # Sample from inputs with low similarity to all codes
            max_sim = torch.max(similarity, dim=1).values

            # Ensure numerical stability for softmax
            neg_max_sim = -max_sim
            neg_max_sim = neg_max_sim - neg_max_sim.max()  # Prevent overflow
            sample_probs = F.softmax(neg_max_sim, dim=0)

            # Add small epsilon to prevent zero probabilities
            sample_probs = sample_probs + 1e-8
            sample_probs = sample_probs / sample_probs.sum()

            # Check if we have valid probabilities
            if torch.isnan(sample_probs).any() or sample_probs.sum() <= 0:
                # Fallback to uniform sampling
                sample_indices = torch.randint(0, z.shape[0], (num_dead,), device=z.device)
            else:
                # Sample indices for reinitialization
                sample_indices = torch.multinomial(
                    sample_probs, num_samples=min(num_dead, z.shape[0]), replacement=True
                )

            # Reinitialize dead codes
            dead_indices = torch.where(dead_codes)[0][:len(sample_indices)]
            with torch.no_grad():
                new_vecs = z.detach()[sample_indices]
                self.codebook.weight[dead_indices] = new_vecs

                # Keep EMA buffers consistent with the overwritten codebook.
                # Without this, the next EMA step recomputes
                # codebook = ema_embedding_sum / smoothed_cluster_size and
                # would snap the revived code back toward its prior
                # (near-zero) EMA state, undoing the revive.
                if self.ema_update:
                    self.ema_cluster_size[dead_indices] = 1.0
                    self.ema_embedding_sum[dead_indices] = new_vecs
                    # Reset usage EMA so the revived code is not declared
                    # dead again on the next step.
                    self.code_usage[dead_indices] = 1.0 / self.num_codes

    def _compute_perplexity(self, one_hot: torch.Tensor) -> torch.Tensor:
        """
        Compute perplexity of code usage.

        Args:
            one_hot: One-hot encoding of code assignments

        Returns:
            perplexity: Measure of how many codes are being used
        """
        avg_probs = torch.mean(one_hot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity

    def get_codebook(self) -> torch.Tensor:
        """Return the codebook embeddings."""
        return self.codebook.weight.data.clone()

    def encode_indices(self, z: torch.Tensor) -> torch.Tensor:
        """
        Encode input to codebook indices.

        Args:
            z: Input tensor (B x D)

        Returns:
            indices: Codebook indices (B,)
        """
        z_norm = F.normalize(z, dim=1)
        codebook_norm = F.normalize(self.codebook.weight, dim=1)
        similarity = torch.einsum("bd,nd->bn", z_norm, codebook_norm)
        indices = torch.argmax(similarity, dim=1)
        return indices

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode codebook indices to embeddings.

        Args:
            indices: Codebook indices (B,)

        Returns:
            z_q: Quantized embeddings (B x D)
        """
        return self.codebook(indices)

    def get_usage_stats(self) -> dict:
        """Get statistics about codebook usage."""
        return {
            'usage': self.code_usage.cpu().numpy(),
            'num_active': (self.code_usage > 1e-3).sum().item(),
            'total_codes': self.num_codes,
            'utilization': (self.code_usage > 1e-3).sum().item() / self.num_codes
        }
