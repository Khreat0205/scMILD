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
    """

    def __init__(
        self,
        num_codes: int = 256,
        code_dim: int = 128,
        decay: float = 0.9,
        commitment_weight: float = 0.25
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
                self.codebook.weight.data.copy_(data[indices])
            else:
                self.codebook.weight.data.uniform_(-1.0 / self.num_codes, 1.0 / self.num_codes)

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

                # Get cluster centers
                centers = kmeans.centroids
                self.codebook.weight.data.copy_(torch.from_numpy(centers))
            except ImportError:
                print("Warning: faiss not installed. Using random initialization instead.")
                self.init_codebook(data, method="random")

        elif method == "uniform":
            # Uniform initialization
            self.codebook.weight.data.uniform_(-1.0 / self.num_codes, 1.0 / self.num_codes)

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
                self.codebook.weight[dead_indices] = z.detach()[sample_indices]

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
