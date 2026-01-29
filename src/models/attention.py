"""
Attention modules for scMILD.

Multiple Instance Learning에서 세포 중요도를 계산하는 어텐션 모듈입니다.
"""

import torch
from torch import nn


class AttentionModule(nn.Module):
    """
    Standard Attention Module for MIL.

    Tanh 활성화 함수를 사용하는 단순한 어텐션 메커니즘입니다.

    Args:
        L: Input feature dimension
        D: Hidden dimension for attention
        K: Number of attention heads (usually 1)
    """

    def __init__(self, L: int, D: int, K: int = 1):
        super().__init__()
        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores.

        Args:
            H: Input features (N x L) where N is number of cells

        Returns:
            A: Attention scores (K x N), unnormalized
        """
        A = self.attention(H)  # N x K
        A = torch.transpose(A, 1, 0)  # K x N
        return A


class GatedAttentionModule(nn.Module):
    """
    Gated Attention Module for MIL.

    Sigmoid 게이트를 사용하여 더 표현력 있는 어텐션을 계산합니다.
    일반적으로 AttentionModule보다 더 좋은 성능을 보입니다.

    Args:
        L: Input feature dimension
        D: Hidden dimension for attention
        K: Number of attention heads (usually 1)
    """

    def __init__(self, L: int, D: int, K: int = 1):
        super().__init__()
        self.L = L
        self.D = D
        self.K = K

        # Value path (Tanh)
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        # Gate path (Sigmoid)
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        # Final projection
        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Compute gated attention scores.

        Args:
            H: Input features (N x L) where N is number of cells

        Returns:
            A: Attention scores (K x N), unnormalized
        """
        A_V = self.attention_V(H)  # N x D
        A_U = self.attention_U(H)  # N x D
        A = self.attention_weights(A_V * A_U)  # Element-wise multiplication, N x K
        A = torch.transpose(A, 1, 0)  # K x N
        return A
