"""
MIL Branch modules for scMILD.

Teacher-Student 구조의 MIL 브랜치들입니다:
- TeacherBranch: 샘플(Bag) 레벨 분류
- StudentBranch: 세포(Instance) 레벨 분류
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


class TeacherBranch(nn.Module):
    """
    Teacher Branch for sample-level classification.

    Attention 메커니즘을 사용하여 세포들을 집계하고 샘플 레벨에서 분류합니다.

    Args:
        input_dims: Input feature dimension
        latent_dims: Hidden layer dimension
        attention_module: Attention module instance (AttentionModule or GatedAttentionModule)
        num_classes: Number of output classes (default: 2)
        activation_function: Activation function class (default: nn.Tanh)
    """

    def __init__(
        self,
        input_dims: int,
        latent_dims: int,
        attention_module: nn.Module,
        num_classes: int = 2,
        activation_function=nn.Tanh
    ):
        super().__init__()
        self.input_dims = input_dims
        self.L = latent_dims
        self.K = 1
        self.D = latent_dims
        self.attention_module = attention_module
        self.num_classes = num_classes

        # Bag-level classifier
        self.bagNN = nn.Sequential(
            nn.Linear(self.input_dims, self.L),
            activation_function(),
            nn.Linear(self.L, self.L),
            activation_function(),
            nn.Linear(self.L, self.num_classes),
        )
        self._initialize_weights()

    def forward(
        self,
        input: torch.Tensor,
        replace_attention_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for sample classification.

        Args:
            input: Cell features (N x input_dims) where N is number of cells in the bag
            replace_attention_scores: Optional pre-computed attention scores to use
                                     instead of computing them (K x N)

        Returns:
            output: Classification logits (num_classes,)
        """
        if replace_attention_scores is not None:
            attention_weights = F.softmax(replace_attention_scores, dim=1)
        else:
            attention_weights = self.attention_module(input)
            attention_weights = F.softmax(attention_weights, dim=1)

        # Aggregate cells using attention weights
        aggregated_instance = torch.mm(attention_weights, input)  # 1 x input_dims
        output = aggregated_instance.squeeze()
        output = self.bagNN(output)
        return output

    def get_attention_weights(self, input: torch.Tensor) -> torch.Tensor:
        """
        Get normalized attention weights for cells.

        Args:
            input: Cell features (N x input_dims)

        Returns:
            attention_weights: Normalized attention weights (1 x N)
        """
        attention_scores = self.attention_module(input)
        attention_weights = F.softmax(attention_scores, dim=1)
        return attention_weights

    def _initialize_weights(self):
        """Initialize weights using Xavier normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)


class StudentBranch(nn.Module):
    """
    Student Branch for cell-level classification.

    개별 세포 레벨에서 분류를 수행합니다.
    Teacher의 attention weights로 학습이 가이드됩니다.

    Args:
        input_dims: Input feature dimension
        latent_dims: Hidden layer dimension
        num_classes: Number of output classes (default: 2)
        activation_function: Activation function class (default: nn.Tanh)
    """

    def __init__(
        self,
        input_dims: int,
        latent_dims: int,
        num_classes: int = 2,
        activation_function=nn.Tanh
    ):
        super().__init__()
        self.input_dims = input_dims
        self.L = latent_dims
        self.K = 1
        self.D = latent_dims
        self.num_classes = num_classes

        # Instance-level classifier
        self.instanceNN = nn.Sequential(
            nn.Linear(self.input_dims, self.L),
            activation_function(),
            nn.Linear(self.L, self.L),
            activation_function(),
            nn.Linear(self.L, self.num_classes)
        )
        self._initialize_weights()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for cell classification.

        Args:
            input: Cell features (N x input_dims) where N is number of cells

        Returns:
            output: Classification logits (N x num_classes)
        """
        output = self.instanceNN(input)
        return output

    def _initialize_weights(self):
        """Initialize weights using Xavier normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
