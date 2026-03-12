"""
Celltype Auxiliary Classifier for scMILD Pretraining.

z_q (quantized representation) 기반 cell type 분류기.
Pretraining 시 auxiliary loss로 사용되며, 다운스트림 MIL에서는 사용하지 않습니다.

Encoder checkpoint에 포함되지 않는 독립 모듈입니다.
"""

import torch
from torch import nn


class CelltypeClassifier(nn.Module):
    """
    Shallow MLP classifier for cell type prediction from z_q.

    Args:
        input_dim: Dimension of z_q (typically latent_dim, e.g., 128)
        n_classes: Number of celltype classes
        hidden_dim: Hidden layer dimension (default: 64)
        n_layers: Number of hidden layers, 1 or 2 (default: 1)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dim: int = 64,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes

        layers = []
        prev_dim = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_classes))

        self.classifier = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q: Quantized latent representation (batch, input_dim)

        Returns:
            logits: Cell type logits (batch, n_classes)
        """
        return self.classifier(z_q)

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
