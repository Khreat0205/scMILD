"""
Autoencoder models for scMILD.

단일 세포 RNA-seq 데이터를 위한 오토인코더 모델들입니다:
- AENB: Negative Binomial loss 기반 기본 오토인코더
- VQ_AENB: Vector Quantized 오토인코더
- VQ_AENB_Conditional: Study/Batch 조건부 VQ 오토인코더
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union

from .quantizer import Quantizer


class AENB(nn.Module):
    """
    Autoencoder with Negative Binomial loss for single-cell RNA-seq data.

    Args:
        input_dim: Dimension of input features (number of genes)
        latent_dim: Dimension of latent space
        device: Device to run the model on
        hidden_layers: List of hidden layer dimensions
        activation_function: Activation function class (default: nn.ReLU)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        device: torch.device,
        hidden_layers: List[int],
        activation_function=nn.ReLU
    ):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.epsilon = 1e-4
        self.hidden_layers = hidden_layers
        self.activation_function = activation_function

        # Encoder
        feature_layers = []
        previous_dim = input_dim
        for layer_dim in self.hidden_layers:
            feature_layers.append(nn.Linear(previous_dim, layer_dim))
            feature_layers.append(self.activation_function())
            previous_dim = layer_dim
        feature_layers.append(nn.Linear(previous_dim, latent_dim))
        self.features = nn.Sequential(*feature_layers)

        # Decoder
        decoder_layers = []
        previous_dim = latent_dim
        for layer_dim in reversed(self.hidden_layers):
            decoder_layers.append(nn.Linear(previous_dim, layer_dim))
            decoder_layers.append(self.activation_function())
            previous_dim = layer_dim
        decoder_layers.append(nn.Linear(previous_dim, input_dim * 2))
        self.decoder_layers = nn.Sequential(*decoder_layers)

        self._initialize_weights()

    def decoder(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent representation to NB parameters.

        Args:
            z: Latent representation

        Returns:
            mu_recon: Reconstructed mean parameters
            theta_recon: Reconstructed dispersion parameters
        """
        decoded = self.decoder_layers(z)
        mu_recon = torch.exp(decoded[:, :self.input_dim]).clamp(1e-6, 1e6)
        theta_recon = F.softplus(decoded[:, self.input_dim:]).clamp(1e-4, 1e4)
        return mu_recon, theta_recon

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        is_train: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through AENB.

        Args:
            x: Input data (batch_size, input_dim)
            y: Labels (not used, kept for compatibility)
            is_train: Whether in training mode

        Returns:
            mu_recon: Reconstructed mean parameters
            theta_recon: Reconstructed dispersion parameters
        """
        encoded_features = self.features(x)
        z = encoded_features
        mu_recon, theta_recon = self.decoder(z)
        return mu_recon, theta_recon

    def _initialize_weights(self):
        """Initialize weights using Xavier normal initialization."""
        for m in self.features.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.decoder_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class VQ_AENB(nn.Module):
    """
    Vector Quantized Autoencoder with Negative Binomial loss.

    Codebook 기반의 이산 잠재 표현을 학습합니다.

    Args:
        input_dim: Dimension of input features (number of genes)
        latent_dim: Dimension of latent space (before quantization)
        device: Device to run the model on
        hidden_layers: List of hidden layer dimensions
        num_codes: Number of codes in the codebook
        commitment_weight: Weight for commitment loss
        activation_function: Activation function class
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        device: torch.device,
        hidden_layers: List[int],
        num_codes: int = 256,
        commitment_weight: float = 0.25,
        activation_function=nn.ReLU
    ):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.epsilon = 1e-4
        self.hidden_layers = hidden_layers
        self.activation_function = activation_function
        self.num_codes = num_codes
        self.commitment_weight = commitment_weight

        # Encoder
        encoder_layers = []
        previous_dim = input_dim
        for layer_dim in self.hidden_layers:
            encoder_layers.append(nn.Linear(previous_dim, layer_dim))
            encoder_layers.append(self.activation_function())
            previous_dim = layer_dim
        encoder_layers.append(nn.Linear(previous_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Quantizer
        self.quantizer = Quantizer(
            num_codes=num_codes,
            code_dim=latent_dim,
            commitment_weight=commitment_weight
        )

        # Decoder
        decoder_layers = []
        previous_dim = latent_dim
        for layer_dim in reversed(self.hidden_layers):
            decoder_layers.append(nn.Linear(previous_dim, layer_dim))
            decoder_layers.append(self.activation_function())
            previous_dim = layer_dim
        decoder_layers.append(nn.Linear(previous_dim, input_dim * 2))
        self.decoder_layers = nn.Sequential(*decoder_layers)

        self._initialize_weights()

    def encoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to continuous latent representation."""
        return self.encoder(x)

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize continuous latent to discrete codes."""
        return self.quantizer(z, return_info=False)

    def decoder(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latent representation to reconstruction parameters."""
        decoded = self.decoder_layers(z)
        mu_recon = torch.exp(decoded[:, :self.input_dim]).clamp(1e-6, 1e6)
        theta_recon = F.softplus(decoded[:, self.input_dim:]).clamp(1e-4, 1e4)
        return mu_recon, theta_recon

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        is_train: bool = True
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through VQ-AENB.

        Args:
            x: Input data (batch_size, input_dim)
            y: Labels (not used, kept for compatibility)
            is_train: Whether in training mode

        Returns:
            mu_recon: Reconstructed mean parameters
            theta_recon: Reconstructed dispersion parameters
            commitment_loss: Loss from vector quantization (only if is_train=True)
        """
        z = self.encoder_forward(x)
        z_q, commitment_loss = self.quantize(z)
        mu_recon, theta_recon = self.decoder(z_q)

        if is_train:
            return mu_recon, theta_recon, commitment_loss
        else:
            return mu_recon, theta_recon

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get quantized features for downstream tasks.

        Args:
            x: Input data

        Returns:
            z_q: Quantized latent representation
        """
        z = self.encoder_forward(x)
        z_q, _ = self.quantize(z)
        return z_q

    def get_codebook_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get codebook indices for input data.

        Args:
            x: Input data

        Returns:
            indices: Codebook indices for each sample
        """
        z = self.encoder_forward(x)
        indices = self.quantizer.encode_indices(z)
        return indices

    def init_codebook(
        self,
        dataloader,
        method: str = "kmeans",
        num_samples: int = 10000
    ):
        """
        Initialize codebook using training data.

        Args:
            dataloader: DataLoader containing training data
            method: Initialization method ("kmeans", "random", "uniform")
            num_samples: Number of samples to use for initialization
        """
        embeddings = []
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                data = batch[0].to(self.device)
                z = self.encoder_forward(data)
                embeddings.append(z)
                count += z.shape[0]
                if count >= num_samples:
                    break

        embeddings = torch.cat(embeddings, dim=0)[:num_samples]
        self.quantizer.init_codebook(embeddings, method=method)
        print(f"Initialized codebook with {method} using {embeddings.shape[0]} samples")

    def _initialize_weights(self):
        """Initialize weights for encoder and decoder."""
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.decoder_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_codebook_usage(self) -> dict:
        """Get statistics about codebook usage."""
        return self.quantizer.get_usage_stats()


class VQ_AENB_Conditional(nn.Module):
    """
    Conditional Vector Quantized Autoencoder with Negative Binomial loss.

    Study/Batch 정보를 Encoder와 Decoder에 주입합니다 (scVI 스타일).
    Cross-study generalization을 위한 organ-mixed codebook을 학습합니다.

    Args:
        input_dim: Dimension of input features (number of genes)
        latent_dim: Dimension of latent space (before quantization)
        device: Device to run the model on
        hidden_layers: List of hidden layer dimensions
        n_studies: Number of unique studies/batches
        study_emb_dim: Dimension of study embedding
        num_codes: Number of codes in the codebook
        commitment_weight: Weight for commitment loss
        activation_function: Activation function class
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        device: torch.device,
        hidden_layers: List[int],
        n_studies: int,
        study_emb_dim: int = 16,
        num_codes: int = 256,
        commitment_weight: float = 0.25,
        activation_function=nn.ReLU
    ):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.epsilon = 1e-4
        self.hidden_layers = hidden_layers
        self.activation_function = activation_function
        self.num_codes = num_codes
        self.commitment_weight = commitment_weight
        self.n_studies = n_studies
        self.study_emb_dim = study_emb_dim

        # Study embedding (shared between encoder and decoder)
        self.study_embedding = nn.Embedding(n_studies, study_emb_dim)

        # Encoder: input_dim + study_emb_dim → latent_dim
        encoder_layers = []
        previous_dim = input_dim + study_emb_dim  # Conditional input
        for layer_dim in self.hidden_layers:
            encoder_layers.append(nn.Linear(previous_dim, layer_dim))
            encoder_layers.append(self.activation_function())
            previous_dim = layer_dim
        encoder_layers.append(nn.Linear(previous_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Quantizer
        self.quantizer = Quantizer(
            num_codes=num_codes,
            code_dim=latent_dim,
            commitment_weight=commitment_weight
        )

        # Decoder: latent_dim + study_emb_dim → input_dim * 2
        decoder_layers = []
        previous_dim = latent_dim + study_emb_dim  # Conditional input
        for layer_dim in reversed(self.hidden_layers):
            decoder_layers.append(nn.Linear(previous_dim, layer_dim))
            decoder_layers.append(self.activation_function())
            previous_dim = layer_dim
        decoder_layers.append(nn.Linear(previous_dim, input_dim * 2))
        self.decoder_layers = nn.Sequential(*decoder_layers)

        self._initialize_weights()

    def encoder_forward(
        self,
        x: torch.Tensor,
        study_ids: torch.Tensor
    ) -> torch.Tensor:
        """Encode input with study condition to continuous latent representation."""
        s_emb = self.study_embedding(study_ids)  # (batch, study_emb_dim)
        x_cond = torch.cat([x, s_emb], dim=-1)   # (batch, input_dim + study_emb_dim)
        return self.encoder(x_cond)

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize continuous latent to discrete codes."""
        return self.quantizer(z, return_info=False)

    def decoder(
        self,
        z: torch.Tensor,
        study_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latent representation with study condition."""
        s_emb = self.study_embedding(study_ids)  # (batch, study_emb_dim)
        z_cond = torch.cat([z, s_emb], dim=-1)   # (batch, latent_dim + study_emb_dim)
        decoded = self.decoder_layers(z_cond)
        mu_recon = torch.exp(decoded[:, :self.input_dim]).clamp(1e-6, 1e6)
        theta_recon = F.softplus(decoded[:, self.input_dim:]).clamp(1e-4, 1e4)
        return mu_recon, theta_recon

    def forward(
        self,
        x: torch.Tensor,
        study_ids: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        is_train: bool = True
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through Conditional VQ-AENB.

        Args:
            x: Input data (batch_size, input_dim)
            study_ids: Study/batch indices (batch_size,)
            y: Labels (not used, kept for compatibility)
            is_train: Whether in training mode

        Returns:
            mu_recon: Reconstructed mean parameters
            theta_recon: Reconstructed dispersion parameters
            commitment_loss: Loss from vector quantization (only if is_train=True)
        """
        z = self.encoder_forward(x, study_ids)
        z_q, commitment_loss = self.quantize(z)
        mu_recon, theta_recon = self.decoder(z_q, study_ids)

        if is_train:
            return mu_recon, theta_recon, commitment_loss
        else:
            return mu_recon, theta_recon

    def features(self, x: torch.Tensor, study_ids: torch.Tensor) -> torch.Tensor:
        """
        Get quantized features for downstream tasks.

        Args:
            x: Input data
            study_ids: Study/batch indices

        Returns:
            z_q: Quantized latent representation
        """
        z = self.encoder_forward(x, study_ids)
        z_q, _ = self.quantize(z)
        return z_q

    def get_codebook_indices(
        self,
        x: torch.Tensor,
        study_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Get codebook indices for input data.

        Args:
            x: Input data
            study_ids: Study/batch indices

        Returns:
            indices: Codebook indices for each sample
        """
        z = self.encoder_forward(x, study_ids)
        indices = self.quantizer.encode_indices(z)
        return indices

    def get_study_embeddings(self) -> np.ndarray:
        """Get learned study embeddings for analysis."""
        return self.study_embedding.weight.detach().cpu().numpy()

    def init_codebook(
        self,
        dataloader,
        method: str = "kmeans",
        num_samples: int = 10000
    ):
        """
        Initialize codebook using training data.

        Args:
            dataloader: DataLoader containing (data, study_ids, ...)
            method: Initialization method ("kmeans", "random", "uniform")
            num_samples: Number of samples to use for initialization
        """
        embeddings = []
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                data = batch[0].to(self.device)
                study_ids = batch[1].to(self.device) if len(batch) > 1 else \
                    torch.zeros(data.shape[0], dtype=torch.long, device=self.device)
                z = self.encoder_forward(data, study_ids)
                embeddings.append(z)
                count += z.shape[0]
                if count >= num_samples:
                    break

        embeddings = torch.cat(embeddings, dim=0)[:num_samples]
        self.quantizer.init_codebook(embeddings, method=method)
        print(f"Initialized codebook with {method} using {embeddings.shape[0]} samples")

    def _initialize_weights(self):
        """Initialize weights for encoder and decoder."""
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.decoder_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_codebook_usage(self) -> dict:
        """Get statistics about codebook usage."""
        return self.quantizer.get_usage_stats()
