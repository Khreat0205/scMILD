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

    Conditional 정보(study, organ 등)를 Encoder와 Decoder에 주입합니다 (scVI 스타일).
    Cross-condition generalization을 위한 mixed codebook을 학습합니다.

    Args:
        input_dim: Dimension of input features (number of genes)
        latent_dim: Dimension of latent space (before quantization)
        device: Device to run the model on
        hidden_layers: List of hidden layer dimensions
        n_conditionals: Number of unique conditional categories (e.g., studies, organs)
        conditional_emb_dim: Dimension of conditional embedding
        num_codes: Number of codes in the codebook
        commitment_weight: Weight for commitment loss
        activation_function: Activation function class
        n_studies: Deprecated, use n_conditionals instead
        study_emb_dim: Deprecated, use conditional_emb_dim instead
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        device: torch.device,
        hidden_layers: List[int],
        n_conditionals: int = None,
        conditional_emb_dim: int = 16,
        num_codes: int = 256,
        commitment_weight: float = 0.25,
        activation_function=nn.ReLU,
        # Deprecated parameters (for backward compatibility)
        n_studies: int = None,
        study_emb_dim: int = None,
    ):
        super().__init__()

        # Backward compatibility: support old parameter names
        if n_conditionals is None and n_studies is not None:
            n_conditionals = n_studies
        if study_emb_dim is not None:
            conditional_emb_dim = study_emb_dim

        if n_conditionals is None:
            raise ValueError("n_conditionals (or n_studies) must be provided")

        self.device = device
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.epsilon = 1e-4
        self.hidden_layers = hidden_layers
        self.activation_function = activation_function
        self.num_codes = num_codes
        self.commitment_weight = commitment_weight
        self.n_conditionals = n_conditionals
        self.conditional_emb_dim = conditional_emb_dim

        # Backward compatibility aliases
        self.n_studies = n_conditionals
        self.study_emb_dim = conditional_emb_dim

        # Conditional embedding (shared between encoder and decoder)
        self.conditional_embedding = nn.Embedding(n_conditionals, conditional_emb_dim)
        # Backward compatibility alias
        self.study_embedding = self.conditional_embedding

        # Encoder: input_dim + conditional_emb_dim → latent_dim
        encoder_layers = []
        previous_dim = input_dim + conditional_emb_dim  # Conditional input
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

        # Decoder: latent_dim + conditional_emb_dim → input_dim * 2
        decoder_layers = []
        previous_dim = latent_dim + conditional_emb_dim  # Conditional input
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
        conditional_ids: torch.Tensor
    ) -> torch.Tensor:
        """Encode input with conditional information to continuous latent representation."""
        c_emb = self.conditional_embedding(conditional_ids)  # (batch, conditional_emb_dim)
        x_cond = torch.cat([x, c_emb], dim=-1)   # (batch, input_dim + conditional_emb_dim)
        return self.encoder(x_cond)

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize continuous latent to discrete codes."""
        return self.quantizer(z, return_info=False)

    def decoder(
        self,
        z: torch.Tensor,
        conditional_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latent representation with conditional information."""
        c_emb = self.conditional_embedding(conditional_ids)  # (batch, conditional_emb_dim)
        z_cond = torch.cat([z, c_emb], dim=-1)   # (batch, latent_dim + conditional_emb_dim)
        decoded = self.decoder_layers(z_cond)
        mu_recon = torch.exp(decoded[:, :self.input_dim]).clamp(1e-6, 1e6)
        theta_recon = F.softplus(decoded[:, self.input_dim:]).clamp(1e-4, 1e4)
        return mu_recon, theta_recon

    def forward(
        self,
        x: torch.Tensor,
        conditional_ids: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        is_train: bool = True
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through Conditional VQ-AENB.

        Args:
            x: Input data (batch_size, input_dim)
            conditional_ids: Conditional category indices (batch_size,)
            y: Labels (not used, kept for compatibility)
            is_train: Whether in training mode

        Returns:
            mu_recon: Reconstructed mean parameters
            theta_recon: Reconstructed dispersion parameters
            commitment_loss: Loss from vector quantization (only if is_train=True)
        """
        z = self.encoder_forward(x, conditional_ids)
        z_q, commitment_loss = self.quantize(z)
        mu_recon, theta_recon = self.decoder(z_q, conditional_ids)

        if is_train:
            return mu_recon, theta_recon, commitment_loss
        else:
            return mu_recon, theta_recon

    def features(self, x: torch.Tensor, conditional_ids: torch.Tensor) -> torch.Tensor:
        """
        Get quantized features for downstream tasks.

        Args:
            x: Input data
            conditional_ids: Conditional category indices

        Returns:
            z_q: Quantized latent representation
        """
        z = self.encoder_forward(x, conditional_ids)
        z_q, _ = self.quantize(z)
        return z_q

    def get_codebook_indices(
        self,
        x: torch.Tensor,
        conditional_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Get codebook indices for input data.

        Args:
            x: Input data
            conditional_ids: Conditional category indices

        Returns:
            indices: Codebook indices for each sample
        """
        z = self.encoder_forward(x, conditional_ids)
        indices = self.quantizer.encode_indices(z)
        return indices

    def get_conditional_embeddings(self) -> np.ndarray:
        """Get learned conditional embeddings for analysis."""
        return self.conditional_embedding.weight.detach().cpu().numpy()

    # Backward compatibility alias
    def get_study_embeddings(self) -> np.ndarray:
        """Deprecated: Use get_conditional_embeddings() instead."""
        return self.get_conditional_embeddings()

    def init_codebook(
        self,
        dataloader,
        method: str = "kmeans",
        num_samples: int = None,
        stratify: bool = False
    ):
        """
        Initialize codebook using training data.

        Args:
            dataloader: DataLoader containing (data, study_ids, ...)
            method: Initialization method ("kmeans", "random", "uniform")
            num_samples: Number of samples to use for initialization.
                         If None, uses all data or at least 40 * num_codes for kmeans.
            stratify: If True, perform stratified sampling based on conditional IDs
                      to ensure all conditions are represented in codebook initialization.
        """
        # For k-means, faiss recommends at least 39 * k training points
        min_samples_for_kmeans = self.num_codes * 40 if method == "kmeans" else 10000
        if num_samples is None:
            num_samples = min_samples_for_kmeans

        embeddings = []
        conditional_ids = []  # For stratified sampling
        count = 0

        # Collect more samples if stratified (to ensure enough samples per group)
        collect_samples = num_samples * 2 if stratify else num_samples

        with torch.no_grad():
            for batch in dataloader:
                data = batch[0].to(self.device)
                cond_ids = batch[1].to(self.device) if len(batch) > 1 else \
                    torch.zeros(data.shape[0], dtype=torch.long, device=self.device)
                z = self.encoder_forward(data, cond_ids)
                embeddings.append(z)
                conditional_ids.append(cond_ids)
                count += z.shape[0]
                if count >= collect_samples:
                    break

        embeddings = torch.cat(embeddings, dim=0)
        conditional_ids = torch.cat(conditional_ids, dim=0)

        # Apply stratified sampling if requested
        if stratify:
            unique_conds = torch.unique(conditional_ids)
            if len(unique_conds) > 1:
                selected_indices = self._stratified_sample(
                    conditional_ids,
                    n_samples=num_samples
                )
                embeddings = embeddings[selected_indices]
                print(f"  Stratified sampling: {len(unique_conds)} groups, "
                      f"{num_samples // len(unique_conds)} samples per group")
            else:
                embeddings = embeddings[:num_samples]
        else:
            embeddings = embeddings[:num_samples]

        self.quantizer.init_codebook(embeddings, method=method)
        print(f"Initialized codebook with {method} using {embeddings.shape[0]} samples")

    def _stratified_sample(
        self,
        group_ids: torch.Tensor,
        n_samples: int
    ) -> torch.Tensor:
        """
        Perform stratified sampling to get equal samples from each group.

        Args:
            group_ids: Tensor of group IDs for each sample
            n_samples: Total number of samples to select

        Returns:
            selected_indices: Indices of selected samples
        """
        unique_groups = torch.unique(group_ids)
        n_groups = len(unique_groups)
        samples_per_group = n_samples // n_groups

        selected = []
        for gid in unique_groups:
            mask = (group_ids == gid)
            indices = torch.where(mask)[0]
            n_select = min(samples_per_group, len(indices))

            # Random permutation for selection
            perm = torch.randperm(len(indices), device=indices.device)[:n_select]
            selected.append(indices[perm])

        return torch.cat(selected)

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
