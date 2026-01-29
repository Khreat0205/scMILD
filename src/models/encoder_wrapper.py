"""
Encoder wrappers for scMILD.

Pretrained VQ-AENB 모델을 MIL 파이프라인에서 사용하기 위한 래퍼 클래스들입니다.
Frozen encoder + trainable projection layer 구조를 지원합니다.
"""

import torch
from torch import nn
from typing import Optional


class VQEncoderWrapper(nn.Module):
    """
    Wrapper for VQ-AENB to use in MIL pipeline.

    Pretrained encoder를 frozen 상태로 사용하고,
    선택적으로 trainable projection layer를 추가합니다.

    Args:
        vq_aenb_model: Pretrained VQ_AENB model
        use_projection: Whether to add a projection layer
        projection_dim: Dimension of projection layer output
    """

    def __init__(
        self,
        vq_aenb_model: nn.Module,
        use_projection: bool = False,
        projection_dim: Optional[int] = None
    ):
        super().__init__()
        self.vq_model = vq_aenb_model
        self.input_dims = vq_aenb_model.latent_dim  # For compatibility with MIL branches
        self.use_projection = use_projection

        if use_projection:
            proj_dim = projection_dim or vq_aenb_model.latent_dim
            self.projection = nn.Sequential(
                nn.Linear(vq_aenb_model.latent_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim)
            )
            self.input_dims = proj_dim  # Update for downstream compatibility
            self._initialize_projection()
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input data (batch_size, input_dim)

        Returns:
            encoded: Encoded features (batch_size, output_dim)
        """
        with torch.no_grad():
            encoded = self.vq_model.features(x)

        if self.projection is not None:
            encoded = self.projection(encoded)

        return encoded

    def freeze_encoder(self):
        """Freeze VQ-AENB encoder parameters, keep projection trainable."""
        for param in self.vq_model.parameters():
            param.requires_grad = False

        if self.projection is not None:
            for param in self.projection.parameters():
                param.requires_grad = True

    def unfreeze_encoder(self):
        """Unfreeze all parameters."""
        for param in self.vq_model.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self):
        """Return only trainable parameters (projection layer if frozen)."""
        if self.projection is not None:
            return self.projection.parameters()
        else:
            return self.parameters()

    def get_codebook_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Get codebook indices for input data."""
        return self.vq_model.get_codebook_indices(x)

    def _initialize_projection(self):
        """Initialize projection layer weights."""
        if self.projection is not None:
            for m in self.projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


class VQEncoderWrapperConditional(nn.Module):
    """
    Wrapper for Conditional VQ-AENB to use in MIL pipeline.

    Study ID를 조건으로 받아 처리하는 Conditional encoder를 위한 래퍼입니다.

    Args:
        vq_aenb_conditional_model: Pretrained VQ_AENB_Conditional model
        use_projection: Whether to add a projection layer
        projection_dim: Dimension of projection layer output
    """

    def __init__(
        self,
        vq_aenb_conditional_model: nn.Module,
        use_projection: bool = False,
        projection_dim: Optional[int] = None
    ):
        super().__init__()
        self.vq_model = vq_aenb_conditional_model
        self.input_dims = vq_aenb_conditional_model.latent_dim
        self.use_projection = use_projection
        self.is_conditional = True  # Flag for conditional handling

        if use_projection:
            proj_dim = projection_dim or vq_aenb_conditional_model.latent_dim
            self.projection = nn.Sequential(
                nn.Linear(vq_aenb_conditional_model.latent_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim)
            )
            self.input_dims = proj_dim
            self._initialize_projection()
        else:
            self.projection = None

    def forward(
        self,
        x: torch.Tensor,
        study_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with study condition.

        Args:
            x: Input data (batch_size, input_dim)
            study_ids: Study/batch indices (batch_size,). Required for conditional model.

        Returns:
            encoded: Encoded features (batch_size, output_dim)
        """
        if study_ids is None:
            # Fallback: use study_id=0 for all samples
            study_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            print("WARNING: study_ids not provided to conditional encoder, using 0")

        with torch.no_grad():
            encoded = self.vq_model.features(x, study_ids)

        if self.projection is not None:
            encoded = self.projection(encoded)

        return encoded

    def freeze_encoder(self):
        """Freeze VQ-AENB encoder parameters, keep projection trainable."""
        for param in self.vq_model.parameters():
            param.requires_grad = False

        if self.projection is not None:
            for param in self.projection.parameters():
                param.requires_grad = True

    def unfreeze_encoder(self):
        """Unfreeze all parameters."""
        for param in self.vq_model.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self):
        """Return only trainable parameters (projection layer if frozen)."""
        if self.projection is not None:
            return self.projection.parameters()
        else:
            return self.parameters()

    def get_codebook_indices(
        self,
        x: torch.Tensor,
        study_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get codebook indices for input data."""
        if study_ids is None:
            study_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.vq_model.get_codebook_indices(x, study_ids)

    def _initialize_projection(self):
        """Initialize projection layer weights."""
        if self.projection is not None:
            for m in self.projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


def create_encoder_wrapper(
    model_path: str,
    device: torch.device,
    model_type: str = "VQ_AENB_Conditional",
    use_projection: bool = True,
    projection_dim: Optional[int] = None,
    freeze: bool = True
) -> nn.Module:
    """
    Factory function to create appropriate encoder wrapper.

    Args:
        model_path: Path to pretrained model checkpoint
        device: Device to load model on
        model_type: Type of encoder model
        use_projection: Whether to add projection layer
        projection_dim: Projection layer output dimension
        freeze: Whether to freeze encoder weights

    Returns:
        Encoder wrapper instance
    """
    from .autoencoder import VQ_AENB, VQ_AENB_Conditional

    checkpoint = torch.load(model_path, map_location=device)

    if model_type == "VQ_AENB_Conditional":
        # Extract model config from checkpoint
        model_config = checkpoint.get('config', {})
        model = VQ_AENB_Conditional(
            input_dim=model_config.get('input_dim'),
            latent_dim=model_config.get('latent_dim'),
            device=device,
            hidden_layers=model_config.get('hidden_layers'),
            n_studies=model_config.get('n_studies'),
            study_emb_dim=model_config.get('study_emb_dim', 16),
            num_codes=model_config.get('num_codes', 256),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        wrapper = VQEncoderWrapperConditional(
            model,
            use_projection=use_projection,
            projection_dim=projection_dim
        )

    elif model_type == "VQ_AENB":
        model_config = checkpoint.get('config', {})
        model = VQ_AENB(
            input_dim=model_config.get('input_dim'),
            latent_dim=model_config.get('latent_dim'),
            device=device,
            hidden_layers=model_config.get('hidden_layers'),
            num_codes=model_config.get('num_codes', 256),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        wrapper = VQEncoderWrapper(
            model,
            use_projection=use_projection,
            projection_dim=projection_dim
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if freeze:
        wrapper.freeze_encoder()

    return wrapper
