"""
scMILD Models

모델 컴포넌트들을 제공합니다:
- Attention modules (AttentionModule, GatedAttentionModule)
- MIL branches (TeacherBranch, StudentBranch)
- Autoencoders (AENB, VQ_AENB, VQ_AENB_Conditional)
- Encoder wrappers (VQEncoderWrapper, VQEncoderWrapperConditional)
"""

from .attention import AttentionModule, GatedAttentionModule
from .branches import TeacherBranch, StudentBranch
from .autoencoder import AENB, VQ_AENB, VQ_AENB_Conditional
from .encoder_wrapper import VQEncoderWrapper, VQEncoderWrapperConditional

__all__ = [
    # Attention
    "AttentionModule",
    "GatedAttentionModule",
    # Branches
    "TeacherBranch",
    "StudentBranch",
    # Autoencoders
    "AENB",
    "VQ_AENB",
    "VQ_AENB_Conditional",
    # Wrappers
    "VQEncoderWrapper",
    "VQEncoderWrapperConditional",
]
