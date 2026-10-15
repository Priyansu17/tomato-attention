"""Model package for the tomato leaf disease detection project (Phase 2D).

Contents:
  * ``backbone``  — torchvision backbone wrapper (Phase 2D: efficientnet_b0 only),
  * ``attention`` — attention modules (baseline / SE / CBAM, one module),
  * ``net``       — full network following the locked architecture.

Locked flow: image -> backbone -> attention -> GAP -> Dense(256)+ReLU ->
Dropout(0.4) -> Dense(10).
"""

from .attention import CBAM, SEBlock, BaselineAttention, build_attention
from .backbone import BackboneSpec, ModelConfigError, build_backbone
from .net import TomatoAttentionNet, build_model

__all__ = [
    "BackboneSpec",
    "BaselineAttention",
    "CBAM",
    "ModelConfigError",
    "SEBlock",
    "TomatoAttentionNet",
    "build_attention",
    "build_backbone",
    "build_model",
]
