"""Backbone factory (Phase 2D).

Wraps the torchvision CNN backbone named by ``model.backbone`` and exposes the feature
extractor that feeds the attention module of the locked architecture:

    image -> backbone (features) -> attention -> GAP -> Dense(256)+ReLU -> Dropout -> Dense(10)

Phase 2D scope (approved decisions):
  * ``efficientnet_b0`` only — ``resnet50`` and ``mobilenet_v3_large`` raise
    :class:`ModelConfigError` instead of silently falling back.
  * ``attention.insertion_stage`` must be ``last`` — stage2/stage3/stage4 raise too.
  * Tests must build with ``pretrained=False`` so ImageNet weights are never downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from ..config import cfg_get

SUPPORTED_BACKBONES = ("efficientnet_b0",)
UNSUPPORTED_BACKBONES = ("resnet50", "mobilenet_v3_large")
SUPPORTED_INSERTION_STAGES = ("last",)
UNSUPPORTED_INSERTION_STAGES = ("stage4", "stage3", "stage2")


class ModelConfigError(ValueError):
    """Raised when a model configuration value is unsupported in this phase."""


@dataclass(frozen=True)
class BackboneSpec:
    """Description of the built backbone."""

    name: str
    num_features: int
    insertion_stage: str


def build_backbone(cfg: Any, pretrained: bool | None = None) -> tuple[nn.Module, BackboneSpec]:
    """Build the configured backbone's feature extractor.

    Args:
        cfg: validated configuration mapping.
        pretrained: override for ``model.pretrained``; tests pass ``False`` so no
            ImageNet weights are ever downloaded during testing.

    Returns:
        ``(feature_extractor, spec)`` where the extractor maps ``(B, 3, 224, 224)``
        inputs to feature maps with ``spec.num_features`` channels.
    """
    name = str(cfg_get(cfg, "model.backbone"))
    stage = str(cfg_get(cfg, "attention.insertion_stage"))
    if name in UNSUPPORTED_BACKBONES:
        raise ModelConfigError(
            f"backbone '{name}' is not supported yet (Phase 2D supports only "
            f"{', '.join(SUPPORTED_BACKBONES)}); it must not silently fall back"
        )
    if name not in SUPPORTED_BACKBONES:
        raise ModelConfigError(
            f"unknown model.backbone '{name}'; supported: {', '.join(SUPPORTED_BACKBONES)}"
        )
    if stage in UNSUPPORTED_INSERTION_STAGES:
        raise ModelConfigError(
            f"attention.insertion_stage '{stage}' is not supported yet (Phase 2D supports "
            "only 'last'); it must not silently fall back"
        )
    if stage not in SUPPORTED_INSERTION_STAGES:
        raise ModelConfigError(
            f"unknown attention.insertion_stage '{stage}'; supported: "
            f"{', '.join(SUPPORTED_INSERTION_STAGES)}"
        )
    if name == "efficientnet_b0":  # only reachable name after the checks above
        features, num_features = _build_efficientnet_b0(cfg, pretrained)
        return features, BackboneSpec(name=name, num_features=num_features,
                                      insertion_stage=stage)
    raise ModelConfigError(f"backbone '{name}' is not wired up")  # pragma: no cover


def _build_efficientnet_b0(cfg: Any, pretrained: bool | None) -> tuple[nn.Module, int]:
    """Return EfficientNet-B0's ``features`` trunk and its output channel count."""
    from torchvision.models import efficientnet_b0

    use_pretrained = (bool(cfg_get(cfg, "model.pretrained"))
                      if pretrained is None else pretrained)
    if use_pretrained:
        try:
            from torchvision.models import EfficientNet_B0_Weights
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        except ImportError:  # very old torchvision without the weights enum
            model = efficientnet_b0(pretrained=True)
    else:
        try:
            model = efficientnet_b0(weights=None)
        except TypeError:  # very old torchvision used the ``pretrained`` flag
            model = efficientnet_b0(pretrained=False)
    num_features = int(getattr(model.classifier[1], "in_features", 1280))
    return model.features, num_features
