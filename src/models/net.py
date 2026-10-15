"""Full classification network (Phase 2D) following the locked architecture.

    image -> backbone -> attention -> GAP -> Dense(256) + ReLU -> Dropout(0.4) -> Dense(10)

Every dimension comes from the ``model`` config section. ``forward`` returns class
logits only; :meth:`TomatoAttentionNet.predict` adds softmax confidence for inference.
No training, evaluation, or Grad-CAM code lives here.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..config import cfg_get
from .attention import build_attention
from .backbone import ModelConfigError, build_backbone

_ACTIVATIONS: dict[str, type[nn.Module]] = {"relu": nn.ReLU}


class TomatoAttentionNet(nn.Module):
    """Backbone + attention + pooled dense head producing one logit per class."""

    def __init__(
        self,
        backbone: nn.Module,
        attention: nn.Module,
        num_features: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
        activation: str = "relu",
        global_pool: str = "avg",
        backbone_name: str = "",
        insertion_stage: str = "last",
    ) -> None:
        super().__init__()
        if global_pool != "avg":
            raise ModelConfigError(
                f"model.global_pool '{global_pool}' is not supported yet (only 'avg')"
            )
        if activation not in _ACTIVATIONS:
            raise ModelConfigError(
                f"model.head.activation '{activation}' is not supported yet "
                f"(only {', '.join(sorted(_ACTIVATIONS))})"
            )
        self.backbone = backbone
        self.attention = attention
        self.backbone_name = backbone_name
        self.insertion_stage = insertion_stage
        self.num_features = int(num_features)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_hidden = nn.Linear(int(num_features), int(hidden_dim))
        self.activation = _ACTIVATIONS[activation]()
        self.dropout = nn.Dropout(float(dropout))
        self.fc_out = nn.Linear(int(hidden_dim), int(num_classes))

    def extract_refined_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return the post-attention feature map (pre-pool) — future Grad-CAM hook."""
        return self.attention(self.backbone(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits with shape ``(batch, num_classes)``."""
        features = self.global_pool(self.extract_refined_features(images)).flatten(1)
        hidden = self.activation(self.fc_hidden(features))
        return self.fc_out(self.dropout(hidden))

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> tuple[int, float]:
        """Return ``(class_index, confidence)`` for a single image (eval mode).

        Accepts ``(3, H, W)`` or ``(1, 3, H, W)``; dropout is inactive in eval mode
        so repeated calls on the same input return the same result.
        """
        was_training = self.training
        self.eval()
        batched = image if image.dim() == 4 else image.unsqueeze(0)
        probabilities = torch.softmax(self(batched), dim=1)
        confidence, index = probabilities.max(dim=1)
        if was_training:
            self.train()
        return int(index[0].item()), float(confidence[0].item())


def build_model(cfg: Any, pretrained: bool | None = None) -> TomatoAttentionNet:
    """Build the full network from config.

    Args:
        cfg: validated configuration mapping.
        pretrained: override for ``model.pretrained``; tests pass ``False`` so no
            ImageNet weights are downloaded during testing.
    """
    backbone, spec = build_backbone(cfg, pretrained=pretrained)
    attention = build_attention(cfg, channels=spec.num_features)
    head = cfg_get(cfg, "model.head") or {}
    return TomatoAttentionNet(
        backbone=backbone,
        attention=attention,
        num_features=spec.num_features,
        num_classes=int(cfg_get(cfg, "model.num_classes")),
        hidden_dim=int(head["hidden_dim"]),
        dropout=float(head["dropout"]),
        activation=str(head.get("activation", "relu")),
        global_pool=str(cfg_get(cfg, "model.global_pool", "avg")),
        backbone_name=spec.name,
        insertion_stage=spec.insertion_stage,
    )
