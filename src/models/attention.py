"""Attention modules (Phase 2D): baseline identity, SE channel attention, CBAM.

Locked CBAM specification (.clinerules section 10), applied after the last backbone
feature block:

* Channel attention FIRST: global average pool + global max pool -> shared MLP with
  reduction ratio 16 -> sum of the two MLP outputs -> sigmoid -> channel-wise multiply.
* Spatial attention SECOND: channel-wise average/max pooling -> concatenate -> 7x7
  convolution -> sigmoid -> spatial multiply.

All parameters come from the ``attention`` config section; nothing is hardcoded.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..config import cfg_get
from .backbone import ModelConfigError


class BaselineAttention(nn.Module):
    """Identity attention: the ablation baseline that re-weights nothing."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the input feature map unchanged."""
        return features


class SEBlock(nn.Module):
    """Squeeze-and-excitation: global average pool -> shared MLP -> sigmoid scale."""

    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        channels = int(channels)
        if channels // reduction_ratio < 1:
            raise ModelConfigError(
                f"SE reduction_ratio {reduction_ratio} is too large for {channels} channels"
            )
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels = features.shape[:2]
        scale = torch.sigmoid(self.excitation(features.mean(dim=(2, 3))))
        return features * scale.view(batch, channels, 1, 1)


class CBAM(nn.Module):
    """Convolutional block attention: channel module first, spatial module second."""

    def __init__(self, channels: int, reduction_ratio: int = 16,
                 spatial_kernel: int = 7) -> None:
        super().__init__()
        channels = int(channels)
        if channels // reduction_ratio < 1:
            raise ModelConfigError(
                f"CBAM reduction_ratio {reduction_ratio} is too large for {channels} channels"
            )
        if spatial_kernel % 2 == 0:
            raise ModelConfigError(
                f"CBAM spatial_kernel must be a positive odd integer, got {spatial_kernel}"
            )
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
        )
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False
        )

    def channel_attention(self, features: torch.Tensor) -> torch.Tensor:
        """Shared-MLP channel attention over global average and global max pooling."""
        batch, channels = features.shape[:2]
        average = features.mean(dim=(2, 3))
        maximum = features.amax(dim=(2, 3))
        scale = torch.sigmoid(self.channel_mlp(average) + self.channel_mlp(maximum))
        return features * scale.view(batch, channels, 1, 1)

    def spatial_attention(self, features: torch.Tensor) -> torch.Tensor:
        """7x7 convolution over concatenated channel-wise average/max maps."""
        pooled = torch.cat(
            [features.mean(dim=1, keepdim=True), features.amax(dim=1, keepdim=True)],
            dim=1,
        )
        return features * torch.sigmoid(self.spatial_conv(pooled))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply channel attention first, then spatial attention."""
        return self.spatial_attention(self.channel_attention(features))


def build_attention(cfg: Any, channels: int) -> nn.Module:
    """Build the configured attention module for ``channels`` feature channels.

    Raises:
        ModelConfigError: for an unknown/unsupported ``attention.type`` or any
            configuration that deviates from the locked CBAM specification.
    """
    kind = str(cfg_get(cfg, "attention.type"))
    if kind == "baseline":
        return BaselineAttention(channels)
    if kind == "se":
        ratio = int(cfg_get(cfg, "attention.se.reduction_ratio"))
        return SEBlock(channels, reduction_ratio=ratio)
    if kind == "cbam":
        if not bool(cfg_get(cfg, "attention.cbam.channel_first")):
            raise ModelConfigError(
                "Phase 2D implements attention.cbam.channel_first=true only "
                "(locked: channel attention then spatial attention)"
            )
        channel_pool = list(cfg_get(cfg, "attention.cbam.channel.pooling") or [])
        spatial_pool = list(cfg_get(cfg, "attention.cbam.spatial.pooling") or [])
        if channel_pool != ["avg", "max"] or spatial_pool != ["avg", "max"]:
            raise ModelConfigError(
                "CBAM pooling must be [avg, max] for both modules (locked specification), "
                f"got channel={channel_pool} spatial={spatial_pool}"
            )
        ratio = int(cfg_get(cfg, "attention.cbam.channel.reduction_ratio"))
        kernel = int(cfg_get(cfg, "attention.cbam.spatial.kernel_size"))
        return CBAM(channels, reduction_ratio=ratio, spatial_kernel=kernel)
    raise ModelConfigError(
        f"unsupported attention.type '{kind}'; supported: baseline, se, cbam"
    )
