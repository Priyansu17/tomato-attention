"""Tests for the Phase 2D model package: backbone, attention modules, network.

No ImageNet weights are ever downloaded here (approved decision 4): every build passes
``pretrained=False`` explicitly or overrides ``model.pretrained`` first, while the
production config keeps ``model.pretrained: true``.
"""

from __future__ import annotations

import copy

import pytest
import torch

from src.config import cfg_get
from src.models import (
    CBAM,
    SEBlock,
    BaselineAttention,
    ModelConfigError,
    TomatoAttentionNet,
    build_attention,
    build_backbone,
    build_model,
)

BATCH, CHANNELS, HEIGHT, WIDTH = 2, 3, 224, 224


def _cfg(real_config, **overrides):
    """Deep copy of the repo config with dotted-key overrides applied."""
    cfg = copy.deepcopy(real_config)
    for key, value in overrides.items():
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return cfg


def test_backbone_is_efficientnet_b0_and_stops_on_unsupported(real_config):
    """Decision 1: EfficientNet-B0 builds; resnet50/mobilenet raise, never fall back."""
    cfg = _cfg(real_config, **{"model.pretrained": False})
    features, spec = build_backbone(cfg, pretrained=False)
    assert spec.name == "efficientnet_b0"
    assert spec.insertion_stage == "last"
    assert spec.num_features == 1280
    output = features(torch.zeros(1, CHANNELS, HEIGHT, WIDTH))
    assert output.shape[:2] == (1, spec.num_features)
    assert output.dim() == 4

    for name in ("resnet50", "mobilenet_v3_large"):
        with pytest.raises(ModelConfigError, match="not supported yet"):
            build_backbone(_cfg(real_config, **{"model.backbone": name}), pretrained=False)
    with pytest.raises(ModelConfigError, match="unknown model.backbone"):
        build_backbone(_cfg(real_config, **{"model.backbone": "vgg16"}), pretrained=False)


def test_insertion_stage_last_only(real_config):
    """Decision 2: stage2/3/4 raise a configuration error instead of falling back."""
    for stage in ("stage4", "stage3", "stage2"):
        cfg = _cfg(real_config, **{"attention.insertion_stage": stage})
        with pytest.raises(ModelConfigError, match="not supported yet"):
            build_backbone(cfg, pretrained=False)
    with pytest.raises(ModelConfigError, match="unknown attention.insertion_stage"):
        build_backbone(
            _cfg(real_config, **{"attention.insertion_stage": "stage9"}),
            pretrained=False,
        )


@pytest.mark.parametrize("kind", ["baseline", "se", "cbam"])
def test_attention_variants_preserve_shape_and_actually_scale(real_config, kind):
    """Each attention type keeps ``(B, C, H, W)``; baseline is identity, others act."""
    cfg = _cfg(real_config, **{"attention.type": kind})
    attention = build_attention(cfg, channels=32)
    features = torch.randn(BATCH, 32, 16, 16)
    refined = attention(features)
    assert refined.shape == features.shape
    assert torch.isfinite(refined).all()
    if kind == "baseline":
        assert isinstance(attention, BaselineAttention)
        assert torch.equal(refined, features)
    else:
        assert not torch.allclose(refined, features), f"{kind} must re-weight features"


def test_attention_type_and_cbam_options_are_validated(real_config):
    """Unknown types and locked-spec deviations raise instead of silently degrading."""
    with pytest.raises(ModelConfigError, match="unsupported attention.type"):
        build_attention(_cfg(real_config, **{"attention.type": "transformer"}), 32)
    with pytest.raises(ModelConfigError, match="channel_first"):
        build_attention(
            _cfg(real_config, **{"attention.cbam.channel_first": False}), 32
        )
    with pytest.raises(ModelConfigError, match="pooling"):
        build_attention(
            _cfg(real_config, **{"attention.cbam.spatial.pooling": ["avg"]}), 32
        )
    with pytest.raises(ModelConfigError, match="too large"):
        build_attention(_cfg(real_config, **{"attention.type": "se"}), channels=8)


def test_cbam_locked_spec_shapes(real_config):
    """Locked CBAM: 16-reduction shared MLP and a 7x7 spatial convolution."""
    cfg = _cfg(real_config, **{"attention.type": "cbam"})
    cbam = build_attention(cfg, channels=32)
    assert isinstance(cbam, CBAM)
    assert cbam.spatial_conv.kernel_size == (7, 7)
    assert cbam.spatial_conv.weight.shape == (1, 2, 7, 7)
    assert cbam.channel_mlp[0].in_features == 32
    assert cbam.channel_mlp[0].out_features == 32 // 16
    assert cbam.channel_mlp[-1].out_features == 32
    assert cfg_get(cfg, "attention.cbam.channel.reduction_ratio") == 16
    assert cfg_get(cfg, "attention.cbam.spatial.kernel_size") == 7

    refined = cbam(torch.randn(BATCH, 32, 16, 16))
    assert refined.shape == (BATCH, 32, 16, 16)
    assert not torch.allclose(refined, torch.randn(BATCH, 32, 16, 16))


def test_se_reduction_ratio_comes_from_config(real_config):
    """SE hidden size follows attention.se.reduction_ratio (32 -> 8 for ratio 4)."""
    cfg = _cfg(real_config, **{"attention.type": "se", "attention.se.reduction_ratio": 4})
    se = build_attention(cfg, channels=32)
    assert isinstance(se, SEBlock)
    assert se.excitation[0].in_features == 32
    assert se.excitation[0].out_features == 8
    assert se(torch.randn(BATCH, 32, 8, 8)).shape == (BATCH, 32, 8, 8)


@pytest.mark.parametrize("kind", ["baseline", "se", "cbam"])
def test_network_logits_shape_for_each_attention(real_config, kind):
    """The locked architecture yields (batch, num_classes) logits for every attention."""
    cfg = _cfg(real_config, **{"attention.type": kind, "model.pretrained": False})
    model = build_model(cfg, pretrained=False).eval()
    logits = model(torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH))
    assert logits.shape == (BATCH, int(cfg_get(cfg, "model.num_classes")))
    assert torch.isfinite(logits).all()


def test_network_head_predict_and_eval_determinism(real_config):
    """Head dims follow config; predict returns class+confidence; eval is deterministic."""
    cfg = _cfg(real_config, **{"model.pretrained": False})
    model = build_model(cfg, pretrained=False)
    assert isinstance(model, TomatoAttentionNet)
    assert model.backbone_name == "efficientnet_b0"
    assert model.insertion_stage == "last"
    assert model.fc_hidden.out_features == 256
    assert model.fc_out.out_features == 10
    assert model.dropout.p == 0.4

    model.eval()
    images = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    with torch.no_grad():
        logits = model(images)
        assert torch.equal(logits, model(images)), "eval mode must be deterministic"
    index, confidence = model.predict(images[0])
    assert 0 <= index < 10 and 0.0 <= confidence <= 1.0
    assert index == int(logits[0].argmax())
    assert confidence == pytest.approx(float(torch.softmax(logits[0], dim=0).max()))


def test_pretrained_flag_and_head_dimensions_are_config_driven(real_config):
    """Decision 4 + no hardcoded dims: config drives weights and head size."""
    assert cfg_get(real_config, "model.pretrained") is True
    cfg = _cfg(real_config, **{"model.pretrained": False,
                               "model.head.hidden_dim": 64,
                               "model.head.dropout": 0.25})
    assert cfg_get(cfg, "model.pretrained") is False
    model = build_model(cfg)  # reads pretrained=False from config: offline build
    assert model.fc_hidden.in_features == 1280
    assert model.fc_hidden.out_features == 64
    assert model.dropout.p == 0.25
    assert model.fc_out.out_features == int(cfg_get(cfg, "model.num_classes"))


@pytest.mark.realdata
def test_one_real_val_batch_passes_through_the_network(real_config):
    """Forward-only check on a real Phase 2C val batch: no training, no downloads."""
    from src.data.dataloaders import build_dataloaders

    cfg = copy.deepcopy(real_config)
    cfg["smoke_test"]["num_workers"] = 0
    bundle = build_dataloaders(cfg, smoke=True)
    model = build_model(cfg, pretrained=False)
    model.eval()
    images, labels = next(iter(bundle.loaders["val"]))
    assert images.shape[1] == int(cfg_get(cfg, "model.input.channels")) == 3
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (labels.shape[0], int(cfg_get(cfg, "model.num_classes")))

