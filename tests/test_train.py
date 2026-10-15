"""Tests for src/train.py — smoke-test mode only (never full training).

All tests use the smoke_test config overrides (1 epoch, tiny subset, 0 workers)
so they finish in seconds. The checkpoint produced is written to a tmp_path
directory, never to checkpoints/.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from src.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def smoke_cfg():
    """Config with num_workers=0 and checkpoints in a temp dir — module-scoped."""
    return load_config(PROJECT_ROOT / "configs" / "config.yaml")


def _override(cfg: dict, **kw) -> dict:
    c = copy.deepcopy(cfg)
    for k, v in kw.items():
        node = c
        parts = k.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = v
    return c


@pytest.mark.realdata
def test_smoke_train_runs_one_epoch(tmp_path, smoke_cfg):
    """Smoke training completes without error and saves checkpoints."""
    from src.train import train

    cfg = _override(
        smoke_cfg,
        **{
            "paths.checkpoints_dir": str(tmp_path / "ckpts"),
            "paths.logs_dir": str(tmp_path / "logs"),
            "smoke_test.num_workers": 0,
            "training.freeze_backbone.enabled": False,
        },
    )
    summary = train(cfg, smoke=True)

    assert summary["total_epochs"] == 1
    assert 0.0 <= summary["best_val_accuracy"] <= 1.0
    assert summary["best_val_loss"] >= 0.0
    assert (tmp_path / "ckpts" / "best_model.pt").exists()
    # history JSON
    history_file = tmp_path / "logs" / "training_history.json"
    assert history_file.exists()
    history = json.loads(history_file.read_text())
    assert len(history) == 1
    assert "train_loss" in history[0]
    assert "val_accuracy" in history[0]


@pytest.mark.realdata
def test_smoke_train_checkpoint_loadable(tmp_path, smoke_cfg):
    """The saved checkpoint can be reloaded and the model runs inference."""
    from src.train import train
    from src.models import build_model

    cfg = _override(
        smoke_cfg,
        **{
            "paths.checkpoints_dir": str(tmp_path / "ckpts"),
            "paths.logs_dir": str(tmp_path / "logs"),
            "smoke_test.num_workers": 0,
            "training.freeze_backbone.enabled": False,
        },
    )
    train(cfg, smoke=True)
    ckpt_path = tmp_path / "ckpts" / "best_model.pt"
    assert ckpt_path.exists()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "epoch" in ckpt

    model = build_model(cfg, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()


@pytest.mark.realdata
@pytest.mark.parametrize("attention", ["baseline", "se", "cbam"])
def test_smoke_train_attention_variants(tmp_path, smoke_cfg, attention):
    """Smoke training works for all three attention variants."""
    from src.train import train

    cfg = _override(
        smoke_cfg,
        **{
            "attention.type": attention,
            "paths.checkpoints_dir": str(tmp_path / "ckpts" / attention),
            "paths.logs_dir": str(tmp_path / "logs" / attention),
            "smoke_test.num_workers": 0,
            "training.freeze_backbone.enabled": False,
        },
    )
    summary = train(cfg, smoke=True)
    assert summary["total_epochs"] == 1
    assert (tmp_path / "ckpts" / attention / "best_model.pt").exists()
