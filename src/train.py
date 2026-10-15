"""Training loop for the Tomato Leaf Disease Detection project.

Implements the full training pipeline from configs/config.yaml:

  Phase A (freeze_backbone.epochs): backbone frozen, head-only training.
  Phase B (remaining epochs): backbone unfrozen, lower LR for backbone params.

Key behaviours (all controlled by config, nothing hardcoded):
  * AdamW with separate LRs for backbone and head.
  * Cosine annealing with linear warm-up.
  * Cross-entropy with label smoothing 0.1.
  * Mixed precision (AMP) on CUDA.
  * Gradient clipping.
  * Early stopping on val_loss (patience, min_delta from config).
  * Best-checkpoint and last-checkpoint saved to checkpoints/.
  * tqdm progress bars; step-level and epoch-level logging.

Entry point (smoke test):
    python -m src.train --smoke-test

Entry point (full training — only with explicit user approval):
    python -m src.train
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from .config import cfg_get, load_config
from .data.dataloaders import build_dataloaders
from .models import build_model
from .utils.logging_utils import get_logger, log_config_summary
from .utils.paths import ensure_dir
from .utils.seed import set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when a monitored metric does not improve."""

    def __init__(self, patience: int, min_delta: float, mode: str = "min") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float | None = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """Return True if training should stop."""
        if self.best is None:
            self.best = value
            return False
        if self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def _build_optimizer(model: nn.Module, cfg: Mapping[str, Any],
                     backbone_frozen: bool) -> AdamW:
    """Build AdamW with separate LR groups for backbone and head.

    When the backbone is frozen its parameters are excluded from the
    optimiser (they have no gradients), so only head params are included.
    After unfreezing, both groups are active.
    """
    lr_head = float(cfg_get(cfg, "optimizer.learning_rate"))
    lr_backbone = float(cfg_get(cfg, "optimizer.backbone_learning_rate"))
    wd = float(cfg_get(cfg, "optimizer.weight_decay"))
    betas = tuple(float(b) for b in (cfg_get(cfg, "optimizer.betas") or [0.9, 0.999]))
    eps = float(cfg_get(cfg, "optimizer.eps", 1e-8))

    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    if backbone_frozen:
        param_groups = [{"params": head_params, "lr": lr_head}]
    else:
        param_groups = [
            {"params": head_params, "lr": lr_head},
            {"params": backbone_params, "lr": lr_backbone},
        ]
    return AdamW(param_groups, weight_decay=wd, betas=betas, eps=eps)


def _build_scheduler(optimizer: AdamW, cfg: Mapping[str, Any],
                     total_steps: int) -> LambdaLR:
    """Linear warm-up then cosine annealing, step-level scheduler."""
    warmup_epochs = int(cfg_get(cfg, "scheduler.warmup_epochs", 3))
    eta_min = float(cfg_get(cfg, "scheduler.eta_min", 1e-6))
    # Determine steps per epoch from optimizer groups (already set)
    # total_steps = steps_per_epoch * max_epochs (passed in)

    # A simple linear-warmup + cosine decay lambda
    def _lr_lambda(step: int) -> float:
        if step < warmup_epochs:
            return float(step + 1) / float(max(1, warmup_epochs))
        progress = (step - warmup_epochs) / max(1, total_steps - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # scale so the floor is eta_min / base_lr (approximate for param groups)
        base_lr = optimizer.param_groups[0]["lr"]
        floor = eta_min / max(base_lr, 1e-12)
        return floor + (1.0 - floor) * cosine

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


# ---------------------------------------------------------------------------
# Core training / evaluation steps
# ---------------------------------------------------------------------------

def _run_epoch(
    loader: Any,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: AdamW | None,
    scaler: Any,
    device: torch.device,
    cfg: Mapping[str, Any],
    phase: str,
    epoch: int,
    scheduler: LambdaLR | None = None,
) -> dict[str, float]:
    """Run one epoch of training or evaluation.

    Args:
        loader: DataLoader for this split.
        optimizer: None for eval.
        scaler: GradScaler (no-op when not CUDA).
        phase: "train" | "val" | "test".
        scheduler: epoch-level LambdaLR (stepped after each train epoch).

    Returns:
        dict with "loss" and "accuracy" keys.
    """
    is_train = phase == "train"
    model.train(is_train)
    grad_accum = int(cfg_get(cfg, "training.grad_accum_steps", 1))
    clip_norm = float(cfg_get(cfg, "training.grad_clip_norm", 1.0))
    use_amp = scaler is not None and device.type == "cuda"

    total_loss = 0.0
    correct = 0
    total = 0
    batches = 0

    bar = tqdm(loader, desc=f"  {phase} ep{epoch}", leave=False, unit="batch")
    with torch.set_grad_enabled(is_train):
        for step, (images, labels) in enumerate(bar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type if device.type in ("cuda", "cpu") else "cpu", enabled=use_amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels) / grad_accum

            if is_train:
                scaler.scale(loss).backward()
                if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            batch_loss = loss.item() * grad_accum
            preds = logits.argmax(dim=1)
            batch_correct = (preds == labels).sum().item()

            total_loss += batch_loss * len(labels)
            correct += batch_correct
            total += len(labels)
            batches += 1
            bar.set_postfix(loss=f"{batch_loss:.4f}",
                            acc=f"{batch_correct/len(labels):.3f}")

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return {"loss": avg_loss, "accuracy": accuracy}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    epoch: int,
    metrics: dict[str, Any],
    path: Path,
    cfg: Mapping[str, Any],
) -> None:
    """Save a checkpoint dict compatible with evaluate.py."""
    ensure_dir(path.parent, cfg)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": dict(cfg),
        },
        path,
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: Mapping[str, Any], smoke: bool = False) -> dict[str, Any]:
    """Run the full training procedure and return the best validation metrics.

    Args:
        cfg: validated configuration mapping.
        smoke: run in smoke-test mode (tiny subset, 1 epoch max).

    Returns:
        dict with keys "best_epoch", "best_val_loss", "best_val_accuracy",
        "history" (list of per-epoch dicts).
    """
    logger = get_logger("tomato.train")
    log_config_summary(cfg, logger)
    seed = set_seed(cfg, logger)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device_cfg = str(cfg_get(cfg, "training.device", "auto")).lower()
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    logger.info("device: %s", device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    bundle = build_dataloaders(cfg, smoke=smoke)
    train_loader = bundle.loaders["train"]
    val_loader = bundle.loaders["val"]
    logger.info(
        "splits — train: %d  val: %d  test: %d",
        len(bundle.datasets["train"]),
        len(bundle.datasets["val"]),
        len(bundle.datasets["test"]),
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(cfg).to(device)
    num_classes = int(cfg_get(cfg, "model.num_classes"))
    label_smoothing = float(cfg_get(cfg, "training.loss.label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # ------------------------------------------------------------------
    # Training schedule parameters
    # ------------------------------------------------------------------
    max_epochs = 1 if smoke else int(cfg_get(cfg, "training.max_epochs"))
    freeze_enabled = bool(cfg_get(cfg, "training.freeze_backbone.enabled", False))
    freeze_epochs = int(cfg_get(cfg, "training.freeze_backbone.epochs", 0))
    if smoke:
        freeze_epochs = 0  # no point freezing for 1 epoch smoke test

    es_enabled = bool(cfg_get(cfg, "training.early_stopping.enabled", False))
    es_patience = int(cfg_get(cfg, "training.early_stopping.patience", 7))
    es_min_delta = float(cfg_get(cfg, "training.early_stopping.min_delta", 0.0))
    es_mode = str(cfg_get(cfg, "training.early_stopping.mode", "min"))
    early_stopper = EarlyStopping(es_patience, es_min_delta, es_mode) if es_enabled else None

    ckpt_dir = Path(cfg_get(cfg, "paths.checkpoints_dir", "checkpoints"))
    best_ckpt = ckpt_dir / str(cfg_get(cfg, "training.checkpoint.filename", "best_model.pt"))
    last_ckpt = ckpt_dir / "last_model.pt"
    ckpt_mode = str(cfg_get(cfg, "training.checkpoint.mode", "min"))

    use_amp = bool(cfg_get(cfg, "training.mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ------------------------------------------------------------------
    # Phase A: freeze backbone
    # ------------------------------------------------------------------
    if freeze_enabled and freeze_epochs > 0:
        logger.info("Phase A: freezing backbone for %d epochs", freeze_epochs)
        for p in model.backbone.parameters():
            p.requires_grad_(False)

    optimizer = _build_optimizer(model, cfg, backbone_frozen=(freeze_enabled and freeze_epochs > 0))
    scheduler = _build_scheduler(optimizer, cfg, total_steps=max_epochs)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # Phase B: unfreeze backbone after freeze_epochs
        if freeze_enabled and epoch == freeze_epochs + 1:
            logger.info("Phase B: unfreezing backbone (epoch %d)", epoch)
            for p in model.backbone.parameters():
                p.requires_grad_(True)
            # Rebuild optimizer so backbone params get their own LR group
            optimizer = _build_optimizer(model, cfg, backbone_frozen=False)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            scheduler = _build_scheduler(optimizer, cfg,
                                         total_steps=max_epochs - freeze_epochs)

        # Train
        train_metrics = _run_epoch(
            train_loader, model, criterion, optimizer, scaler,
            device, cfg, "train", epoch, scheduler,
        )
        # Val
        val_metrics = _run_epoch(
            val_loader, model, criterion, None, scaler,
            device, cfg, "val", epoch,
        )

        # Step scheduler (epoch-level)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "lr_head": optimizer.param_groups[0]["lr"],
            "epoch_time_s": round(epoch_time, 1),
        }
        history.append(epoch_record)

        logger.info(
            "epoch %3d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f | %.1fs",
            epoch, max_epochs,
            train_metrics["loss"], train_metrics["accuracy"],
            val_metrics["loss"], val_metrics["accuracy"],
            epoch_time,
        )

        # Checkpoint
        is_best = (
            (ckpt_mode == "min" and val_metrics["loss"] < best_val_loss) or
            (ckpt_mode == "max" and val_metrics["accuracy"] > best_val_acc)
        )
        if is_best:
            best_val_loss = val_metrics["loss"]
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            _save_checkpoint(model, optimizer, epoch, val_metrics, best_ckpt, cfg)
            logger.info("  [best] saved checkpoint (val_loss=%.4f)", best_val_loss)

        if bool(cfg_get(cfg, "training.checkpoint.save_last", True)):
            _save_checkpoint(model, optimizer, epoch, val_metrics, last_ckpt, cfg)

        # Early stopping
        if early_stopper is not None and early_stopper(val_metrics["loss"]):
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    # ------------------------------------------------------------------
    # Save history
    # ------------------------------------------------------------------
    logs_dir = Path(cfg_get(cfg, "paths.logs_dir", "results/logs"))
    ensure_dir(logs_dir, cfg)
    history_path = logs_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    logger.info("training history -> %s", history_path)

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_acc,
        "total_epochs": len(history),
        "history": history,
    }
    logger.info(
        "Training complete. Best epoch %d — val_loss=%.4f val_acc=%.4f",
        best_epoch, best_val_loss, best_val_acc,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the Tomato Attention model. Full training requires explicit approval."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Smoke-test mode: tiny subset, at most 1 epoch. Safe to run anytime."
    )
    parser.add_argument(
        "--attention", default=None,
        help="Override attention.type (baseline | se | cbam) for ablation runs."
    )
    parser.add_argument(
        "--backbone", default=None,
        help="Override model.backbone for ablation runs."
    )
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {}
    if args.attention:
        overrides["attention.type"] = args.attention
    if args.backbone:
        overrides["model.backbone"] = args.backbone

    cfg = load_config(args.config, overrides=overrides or None)
    train(cfg, smoke=args.smoke_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
