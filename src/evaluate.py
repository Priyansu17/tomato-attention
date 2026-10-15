"""Evaluation script for the Tomato Leaf Disease Detection project.

Loads a saved checkpoint and evaluates the model on the test split, producing:
  - Accuracy, macro-averaged precision / recall / F1 / ROC-AUC
  - Per-class classification report (text + CSV)
  - Normalised confusion matrix (PNG)
  - Per-sample predictions CSV (for error analysis)
  - Inference time (mean ms/image)
  - JSON metrics summary

All paths come from configs/config.yaml; nothing is hardcoded.

Usage:
    python -m src.evaluate \\
        --checkpoint checkpoints/best_model.pt \\
        [--config configs/config.yaml] \\
        [--split test]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import cfg_get, load_config
from .data.dataloaders import build_dataloaders
from .data.splitting import SPLIT_NAMES
from .models import build_model
from .utils.logging_utils import get_logger, log_config_summary
from .utils.paths import ensure_dir
from .utils.seed import set_seed


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def _collect_predictions(
    loader: Any,
    model: nn.Module,
    device: torch.device,
    cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Forward pass over ``loader`` and collect labels, predictions and probabilities.

    Returns:
        ``(labels, predictions, probabilities, ms_per_image)`` where
        ``probabilities`` has shape ``(N, num_classes)``.
    """
    model.eval()
    use_amp = device.type == "cuda"

    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[list[float]] = []
    total_images = 0
    elapsed = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            t0 = time.perf_counter()
            with torch.autocast(device_type=device.type if device.type in ("cuda", "cpu") else "cpu", enabled=use_amp):
                logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - t0

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            total_images += len(labels)

    ms_per_image = (elapsed / max(total_images, 1)) * 1000.0
    return (
        np.array(all_labels, dtype=np.int64),
        np.array(all_preds, dtype=np.int64),
        np.array(all_probs, dtype=np.float32),
        ms_per_image,
    )


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    ms_per_image: float,
) -> dict[str, Any]:
    """Compute the full metric suite and return a serialisable dict."""
    acc = float(accuracy_score(labels, predictions))
    prec = float(precision_score(labels, predictions, average="macro", zero_division=0))
    rec = float(recall_score(labels, predictions, average="macro", zero_division=0))
    f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))

    # ROC-AUC (OVR, macro) — requires at least 2 classes present
    try:
        roc = float(roc_auc_score(labels, probabilities, multi_class="ovr",
                                   average="macro"))
    except ValueError:
        roc = float("nan")

    # Per-class report dict
    report_dict = classification_report(
        labels, predictions, target_names=class_names,
        output_dict=True, zero_division=0,
    )

    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "roc_auc_ovr_macro": roc,
        "ms_per_image": round(ms_per_image, 3),
        "num_samples": int(len(labels)),
        "classification_report": report_dict,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _save_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    out_path: Path,
    cfg: Mapping[str, Any],
) -> None:
    """Save a normalised confusion matrix figure."""
    ensure_dir(out_path.parent, cfg)
    cm = confusion_matrix(labels, predictions, normalize="true")
    n = len(class_names)
    fig_size = max(8, n * 1.1)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Normalised Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    dpi = int(cfg_get(cfg, "evaluation.figure_dpi", 150))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_predictions_csv(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    out_path: Path,
    cfg: Mapping[str, Any],
) -> None:
    """Save per-sample predictions CSV for error analysis."""
    ensure_dir(out_path.parent, cfg)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["sample_idx", "true_label", "true_class",
                  "pred_label", "pred_class", "correct", "confidence"]
        header += [f"prob_{name}" for name in class_names]
        writer.writerow(header)
        for i, (true, pred) in enumerate(zip(labels, predictions)):
            conf = float(probabilities[i, pred])
            row = [i, int(true), class_names[int(true)],
                   int(pred), class_names[int(pred)],
                   int(true == pred), round(conf, 6)]
            row += [round(float(p), 6) for p in probabilities[i]]
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main evaluate function
# ---------------------------------------------------------------------------

def evaluate(
    cfg: Mapping[str, Any],
    checkpoint_path: str | Path,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate the model at ``checkpoint_path`` on ``split`` and write artifacts.

    Returns the metrics dict.
    """
    logger = get_logger("tomato.evaluate")
    log_config_summary(cfg, logger)
    set_seed(cfg, logger)

    # Device
    device_cfg = str(cfg_get(cfg, "training.device", "auto")).lower()
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    logger.info("device: %s", device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    logger.info("loaded checkpoint from epoch %d", ckpt.get("epoch", "?"))

    # Build model and load weights
    model = build_model(cfg, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Data — eval uses production loaders (not smoke), test split only
    bundle = build_dataloaders(cfg, smoke=False)
    loader = bundle.loaders[split]
    class_names = list(cfg_get(cfg, "data.class_names"))
    logger.info("evaluating on '%s' split (%d samples)", split, len(bundle.datasets[split]))

    # Predictions
    labels, predictions, probabilities, ms_per_image = _collect_predictions(
        loader, model, device, cfg
    )

    # Metrics
    metrics = compute_metrics(labels, predictions, probabilities, class_names, ms_per_image)
    logger.info(
        "accuracy=%.4f  f1_macro=%.4f  roc_auc=%.4f  ms/img=%.2f",
        metrics["accuracy"], metrics["f1_macro"],
        metrics["roc_auc_ovr_macro"], metrics["ms_per_image"],
    )

    # Print classification report
    report_text = classification_report(
        labels, predictions, target_names=class_names,
        digits=int(cfg_get(cfg, "evaluation.classification_report.digits", 4)),
        zero_division=0,
    )
    print("\nClassification Report:\n" + report_text)

    # Output paths
    figs_dir = Path(cfg_get(cfg, "paths.figures_dir", "results/figures"))
    preds_dir = Path(cfg_get(cfg, "paths.predictions_dir", "results/predictions"))
    results_dir = Path(cfg_get(cfg, "paths.results_dir", "results"))
    fmt = str(cfg_get(cfg, "evaluation.figure_format", "png"))

    # Confusion matrix
    if bool(cfg_get(cfg, "evaluation.confusion_matrix.enabled", True)):
        cm_path = figs_dir / f"confusion_matrix_{split}.{fmt}"
        _save_confusion_matrix(labels, predictions, class_names, cm_path, cfg)
        logger.info("confusion matrix → %s", cm_path)

    # Predictions CSV
    if bool(cfg_get(cfg, "evaluation.save_predictions", True)):
        pred_path = preds_dir / f"predictions_{split}.csv"
        _save_predictions_csv(
            labels, predictions, probabilities, class_names, pred_path, cfg
        )
        logger.info("predictions → %s", pred_path)

    # Metrics JSON
    metrics_path = results_dir / f"metrics_{split}.json"
    ensure_dir(results_dir, cfg)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    logger.info("metrics → %s", metrics_path)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the .pt checkpoint file.")
    parser.add_argument("--split", default="test", choices=list(SPLIT_NAMES),
                        help="Which split to evaluate on (default: test).")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    evaluate(cfg, args.checkpoint, split=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
