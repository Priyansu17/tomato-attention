"""Build outputs/ablation.csv and outputs/ablation.md from per-variant metrics.

Run AFTER all three evaluate runs have finished:

    python -m src.ablation

Reads:
    results/metrics_test_baseline.json
    results/metrics_test_se.json
    results/metrics_test_cbam.json
    results/logs/training_history_best_model_<variant>.json  (optional)

Writes:
    outputs/ablation.csv
    outputs/ablation.md
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANTS = ["baseline", "se", "cbam"]
RESULTS_DIR = Path("results")
OUTPUTS_DIR = Path("outputs")
LOGS_DIR = RESULTS_DIR / "logs"

METRIC_COLS = [
    "variant",
    "accuracy",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "roc_auc_ovr_macro",
    "ms_per_image",
    "best_epoch",
    "total_epochs",
    "best_val_accuracy",
    "best_val_loss",
]


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def _load_metrics(variant: str) -> dict:
    path = RESULTS_DIR / f"metrics_test_{variant}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {path}\n"
            "Run evaluate.py for this variant first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_history(variant: str) -> dict:
    """Return training summary dict, or empty dict if not found."""
    path = LOGS_DIR / f"training_history_best_model_{variant}.json"
    if not path.exists():
        return {}
    try:
        epochs: list = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(epochs, list) or not epochs:
            return {}
        # Find best epoch by lowest val_loss
        best = min(epochs, key=lambda e: e.get("val_loss", float("inf")))
        return {
            "best_epoch": best.get("epoch"),
            "total_epochs": len(epochs),
            "best_val_accuracy": best.get("val_accuracy"),
            "best_val_loss": best.get("val_loss"),
        }
    except Exception:
        return {}


def _build_row(variant: str) -> dict:
    m = _load_metrics(variant)
    h = _load_history(variant)

    row: dict = {"variant": variant}

    # Test-set metrics
    row["accuracy"] = m.get("accuracy")
    row["f1_macro"] = m.get("f1_macro")
    row["precision_macro"] = m.get("precision_macro")
    row["recall_macro"] = m.get("recall_macro")
    row["roc_auc_ovr_macro"] = m.get("roc_auc_ovr_macro")
    row["ms_per_image"] = m.get("ms_per_image")

    # Training history (best epoch / convergence)
    row["best_epoch"] = h.get("best_epoch")
    row["total_epochs"] = h.get("total_epochs")
    row["best_val_accuracy"] = h.get("best_val_accuracy")
    row["best_val_loss"] = h.get("best_val_loss")

    return row


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV  -> {path}")


def _fmt(value, fmt=".4f") -> str:
    if value is None:
        return "—"
    try:
        if fmt == "d":
            return format(int(value), "d")
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)



def _write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Ablation Study — Tomato Leaf Disease Detection",
        "",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  "
        f"(seed 42, EfficientNet-B0 backbone, 70/15/15 split)_",
        "",
        "## Test-Set Results",
        "",
        "| Variant | Accuracy | F1 Macro | Precision | Recall | ROC-AUC | ms/img |",
        "|---------|----------|----------|-----------|--------|---------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']:8s} "
            f"| {_fmt(r['accuracy'])} "
            f"| {_fmt(r['f1_macro'])} "
            f"| {_fmt(r['precision_macro'])} "
            f"| {_fmt(r['recall_macro'])} "
            f"| {_fmt(r['roc_auc_ovr_macro'])} "
            f"| {_fmt(r['ms_per_image'], '.2f')} |"
        )

    lines += [
        "",
        "## Training Convergence",
        "",
        "| Variant | Best Epoch | Total Epochs | Val Acc (best) | Val Loss (best) |",
        "|---------|------------|--------------|----------------|-----------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']:8s} "
            f"| {_fmt(r['best_epoch'], 'd'):>10s} "
            f"| {_fmt(r['total_epochs'], 'd'):>12s} "
            f"| {_fmt(r['best_val_accuracy'])} "
            f"| {_fmt(r['best_val_loss'])} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- **baseline**: EfficientNet-B0 backbone → GAP → Dense(256) → Dropout(0.4) → Dense(10). No attention.",
        "- **se**: Squeeze-and-Excitation channel attention added after backbone features.",
        "- **cbam**: Channel + Spatial attention (CBAM). Channel: reduction=16, avg+max pool. Spatial: 7×7 conv.",
        "- All three variants trained with **identical** seed, split, optimiser, and schedule.",
        "- Grad-CAM images are in `results/gradcam/<variant>/`.",
        "",
        "> Results labelled **unverified** if the checkpoint cannot be reproduced from the current code and config.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"MD   -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build ablation CSV and Markdown from per-variant metrics JSONs."
    )
    parser.add_argument(
        "--variants", nargs="+", default=VARIANTS, choices=VARIANTS,
        help="Which variants to include (default: all three).",
    )
    parser.add_argument(
        "--out-dir", default=str(OUTPUTS_DIR),
        help="Output directory (default: outputs/).",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    rows = []
    missing = []

    for variant in args.variants:
        try:
            row = _build_row(variant)
            rows.append(row)
            print(
                f"[{variant:8s}] accuracy={_fmt(row['accuracy'])}  "
                f"f1={_fmt(row['f1_macro'])}  "
                f"roc_auc={_fmt(row['roc_auc_ovr_macro'])}"
            )
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            missing.append(variant)

    if not rows:
        print("No metrics found. Run evaluate.py for each variant first.")
        return 1

    if missing:
        print(f"\nWarning: missing variants: {missing}. Ablation table will be partial.")

    _write_csv(rows, out_dir / "ablation.csv")
    _write_markdown(rows, out_dir / "ablation.md")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
