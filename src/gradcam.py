"""Grad-CAM explainability for the Tomato Leaf Disease Detection project.

Generates gradient-weighted class activation maps on the ``post_attention``
feature map (as configured by ``gradcam.target``), overlaid on the input image.

Supports:
  - ``gradcam``       — standard Grad-CAM (Selvaraju et al.)
  - ``gradcam_plus_plus`` — Grad-CAM++ (only gradcam used in baseline per config)

Output: a grid of ``gradcam.num_samples`` images written to
``paths.gradcam_dir/``.

Usage:
    python -m src.gradcam \\
        --checkpoint checkpoints/best_model.pt \\
        [--config configs/config.yaml] \\
        [--split val] \\
        [--num-samples 8]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .config import cfg_get, load_config
from .data.dataloaders import build_dataloaders
from .data.transforms import build_eval_transform
from .models import TomatoAttentionNet, build_model
from .utils.logging_utils import get_logger, log_config_summary
from .utils.paths import ensure_dir
from .utils.seed import set_seed


# ---------------------------------------------------------------------------
# Hook-based Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """Hooks into a target layer to extract activations and gradients.

    Args:
        model: the ``TomatoAttentionNet`` instance.
        target_layer: the ``nn.Module`` layer to hook.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module: nn.Module, inp: Any, out: torch.Tensor) -> None:
        self.activations = out.detach()

    def _save_gradients(self, module: nn.Module, grad_in: Any, grad_out: Any) -> None:
        self.gradients = grad_out[0].detach()

    def generate(self, image_tensor: torch.Tensor,
                 class_idx: int | None = None) -> np.ndarray:
        """Return the Grad-CAM heatmap (H, W) in [0, 1] for ``class_idx``.

        Args:
            image_tensor: ``(1, 3, H, W)`` tensor.
            class_idx: target class; defaults to the predicted class.

        Returns:
            Numpy array ``(H, W)`` in [0, 1].
        """
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(False)

        logits = self.model(image_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Global average pooling of gradients → channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32)

    def remove_hooks(self) -> None:
        """Remove the registered hooks to free memory."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def _get_target_layer(model: TomatoAttentionNet, cfg: Mapping[str, Any]) -> nn.Module:
    """Return the layer to hook for Grad-CAM based on ``gradcam.target``.

    ``post_attention``: the attention module itself (its output = post-attention
    features, the spec default).
    ``pre_attention``: the backbone (output = pre-attention features).
    ``auto``: same as ``post_attention`` (spec default).
    """
    target = str(cfg_get(cfg, "gradcam.target", "post_attention")).lower()
    if target in ("post_attention", "auto"):
        return model.attention
    if target == "pre_attention":
        return model.backbone
    raise ValueError(f"Unknown gradcam.target '{target}'. Use post_attention or pre_attention.")


def _overlay(image_np: np.ndarray, cam: np.ndarray,
             alpha: float, colormap: str) -> np.ndarray:
    """Overlay a CAM heatmap on an image.

    Args:
        image_np: ``(H, W, 3)`` uint8 array.
        cam: ``(h, w)`` float array in [0, 1].
        alpha: overlay opacity.
        colormap: matplotlib colormap name.

    Returns:
        ``(H, W, 3)`` uint8 blended array.
    """
    h, w = image_np.shape[:2]
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ).astype(np.float32) / 255.0
    cmap = plt.get_cmap(colormap)
    heatmap = (cmap(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    blended = (image_np * (1 - alpha) + heatmap * alpha).clip(0, 255).astype(np.uint8)
    return blended


def _denormalize(tensor: torch.Tensor, mean: list[float],
                 std: list[float]) -> np.ndarray:
    """Convert a normalised ``(3, H, W)`` tensor to a ``(H, W, 3)`` uint8 array."""
    t = tensor.cpu().clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    t = t.permute(1, 2, 0).clamp(0, 1).numpy()
    return (t * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main generate function
# ---------------------------------------------------------------------------

def generate_gradcam(
    cfg: Mapping[str, Any],
    checkpoint_path: str | Path,
    split: str = "val",
    num_samples: int | None = None,
    run_name: str | None = None,
) -> list[Path]:
    """Generate and save Grad-CAM overlays for ``num_samples`` images.

    Args:
        cfg: configuration mapping.
        checkpoint_path: path to the saved ``.pt`` checkpoint.
        split: which data split to sample from (default: ``val``).
        num_samples: override ``gradcam.num_samples`` from config.
        run_name: optional variant label (e.g. ``'cbam'``).  When given,
            images are saved to a subdirectory
            ``paths.gradcam_dir/<run_name>/`` so variants do not overwrite
            each other.

    Returns:
        List of saved figure paths.
    """
    logger = get_logger("tomato.gradcam")
    log_config_summary(cfg, logger)
    set_seed(cfg, logger)

    device_cfg = str(cfg_get(cfg, "training.device", "auto")).lower()
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    logger.info("device: %s", device)

    # Load checkpoint first so we can read its saved config
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    logger.info("loaded checkpoint from epoch %d", ckpt.get("epoch", "?"))

    # Merge checkpoint config so the model architecture (attention.type) matches
    # what was trained, regardless of what configs/config.yaml currently says.
    ckpt_config = ckpt.get("config", {})
    if ckpt_config:
        cfg = {**cfg, **ckpt_config}
        logger.info(
            "merged checkpoint config; attention.type=%s",
            cfg_get(cfg, "attention.type", "?"),
        )

    # Build model and load weights
    model = build_model(cfg, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()


    # Target layer
    target_layer = _get_target_layer(model, cfg)
    gradcam = GradCAM(model, target_layer)

    # Data
    n_samples = num_samples or int(cfg_get(cfg, "gradcam.num_samples", 8))
    bundle = build_dataloaders(cfg, smoke=False)
    dataset = bundle.datasets[split]
    rows = bundle.rows[split]

    # Config
    class_names = list(cfg_get(cfg, "data.class_names"))
    mean = list(cfg_get(cfg, "model.normalization.mean"))
    std = list(cfg_get(cfg, "model.normalization.std"))
    alpha = float(cfg_get(cfg, "gradcam.alpha", 0.5))
    colormap = str(cfg_get(cfg, "gradcam.colormap", "jet"))
    fmt = str(cfg_get(cfg, "evaluation.figure_format", "png"))
    dpi = int(cfg_get(cfg, "evaluation.figure_dpi", 150))

    # Use a variant-specific subdirectory when run_name is given
    base_gradcam_dir = Path(cfg_get(cfg, "paths.gradcam_dir", "results/gradcam"))
    gradcam_dir = base_gradcam_dir / run_name if run_name else base_gradcam_dir
    ensure_dir(gradcam_dir, cfg)

    saved: list[Path] = []
    indices = list(range(min(n_samples, len(dataset))))

    for i, idx in enumerate(indices):
        image_tensor, true_label = dataset[idx]
        image_np = _denormalize(image_tensor, mean, std)
        input_batch = image_tensor.unsqueeze(0).to(device)

        with torch.enable_grad():
            cam = gradcam.generate(input_batch)

        # Model prediction
        with torch.no_grad():
            logits = model(input_batch)
        pred_label = int(logits.argmax(dim=1).item())
        pred_conf = float(torch.softmax(logits, dim=1)[0, pred_label].item())

        overlay = _overlay(image_np, cam, alpha, colormap)
        true_name = class_names[true_label].replace("Tomato___", "")
        pred_name = class_names[pred_label].replace("Tomato___", "")
        correct = "[OK]" if pred_label == true_label else "[X]"

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(image_np)
        axes[0].set_title(f"Original\nTrue: {true_name}", fontsize=9)
        axes[0].axis("off")
        axes[1].imshow(overlay)
        axes[1].set_title(
            f"Grad-CAM {correct}\nPred: {pred_name} ({pred_conf:.2%})", fontsize=9
        )
        axes[1].axis("off")
        title = f"Sample {idx} | {split}"
        if run_name:
            title += f" | {run_name}"
        fig.suptitle(title, fontsize=10, fontweight="bold")
        fig.tight_layout()

        out_path = gradcam_dir / f"gradcam_{split}_{i:03d}.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        logger.info("  [%d/%d] %s -> %s", i + 1, len(indices), correct, out_path.name)

    gradcam.remove_hooks()

    # Save a summary grid
    if len(saved) > 1:
        n_cols = min(4, len(saved))
        n_rows = (len(saved) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        axes_flat = np.array(axes).flatten() if n_rows * n_cols > 1 else [axes]
        for ax in axes_flat:
            ax.axis("off")
        for ax, path in zip(axes_flat, saved):
            img = plt.imread(path)
            ax.imshow(img)
            ax.axis("off")
        grid_title = f"Grad-CAM Grid — {split} split"
        if run_name:
            grid_title += f" ({run_name})"
        fig.suptitle(grid_title, fontsize=12, fontweight="bold")
        fig.tight_layout()
        grid_path = gradcam_dir / f"gradcam_{split}_grid.{fmt}"
        fig.savefig(grid_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("grid -> %s", grid_path)

    logger.info("Grad-CAM complete: %d images saved to %s", len(saved), gradcam_dir)
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the .pt checkpoint file.")
    parser.add_argument("--split", default="val",
                        help="Dataset split to visualise (default: val).")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of images (default: gradcam.num_samples from config).")
    parser.add_argument(
        "--run-name", default=None,
        help=(
            "Variant label (e.g. 'cbam'). Images are saved to "
            "paths.gradcam_dir/<run-name>/ so variants do not overwrite each other."
        ),
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    generate_gradcam(cfg, args.checkpoint, split=args.split,
                     num_samples=args.num_samples, run_name=args.run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
