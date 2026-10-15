"""Streamlit demo application for Tomato Leaf Disease Detection.

Upload a tomato leaf image, select the attention variant, and the app:
  1. Loads the corresponding checkpoint from checkpoints/
  2. Runs inference and shows the predicted disease class + confidence bar
  3. Generates a Grad-CAM overlay and displays it side-by-side

Requirements: a trained checkpoint must exist in checkpoints/.

Run with:
    .\\venv\\python.exe -m streamlit run app/app.py
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from typing import Any

# Ensure the project root is on the Python path when run from app/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.config import cfg_get, load_config
from src.gradcam import GradCAM, _denormalize, _get_target_layer, _overlay
from src.models import build_model
from src.data.transforms import build_eval_transform

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tomato Leaf Disease Detector",
    page_icon="🍅",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
ATTENTION_VARIANTS = ["cbam", "se", "baseline"]
VARIANT_LABELS = {
    "cbam": "CBAM (channel + spatial attention)",
    "se": "SE (channel attention only)",
    "baseline": "Baseline (no attention)",
}

# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading config…")
def _load_cfg() -> dict[str, Any]:
    return load_config(CONFIG_PATH)


@st.cache_resource(show_spinner="Loading model…")
def _load_model(attention_type: str, checkpoint_path: str) -> tuple[Any, Any]:
    """Load model and config; cached so the model is not reloaded on every interaction."""
    cfg = load_config(CONFIG_PATH, overrides={"attention.type": attention_type})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def _find_checkpoint(variant: str) -> Path | None:
    """Look for a checkpoint matching the variant name, fallback to best_model.pt."""
    for pattern in [f"best_model_{variant}.pt", f"{variant}_best.pt", "best_model.pt"]:
        p = CHECKPOINT_DIR / pattern
        if p.exists():
            return p
    # any .pt file
    candidates = list(CHECKPOINT_DIR.glob("*.pt"))
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Inference + Grad-CAM
# ---------------------------------------------------------------------------

def run_inference(
    pil_image: Image.Image,
    model: Any,
    cfg: dict[str, Any],
) -> tuple[int, float, np.ndarray]:
    """Return ``(class_idx, confidence, probabilities)``."""
    device = next(model.parameters()).device
    transform = build_eval_transform(cfg)
    tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())
    return pred, float(probs[pred]), probs


def make_gradcam_overlay(
    pil_image: Image.Image,
    model: Any,
    cfg: dict[str, Any],
) -> np.ndarray:
    """Return the Grad-CAM overlay as a ``(H, W, 3)`` uint8 numpy array."""
    device = next(model.parameters()).device
    transform = build_eval_transform(cfg)
    tensor = transform(pil_image).unsqueeze(0).to(device)
    mean = list(cfg_get(cfg, "model.normalization.mean"))
    std_v = list(cfg_get(cfg, "model.normalization.std"))
    alpha = float(cfg_get(cfg, "gradcam.alpha", 0.5))
    colormap = str(cfg_get(cfg, "gradcam.colormap", "jet"))

    target_layer = _get_target_layer(model, cfg)
    gcam = GradCAM(model, target_layer)
    with torch.enable_grad():
        cam = gcam.generate(tensor)
    gcam.remove_hooks()

    image_np = _denormalize(tensor.squeeze(0).cpu(), mean, std_v)
    return _overlay(image_np, cam, alpha, colormap)


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🍅 Tomato Leaf Disease Detector")
    st.markdown(
        "Upload a tomato leaf image to classify its disease and visualise where "
        "the model is looking using **Grad-CAM**."
    )

    cfg_base = _load_cfg()
    class_names = list(cfg_get(cfg_base, "data.class_names"))
    friendly_names = [n.replace("Tomato___", "").replace("_", " ") for n in class_names]

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        variant = st.selectbox(
            "Attention variant",
            options=ATTENTION_VARIANTS,
            format_func=lambda v: VARIANT_LABELS[v],
        )
        show_gradcam = st.checkbox("Show Grad-CAM overlay", value=True)
        st.markdown("---")
        st.markdown("**Device:** " + ("CUDA 🚀" if torch.cuda.is_available() else "CPU"))
        ckpt = _find_checkpoint(variant)
        if ckpt:
            st.success(f"Checkpoint: `{ckpt.name}`")
        else:
            st.error(
                "No checkpoint found in `checkpoints/`. "
                "Train the model first with `python -m src.train`."
            )

    # File uploader
    uploaded = st.file_uploader(
        "Choose a tomato leaf image", type=["jpg", "jpeg", "png"]
    )

    if uploaded is None:
        st.info("👆 Upload an image to get started.")
        st.markdown(
            """
            **Classes recognised:**

            """ + "\n".join(f"- {n}" for n in friendly_names)
        )
        return

    if ckpt is None:
        st.error("Please train the model first (see sidebar).")
        return

    # Load image
    pil_image = Image.open(uploaded).convert("RGB")

    # Load model (cached)
    with st.spinner(f"Loading {VARIANT_LABELS[variant]}…"):
        model, cfg = _load_model(variant, str(ckpt))

    # Inference
    with st.spinner("Running inference…"):
        pred_idx, confidence, probs = run_inference(pil_image, model, cfg)

    pred_class = friendly_names[pred_idx]
    is_healthy = "healthy" in class_names[pred_idx].lower()

    # ---- Results layout ----
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Input Image")
        st.image(pil_image, use_container_width=True)

    with col2:
        st.subheader("Prediction")
        colour = "🟢" if is_healthy else "🔴"
        st.metric(label="Predicted class", value=f"{colour} {pred_class}")
        st.metric(label="Confidence", value=f"{confidence:.1%}")
        st.progress(float(confidence))

        # Top-3 predictions
        st.markdown("**Top predictions:**")
        top3_idx = np.argsort(probs)[::-1][:3]
        for i in top3_idx:
            st.markdown(f"- `{friendly_names[i]}` — {probs[i]:.1%}")

    with col3:
        if show_gradcam:
            st.subheader("Grad-CAM Overlay")
            with st.spinner("Generating Grad-CAM…"):
                overlay_np = make_gradcam_overlay(pil_image, model, cfg)
            st.image(overlay_np, use_container_width=True,
                     caption=f"Attention via {VARIANT_LABELS[variant]}")
        else:
            st.subheader("Probability Bar Chart")
            fig, ax = plt.subplots(figsize=(5, 4))
            y_pos = np.arange(len(class_names))
            ax.barh(y_pos, probs, color=["green" if i == pred_idx else "steelblue"
                                          for i in range(len(class_names))])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(friendly_names, fontsize=8)
            ax.set_xlabel("Probability")
            ax.set_xlim(0, 1)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # Full probability table
    with st.expander("Show all class probabilities"):
        data = {
            "Class": friendly_names,
            "Probability": [f"{p:.4f}" for p in probs],
        }
        st.dataframe(data, use_container_width=True)


if __name__ == "__main__":
    main()
