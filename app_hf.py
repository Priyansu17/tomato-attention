"""Hugging Face Spaces entry point — Gradio wrapper for the Tomato Disease Detector.

Reuses all existing src/ model, config, and Grad-CAM code.
Launch command (set in README.md YAML header): python app_hf.py
"""

from __future__ import annotations

import base64
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup — ensures src/ is importable whether run from repo root or HF
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg_get, load_config  # noqa: E402
from src.data.transforms import build_eval_transform  # noqa: E402
from src.gradcam import GradCAM, _denormalize, _get_target_layer, _overlay  # noqa: E402
from src.models import build_model  # noqa: E402
from src.validator import validate_image_foliage, validate_tomato_prediction  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
ATTENTION_VARIANTS = ("baseline", "se", "cbam")

CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "severity": "🔴 High",
        "cause": "Xanthomonas species (bacteria)",
        "treatment": "Apply copper-based bactericides; use certified disease-free seeds; avoid overhead irrigation.",
    },
    "Tomato___Early_blight": {
        "severity": "🟡 Medium",
        "cause": "Alternaria solani (fungus)",
        "treatment": "Apply chlorothalonil or mancozeb; mulch soil to prevent splash; remove infected lower leaves.",
    },
    "Tomato___Late_blight": {
        "severity": "🔴 High",
        "cause": "Phytophthora infestans (oomycete)",
        "treatment": "Apply metalaxyl immediately; remove and destroy all infected material; plant resistant varieties.",
    },
    "Tomato___Leaf_Mold": {
        "severity": "🟡 Medium",
        "cause": "Passalora fulva (fungus)",
        "treatment": "Improve ventilation; apply chlorothalonil or mancozeb; maintain humidity below 85%.",
    },
    "Tomato___Septoria_leaf_spot": {
        "severity": "🟡 Medium",
        "cause": "Septoria lycopersici (fungus)",
        "treatment": "Apply chlorothalonil every 7–10 days; mulch soil; use drip irrigation.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "severity": "🟡 Medium",
        "cause": "Tetranychus urticae (mite)",
        "treatment": "Apply abamectin or insecticidal soap; release predatory mites; maintain adequate irrigation.",
    },
    "Tomato___Target_Spot": {
        "severity": "🟡 Medium",
        "cause": "Corynespora cassiicola (fungus)",
        "treatment": "Apply azoxystrobin or pyraclostrobin; remove crop residues; practice 2-year rotation.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "severity": "🔴 High",
        "cause": "TYLCV transmitted by silverleaf whitefly",
        "treatment": "No cure — control whiteflies with imidacloprid; use reflective mulch; plant resistant varieties.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "severity": "🟡 Medium",
        "cause": "Tomato Mosaic Virus (ToMV)",
        "treatment": "No chemical cure; remove infected plants; sanitize all tools; use virus-free certified seeds.",
    },
    "Tomato___healthy": {
        "severity": "🟢 None",
        "cause": "No disease detected",
        "treatment": "Continue regular monitoring, fertilization, and crop rotation.",
    },
}

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------
_model_cache: dict[str, tuple[Any, dict]] = {}


def _find_checkpoint(variant: str) -> Path | None:
    for pattern in [f"best_model_{variant}.pt", f"{variant}_best.pt"]:
        p = CHECKPOINT_DIR / pattern
        if p.exists():
            return p
    candidates = [p for p in CHECKPOINT_DIR.glob("*.pt") if "smoke" not in str(p) and "last" not in str(p)]
    return candidates[0] if candidates else None


def _load_model(variant: str) -> tuple[Any, dict]:
    if variant in _model_cache:
        return _model_cache[variant]

    ckpt_path = _find_checkpoint(variant)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found for variant '{variant}'.")

    cfg = load_config(CONFIG_PATH, overrides={"attention.type": variant})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, pretrained=False).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if "config" in ckpt and isinstance(ckpt["config"], dict) and "attention" in ckpt["config"]:
        cfg["attention"] = {**cfg.get("attention", {}), **ckpt["config"]["attention"]}

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _model_cache[variant] = (model, cfg)
    return model, cfg


# ---------------------------------------------------------------------------
# Inference + Grad-CAM
# ---------------------------------------------------------------------------

def _run_inference(pil_image: Image.Image, model: Any, cfg: dict) -> tuple[int, float, list[float], np.ndarray]:
    device = next(model.parameters()).device
    transform = build_eval_transform(cfg)
    tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    logits_np = logits[0].cpu().numpy()
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())
    return pred, float(probs[pred]), probs.tolist(), logits_np


def _make_gradcam(pil_image: Image.Image, model: Any, cfg: dict) -> Image.Image | None:
    try:
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
        overlay_np = _overlay(image_np, cam, alpha, colormap)
        return Image.fromarray(overlay_np.astype(np.uint8))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main predict function (called by Gradio)
# ---------------------------------------------------------------------------

def predict(image: Image.Image, variant: str) -> tuple:
    """
    Returns: (gradcam_image, result_markdown)
    """
    if image is None:
        return None, "⚠️ Please upload a tomato leaf image."

    # Tier 1: Foliage Check
    is_foliage, foliage_msg = validate_image_foliage(image)
    if not is_foliage:
        return None, (
            "### ⚠️ Non-Plant Object Detected\n\n"
            f"**Validation Notice**: {foliage_msg}\n\n"
            "Please upload a clear photograph of an actual tomato (*Solanum lycopersicum*) leaf."
        )

    variant = variant.lower()
    if variant not in ATTENTION_VARIANTS:
        variant = "cbam"

    try:
        model, cfg = _load_model(variant)
    except Exception as e:
        return None, f"❌ Model load failed: {e}"

    try:
        pred_idx, confidence, probs, logits_np = _run_inference(image, model, cfg)
    except Exception as e:
        return None, f"❌ Inference failed: {e}"

    # Tier 2: Tomato In-Distribution Check
    is_tomato, ood_msg = validate_tomato_prediction(logits_np, np.array(probs))
    if not is_tomato:
        return None, (
            "### ⚠️ Unrecognized / Non-Tomato Foliage\n\n"
            f"**Diagnostic Notice**: {ood_msg}\n\n"
            "The leaf in this photograph does not match recognized tomato leaf pathology patterns. "
            "Please verify that the leaf is from a tomato plant."
        )

    class_name = CLASS_NAMES[pred_idx]
    friendly = class_name.replace("Tomato___", "").replace("_", " ")
    info = DISEASE_INFO.get(class_name, {})

    # Grad-CAM overlay
    gradcam_img = _make_gradcam(image, model, cfg)

    # Build class probability markdown table
    prob_rows = ""
    sorted_probs = sorted(enumerate(probs), key=lambda x: -x[1])
    for rank, (idx, prob) in enumerate(sorted_probs[:5]):
        fname = CLASS_NAMES[idx].replace("Tomato___", "").replace("_", " ")
        bar = "█" * int(prob * 20)
        marker = " ← **Predicted**" if idx == pred_idx else ""
        prob_rows += f"| {rank+1} | {fname} | {bar} | {prob*100:.2f}%{marker} |\n"

    is_healthy = class_name == "Tomato___healthy"
    status_emoji = "✅" if is_healthy else "⚠️"
    severity_color = info.get("severity", "Unknown")

    result_md = f"""
## {status_emoji} Detection Result

| Field | Value |
|-------|-------|
| **Disease** | **{friendly}** |
| **Confidence** | **{confidence*100:.1f}%** |
| **Severity** | {severity_color} |
| **Variant Used** | `{variant.upper()}` |

---

### 🔬 Cause
{info.get('cause', 'N/A')}

### 💊 Recommended Treatment
{info.get('treatment', 'N/A')}

---

### 📊 Top-5 Class Probabilities
| Rank | Class | Probability Bar | Score |
|------|-------|-----------------|-------|
{prob_rows}
"""
    return gradcam_img, result_md


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# 🍅 Tomato Disease Intelligence

**AI-powered tomato leaf disease detection** using EfficientNet-B0 with configurable attention mechanisms.
- 📊 **99.38% test accuracy** across 10 disease categories
- 🔬 **Grad-CAM visualisation** shows exactly where the model looks
- 🧠 Three variants: **CBAM** (dual-attention), **SE** (channel attention), **Baseline**

Upload a tomato leaf image and select an attention variant to get an instant diagnosis.

> *Dataset: PlantVillage (18,160 images) | Model: EfficientNet-B0 + CBAM attention*
"""

EXAMPLES = []

with gr.Blocks(title="Tomato Disease Intelligence") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📷 Upload Tomato Leaf Image",
                type="pil",
                height=300,
            )
            variant_radio = gr.Radio(
                choices=["cbam", "se", "baseline"],
                value="cbam",
                label="🧠 Attention Variant",
                info="CBAM = best accuracy (99.38%) | SE = channel attention | Baseline = no attention",
            )
            submit_btn = gr.Button("🔍 Analyze Leaf", variant="primary", size="lg")

        with gr.Column(scale=1):
            gradcam_output = gr.Image(
                label="🔬 Grad-CAM Attention Map",
                height=300,
            )
            result_output = gr.Markdown(label="📋 Diagnosis Report")

    submit_btn.click(
        fn=predict,
        inputs=[image_input, variant_radio],
        outputs=[gradcam_output, result_output],
    )

    gr.Markdown("""
---
### About This Project
This application demonstrates tomato leaf disease classification using three attention mechanism variants:
- **CBAM** (Convolutional Block Attention Module): applies both channel and spatial attention — best performer at **99.38% test accuracy**
- **SE** (Squeeze-and-Excitation): channel attention only — **99.07% test accuracy**
- **Baseline**: no attention module — **98.15% test accuracy**

*Built with PyTorch · EfficientNet-B0 · FastAPI · Gradio*
""")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="green", secondary_hue="red"))
