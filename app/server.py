"""FastAPI backend for Tomato Leaf Disease Detection.

Serves the custom 3D frontend (app/static/index.html) and exposes a
prediction API that runs the trained CBAM / SE / baseline model.

Run with:
    .\\venv\\python.exe -m uvicorn app.server:app --host 0.0.0.0 --port 8000

Then open: http://localhost:8000
"""

from __future__ import annotations

import base64
import sys
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg_get, load_config  # noqa: E402
from src.data.transforms import build_eval_transform  # noqa: E402
from src.gradcam import GradCAM, _denormalize, _get_target_layer, _overlay  # noqa: E402
from src.models import build_model  # noqa: E402
from src.validator import validate_image_foliage, validate_image_context, validate_tomato_prediction  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
STATIC_DIR = Path(__file__).resolve().parent / "static"
ATTENTION_VARIANTS = ("baseline", "se", "cbam")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Tomato Disease Detector API",
    description="CBAM attention network for tomato leaf disease classification",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------
_model_cache: dict[str, tuple[Any, dict]] = {}


def _find_checkpoint(variant: str) -> Path | None:
    """Return best checkpoint for the given variant."""
    for pattern in [f"best_model_{variant}.pt", f"{variant}_best.pt", "best_model.pt"]:
        p = CHECKPOINT_DIR / pattern
        if p.exists():
            return p
    candidates = [p for p in CHECKPOINT_DIR.glob("*.pt") if "smoke" not in str(p)]
    return candidates[0] if candidates else None


def _load_model(variant: str) -> tuple[Any, dict]:
    """Load and cache model for the given variant."""
    if variant in _model_cache:
        return _model_cache[variant]

    ckpt_path = _find_checkpoint(variant)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found for variant '{variant}' in {CHECKPOINT_DIR}. "
            "Run training first."
        )

    cfg = load_config(CONFIG_PATH, overrides={"attention.type": variant})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, pretrained=False).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # Merge checkpoint config so architecture matches exactly
    if "config" in ckpt:
        saved_cfg = ckpt["config"]
        if isinstance(saved_cfg, dict) and "attention" in saved_cfg:
            cfg["attention"] = {**cfg.get("attention", {}), **saved_cfg["attention"]}

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _model_cache[variant] = (model, cfg)
    return model, cfg


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _run_inference(
    pil_image: Image.Image,
    model: Any,
    cfg: dict,
) -> tuple[int, float, list[float], np.ndarray]:
    """Return (class_idx, confidence, all_probs, raw_logits_np)."""
    device = next(model.parameters()).device
    transform = build_eval_transform(cfg)
    tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    logits_np = logits[0].cpu().numpy()
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())
    return pred, float(probs[pred]), probs.tolist(), logits_np


def _make_gradcam_b64(
    pil_image: Image.Image,
    model: Any,
    cfg: dict,
) -> str:
    """Return Grad-CAM overlay as base64-encoded PNG string."""
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

    # Convert numpy uint8 array to base64 PNG
    pil_overlay = Image.fromarray(overlay_np.astype(np.uint8))
    buf = BytesIO()
    pil_overlay.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the main frontend HTML."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found. Check app/static/index.html.")
    return FileResponse(str(index_path), media_type="text/html")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = None
    if device == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return {
        "status": "ok",
        "device": device,
        "gpu": gpu_name,
        "checkpoints": {
            v: (_find_checkpoint(v) is not None) for v in ATTENTION_VARIANTS
        },
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    variant: str = Form(default="cbam"),
):
    """Run inference on an uploaded tomato leaf image."""

    # Validate variant
    if variant not in ATTENTION_VARIANTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid variant '{variant}'. Choose from: {ATTENTION_VARIANTS}",
        )

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported.",
        )

    # Read image
    try:
        image_bytes = await file.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    # 1. Tier 1: Foliage & Chlorophyll Optical Guard
    is_foliage, foliage_msg = validate_image_foliage(pil_image)
    if not is_foliage:
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        original_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        raw_cfg = load_config(CONFIG_PATH)
        class_names: list[str] = list(cfg_get(raw_cfg, "data.class_names"))
        friendly_names = [n.replace("Tomato___", "").replace("_", " ") for n in class_names]

        return JSONResponse({
            "is_valid": False,
            "class_idx": -1,
            "class_name": "Invalid_Non_Plant",
            "friendly_name": "Non-Plant Object Detected",
            "confidence": 0.0,
            "probs": [0.0] * len(class_names),
            "class_names": class_names,
            "friendly_names": friendly_names,
            "rejection_reason": foliage_msg,
            "guidance": "The uploaded photo does not contain organic plant foliage. Please upload a clear photo of a tomato leaf.",
            "gradcam_b64": None,
            "original_b64": original_b64,
            "variant": variant,
        })

    # 1.5. Tier 1.5: Studio / Mortar Context Guard
    # Rejects images with bright white background + oversaturated non-leaf colours.
    # Calibrated: 0% false-rejection on 500 PlantVillage tomato leaves (2026-10-27).
    is_context, context_msg = validate_image_context(pil_image)
    if not is_context:
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        original_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        raw_cfg = load_config(CONFIG_PATH)
        class_names_t15: list[str] = list(cfg_get(raw_cfg, "data.class_names"))
        friendly_names_t15 = [n.replace("Tomato___", "").replace("_", " ") for n in class_names_t15]

        return JSONResponse({
            "is_valid": False,
            "class_idx": -1,
            "class_name": "Invalid_Non_Tomato",
            "friendly_name": "Unrecognized / Non-Tomato Foliage",
            "confidence": 0.0,
            "probs": [0.0] * len(class_names_t15),
            "class_names": class_names_t15,
            "friendly_names": friendly_names_t15,
            "rejection_reason": context_msg,
            "guidance": "Please upload a clear, close-up photo of a tomato leaf in a field or garden. Avoid photos taken inside bowls, mortars, or on white surfaces.",
            "gradcam_b64": None,
            "original_b64": original_b64,
            "variant": variant,
        })

    # Load model
    try:
        model, cfg = _load_model(variant)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model loading failed: {e}\n{traceback.format_exc()}",
        )

    # Inference
    try:
        pred_idx, confidence, probs, logits_np = _run_inference(pil_image, model, cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # Class names
    class_names: list[str] = list(cfg_get(cfg, "data.class_names"))
    friendly_names = [n.replace("Tomato___", "").replace("_", " ") for n in class_names]

    # 2. Tier 2: Tomato In-Distribution & Feature Activation Guard
    is_tomato, ood_msg = validate_tomato_prediction(logits_np, np.array(probs))
    if not is_tomato:
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        original_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return JSONResponse({
            "is_valid": False,
            "class_idx": -1,
            "class_name": "Invalid_Non_Tomato",
            "friendly_name": "Unrecognized / Non-Tomato Foliage",
            "confidence": round(confidence, 6),
            "probs": [round(p, 6) for p in probs],
            "class_names": class_names,
            "friendly_names": friendly_names,
            "rejection_reason": ood_msg,
            "guidance": "The foliage in this photo does not match known tomato leaf patterns. Please ensure the leaf is from a tomato plant.",
            "gradcam_b64": None,
            "original_b64": original_b64,
            "variant": variant,
        })

    pred_class = class_names[pred_idx]
    pred_friendly = friendly_names[pred_idx]

    # Grad-CAM
    gradcam_b64 = None
    try:
        gradcam_b64 = _make_gradcam_b64(pil_image, model, cfg)
    except Exception as e:
        # Non-fatal: return prediction without Grad-CAM
        gradcam_b64 = None

    # Return original image as base64 too (for display)
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    original_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse({
        "is_valid": True,
        "class_idx": pred_idx,
        "class_name": pred_class,
        "friendly_name": pred_friendly,
        "confidence": round(confidence, 6),
        "probs": [round(p, 6) for p in probs],
        "class_names": class_names,
        "friendly_names": friendly_names,
        "gradcam_b64": gradcam_b64,
        "original_b64": original_b64,
        "variant": variant,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=False)
