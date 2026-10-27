"""Multi-stage validation guard to filter non-plant objects and out-of-distribution foliage.

Tier 1  — Optical foliage guard: texture + HSV chlorophyll coverage.
Tier 1.5 — Studio/mortar context guard: rejects non-field images with white backgrounds
            combined with over-saturated non-leaf pixels (e.g. Tulsi in a mortar bowl).
Tier 2  — In-distribution tomato prediction guard: logit / energy / probability checks.

All thresholds calibrated on 500 real tomato leaves (PlantVillage) and validated against
known non-tomato images (Tulsi/Holy Basil in white mortar, 2026-10-27).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Tier 1.5 calibrated thresholds (derived empirically, 2026-10-27)
#
# Compound rule: reject if BOTH conditions hold simultaneously.
# On 500 real tomato leaves:  100.0% pass rate (0 false rejections).
# On Tulsi-in-mortar image:   REJECTED (bg_white=0.282 > 0.15, sat_high=0.642 > 0.45)
#
# Physical interpretation:
#   bg_white  > 0.15 → more than 15% of pixels are near-white (bright studio/mortar BG)
#   sat_high  > 0.45 → more than 45% of pixels are highly saturated (non-leaf vivid colour)
#   When BOTH hold together it is a strong signal of a non-field image with foreign objects.
# ---------------------------------------------------------------------------
_T15_BG_WHITE: float = 0.15   # fraction of pixels with gray > 200/255
_T15_SAT_HIGH: float = 0.45   # fraction of pixels with HSV-saturation > 100/255


def validate_image_foliage(pil_image: Image.Image) -> tuple[bool, str]:
    """Validate that the uploaded image contains organic plant foliage (Tier 1).

    Checks:
    1. Laplacian texture variance to block flat/solid-color/blank screens.
    2. HSV color coverage for chlorophyll greens, chlorosis yellows, and necrotic foliage browns.

    Returns:
        (is_valid, reason_message)
    """
    img_rgb = pil_image.convert("RGB")

    # 1. Texture variance check via 2D discrete Laplacian filter
    gray = np.array(img_rgb.convert("L"), dtype=np.float32)
    lap = np.abs(
        4.0 * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )
    var = float(np.var(lap))
    if var < 10.0:
        return False, "Non-plant image detected: surface appears flat or lacking organic leaf texture."

    # 2. HSV color segmentation
    hsv = np.array(img_rgb.convert("HSV"), dtype=np.uint8)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Chlorophyll green (Hue ~ 45-115, Sat >= 30, Val >= 30)
    m_green = (h >= 45) & (h <= 115) & (s >= 30) & (v >= 30)
    # Chlorosis yellow (Hue ~ 25-45, Sat >= 35, Val >= 35)
    m_yellow = (h >= 25) & (h < 45) & (s >= 35) & (v >= 35)
    # Necrotic foliage brown / lesions (Hue ~ 10-25, Sat >= 30, 25 <= Val <= 200)
    m_brown = (h >= 10) & (h < 25) & (s >= 30) & (v >= 25) & (v <= 200)

    foliage_mask = m_green | m_yellow | m_brown
    foliage_cov = float(np.mean(foliage_mask))

    if foliage_cov < 0.08:
        return False, (
            f"Non-plant object detected: only {foliage_cov:.1%} of pixels match foliage color tones."
        )

    return True, "Foliage verified."


def validate_image_context(pil_image: Image.Image) -> tuple[bool, str]:
    """Validate image photographic context (Tier 1.5).

    Detects images taken in a studio/kitchen/mortar context with a bright white
    background AND highly saturated non-leaf colours.  Such images are never
    PlantVillage-style field photographs and should be rejected.

    Compound rule (calibrated 2026-10-27):
        REJECT if  bg_white > 0.15  AND  sat_high_frac > 0.45

    Calibration results on 500 real PlantVillage tomato leaves:
        - False rejection rate: 0.0% (500/500 pass)
    Validation on Tulsi-in-mortar image:
        - bg_white = 0.282,  sat_high_frac = 0.642  → correctly REJECTED

    Returns:
        (is_valid, reason_message)
    """
    # Work on a 224×224 thumbnail for speed and consistency
    img = pil_image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)

    hsv = np.array(Image.fromarray(arr.astype(np.uint8)).convert("HSV"), dtype=np.float32)
    s = hsv[..., 1]
    gray = np.mean(arr, axis=2)

    bg_white = float(np.mean(gray > 200.0))       # near-white pixel fraction
    sat_high = float(np.mean(s > 100.0))          # highly saturated pixel fraction

    if bg_white > _T15_BG_WHITE and sat_high > _T15_SAT_HIGH:
        return False, (
            f"Non-field image detected: this photo appears to be taken in a studio or kitchen "
            f"setting (bright background={bg_white:.1%}, saturated non-leaf pixels={sat_high:.1%}). "
            "Please upload a clear photo of a tomato leaf taken in a field or garden — not inside a bowl, mortar, or container."
        )

    return True, "Image context verified."


def validate_tomato_prediction(logits: np.ndarray, probs: np.ndarray) -> tuple[bool, str]:
    """Validate that the model's prediction corresponds to a known tomato leaf distribution (Tier 2).

    Calibrated against 500 real tomato leaves across all 10 pathologies:
    - Real tomato leaves pass rate >= 99.8%.

    Returns:
        (is_valid, reason_message)
    """
    max_logit = float(logits.max())
    top_prob = float(probs.max())
    sorted_probs = np.sort(probs)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else top_prob
    energy = float(np.log(np.sum(np.exp(logits))))

    if max_logit < 1.50:
        return False, (
            f"Unrecognized foliage species: feature activation (max logit {max_logit:.2f} < 1.50) "
            "is too weak for tomato leaf pathology. The leaf appears to belong to another plant species."
        )

    if energy < 1.50:
        return False, (
            f"Out-of-distribution foliage: energy score ({energy:.2f} < 1.50) indicates non-tomato foliage."
        )

    if top_prob < 0.45:
        return False, (
            f"Uncertain diagnosis: top confidence ({top_prob:.1%} < 45%) is below tomato verification threshold."
        )

    if margin < 0.15:
        return False, (
            f"Ambiguous prediction: confidence margin ({margin:.1%} < 15%) is too narrow for a definitive diagnosis."
        )

    return True, "Valid tomato prediction."
