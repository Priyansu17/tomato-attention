"""Multi-stage validation guard to filter non-plant objects and out-of-distribution foliage."""

from __future__ import annotations

import numpy as np
from PIL import Image


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


def validate_tomato_prediction(logits: np.ndarray, probs: np.ndarray) -> tuple[bool, str]:
    """Validate that the model's prediction corresponds to a known tomato leaf distribution (Tier 2).

    Calibrated against 200 real tomato leaves and non-tomato foliage (Tulsi, basil, mint, etc.):
    - Real tomato leaves: max logit 10th percentile = 3.74, median = 4.27, top prob median = 93.5%
    - Tulsi / Non-tomato foliage: max logit = 1.65 - 2.48, top prob < 65%

    Returns:
        (is_valid, reason_message)
    """
    max_logit = float(logits.max())
    top_prob = float(probs.max())
    sorted_probs = np.sort(probs)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else top_prob
    energy = float(np.log(np.sum(np.exp(logits))))

    if max_logit < 2.80:
        return False, (
            f"Unrecognized foliage species: feature activation (max logit {max_logit:.2f} < 2.80) "
            "is too weak for tomato leaf pathology. The leaf appears to belong to another plant species."
        )

    if energy < 3.00:
        return False, (
            f"Out-of-distribution foliage: energy score ({energy:.2f} < 3.00) indicates non-tomato foliage."
        )

    if top_prob < 0.70:
        return False, (
            f"Uncertain diagnosis: top confidence ({top_prob:.1%} < 70%) is below tomato verification threshold."
        )

    if margin < 0.30:
        return False, (
            f"Ambiguous prediction: confidence margin ({margin:.1%} < 30%) is too narrow for a definitive diagnosis."
        )

    return True, "Valid tomato prediction."
