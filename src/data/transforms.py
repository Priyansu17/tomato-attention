"""Config-driven torchvision transforms (Phase 2C).

Every augmentation parameter is read from ``configs/config.yaml`` via the ``augmentation``
section; nothing is hardcoded here (project rule). The normalization statistics come from
``model.normalization`` (ImageNet mean/std) so the transforms match the locked architecture.

Train pipeline (in the order the keys appear in the config):
    RandomResizedCrop -> Rotation -> HorizontalFlip -> [VerticalFlip] -> ColorJitter
    -> [RandomErasing] -> ToTensor -> Normalize

Eval pipeline (train/val/test evaluation and inference):
    Resize -> CenterCrop -> ToTensor -> Normalize
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torchvision.transforms as T
from torch import Tensor

DEFAULT_CROP_SCALE: tuple[float, float] = (0.8, 1.0)
DEFAULT_CROP_RATIO: tuple[float, float] = (0.9, 1.111)
DEFAULT_ROTATION_DEGREES: float = 15.0
DEFAULT_JITTER = {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05}


class TransformConfigError(ValueError):
    """Raised when the augmentation/normalization configuration is unusable."""


def _enabled(settings: Mapping[str, Any], key: str, default: bool = False) -> bool:
    """Return ``settings[key].enabled`` (default ``default``)."""
    node = settings.get(key) or {}
    return bool(node.get("enabled", default))


def _triple(values: Any, key: str) -> tuple[float, float, float]:
    """Validate a 3-element sequence of numbers from the config."""
    if (not isinstance(values, Sequence) or isinstance(values, str)
            or len(values) != 3 or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                           for v in values)):
        raise TransformConfigError(f"{key} must be a 3-element sequence of numbers")
    return float(values[0]), float(values[1]), float(values[2])


def normalization(cfg: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
    """Return the configured ``(mean, std)`` normalization tensors."""
    norm = cfg.get("model", {}).get("normalization", {})
    mean = _triple(norm.get("mean"), "model.normalization.mean")
    std = _triple(norm.get("std"), "model.normalization.std")
    if any(value == 0 for value in std):
        raise TransformConfigError("model.normalization.std must not contain zeros")
    return Tensor(mean), Tensor(std)


def build_eval_transform(cfg: Mapping[str, Any]) -> T.Compose:
    """Deterministic eval transform: Resize -> CenterCrop -> ToTensor -> Normalize."""
    eval_cfg = cfg.get("augmentation", {}).get("eval", {})
    resize = int(eval_cfg.get("resize", 256))
    crop = int(eval_cfg.get("center_crop", cfg.get("data", {}).get("image_size", 224)))
    if crop > resize:
        raise TransformConfigError(
            f"augmentation.eval.center_crop ({crop}) cannot exceed resize ({resize})"
        )
    mean, std = normalization(cfg)
    return T.Compose([
        T.Resize(resize),
        T.CenterCrop(crop),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def build_train_transform(cfg: Mapping[str, Any]) -> T.Compose:
    """Stochastic train transform driven entirely by the ``augmentation.train`` section."""
    aug = cfg.get("augmentation", {}).get("train", {})
    image_size = int(cfg.get("data", {}).get("image_size", 224))
    steps: list[Any] = []

    crop_cfg = aug.get("random_resized_crop") or {}
    if crop_cfg.get("enabled", True):
        scale = crop_cfg.get("scale") or DEFAULT_CROP_SCALE
        ratio = crop_cfg.get("ratio") or DEFAULT_CROP_RATIO
        steps.append(T.RandomResizedCrop(image_size, scale=tuple(scale), ratio=tuple(ratio)))
    else:
        steps.append(T.Resize(image_size))
        steps.append(T.CenterCrop(image_size))

    rotation_cfg = aug.get("rotation") or {}
    if _enabled(aug, "rotation"):
        steps.append(T.RandomRotation(float(rotation_cfg.get("degrees", DEFAULT_ROTATION_DEGREES))))

    flip_cfg = aug.get("horizontal_flip") or {}
    if _enabled(aug, "horizontal_flip"):
        steps.append(T.RandomHorizontalFlip(p=float(flip_cfg.get("p", 0.5))))
    vflip_cfg = aug.get("vertical_flip") or {}
    if _enabled(aug, "vertical_flip"):
        steps.append(T.RandomVerticalFlip(p=float(vflip_cfg.get("p", 0.5))))

    jitter_cfg = aug.get("color_jitter") or {}
    if _enabled(aug, "color_jitter"):
        steps.append(T.ColorJitter(
            brightness=float(jitter_cfg.get("brightness", DEFAULT_JITTER["brightness"])),
            contrast=float(jitter_cfg.get("contrast", DEFAULT_JITTER["contrast"])),
            saturation=float(jitter_cfg.get("saturation", DEFAULT_JITTER["saturation"])),
            hue=float(jitter_cfg.get("hue", DEFAULT_JITTER["hue"])),
        ))

    steps.append(T.ToTensor())

    erasing_cfg = aug.get("random_erasing") or {}
    if _enabled(aug, "random_erasing"):
        steps.append(T.RandomErasing(p=float(erasing_cfg.get("p", 0.25))))

    mean, std = normalization(cfg)
    steps.append(T.Normalize(mean=mean, std=std))
    return T.Compose(steps)
