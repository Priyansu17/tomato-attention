"""Configuration loading and validation.

This is the single place in the project that reads ``configs/config.yaml``; every other
module receives the parsed mapping. Hyperparameters must never be hardcoded in Python
sources (see ``.clinerules``).

The order of ``data.class_names`` is authoritative for label ids: labels are derived from
the config order only, never from filesystem, sorted or tool-specific enumeration order.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")

REQUIRED_SECTIONS: tuple[str, ...] = (
    "project",
    "seed",
    "data",
    "model",
    "attention",
    "training",
    "optimizer",
    "scheduler",
    "augmentation",
    "evaluation",
    "gradcam",
    "paths",
    "smoke_test",
)

REQUIRED_KEYS: tuple[str, ...] = (
    "project.name",
    "seed.value",
    "data.root",
    "data.image_size",
    "data.num_classes",
    "data.class_names",
    "data.split.train",
    "data.split.val",
    "data.split.test",
    "data.duplicate_check.enabled",
    "data.duplicate_check.method",
    "data.loader.batch_size",
    "data.loader.eval_batch_size",
    "data.loader.num_workers",
    "model.backbone",
    "model.pretrained",
    "model.num_classes",
    "model.head.hidden_dim",
    "model.head.dropout",
    "attention.type",
    "attention.insertion_stage",
    "attention.cbam.channel.reduction_ratio",
    "attention.cbam.spatial.kernel_size",
    "training.max_epochs",
    "training.loss.label_smoothing",
    "optimizer.learning_rate",
    "optimizer.backbone_learning_rate",
    "optimizer.weight_decay",
    "paths.checkpoints_dir",
    "paths.results_dir",
    "paths.manifests_dir",
    "smoke_test.enabled",
    "smoke_test.max_epochs",
    "smoke_test.subset_size",
)

_MISSING = object()


class ConfigError(ValueError):
    """Raised when the configuration is missing, malformed or inconsistent."""


def cfg_get(cfg: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Return a nested configuration value addressed by a dotted key."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def class_label_map(cfg: Mapping[str, Any]) -> dict[str, int]:
    """Return ``{class_name: label_id}`` using the config order as the only source.

    Label ids are assigned as ``enumerate(data.class_names)``. Filesystem order,
    alphabetical order and any tool-specific sort order must never be used, because
    the same folder set can be enumerated differently by different tools.
    """
    names = list(cfg_get(cfg, "data.class_names", []) or [])
    if len(names) != len(set(names)):
        raise ConfigError("data.class_names contains duplicate entries")
    return {name: label for label, name in enumerate(names)}


def config_fingerprint(cfg: Mapping[str, Any]) -> str:
    """Return a stable sha256 fingerprint of the resolved configuration."""
    payload = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: str | Path = DEFAULT_CONFIG_PATH,
                overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load, override and validate the YAML configuration.

    Args:
        path: path to the YAML file.
        overrides: optional mapping of dotted keys to values, e.g.
            ``{"data.root": "C:/tmp/raw"}``.

    Raises:
        ConfigError: if the file is missing/invalid or validation fails.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a top-level mapping")
    cfg: dict[str, Any] = raw
    for dotted, value in (overrides or {}).items():
        _set_dotted(cfg, dotted, value)
    validate_config(cfg)
    return cfg


def _set_dotted(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``cfg`` at ``dotted_key``, creating intermediate mappings."""
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Validate structure and the important invariants of the configuration.

    Raises:
        ConfigError: on the first violated invariant, with an actionable message.
    """
    invalid_sections = [s for s in REQUIRED_SECTIONS if not isinstance(cfg.get(s), Mapping)]
    if invalid_sections:
        raise ConfigError(f"missing or invalid config sections: {invalid_sections}")

    missing_keys = [k for k in REQUIRED_KEYS if cfg_get(cfg, k, _MISSING) is _MISSING]
    if missing_keys:
        raise ConfigError(f"missing required config keys: {missing_keys}")

    class_names = cfg_get(cfg, "data.class_names")
    if (not isinstance(class_names, Sequence) or isinstance(class_names, str)
            or not all(isinstance(name, str) for name in class_names)):
        raise ConfigError("data.class_names must be a sequence of strings")
    num_classes = cfg_get(cfg, "data.num_classes")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 1:
        raise ConfigError("data.num_classes must be a positive integer")
    if len(class_names) != num_classes:
        raise ConfigError(
            f"len(data.class_names)={len(class_names)} != data.num_classes={num_classes}"
        )
    image_size = cfg_get(cfg, "data.image_size")
    if not isinstance(image_size, int) or not 1 <= image_size <= 4096:
        raise ConfigError("data.image_size must be an integer between 1 and 4096")

    ratios = [cfg_get(cfg, f"data.split.{name}", None) for name in ("train", "val", "test")]
    if not all(isinstance(r, (int, float)) and not isinstance(r, bool) and r > 0 for r in ratios):
        raise ConfigError("data.split.{train,val,test} must be positive numbers")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ConfigError(f"data.split ratios must sum to 1.0, got {sum(ratios)}")

    reduction = cfg_get(cfg, "attention.cbam.channel.reduction_ratio")
    if not isinstance(reduction, int) or isinstance(reduction, bool) or reduction < 1:
        raise ConfigError("attention.cbam.channel.reduction_ratio must be an integer >= 1")
    kernel = cfg_get(cfg, "attention.cbam.spatial.kernel_size")
    if not isinstance(kernel, int) or isinstance(kernel, bool) or kernel < 1 or kernel % 2 == 0:
        raise ConfigError("attention.cbam.spatial.kernel_size must be a positive odd integer")
    if cfg_get(cfg, "attention.type") not in {"baseline", "se", "cbam"}:
        raise ConfigError("attention.type must be one of: baseline, se, cbam")

    dropout = cfg_get(cfg, "model.head.dropout")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0.0 <= float(dropout) < 1.0:
        raise ConfigError("model.head.dropout must be in [0, 1)")
    smoothing = cfg_get(cfg, "training.loss.label_smoothing")
    if not isinstance(smoothing, (int, float)) or isinstance(smoothing, bool) or not 0.0 <= float(smoothing) < 1.0:
        raise ConfigError("training.loss.label_smoothing must be in [0, 1)")
    for key in ("optimizer.learning_rate", "optimizer.backbone_learning_rate",
                "optimizer.weight_decay"):
        value = cfg_get(cfg, key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ConfigError(f"{key} must be a non-negative number")

    for key in ("data.loader.batch_size", "data.loader.eval_batch_size",
                "data.loader.num_workers", "training.max_epochs", "smoke_test.max_epochs",
                "smoke_test.subset_size"):
        value = cfg_get(cfg, key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ConfigError(f"{key} must be a non-negative integer")
    if cfg_get(cfg, "data.loader.batch_size") < 1:
        raise ConfigError("data.loader.batch_size must be >= 1")
    if cfg_get(cfg, "training.max_epochs") < 1:
        raise ConfigError("training.max_epochs must be >= 1")
    if cfg_get(cfg, "smoke_test.max_epochs") > 1:
        raise ConfigError("smoke_test.max_epochs must be <= 1")
    if cfg_get(cfg, "smoke_test.subset_size") < 1:
        raise ConfigError("smoke_test.subset_size must be >= 1")
    if not isinstance(cfg_get(cfg, "seed.value"), int) or isinstance(cfg_get(cfg, "seed.value"), bool):
        raise ConfigError("seed.value must be an integer")
