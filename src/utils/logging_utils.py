"""Logging helpers shared by the project scripts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Mapping

from .paths import ensure_dir

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "tomato", level: int = logging.INFO,
               log_file: str | Path | None = None,
               cfg: Mapping[str, Any] | None = None) -> logging.Logger:
    """Return a configured logger writing to stdout and optionally to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter(_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_file is not None:
        target = Path(log_file)
        if cfg is not None:
            target = ensure_dir(target.parent, cfg) / target.name
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def log_config_summary(cfg: Mapping[str, Any], logger: logging.Logger) -> None:
    """Log the configuration values that identify and shape a run."""
    from ..config import cfg_get, config_fingerprint

    logger.info("config fingerprint: %s", config_fingerprint(cfg))
    for key in (
        "project.name",
        "project.phase",
        "seed.value",
        "data.root",
        "data.image_size",
        "data.num_classes",
        "data.split.train",
        "data.split.val",
        "data.split.test",
        "model.backbone",
        "attention.type",
        "training.max_epochs",
        "optimizer.learning_rate",
        "smoke_test.enabled",
    ):
        logger.info("%-28s = %r", key, cfg_get(cfg, key))
