"""Global seeding for reproducible runs (Phase 2C).

Every script must call :func:`set_seed` once, with ``seed.value`` from
``configs/config.yaml``; no module may hardcode a seed (project rule).
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Mapping

import numpy as np
import torch

from ..config import cfg_get


def set_seed(cfg: Mapping[str, Any], logger: logging.Logger | None = None) -> int:
    """Seed python, numpy and torch from ``seed.value`` and return the seed used.

    When ``seed.deterministic`` is true, deterministic algorithm selection is requested
    (``error_if_possible=False`` keeps CUDA nondeterministic kernels usable, as some
    torchvision CUDA kernels have no deterministic implementation). ``seed.cudnn_benchmark``
    is applied as configured; it must stay ``false`` while determinism is requested.
    """
    seed = int(cfg_get(cfg, "seed.value"))
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    deterministic = bool(cfg_get(cfg, "seed.deterministic", True))
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = bool(cfg_get(cfg, "seed.cudnn_benchmark", False))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if logger is not None:
        logger.info("seed=%s deterministic=%s cudnn_benchmark=%s",
                    seed, deterministic, torch.backends.cudnn.benchmark)
    return seed
