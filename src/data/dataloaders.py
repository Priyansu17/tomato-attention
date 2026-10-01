"""Config-driven DataLoader factory (Phase 2C).

All knobs (batch sizes, workers, shuffling, drop-last, pinning, augmentation, seeds,
smoke-test overrides) are read from ``configs/config.yaml``. ``data/raw`` remains
read-only: loading uses the persisted ``splits.csv`` manifest only.

Smoke mode uses the tiny ``smoke_test`` overrides with a deterministic, class-balanced
row subset (``limit_rows``), honoring ``smoke_test.max_epochs: 1``.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import cfg_get, load_config
from ..utils.logging_utils import get_logger, log_config_summary
from ..utils.paths import dataset_root, manifests_dir
from ..utils.seed import set_seed
from .dataset import LeafImageDataset, SplitRow, limit_rows, load_split_rows
from .splitting import SPLITS_FILENAME
from .transforms import build_eval_transform, build_train_transform

SPLITS = ("train", "val", "test")


class DataLoaderConfigError(ValueError):
    """Raised when loader configuration is unusable."""


@dataclass(frozen=True)
class DataLoaderBundle:
    """The three loaders plus the datasets and row subsets they were built from."""

    loaders: dict[str, DataLoader]
    datasets: dict[str, LeafImageDataset]
    rows: dict[str, list[SplitRow]]


def worker_init(worker_id: int) -> None:
    """Seed ``random``/``numpy`` inside each DataLoader worker from torch's base seed.

    ``torch.utils.data`` already seeds ``torch`` per worker; this makes the python and
    numpy RNGs (used by some transforms) worker-deterministic as well.
    """
    info = torch.utils.data.get_worker_info()
    base = int(info.seed) if info is not None else torch.initial_seed()
    random.seed(base)
    np.random.seed(base % (2**32))


def _make_loader(dataset: LeafImageDataset, *, batch_size: int, shuffle: bool,
                 num_workers: int, pin_memory: bool, drop_last: bool, seed: int,
                 profile: str) -> DataLoader:
    """Build one DataLoader from validated, config-provided parameters."""
    if batch_size < 1:
        raise DataLoaderConfigError(f"batch_size must be >= 1, got {batch_size}")
    if num_workers < 0:
        raise DataLoaderConfigError(f"num_workers must be >= 0, got {num_workers}")
    if profile not in SPLITS:
        raise DataLoaderConfigError(f"unknown loader profile: {profile}")
    if drop_last and len(dataset) < batch_size:
        raise DataLoaderConfigError(
            f"drop_last=True with {len(dataset)} samples < batch_size={batch_size}"
        )
    kwargs: dict[str, Any] = {}
    if shuffle:
        # Dedicated generator => the epoch-1 sample order depends only on the seed.
        generator = torch.Generator()
        generator.manual_seed(seed)
        kwargs["generator"] = generator
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init,
        persistent_workers=num_workers > 0,
        **kwargs,
    )


def build_dataloaders(cfg: Mapping[str, Any], smoke: bool = False,
                      rows: Mapping[str, list[SplitRow]] | None = None,
                      manifest: str | None = None) -> DataLoaderBundle:
    """Build train/val/test loaders from config. ``data/raw`` is never written.

    Args:
        cfg: validated configuration mapping.
        smoke: apply the tiny ``smoke_test`` overrides (subsets, batch size, workers,
            shuffle) instead of the production loader settings.
        rows: pre-computed split rows (tests); defaults to the persisted manifest.
        manifest: path to ``splits.csv``; defaults to ``paths.manifests_dir/splits.csv``.
    """
    seed = int(cfg_get(cfg, "seed.value"))
    set_seed(cfg)
    if rows is None:
        base = manifest or str(manifests_dir(cfg) / SPLITS_FILENAME)
        rows = {split: load_split_rows(base, split) for split in SPLITS}

    root = dataset_root(cfg)
    subset_kwargs: dict[str, Any]
    if smoke:
        subset_size = int(cfg_get(cfg, "smoke_test.subset_size"))
        batch_size = int(cfg_get(cfg, "smoke_test.batch_size",
                                 cfg_get(cfg, "data.loader.batch_size")))
        num_workers = int(cfg_get(cfg, "smoke_test.num_workers"))
        shuffle = bool(cfg_get(cfg, "smoke_test.shuffle", False))
        pin_memory = False
        subset_kwargs = {"seed": seed, "per_class_cap": None}
    else:
        subset_size = 0
        batch_size = int(cfg_get(cfg, "data.loader.batch_size"))
        num_workers = int(cfg_get(cfg, "data.loader.num_workers"))
        shuffle = bool(cfg_get(cfg, "data.loader.shuffle_train", True))
        pin_memory = (bool(cfg_get(cfg, "data.loader.pin_memory", False))
                      and torch.cuda.is_available())
        subset_kwargs = {}

    train_batch = int(cfg_get(cfg, "training.batch_size", batch_size))
    eval_batch = int(cfg_get(cfg, "data.loader.eval_batch_size", batch_size))
    drop_last = bool(cfg_get(cfg, "data.loader.drop_last", False))

    subsets: dict[str, list[SplitRow]] = {}
    datasets: dict[str, LeafImageDataset] = {}
    loaders: dict[str, DataLoader] = {}
    for split in SPLITS:
        split_rows = list(rows[split])
        if not split_rows:
            raise DataLoaderConfigError(f"no rows available for split '{split}'")
        if smoke and subset_size:
            split_rows = limit_rows(split_rows, subset_size, **subset_kwargs)
        subsets[split] = split_rows
        transform = build_train_transform(cfg) if split == "train" else build_eval_transform(cfg)
        datasets[split] = LeafImageDataset(split_rows, root, transform)
        loaders[split] = _make_loader(
            datasets[split],
            batch_size=train_batch if split == "train" else eval_batch,
            shuffle=shuffle if split == "train" else False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last if split == "train" else False,
            seed=seed,
            profile=split,
        )
    return DataLoaderBundle(loaders=loaders, datasets=datasets, rows=subsets)


def main(argv: list[str] | None = None) -> int:
    """CLI: print dataset sizes and fetch one batch per split to verify the pipeline."""
    parser = argparse.ArgumentParser(description="Build and smoke-verify the DataLoaders.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--smoke", action="store_true", help="apply smoke_test overrides")
    parser.add_argument("--workers", type=int, default=None,
                        help="override num_workers (use 0 when the worker port is blocked)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    logger = get_logger("tomato.dataloaders")
    log_config_summary(cfg, logger)
    if args.workers is not None:
        cfg["data"]["loader"]["num_workers"] = args.workers
        cfg["smoke_test"]["num_workers"] = args.workers

    bundle = build_dataloaders(cfg, smoke=args.smoke)
    for split in SPLITS:
        images, labels = next(iter(bundle.loaders[split]))
        logger.info("%s: %d rows, dataset=%d, batch %s labels %s",
                    split, len(bundle.rows[split]), len(bundle.datasets[split]),
                    tuple(images.shape), labels.tolist()[:4])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
