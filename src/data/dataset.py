"""Torch dataset over the persisted split manifest (Phase 2C).

``data/raw`` stays read-only: images are only ever opened for reading, and labels come
from the manifest's ``class_index`` column, which was created exclusively from
``config.data.class_names`` order (decision D12) — never re-derived here.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from ..config import load_config
from ..utils.logging_utils import get_logger, log_config_summary
from ..utils.paths import dataset_root, manifests_dir
from .splitting import SPLITS_FILENAME, SPLIT_FIELDS, SPLIT_NAMES

Label = tuple[Any, int]


class DatasetError(RuntimeError):
    """Raised when split rows or image files cannot be loaded."""


@dataclass(frozen=True)
class SplitRow:
    """One row of ``results/manifests/splits.csv``."""

    rel_path: str
    class_name: str
    class_index: int
    sha256: str
    group_id: int
    split: str


def load_split_rows(path: str | Path, split: str | None = None) -> list[SplitRow]:
    """Load split rows from a manifest CSV, optionally filtered by ``split``.

    Raises:
        DatasetError: if the manifest is missing, malformed, or names an unknown split.
    """
    source = Path(path)
    if not source.is_file():
        raise DatasetError(f"split manifest not found: {source}")
    if split is not None and split not in SPLIT_NAMES:
        raise DatasetError(f"unknown split '{split}', expected one of {SPLIT_NAMES}")
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != tuple(SPLIT_FIELDS):
            raise DatasetError(f"unexpected columns in {source}, expected {SPLIT_FIELDS}")
        rows = [
            SplitRow(
                rel_path=row["rel_path"],
                class_name=row["class_name"],
                class_index=int(row["class_index"]),
                sha256=row["sha256"],
                group_id=int(row["group_id"]),
                split=row["split"],
            )
            for row in reader
        ]
    unknown = {row.split for row in rows} - set(SPLIT_NAMES)
    if unknown:
        raise DatasetError(f"manifest {source} contains unknown splits: {sorted(unknown)}")
    return rows if split is None else [row for row in rows if row.split == split]


class LeafImageDataset(Dataset):
    """RGB leaf images addressed by split rows, returning ``(image_tensor, label)``.

    The label is ``row.class_index`` as written by the Phase 2B splitter (config label
    order). Decoding failures raise instead of being silently skipped.
    """

    def __init__(self, rows: Sequence[SplitRow], root: str | Path,
                 transform: Any | None = None) -> None:
        if not rows:
            raise DatasetError("LeafImageDataset requires at least one split row")
        self.rows = list(rows)
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def path_for(self, index: int) -> Path:
        """Absolute path of the image at ``index`` (read-only access)."""
        return self.root / self.rows[index].rel_path

    def __getitem__(self, index: int) -> Label:
        row = self.rows[index]
        path = self.root / row.rel_path
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.load()
                sample = image if self.transform is None else self.transform(image)
        except FileNotFoundError as exc:
            raise DatasetError(f"image listed in the manifest is missing: {path}") from exc
        except UnidentifiedImageError as exc:
            raise DatasetError(f"image cannot be decoded: {path}") from exc
        return sample, row.class_index



def limit_rows(rows: Sequence[SplitRow], limit: int, seed: int,
               per_class_cap: int | None = None) -> list[SplitRow]:
    """Deterministically return up to ``limit`` rows for smoke-test subsets.

    Rows are deduplicated by ``rel_path``, sorted, shuffled with ``random.Random(seed)``
    and then selected round-robin across classes, so tiny subsets stay balanced;
    ``per_class_cap`` optionally bounds how many rows one class contributes.
    """
    if limit < 1:
        raise DatasetError(f"limit must be >= 1, got {limit}")
    unique: dict[str, SplitRow] = {}
    for row in rows:
        unique[row.rel_path] = row
    rng = random.Random(seed)
    by_class: dict[int, list[SplitRow]] = defaultdict(list)
    for row in sorted(unique.values(), key=lambda r: r.rel_path):
        by_class[row.class_index].append(row)
    for items in by_class.values():
        rng.shuffle(items)
    if per_class_cap is not None:
        by_class = {cid: items[:per_class_cap] for cid, items in by_class.items()}
    selected: list[SplitRow] = []
    class_ids = sorted(by_class)
    offsets: Counter = Counter()
    while len(selected) < limit:
        progressed = False
        for class_id in class_ids:
            items = by_class[class_id]
            if offsets[class_id] < len(items):
                selected.append(items[offsets[class_id]])
                offsets[class_id] += 1
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: report per-split dataset sizes and one decoded sample shape."""
    parser = argparse.ArgumentParser(description="Inspect the split dataset (read-only).")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", default="train", choices=SPLIT_NAMES)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    logger = get_logger("tomato.dataset")
    log_config_summary(cfg, logger)

    rows = load_split_rows(manifests_dir(cfg) / SPLITS_FILENAME, args.split)
    counts = Counter(row.class_index for row in rows)
    print(f"split={args.split} rows={len(rows)} root={dataset_root(cfg)}")
    for class_id in sorted(counts):
        print(f"  class {class_id:>2}: {counts[class_id]}")

    from .transforms import build_eval_transform

    dataset = LeafImageDataset(rows[:4], dataset_root(cfg), build_eval_transform(cfg))
    image, label = dataset[0]
    print(f"first sample: tensor {tuple(image.shape)} label={label} "
          f"range=({image.min():.3f}, {image.max():.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
