"""Dataset discovery, validation and manifest generation.

``data/raw`` is strictly read-only: this module only reads the dataset and writes its
manifest into ``paths.manifests_dir`` (guarded against the dataset root).

Label ids come from :func:`src.config.class_label_map`, i.e. from the order of
``data.class_names`` in ``configs/config.yaml`` -- never from filesystem, sorted or
tool-specific enumeration order.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from ..config import cfg_get, class_label_map, load_config
from ..utils.logging_utils import get_logger, log_config_summary
from ..utils.paths import (
    assert_not_inside_dataset_root,
    dataset_root,
    ensure_dir,
    manifests_dir,
)

DEFAULT_SPLIT_DIR_NAMES: tuple[str, ...] = ("train", "val", "valid", "validation", "test")

CSV_FIELDS: tuple[str, ...] = (
    "rel_path",
    "class_name",
    "class_index",
    "size_bytes",
    "width",
    "height",
    "mode",
)


class DatasetLayoutError(RuntimeError):
    """Raised when the dataset folders do not match ``data.class_names``."""


class DatasetImageError(RuntimeError):
    """Raised when an image file cannot be read."""


@dataclass(frozen=True)
class ImageRecord:
    """One image of the dataset index."""

    rel_path: str
    class_name: str
    class_index: int
    size_bytes: int
    width: int
    height: int
    mode: str

    def as_row(self) -> dict[str, Any]:
        """Return the record as a plain dict in CSV column order."""
        return asdict(self)


def normalize_extensions(extensions: Sequence[str]) -> tuple[str, ...]:
    """Return lower-cased, dot-prefixed, de-duplicated file extensions."""
    normalized = {(ext if str(ext).startswith(".") else f".{ext}").lower() for ext in extensions}
    return tuple(sorted(normalized))


def discover_class_dirs(root: str | Path, expected: Sequence[str],
                        split_names: Sequence[str] = DEFAULT_SPLIT_DIR_NAMES) -> dict[str, Path]:
    """Return ``{class_name: folder}`` ordered exactly like ``expected``.

    Both layouts are supported: ``root/<class>/`` and ``root/<split>/<class>/``.

    Raises:
        DatasetLayoutError: if the root is missing/empty or the discovered folders do not
            match ``expected`` exactly (missing and/or unexpected entries are reported).
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise DatasetLayoutError(f"dataset root not found: {root_path}")
    top_level = {path.name: path for path in sorted(root_path.iterdir()) if path.is_dir()}
    if not top_level:
        raise DatasetLayoutError(f"no class folders found under {root_path}")

    found: dict[str, Path]
    if set(top_level) & set(split_names):
        found = {}
        for split_name in split_names:
            split_dir = top_level.get(split_name)
            if split_dir is None:
                continue
            for child in sorted(split_dir.iterdir()):
                if child.is_dir():
                    found.setdefault(child.name, child)
    else:
        found = top_level

    expected_set = set(expected)
    missing = sorted(expected_set - set(found))
    unexpected = sorted(set(found) - expected_set)
    if missing or unexpected:
        raise DatasetLayoutError(
            "class folders do not match data.class_names: "
            f"missing={missing} unexpected={unexpected}"
        )
    return {name: found[name] for name in expected}


def iter_image_files(class_dir: str | Path, extensions: Sequence[str]) -> list[Path]:
    """Return the image files in ``class_dir`` (case-insensitive extensions)."""
    suffixes = set(normalize_extensions(extensions))
    return [
        path
        for path in sorted(Path(class_dir).rglob("*"))
        if path.is_file() and path.suffix.lower() in suffixes
    ]


def read_image_header(path: str | Path) -> tuple[int, int, str]:
    """Return ``(width, height, mode)`` using a header-only read (no pixel decoding).

    Raises:
        DatasetImageError: if the file is not a readable image.
    """
    try:
        with Image.open(path) as image:
            return int(image.width), int(image.height), str(image.mode)
    except Exception as exc:  # noqa: BLE001 - re-raised with file context
        raise DatasetImageError(f"cannot read image header for {path}: {exc}") from exc


def build_index(cfg: Mapping[str, Any], root: str | Path | None = None,
                extensions: Sequence[str] | None = None) -> list[ImageRecord]:
    """Build the image index of the dataset described by ``cfg`` (read-only)."""
    root_path = dataset_root(cfg) if root is None else Path(root)
    class_names = list(cfg_get(cfg, "data.class_names", []) or [])
    suffixes = list(extensions if extensions is not None
                    else cfg_get(cfg, "data.extensions", [".jpg", ".jpeg", ".png"]))
    labels = class_label_map(cfg)

    records: list[ImageRecord] = []
    for class_name, folder in discover_class_dirs(root_path, class_names).items():
        for path in iter_image_files(folder, suffixes):
            width, height, mode = read_image_header(path)
            records.append(
                ImageRecord(
                    rel_path=path.relative_to(root_path).as_posix(),
                    class_name=class_name,
                    class_index=labels[class_name],
                    size_bytes=path.stat().st_size,
                    width=width,
                    height=height,
                    mode=mode,
                )
            )
    return records


def index_summary(records: Sequence[ImageRecord]) -> dict[str, Any]:
    """Summarise an index: counts per class (label order), dimensions, modes, bytes."""
    counts = Counter(record.class_name for record in records)
    dimensions = Counter(f"{record.width}x{record.height}" for record in records)
    modes = Counter(record.mode for record in records)
    sizes = [record.size_bytes for record in records]
    per_class = [
        {"class_index": index, "class_name": name, "count": counts[name]}
        for index, name in sorted({(r.class_index, r.class_name) for r in records})
    ]
    return {
        "num_images": len(records),
        "num_classes": len(counts),
        "per_class": per_class,
        "dimensions": dict(dimensions.most_common()),
        "modes": dict(modes.most_common()),
        "bytes_total": sum(sizes),
        "bytes_min": min(sizes) if sizes else 0,
        "bytes_max": max(sizes) if sizes else 0,
    }


def write_index_csv(records: Sequence[ImageRecord], out_path: str | Path,
                    cfg: Mapping[str, Any]) -> Path:
    """Write the index manifest as CSV, refusing paths inside the dataset root."""
    target = assert_not_inside_dataset_root(out_path, cfg)
    ensure_dir(target.parent, cfg)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_row())
    return target


def format_summary(summary: Mapping[str, Any], root: str | Path,
                   manifest: str | Path | None = None) -> str:
    """Render a human-readable dataset index summary."""
    lines = [
        f"dataset root     : {root}",
        f"images           : {summary['num_images']}",
        f"classes          : {summary['num_classes']}",
        f"dimensions       : {summary['dimensions']}",
        f"colour modes     : {summary['modes']}",
        f"bytes total (MB) : {summary['bytes_total'] / (1024 * 1024):.1f}",
        f"bytes min / max  : {summary['bytes_min']} / {summary['bytes_max']}",
        "per class (label order from config.data.class_names):",
    ]
    for entry in summary["per_class"]:
        lines.append(
            f"  {entry['class_index']:>2}  {entry['class_name']:<47} {entry['count']:>6}"
        )
    if manifest is not None:
        lines.append(f"manifest         : {manifest}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: build and persist the dataset index manifest."""
    parser = argparse.ArgumentParser(
        description="Build the dataset index manifest (read-only w.r.t. data/raw)."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out", default=None, help="output CSV path")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    logger = get_logger("tomato.discovery")
    log_config_summary(cfg, logger)

    records = build_index(cfg)
    root = dataset_root(cfg)
    summary = index_summary(records)
    out_path = Path(args.out) if args.out else manifests_dir(cfg) / "dataset_index.csv"
    print(format_summary(summary, root, out_path))
    written = write_index_csv(records, out_path, cfg)
    logger.info("wrote %s (%d records)", written, len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
