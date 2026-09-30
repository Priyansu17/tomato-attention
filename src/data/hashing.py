"""Deterministic SHA-256 content hashing and exact-duplicate group detection.

``data/raw`` is read-only: files are only ever opened for reading. Hash and
duplicate-group metadata are written to ``paths.manifests_dir`` (never inside the
dataset root).

Only byte-identical images are grouped together - near-duplicate detection is out of
scope (decision D9).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import cfg_get, class_label_map, load_config
from ..utils.logging_utils import get_logger, log_config_summary
from ..utils.paths import assert_not_inside_dataset_root, dataset_root, ensure_dir, manifests_dir
from .discovery import build_index

DEFAULT_CHUNK_SIZE = 1024 * 1024
HASHES_FILENAME = "hashes.csv"
DUP_GROUPS_FILENAME = "duplicate_groups.csv"
HASH_FIELDS = ("rel_path", "class_name", "class_index", "size_bytes", "sha256", "group_id")
DUP_GROUP_FIELDS = ("group_id", "sha256", "member_count", "members")


class HashingError(RuntimeError):
    """Raised when a file cannot be read for hashing."""


@dataclass(frozen=True)
class FileHash:
    """SHA-256 content hash of one dataset file."""

    rel_path: str
    class_name: str
    class_index: int
    size_bytes: int
    sha256: str
    group_id: int = -1


def hash_file(path: str | Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return the SHA-256 hex digest of ``path`` read in chunks of ``chunk_size``.

    Reads the file in a streaming fashion so large files never load fully into memory.

    Raises:
        HashingError: if ``chunk_size`` is invalid or the path is not a readable file.
    """
    if chunk_size < 1:
        raise HashingError(f"chunk_size must be >= 1, got {chunk_size}")
    target = Path(path)
    if not target.is_file():
        raise HashingError(f"cannot hash, not a regular file: {target}")
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            while True:
                block = handle.read(chunk_size)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise HashingError(f"cannot read {target}: {exc}") from exc
    return digest.hexdigest()


def hash_records(records: Sequence[Any], cfg: Mapping[str, Any],
                 base_dir: str | Path, chunk_size: int | None = None) -> list[FileHash]:
    """Hash every discovered record under ``base_dir`` (read-only).

    ``class_index`` is taken exclusively from ``config.data.class_names`` (decision D12).
    """
    size = int(chunk_size if chunk_size is not None
               else cfg_get(cfg, "data.hashing.chunk_size", DEFAULT_CHUNK_SIZE))
    labels = class_label_map(cfg)
    root = Path(base_dir)
    hashed: list[FileHash] = []
    for record in records:
        if record.class_name not in labels:
            raise HashingError(f"class not present in config.data.class_names: {record.class_name}")
        hashed.append(
            FileHash(
                rel_path=record.rel_path,
                class_name=record.class_name,
                class_index=labels[record.class_name],
                size_bytes=record.size_bytes,
                sha256=hash_file(root / record.rel_path, size),
            )
        )
    return hashed


def assign_group_ids(hashes: Sequence[FileHash]) -> list[FileHash]:
    """Assign deterministic group ids ordered by SHA-256 ascending."""
    unique = sorted({item.sha256 for item in hashes})
    ids = {sha: index for index, sha in enumerate(unique)}
    return [
        FileHash(item.rel_path, item.class_name, item.class_index,
                 item.size_bytes, item.sha256, ids[item.sha256])
        for item in hashes
    ]


def groups_by_id(hashes: Sequence[FileHash]) -> dict[int, list[FileHash]]:
    """Return ``{group_id: [members]}`` preserving input order inside each group."""
    grouped: dict[int, list[FileHash]] = {}
    for item in hashes:
        grouped.setdefault(item.group_id, []).append(item)
    return grouped


def duplicate_group_rows(hashes: Sequence[FileHash]) -> list[dict[str, Any]]:
    """Return one row per group with two or more members, ordered by group id."""
    rows: list[dict[str, Any]] = []
    for group_id, members in sorted(groups_by_id(hashes).items()):
        if len(members) < 2:
            continue
        rows.append({
            "group_id": group_id,
            "sha256": members[0].sha256,
            "member_count": len(members),
            "members": ";".join(sorted(member.rel_path for member in members)),
        })
    return rows


def hash_summary(hashes: Sequence[FileHash]) -> dict[str, Any]:
    """Summarise hashing results: totals, duplicate groups and size distribution."""
    grouped = groups_by_id(hashes)
    sizes = Counter(len(members) for members in grouped.values())
    duplicate_groups = [gid for gid, members in grouped.items() if len(members) >= 2]
    per_class = Counter(item.class_name for item in hashes)
    unique_per_class: dict[str, set[str]] = {}
    for item in hashes:
        unique_per_class.setdefault(item.class_name, set()).add(item.sha256)
    return {
        "total_images": len(hashes),
        "unique_sha256": len(grouped),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_files": sum(len(grouped[gid]) for gid in duplicate_groups),
        "redundant_copies": len(hashes) - len(grouped),
        "max_group_size": max((len(m) for m in grouped.values()), default=0),
        "group_size_distribution": dict(sorted(sizes.items())),
        "per_class_images": dict(per_class),
        "per_class_unique": {name: len(values) for name, values in sorted(unique_per_class.items())},
    }


def write_hashes_csv(hashes: Sequence[FileHash], out_path: str | Path,
                     cfg: Mapping[str, Any]) -> Path:
    """Write the hash manifest as CSV, refusing paths inside the dataset root."""
    target = assert_not_inside_dataset_root(out_path, cfg)
    ensure_dir(target.parent, cfg)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HASH_FIELDS))
        writer.writeheader()
        for item in hashes:
            writer.writerow(asdict(item))
    return target


def write_duplicate_groups_csv(rows: Sequence[Mapping[str, Any]], out_path: str | Path,
                               cfg: Mapping[str, Any]) -> Path:
    """Write the duplicate-group manifest as CSV (never inside the dataset root)."""
    target = assert_not_inside_dataset_root(out_path, cfg)
    ensure_dir(target.parent, cfg)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DUP_GROUP_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return target


def load_hashes(path: str | Path) -> list[FileHash]:
    """Read a previously written hash manifest back into ``FileHash`` objects.

    Raises:
        HashingError: if the manifest does not exist.
    """
    source = Path(path)
    if not source.is_file():
        raise HashingError(f"hash manifest not found: {source}")
    hashes: list[FileHash] = []
    with source.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            hashes.append(
                FileHash(
                    rel_path=row["rel_path"],
                    class_name=row["class_name"],
                    class_index=int(row["class_index"]),
                    size_bytes=int(row["size_bytes"]),
                    sha256=row["sha256"],
                    group_id=int(row["group_id"]),
                )
            )
    if not hashes:
        raise HashingError(f"hash manifest is empty: {source}")
    return hashes


def format_summary(summary: Mapping[str, Any]) -> str:
    """Render a human-readable hashing summary."""
    distribution = summary["group_size_distribution"]
    dist_text = ", ".join(f"{size}x{count}" for size, count in distribution.items())
    lines = [
        f"total images           : {summary['total_images']}",
        f"unique sha256          : {summary['unique_sha256']}",
        f"duplicate groups       : {summary['duplicate_groups']}",
        f"files in duplicate grps: {summary['duplicate_files']}",
        f"redundant copies       : {summary['redundant_copies']}",
        f"max group size         : {summary['max_group_size']}",
        f"group size distribution: {dist_text or '-'}",
        "per class (images / unique):",
    ]
    images = summary["per_class_images"]
    unique = summary["per_class_unique"]
    for name, count in images.items():
        lines.append(f"  {name:<47} {count:>6} / {unique[name]:>6}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: hash the dataset and persist hash/duplicate manifests."""
    parser = argparse.ArgumentParser(
        description="Compute SHA-256 hashes and detect exact duplicates (read-only)."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out-dir", default=None, help="output directory for manifests")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    logger = get_logger("tomato.hashing")
    log_config_summary(cfg, logger)

    records = build_index(cfg)
    hashes = assign_group_ids(hash_records(records, cfg, dataset_root(cfg)))
    out_dir = Path(args.out_dir) if args.out_dir else manifests_dir(cfg)
    rows = duplicate_group_rows(hashes)

    hashes_path = write_hashes_csv(hashes, out_dir / HASHES_FILENAME, cfg)
    groups_path = write_duplicate_groups_csv(rows, out_dir / DUP_GROUPS_FILENAME, cfg)

    print(format_summary(hash_summary(hashes)))
    logger.info("wrote %s (%d rows)", hashes_path, len(hashes))
    logger.info("wrote %s (%d duplicate groups)", groups_path, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
