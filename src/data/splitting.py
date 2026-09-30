"""Deterministic stratified group split (0.70 / 0.15 / 0.15) with intact duplicate groups.

The split is computed from the SHA-256 groups produced by :mod:`src.data.hashing`:

* every member of an exact-duplicate group lands in the SAME split (decision D8),
* overall and per-class proportions stay within ``data.splitting.tolerance`` (decision D4),
* repeated runs with the same seed yield identical assignments,
* ``class_index`` comes exclusively from ``config.data.class_names`` (decision D12).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import cfg_get, config_fingerprint, load_config
from ..utils.logging_utils import get_logger, log_config_summary
from ..utils.paths import assert_not_inside_dataset_root, ensure_dir, manifests_dir
from .hashing import HASHES_FILENAME, FileHash, load_hashes

SPLIT_NAMES = ("train", "val", "test")
SPLITS_FILENAME = "splits.csv"
SUMMARY_FILENAME = "split_summary.json"
SPLIT_FIELDS = ("rel_path", "class_name", "class_index", "sha256", "group_id", "split")


class SplitError(RuntimeError):
    """Raised when a valid split cannot be produced."""


def ratios_from_config(cfg: Mapping[str, Any]) -> dict[str, float]:
    """Return the configured ``{split: ratio}`` mapping for train/val/test."""
    return {name: float(cfg_get(cfg, f"data.split.{name}")) for name in SPLIT_NAMES}


def tolerance_from_config(cfg: Mapping[str, Any]) -> float:
    """Return ``data.splitting.tolerance`` (max allowed ratio deviation)."""
    return float(cfg_get(cfg, "data.splitting.tolerance", 0.01))


def group_members(hashes: Sequence[FileHash]) -> dict[int, list[FileHash]]:
    """Return ``{group_id: [members]}``, the units the split is assigned on."""
    grouped: dict[int, list[FileHash]] = {}
    for item in hashes:
        grouped.setdefault(item.group_id, []).append(item)
    return grouped


def _counts(assignment: Mapping[int, str], groups: Mapping[int, Sequence[FileHash]],
            total: int) -> tuple[dict[str, Counter], Counter]:
    """Recompute per-split/per-class and per-split global image counts."""
    counts = {split: Counter() for split in SPLIT_NAMES}
    global_counts: Counter = Counter()
    for group_id, split in assignment.items():
        members = groups[group_id]
        for class_index, size in Counter(m.class_index for m in members).items():
            counts[split][class_index] += size
        global_counts[split] += len(members)
    return counts, global_counts


def _objective(counts: Mapping[str, Counter], global_counts: Counter,
               class_totals: Counter, total: int, ratios: Mapping[str, float]) -> float:
    """Total absolute ratio deviation (global + per class), in fraction units."""
    score = 0.0
    for split in SPLIT_NAMES:
        score += abs(global_counts[split] / total - ratios[split])
        for class_index, class_total in class_totals.items():
            score += abs(counts[split][class_index] / class_total - ratios[split])
    return score


def max_deviation(counts: Mapping[str, Counter], global_counts: Counter,
                  class_totals: Counter, total: int, ratios: Mapping[str, float]) -> float:
    """Return the worst absolute ratio deviation over all splits and classes."""
    worst = 0.0
    for split in SPLIT_NAMES:
        worst = max(worst, abs(global_counts[split] / total - ratios[split]))
        for class_index, class_total in class_totals.items():
            worst = max(worst, abs(counts[split][class_index] / class_total - ratios[split]))
    return worst


def per_class_bounds(class_totals: Counter, tolerance: float) -> dict[int, float]:
    """Return the per-class ratio deviation bound for each class.

    The bound is the configured ``tolerance`` or the fraction that a one-image
    error represents, whichever is larger. A one-image rounding error is
    unavoidable for tiny classes, where a single image shifts the fraction a lot.
    """
    return {index: max(tolerance, 1.0 / max(1, total)) for index, total in class_totals.items()}


def is_within_tolerance(counts: Mapping[str, Counter], global_counts: Counter,
                        class_totals: Counter, total: int,
                        ratios: Mapping[str, float], tolerance: float) -> bool:
    """Return True when global and per-class ratios are all within tolerance.

    Global ratios must be within ``tolerance``; each class ratio must be within the
    class-aware bound from :func:`per_class_bounds`.
    """
    for split in SPLIT_NAMES:
        if abs(global_counts[split] / total - ratios[split]) > tolerance:
            return False
    bounds = per_class_bounds(class_totals, tolerance)
    for class_index, class_total in class_totals.items():
        for split in SPLIT_NAMES:
            if abs(counts[split][class_index] / class_total - ratios[split]) > bounds[class_index]:
                return False
    return True


def _repair(assignment: dict[int, str], groups: Mapping[int, Sequence[FileHash]],
            class_totals: Counter, total: int, ratios: Mapping[str, float],
            tolerance: float, max_passes: int = 6) -> tuple[dict[int, str], dict, Counter]:
    """Deterministically move whole groups between splits until within tolerance.

    Only whole groups are ever moved, so duplicate groups stay intact (decision D8).
    """
    counts, global_counts = _counts(assignment, groups, total)
    score = _objective(counts, global_counts, class_totals, total, ratios)
    order = sorted(groups, key=lambda gid: (len(groups[gid]), gid))
    for _ in range(max_passes):
        if is_within_tolerance(counts, global_counts, class_totals, total, ratios, tolerance):
            break
        moved = False
        for group_id in order:
            members = groups[group_id]
            per_class = Counter(member.class_index for member in members)
            size = len(members)
            source = assignment[group_id]
            for destination in SPLIT_NAMES:
                if destination == source:
                    continue
                for class_index, count in per_class.items():
                    counts[source][class_index] -= count
                    counts[destination][class_index] += count
                global_counts[source] -= size
                global_counts[destination] += size
                candidate = _objective(counts, global_counts, class_totals, total, ratios)
                if candidate < score - 1e-12:
                    score = candidate
                    source = destination
                    assignment[group_id] = destination
                    moved = True
                else:
                    for class_index, count in per_class.items():
                        counts[destination][class_index] -= count
                        counts[source][class_index] += count
                    global_counts[destination] -= size
                    global_counts[source] += size
        if not moved:
            break
    return assignment, counts, global_counts


def split_groups(hashes: Sequence[FileHash], ratios: Mapping[str, float],
                 tolerance: float, seed: int) -> dict[str, str]:
    """Assign every image to train/val/test, deterministically and group-atomically.

    Args:
        hashes: dataset file hashes with ``group_id`` already assigned.
        ratios: target split ratios (must cover train/val/test).
        tolerance: max allowed absolute deviation of achieved from target ratios,
            evaluated both globally and per class.
        seed: reproducible seed used for deterministic tie-breaking.

    Returns:
        ``{rel_path: split_name}``.

    Raises:
        SplitError: if the input is empty, ratios are invalid, or the achieved
            proportions cannot be brought inside ``tolerance``.
    """
    if not hashes:
        raise SplitError("cannot split an empty collection of hashes")
    missing = [name for name in SPLIT_NAMES if name not in ratios]
    if missing:
        raise SplitError(f"missing split ratios: {missing}")
    if sum(ratios[name] for name in SPLIT_NAMES) <= 0:
        raise SplitError("split ratios must sum to a positive number")

    groups = group_members(hashes)
    class_totals = Counter(item.class_index for item in hashes)
    total = len(hashes)
    target = {s: {c: ratios[s] * n for c, n in class_totals.items()} for s in SPLIT_NAMES}
    global_target = {s: ratios[s] * total for s in SPLIT_NAMES}

    # Deterministic processing order: largest groups first (they are hardest to fit),
    # with ties broken by a seeded RNG and finally by the SHA-256 value.
    ordered = sorted(
        ((group_id, members[0].sha256, members) for group_id, members in groups.items()),
        key=lambda item: item[1],
    )
    rng = random.Random(seed)
    tie_break = {group_id: rng.random() for group_id, _, _ in ordered}
    ordered.sort(key=lambda item: (-len(item[2]), tie_break[item[0]], item[1]))

    counts = {split: Counter() for split in SPLIT_NAMES}
    global_counts: Counter = Counter()
    assignment: dict[int, str] = {}
    for group_id, _sha, members in ordered:
        per_class = Counter(member.class_index for member in members)
        best_split, best_score = SPLIT_NAMES[0], None
        for split in SPLIT_NAMES:  # fixed evaluation order keeps ties deterministic
            score = sum(
                abs(counts[split][c] + n - target[split][c]) for c, n in per_class.items()
            )
            score += abs(global_counts[split] + len(members) - global_target[split])
            if best_score is None or score < best_score:
                best_split, best_score = split, score
        assignment[group_id] = best_split
        for class_index, count in per_class.items():
            counts[best_split][class_index] += count
        global_counts[best_split] += len(members)

    counts, global_counts = _counts(assignment, groups, total)
    if not is_within_tolerance(counts, global_counts, class_totals, total, ratios, tolerance):
        assignment, counts, global_counts = _repair(
            assignment, groups, class_totals, total, ratios, tolerance
        )
    worst = max_deviation(counts, global_counts, class_totals, total, ratios)
    if not is_within_tolerance(counts, global_counts, class_totals, total, ratios, tolerance):
        raise SplitError(
            f"split proportions outside tolerance: max deviation {worst:.4f} > {tolerance}"
        )
    return {item.rel_path: assignment[item.group_id] for item in hashes}


def split_rows(hashes: Sequence[FileHash], assignment: Mapping[str, str]) -> list[dict[str, Any]]:
    """Return CSV rows for ``splits.csv`` in the input order of ``hashes``."""
    return [
        {
            "rel_path": item.rel_path,
            "class_name": item.class_name,
            "class_index": item.class_index,
            "sha256": item.sha256,
            "group_id": item.group_id,
            "split": assignment[item.rel_path],
        }
        for item in hashes
    ]


def crossing_group_ids(hashes: Sequence[FileHash],
                       assignment: Mapping[str, str]) -> list[int]:
    """Return group ids assigned to more than one split (must always be empty)."""
    splits_by_group: dict[int, set[str]] = {}
    for item in hashes:
        splits_by_group.setdefault(item.group_id, set()).add(assignment[item.rel_path])
    return sorted(gid for gid, names in splits_by_group.items() if len(names) > 1)


def build_summary(hashes: Sequence[FileHash], assignment: Mapping[str, str],
                  ratios: Mapping[str, float], tolerance: float, seed: int,
                  cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Build the persisted ``split_summary.json`` payload."""
    groups = group_members(hashes)
    class_totals = Counter(item.class_index for item in hashes)
    group_assignment = {gid: assignment[members[0].rel_path]
                        for gid, members in groups.items()}
    counts, global_counts = _counts(group_assignment, groups, len(hashes))
    index_to_name = {item.class_index: item.class_name for item in hashes}

    per_class: list[dict[str, Any]] = []
    for class_index in sorted(class_totals):
        entry: dict[str, Any] = {
            "class_index": class_index,
            "class_name": index_to_name[class_index],
            "total": class_totals[class_index],
        }
        for split in SPLIT_NAMES:
            entry[split] = counts[split][class_index]
            entry[f"{split}_ratio"] = round(
                counts[split][class_index] / class_totals[class_index], 6
            )
        per_class.append(entry)

    duplicate_groups = [gid for gid, members in groups.items() if len(members) >= 2]
    crossing = crossing_group_ids(hashes, assignment)
    total = len(hashes)
    achieved = {split: global_counts[split] / total for split in SPLIT_NAMES}
    worst = max_deviation(counts, global_counts, class_totals, total, ratios)
    return {
        "seed": seed,
        "config_fingerprint": config_fingerprint(cfg),
        "tolerance": tolerance,
        "target_ratios": {name: ratios[name] for name in SPLIT_NAMES},
        "achieved_ratios": {name: round(achieved[name], 6) for name in SPLIT_NAMES},
        "max_abs_deviation": round(worst, 6),
        "within_tolerance": bool(is_within_tolerance(counts, global_counts, class_totals, total, ratios, tolerance)),
        "total_images": total,
        "total_groups": len(groups),
        "duplicate_groups": len(duplicate_groups),
        "counts": {split: global_counts[split] for split in SPLIT_NAMES},
        "per_class": per_class,
        "groups_crossing_splits": len(crossing),
        "crossing_group_ids": crossing[:50],
    }


def write_splits_csv(rows: Sequence[Mapping[str, Any]], out_path: str | Path,
                     cfg: Mapping[str, Any]) -> Path:
    """Write the split manifest as CSV, refusing paths inside the dataset root."""
    target = assert_not_inside_dataset_root(out_path, cfg)
    ensure_dir(target.parent, cfg)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SPLIT_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return target


def write_summary_json(summary: Mapping[str, Any], out_path: str | Path,
                       cfg: Mapping[str, Any]) -> Path:
    """Write the split summary as pretty-printed JSON outside the dataset root."""
    target = assert_not_inside_dataset_root(out_path, cfg)
    ensure_dir(target.parent, cfg)
    payload = json.dumps(dict(summary), indent=2, sort_keys=True)
    target.write_text(payload + "\n", encoding="utf-8")
    return target


def format_summary(summary: Mapping[str, Any]) -> str:
    """Render a human-readable split summary."""
    counts = summary["counts"]
    target = summary["target_ratios"]
    achieved = summary["achieved_ratios"]
    lines = [
        f"total images          : {summary['total_images']}",
        f"total groups          : {summary['total_groups']}",
        f"duplicate groups      : {summary['duplicate_groups']}",
        f"seed / tolerance      : {summary['seed']} / {summary['tolerance']}",
        "split   count   target   achieved   deviation",
    ]
    for split in SPLIT_NAMES:
        lines.append(
            f"{split:<7} {counts[split]:>6}   {target[split]:>6.3f}   {achieved[split]:>7.4f}"
            f"   {abs(achieved[split] - target[split]):>9.4f}"
        )
    lines.append(
        f"max deviation {summary['max_abs_deviation']} within tolerance: "
        f"{summary['within_tolerance']}"
    )
    lines.append(f"groups crossing splits: {summary['groups_crossing_splits']}")
    lines.append("per class (label order):")
    for entry in summary["per_class"]:
        lines.append(
            f"  {entry['class_index']:>2}  {entry['class_name']:<47} "
            f"total={entry['total']:>6}  train={entry['train']:>6} "
            f"val={entry['val']:>5} test={entry['test']:>5}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: split the hashed dataset and persist split manifests."""
    parser = argparse.ArgumentParser(
        description="Deterministic stratified group split (read-only wrt data/raw)."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out-dir", default=None, help="directory holding the manifests")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    logger = get_logger("tomato.splitting")
    log_config_summary(cfg, logger)

    out_dir = Path(args.out_dir) if args.out_dir else manifests_dir(cfg)
    hashes = load_hashes(out_dir / HASHES_FILENAME)
    ratios = ratios_from_config(cfg)
    tolerance = tolerance_from_config(cfg)
    seed = int(cfg_get(cfg, "seed.value"))

    assignment = split_groups(hashes, ratios, tolerance, seed)
    rows = split_rows(hashes, assignment)
    summary = build_summary(hashes, assignment, ratios, tolerance, seed, cfg)

    splits_path = write_splits_csv(rows, out_dir / SPLITS_FILENAME, cfg)
    summary_path = write_summary_json(summary, out_dir / SUMMARY_FILENAME, cfg)

    print(format_summary(summary))
    logger.info("wrote %s (%d rows)", splits_path, len(rows))
    logger.info("wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
