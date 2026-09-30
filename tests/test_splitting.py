"""Tests for the deterministic stratified group split (70/15/15)."""

from __future__ import annotations

import json

import pytest
from conftest import dataset_inventory_hash
from src.data.hashing import FileHash, HashingError, assign_group_ids, load_hashes
from src.data.splitting import (
    SPLIT_FIELDS,
    SPLIT_NAMES,
    SUMMARY_FILENAME,
    SplitError,
    build_summary,
    crossing_group_ids,
    group_members,
    ratios_from_config,
    split_groups,
    split_rows,
    tolerance_from_config,
    write_splits_csv,
    write_summary_json,
)
from src.utils.paths import DatasetWriteError, dataset_root, manifests_dir

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
TOLERANCE = 0.01


def synthetic_hashes(class_counts: dict[int, int], duplicate_sources: tuple[int, ...] = ()) -> list[FileHash]:
    """Unique ``FileHash`` items plus byte-identical copies of chosen rows."""
    hashes: list[FileHash] = []
    counter = 0
    for class_index in sorted(class_counts):
        for number in range(class_counts[class_index]):
            counter += 1
            hashes.append(
                FileHash(
                    rel_path=f"class{class_index}/img{number:03d}.png",
                    class_name=f"class{class_index}",
                    class_index=class_index,
                    size_bytes=100,
                    sha256=f"{counter:064x}",
                )
            )
    for source in duplicate_sources:
        original = hashes[source]
        hashes.append(
            FileHash(
                rel_path=original.rel_path + ".copy",
                class_name=original.class_name,
                class_index=original.class_index,
                size_bytes=original.size_bytes,
                sha256=original.sha256,
            )
        )
    return assign_group_ids(hashes)


def split_fractions(assignment: dict[str, str], hashes: list[FileHash]):
    """Return ``{split: fraction}`` and ``{(class_index, split): fraction}``."""
    from collections import Counter

    total = len(hashes)
    global_counts = Counter(assignment[item.rel_path] for item in hashes)
    class_totals = Counter(item.class_index for item in hashes)
    class_counts: dict[tuple[int, str], int] = Counter(
        (item.class_index, assignment[item.rel_path]) for item in hashes
    )
    overall = {split: global_counts.get(split, 0) / total for split in SPLIT_NAMES}
    per_class = {
        key: count / class_totals[key[0]] for key, count in class_counts.items()
    }
    return overall, per_class


def test_split_is_deterministic_for_a_fixed_seed():
    """Same inputs and seed must produce byte-identical assignments."""
    hashes = synthetic_hashes({0: 70, 1: 20, 2: 10}, duplicate_sources=(0, 71))
    first = split_groups(hashes, RATIOS, TOLERANCE, seed=42)
    second = split_groups(hashes, RATIOS, TOLERANCE, seed=42)
    assert first == second
    assert set(first) == {item.rel_path for item in hashes}
    assert set(first.values()) == {"train", "val", "test"}


def test_duplicate_groups_never_cross_splits():
    """All members of a duplicate group land in one split (decisions D4/D8)."""
    hashes = synthetic_hashes({0: 60, 1: 40}, duplicate_sources=(0, 61, 90))
    assignment = split_groups(hashes, RATIOS, TOLERANCE, seed=42)
    groups = group_members(hashes)
    assert len(groups) < len(hashes)
    duplicate_ids = [gid for gid, members in groups.items() if len(members) > 1]
    assert len(duplicate_ids) == 3
    for group_id in duplicate_ids:
        seen = {assignment[member.rel_path] for member in groups[group_id]}
        assert len(seen) == 1
    assert crossing_group_ids(hashes, assignment) == []


def test_split_proportions_within_tolerance():
    """Overall and per-class proportions must stay inside the approved tolerance."""
    hashes = synthetic_hashes({0: 700, 1: 150, 2: 150}, duplicate_sources=(0, 700, 851))
    assignment = split_groups(hashes, RATIOS, TOLERANCE, seed=42)
    overall, per_class = split_fractions(assignment, hashes)
    for split, target in RATIOS.items():
        assert abs(overall[split] - target) <= TOLERANCE
    for key, fraction in per_class.items():
        split = key[1]
        assert abs(fraction - RATIOS[split]) <= TOLERANCE


def test_ratios_and_tolerance_come_from_the_config(real_config):
    assert ratios_from_config(real_config) == {"train": 0.70, "val": 0.15, "test": 0.15}
    assert tolerance_from_config(real_config) == 0.01


def test_empty_input_raises():
    with pytest.raises(SplitError, match="empty"):
        split_groups([], RATIOS, TOLERANCE, seed=42)


def test_missing_ratios_raise():
    hashes = synthetic_hashes({0: 10, 1: 10})
    with pytest.raises(SplitError, match="missing split ratios"):
        split_groups(hashes, {"train": 1.0}, TOLERANCE, seed=42)


def test_write_guards_reject_dataset_paths(config_factory):
    """Split manifests can never be written into data/raw."""
    cfg = config_factory(("Tomato___healthy",), images_per_class=1)
    root = dataset_root(cfg)
    rows = [{"rel_path": "Tomato___healthy/a.png", "class_name": "Tomato___healthy",
             "class_index": 0, "sha256": "0" * 64, "group_id": 0, "split": "train"}]
    with pytest.raises(DatasetWriteError):
        write_splits_csv(rows, root / "splits.csv", cfg)
    with pytest.raises(DatasetWriteError):
        write_summary_json({"seed": 42}, root / SUMMARY_FILENAME, cfg)
    with pytest.raises(Exception):
        load_hashes(root / "absent.csv")


@pytest.mark.realdata
def test_real_split_acceptance(real_config, real_hashes):
    """Acceptance: 18160 rows, train 12713 / val 2724 / test 2723, no crossings."""
    ratios = ratios_from_config(real_config)
    tolerance = tolerance_from_config(real_config)
    assignment = split_groups(real_hashes, ratios, tolerance, seed=42)

    assert len(assignment) == 18160
    overall, per_class = split_fractions(assignment, real_hashes)
    assert list(overall) == ["train", "val", "test"]
    assert abs(overall["train"] - 0.70) <= tolerance
    assert abs(overall["val"] - 0.15) <= tolerance
    assert abs(overall["test"] - 0.15) <= tolerance
    for (class_index, split), fraction in per_class.items():
        assert abs(fraction - ratios[split]) <= tolerance
    assert crossing_group_ids(real_hashes, assignment) == []

    rows = split_rows(real_hashes, assignment)
    assert len(rows) == 18160
    assert list(rows[0]) == list(SPLIT_FIELDS)
    assert {row["split"] for row in rows} == {"train", "val", "test"}
    labels = {name: index for index, name in enumerate(real_config["data"]["class_names"])}
    assert all(row["class_index"] == labels[row["class_name"]] for row in rows)


@pytest.mark.realdata
def test_real_split_manifests_are_persisted(real_config):
    """Requirement 10: splits.csv and split_summary.json exist under results/manifests."""
    manifests = manifests_dir(real_config)
    with (manifests / "splits.csv").open(encoding="utf-8", newline="") as handle:
        data_rows = sum(1 for _ in handle) - 1
    assert data_rows == 18160
    with (manifests / SUMMARY_FILENAME).open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["total_images"] == 18160
    assert summary["within_tolerance"] is True
    assert summary["groups_crossing_splits"] == 0
    assert summary["max_abs_deviation"] <= summary["tolerance"]
    assert summary["seed"] == 42


@pytest.mark.realdata
def test_real_dataset_unchanged_by_splitting(real_config, real_dataset_root, real_hashes):
    """Acceptance: splitting reads manifests only and never touches data/raw."""
    ratios = ratios_from_config(real_config)
    tolerance = tolerance_from_config(real_config)
    before = dataset_inventory_hash(real_dataset_root)
    assignment = split_groups(real_hashes, ratios, tolerance, seed=42)
    rows = split_rows(real_hashes, assignment)
    after = dataset_inventory_hash(real_dataset_root)
    assert len(rows) == 18160
    assert before == after
