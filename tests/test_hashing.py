"""Tests for streaming SHA-256 hashing and exact-duplicate group detection."""

from __future__ import annotations

import hashlib
import shutil

import pytest
from conftest import dataset_inventory_hash
from src.data.discovery import build_index
from src.data.hashing import (
    DUP_GROUP_FIELDS,
    DUP_GROUPS_FILENAME,
    HASHES_FILENAME,
    HashingError,
    assign_group_ids,
    duplicate_group_rows,
    groups_by_id,
    hash_file,
    hash_records,
    hash_summary,
    load_hashes,
    write_duplicate_groups_csv,
    write_hashes_csv,
)
from src.utils.paths import DatasetWriteError, dataset_root, manifests_dir

ORDER_PROBE_CLASSES = ("Tomato___healthy", "Tomato___Early_blight", "Tomato___Bacterial_spot")


def test_hash_is_deterministic_and_matches_hashlib(tmp_path):
    """Hashing is stable for one file and agrees with hashlib.sha256."""
    target = tmp_path / "payload.bin"
    target.write_bytes(b"tomato-attention" * 1000)
    expected = hashlib.sha256(target.read_bytes()).hexdigest()
    assert hash_file(target) == expected
    assert hash_file(target) == hash_file(target)


def test_chunked_hashing_is_independent_of_chunk_size(tmp_path):
    """Streaming chunks must produce the digest of the whole file (requirement 3)."""
    payload = bytes(range(256)) * 7  # 1792 bytes, crossing many chunk boundaries
    target = tmp_path / "chunked.bin"
    target.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    for chunk_size in (1, 7, 128, 1024, 1024 * 1024):
        assert hash_file(target, chunk_size=chunk_size) == expected


def test_different_content_gives_different_hash(tmp_path):
    first, second = tmp_path / "a.bin", tmp_path / "b.bin"
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")
    assert hash_file(first) != hash_file(second)


def test_invalid_input_raises_hashing_error(tmp_path):
    with pytest.raises(HashingError, match="not a regular file"):
        hash_file(tmp_path / "missing.jpg")
    with pytest.raises(HashingError, match="not a regular file"):
        hash_file(tmp_path)  # a directory is not hashable
    valid = tmp_path / "ok.bin"
    valid.write_bytes(b"x")
    with pytest.raises(HashingError, match="chunk_size"):
        hash_file(valid, chunk_size=0)


def test_duplicate_groups_are_detected_across_classes(config_factory):
    """Byte-identical copies in different classes form one duplicate group (D8)."""
    cfg = config_factory(("Tomato___healthy", "Tomato___Bacterial_spot"), images_per_class=1)
    root = dataset_root(cfg)
    source = next((root / "Tomato___healthy").iterdir())
    shutil.copyfile(source, root / "Tomato___Bacterial_spot" / f"copy{source.suffix}")

    records = build_index(cfg)
    hashes = assign_group_ids(hash_records(records, cfg, root))
    summary = hash_summary(hashes)

    assert summary["total_images"] == 3
    assert summary["unique_sha256"] == 2
    assert summary["duplicate_groups"] == 1
    assert summary["redundant_copies"] == 1
    assert summary["max_group_size"] == 2
    assert summary["group_size_distribution"] == {1: 1, 2: 1}

    rows = duplicate_group_rows(hashes)
    assert len(rows) == 1
    assert rows[0]["member_count"] == 2
    assert set(rows[0]) == set(DUP_GROUP_FIELDS)
    assert len(rows[0]["members"].split(";")) == 2

    sizes = [len(members) for members in groups_by_id(hashes).values()]
    assert sorted(sizes) == [1, 2]


def test_class_index_comes_only_from_config_order(config_factory):
    """Decision D12: label ids follow ``config.data.class_names``, not file order."""
    cfg = config_factory(ORDER_PROBE_CLASSES, images_per_class=1)
    hashes = assign_group_ids(hash_records(build_index(cfg), cfg, dataset_root(cfg)))
    assert {item.class_name: item.class_index for item in hashes} == {
        "Tomato___healthy": 0,
        "Tomato___Early_blight": 1,
        "Tomato___Bacterial_spot": 2,
    }
    assert sorted(ORDER_PROBE_CLASSES) != list(ORDER_PROBE_CLASSES)


def test_hash_manifests_roundtrip_and_write_guards(config_factory, tmp_path):
    """CSV round-trip works and manifests can never be written into data/raw."""
    cfg = config_factory(("Tomato___healthy",), images_per_class=2)
    root = dataset_root(cfg)
    hashes = assign_group_ids(hash_records(build_index(cfg), cfg, root))

    with pytest.raises(DatasetWriteError):
        write_hashes_csv(hashes, root / HASHES_FILENAME, cfg)
    with pytest.raises(DatasetWriteError):
        write_duplicate_groups_csv(duplicate_group_rows(hashes), root / DUP_GROUPS_FILENAME, cfg)

    out = write_hashes_csv(hashes, tmp_path / "manifests" / HASHES_FILENAME, cfg)
    assert set(out.read_text(encoding="utf-8").splitlines()[0].split(",")) == {
        "rel_path", "class_name", "class_index", "size_bytes", "sha256", "group_id"
    }
    reloaded = load_hashes(out)
    assert reloaded == list(hashes)


def test_missing_hash_manifest_raises(real_config, tmp_path):
    with pytest.raises(HashingError, match="not found"):
        load_hashes(tmp_path / "absent.csv")


@pytest.mark.realdata
def test_real_dataset_hash_acceptance(real_config, real_hashes):
    """Acceptance: 18160 images, 18146 unique, 14 duplicate groups of size 2."""
    assert len(real_hashes) == 18160
    assert all(len(item.sha256) == 64 for item in real_hashes)
    assert all(set(item.sha256) <= set("0123456789abcdef") for item in real_hashes)

    labels = {name: index for index, name in enumerate(real_config["data"]["class_names"])}
    assert all(item.class_index == labels[item.class_name] for item in real_hashes)

    summary = hash_summary(real_hashes)
    assert summary["total_images"] == 18160
    assert summary["unique_sha256"] == 18146
    assert summary["duplicate_groups"] == 14
    assert summary["duplicate_files"] == 28
    assert summary["redundant_copies"] == 14
    assert summary["max_group_size"] == 2
    assert summary["group_size_distribution"] == {1: 18132, 2: 14}
    assert summary["unique_sha256"] == summary["total_images"] - summary["redundant_copies"]
    assert sum(summary["per_class_images"].values()) == 18160
    assert summary["per_class_images"]["Tomato___Late_blight"] == 1909
    assert summary["per_class_unique"]["Tomato___Late_blight"] == 1901
    assert summary["per_class_unique"]["Tomato___healthy"] == 1585


@pytest.mark.realdata
def test_real_hashes_manifests_are_persisted(real_config):
    """Requirement 10: hash/duplicate metadata exists under results/manifests."""
    manifests = manifests_dir(real_config)
    with (manifests / HASHES_FILENAME).open(encoding="utf-8", newline="") as handle:
        data_rows = sum(1 for _ in handle) - 1
    assert data_rows == 18160
    with (manifests / DUP_GROUPS_FILENAME).open(encoding="utf-8", newline="") as handle:
        group_rows = sum(1 for _ in handle) - 1
    assert group_rows == 14


@pytest.mark.realdata
def test_real_dataset_unchanged_by_hashing(real_config, real_dataset_root, real_hashes):
    """Requirement 4: hashing real files never modifies data/raw."""
    chunk_size = int(real_config["data"]["hashing"]["chunk_size"])
    before = dataset_inventory_hash(real_dataset_root)
    sample = real_hashes[:200]
    digests = [hash_file(real_dataset_root / item.rel_path, chunk_size) for item in sample]
    after = dataset_inventory_hash(real_dataset_root)

    assert len(digests) == 200
    assert digests == [item.sha256 for item in sample]
    assert before == after
