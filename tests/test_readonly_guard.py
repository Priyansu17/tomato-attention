"""Tests proving the pipeline cannot write inside the read-only dataset root."""

from __future__ import annotations

import pytest

from conftest import dataset_inventory_hash
from src.data.discovery import build_index, write_index_csv
from src.utils.paths import (
    DatasetWriteError,
    assert_not_inside_dataset_root,
    dataset_root,
    ensure_dir,
    manifests_dir,
)


def test_guard_rejects_dataset_root_and_children(config_factory):
    """Writing at or below the dataset root must always be refused."""
    cfg = config_factory(("Tomato___healthy",), images_per_class=1)
    root = dataset_root(cfg)
    targets = (
        root,
        root / "dataset_index.csv",
        root / "Tomato___healthy" / "copy.png",
        root / "nested" / "deep" / "x.json",
    )
    for target in targets:
        with pytest.raises(DatasetWriteError):
            assert_not_inside_dataset_root(target, cfg)


def test_guard_allows_paths_outside_the_dataset(config_factory, tmp_path):
    """Output paths outside the dataset root are allowed and created."""
    cfg = config_factory(("Tomato___healthy",), images_per_class=1)
    resolved = assert_not_inside_dataset_root(tmp_path / "manifests" / "a.csv", cfg)
    assert resolved.is_absolute()
    assert ensure_dir(tmp_path / "results" / "logs", cfg).is_dir()


def test_manifests_dir_is_created_outside_the_dataset(config_factory):
    """The manifests directory never lives inside data/raw."""
    cfg = config_factory(("Tomato___healthy",), images_per_class=1)
    target = manifests_dir(cfg).resolve()
    assert target.is_dir()
    assert dataset_root(cfg).resolve() not in target.parents


def test_write_index_csv_refuses_dataset_path(config_factory):
    """Even the manifest writer refuses to touch the dataset root."""
    cfg = config_factory(("Tomato___healthy",), images_per_class=1)
    records = build_index(cfg)
    with pytest.raises(DatasetWriteError):
        write_index_csv(records, dataset_root(cfg) / "dataset_index.csv", cfg)


@pytest.mark.realdata
def test_real_dataset_unchanged_by_discovery(real_config, real_dataset_root):
    """Acceptance: discovery must not modify anything under data/raw."""
    before = dataset_inventory_hash(real_dataset_root)
    records = build_index(real_config)
    after = dataset_inventory_hash(real_dataset_root)
    assert records, "expected the real dataset to produce index records"
    assert before == after, "discovery modified the read-only dataset"
