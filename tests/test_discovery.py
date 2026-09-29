"""Tests for dataset discovery, validation, indexing and the manifest writer."""

from __future__ import annotations

import csv

import pytest

from conftest import write_image
from src.data.discovery import (
    CSV_FIELDS,
    DatasetImageError,
    DatasetLayoutError,
    build_index,
    discover_class_dirs,
    index_summary,
    iter_image_files,
    normalize_extensions,
    write_index_csv,
)
from src.utils.paths import dataset_root

ORDER_PROBE_CLASSES = ("Tomato___healthy", "Tomato___Early_blight", "Tomato___Bacterial_spot")


def test_normalize_extensions():
    assert normalize_extensions(["JPG", ".jpeg", "png", ".JPG"]) == (".jpeg", ".jpg", ".png")


def test_flat_layout_and_label_ids_follow_config_order(config_factory):
    """QA2 end-to-end: labels come from the config order, not the filesystem order."""
    cfg = config_factory(ORDER_PROBE_CLASSES, images_per_class=2)
    records = build_index(cfg)
    indices: dict[str, set[int]] = {}
    for record in records:
        indices.setdefault(record.class_name, set()).add(record.class_index)
    assert indices == {
        "Tomato___healthy": {0},
        "Tomato___Early_blight": {1},
        "Tomato___Bacterial_spot": {2},
    }
    assert len(records) == 6
    assert all(record.rel_path.startswith("Tomato___") for record in records)
    assert all(record.rel_path.count("/") == 1 for record in records)


def test_split_aware_layout_is_supported(tmp_path):
    root = tmp_path / "raw"
    write_image(root / "train" / ORDER_PROBE_CLASSES[0] / "a.png")
    write_image(root / "val" / ORDER_PROBE_CLASSES[1] / "b.png")
    write_image(root / "test" / ORDER_PROBE_CLASSES[2] / "c.png")
    dirs = discover_class_dirs(root, ORDER_PROBE_CLASSES)
    assert list(dirs) == list(ORDER_PROBE_CLASSES)
    assert {path.parent.name for path in dirs.values()} == {"train", "val", "test"}


def test_missing_root_raises(tmp_path):
    with pytest.raises(DatasetLayoutError, match="not found"):
        discover_class_dirs(tmp_path / "absent", ("Tomato___healthy",))


def test_missing_class_raises(tmp_path):
    root = tmp_path / "raw"
    write_image(root / "Tomato___healthy" / "a.png")
    with pytest.raises(DatasetLayoutError, match="missing"):
        discover_class_dirs(root, ("Tomato___healthy", "Tomato___Bacterial_spot"))


def test_unexpected_class_raises(tmp_path):
    root = tmp_path / "raw"
    write_image(root / "Tomato___healthy" / "a.png")
    write_image(root / "Tomato___Unknown" / "a.png")
    with pytest.raises(DatasetLayoutError, match="unexpected"):
        discover_class_dirs(root, ("Tomato___healthy",))


def test_extensions_are_case_insensitive_and_other_files_ignored(tmp_path):
    folder = tmp_path / "raw" / "Tomato___healthy"
    write_image(folder / "upper.PNG")
    write_image(folder / "lower.png")
    write_image(folder / "photo.jpg")
    (folder / "notes.txt").write_text("not an image", encoding="utf-8")
    names = [path.name for path in iter_image_files(folder, [".jpg", ".png"])]
    assert names == ["lower.png", "photo.jpg", "upper.PNG"]


def test_corrupt_image_raises(config_factory):
    cfg = config_factory(("Tomato___healthy",), images_per_class=1)
    broken = dataset_root(cfg) / "Tomato___healthy" / "broken.png"
    broken.write_bytes(b"definitely not a png")
    with pytest.raises(DatasetImageError):
        build_index(cfg)


def test_empty_class_folder_yields_no_records(config_factory):
    cfg = config_factory(("Tomato___healthy", "Tomato___Bacterial_spot"), images_per_class=1)
    for path in (dataset_root(cfg) / "Tomato___Bacterial_spot").iterdir():
        path.unlink()
    summary = index_summary(build_index(cfg))
    assert summary["num_images"] == 1
    assert summary["num_classes"] == 1


def test_index_summary_and_csv_output(config_factory, tmp_path):
    cfg = config_factory(ORDER_PROBE_CLASSES, images_per_class=2)
    records = build_index(cfg)
    summary = index_summary(records)
    assert summary["num_images"] == 6
    assert summary["num_classes"] == 3
    assert [entry["class_index"] for entry in summary["per_class"]] == [0, 1, 2]
    assert sum(entry["count"] for entry in summary["per_class"]) == 6
    assert set(summary["modes"]) == {"RGB"}

    target = write_index_csv(records, tmp_path / "manifests" / "dataset_index.csv", cfg)
    with target.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert list(rows[0]) == list(CSV_FIELDS)
    assert {row["class_name"] for row in rows} == set(ORDER_PROBE_CLASSES)
    assert {int(row["class_index"]) for row in rows} == {0, 1, 2}


@pytest.mark.realdata
def test_real_dataset_acceptance(real_config, real_dataset_root):
    """Acceptance for 2A: 18160 images, 10 classes, label ids in config order."""
    records = build_index(real_config)
    summary = index_summary(records)

    assert summary["num_images"] == 18160
    assert summary["num_classes"] == 10
    assert [entry["class_index"] for entry in summary["per_class"]] == list(range(10))
    assert [entry["class_name"] for entry in summary["per_class"]] == list(
        real_config["data"]["class_names"]
    )
    assert sum(entry["count"] for entry in summary["per_class"]) == 18160
    assert len({record.rel_path for record in records}) == 18160
    assert all(record.width > 0 and record.height > 0 for record in records)
    dominant = max(summary["dimensions"].values())
    assert dominant / summary["num_images"] >= 0.95
