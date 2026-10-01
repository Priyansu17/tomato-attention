"""Tests for the torch Dataset and split-manifest loading (Phase 2C).

Small synthetic JPEGs inside pytest's ``tmp_path`` are used (never ``data/raw``) so
label order, manifest validation, subset determinism and read-only safety are all
checked against real files. Manifest columns come from ``SPLIT_FIELDS`` so this test
breaks loudly if the persisted schema ever drifts.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.config import cfg_get, load_config
from src.data.dataset import (
    DatasetError,
    LeafImageDataset,
    SplitRow,
    limit_rows,
    load_split_rows,
)
from src.data.splitting import SPLIT_FIELDS, SPLIT_NAMES, SPLITS_FILENAME
from src.data.transforms import build_eval_transform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_CLASSES = load_config(PROJECT_ROOT / "configs" / "config.yaml")["data"]["class_names"]


def write_fake_dataset(root: Path, classes: list[str], per_class: dict[str, int]) -> list[dict]:
    """Write synthetic JPEGs under ``root/<class>/<i>.jpg``; return manifest-ready rows.

    The first classes get more images than the rest so balanced smoke subsets must
    down-sample them; ``class_index`` follows the configured label order, which for
    these names is NOT alphabetical order.
    """
    records: list[dict] = []
    index = 0
    for class_name in classes:
        folder = root / class_name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_class[class_name]):
            path = folder / f"{i}.jpg"
            Image.effect_noise((64, 64), 60 + index % 7).convert("RGB").save(
                path, format="JPEG", quality=90
            )
            records.append({"rel_path": f"{class_name}/{i}.jpg", "class_name": class_name,
                            "class_index": classes.index(class_name),
                            "sha256": f"{index:064x}", "group_id": index})
            index += 1
    return records


def write_manifest(cfg: dict, records: list[dict], train_n: int, val_n: int,
                   header: tuple[str, ...] = SPLIT_FIELDS) -> Path:
    """Stratified split (all classes in train) writing splits.csv with exact counts.

    Per-class quotas keep every class in train and the global totals exact:
    train gets ``train_n // n_classes`` per class plus one extra for the first
    ``train_n % n_classes`` classes (same for val); test takes the remainder.
    Rows are grouped by class so the manifest never clusters one class per split.
    """
    manifest = Path(cfg["paths"]["manifests_dir"]) / SPLITS_FILENAME
    manifest.parent.mkdir(parents=True, exist_ok=True)
    classes = sorted({record["class_name"] for record in records})
    n_classes = len(classes)
    train_base, train_extra = divmod(train_n, n_classes)
    val_base, val_extra = divmod(val_n, n_classes)
    quotas = {
        name: (train_base + (1 if i < train_extra else 0),
               val_base + (1 if i < val_extra else 0))
        for i, name in enumerate(classes)
    }
    by_class: dict[str, list[dict]] = {name: [] for name in classes}
    for record in records:
        by_class[record["class_name"]].append(record)
    for name, group in by_class.items():
        need_train, need_val = quotas[name]
        assert len(group) >= need_train + need_val + 1, (
            f"class {name} has {len(group)} images, cannot fill quotas "
            f"train={need_train} val={need_val} test>=1"
        )
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for split_index, split in enumerate(SPLIT_NAMES):
            for name in classes:
                group = by_class[name]
                need_train, need_val = quotas[name]
                start = 0 if split_index == 0 else need_train if split_index == 1 else need_train + need_val
                end = need_train if split_index == 0 else need_train + need_val if split_index == 1 else len(group)
                for record in group[start:end]:
                    writer.writerow([record[field] for field in SPLIT_FIELDS[:-1]] + [split])
    return manifest


@pytest.fixture()
def fake_dataset(config_factory, real_config):
    """Config rooted in a temp dir with 106 synthetic images and a splits manifest."""
    per_class = {name: count for name, count in zip(REAL_CLASSES, (14, 12, 10))}
    per_class.update({name: 10 for name in REAL_CLASSES[3:]})
    cfg = config_factory(REAL_CLASSES, images_per_class=0)
    root = Path(cfg["data"]["root"])
    records = write_fake_dataset(root, list(cfg["data"]["class_names"]), per_class)
    manifest = write_manifest(cfg, records, train_n=70, val_n=12)
    rows = {split: load_split_rows(manifest, split) for split in SPLIT_NAMES}
    return {"cfg": cfg, "root": root, "manifest": manifest, "rows": rows,
            "records": records, "classes": list(cfg["data"]["class_names"])}


def test_load_split_rows_round_trips_fields_and_label_order(fake_dataset: dict) -> None:
    """Check: manifest rows round-trip with config label ids and readable paths."""
    rows = fake_dataset["rows"]
    assert [len(rows[s]) for s in SPLIT_NAMES] == [70, 12, 24]
    assert load_split_rows(fake_dataset["manifest"]) == [
        row for s in SPLIT_NAMES for row in rows[s]
    ]
    for split in SPLIT_NAMES:
        for row in rows[split]:
            assert row.split == split
            assert (fake_dataset["root"] / row.rel_path).is_file()
            assert row.class_index == fake_dataset["classes"].index(row.class_name)
            assert row.sha256 and row.group_id >= 0


def test_load_split_rows_rejects_unknown_split_missing_file_and_header(
    fake_dataset: dict, tmp_path: Path
) -> None:
    """Check: clear DatasetError for bad split names, absent files, wrong columns."""
    manifest = fake_dataset["manifest"]
    with pytest.raises(DatasetError, match="unknown split"):
        load_split_rows(manifest, "holdout")
    with pytest.raises(DatasetError, match="not found"):
        load_split_rows(manifest.parent / "absent.csv", "train")
    bad_header = tmp_path / "bad.csv"
    bad_header.write_text("rel_path,class_name,class_index,sha256,group_id,split,X\n",
                          encoding="utf-8")
    with pytest.raises(DatasetError, match="unexpected columns"):
        load_split_rows(bad_header, "train")


def test_dataset_returns_tensor_and_config_label(fake_dataset: dict) -> None:
    """Check: dataset yields normalized 3x224x224 tensors labelled by class_index."""
    cfg, rows = fake_dataset["cfg"], fake_dataset["rows"]
    dataset = LeafImageDataset(rows["val"], fake_dataset["root"], build_eval_transform(cfg))
    assert len(dataset) == 12
    image, label = dataset[0]
    assert tuple(image.shape) == (3, 224, 224)
    assert isinstance(image, torch.Tensor) and image.dtype == torch.float32
    assert label == rows["val"][0].class_index
    assert 0 <= label < int(cfg["data"]["num_classes"])
    assert {dataset[i][1] for i in range(len(dataset))} == {
        row.class_index for row in rows["val"]
    }
    without_transform = LeafImageDataset(rows["val"], fake_dataset["root"])
    pil_image, pil_label = without_transform[0]
    assert pil_image.size == (64, 64) and pil_label == label


def test_limit_rows_is_deterministic_class_balanced_and_capped(fake_dataset: dict) -> None:
    """Check: smoke subsets repeat by seed, stay balanced, honor caps and limits."""
    cfg, train_rows = fake_dataset["cfg"], fake_dataset["rows"]["train"]
    assert len(train_rows) == 70
    subset_a = limit_rows(train_rows, limit=33, seed=int(cfg["seed"]["value"]))
    subset_b = limit_rows(train_rows, limit=33, seed=int(cfg["seed"]["value"]))
    assert [row.rel_path for row in subset_a] == [row.rel_path for row in subset_b]
    assert len(set(subset_a)) == 33
    assert {row.split for row in subset_a} == {"train"}
    counts: dict[int, int] = {}
    for row in subset_a:
        counts[row.class_index] = counts.get(row.class_index, 0) + 1
    assert len(counts) == len(fake_dataset["classes"])
    assert max(counts.values()) - min(counts.values()) <= 1

    same_seed_other_limit = limit_rows(train_rows, limit=24,
                                       seed=int(cfg["seed"]["value"]))
    assert [row.rel_path for row in same_seed_other_limit] == [
        row.rel_path for row in subset_a[:24]
    ], "same seed + smaller limit must be a prefix of the larger subset"
    reseded = limit_rows(train_rows, limit=33, seed=int(cfg["seed"]["value"]) + 1)
    assert {row.rel_path for row in reseded} != {row.rel_path for row in subset_a}

    capped = limit_rows(train_rows, limit=500, seed=1, per_class_cap=2)
    assert len(capped) == 2 * len(fake_dataset["classes"])
    assert max(sum(1 for row in capped if row.class_index == cid)
               for cid in range(len(fake_dataset["classes"]))) == 2
    full = limit_rows(train_rows, limit=500, seed=1)
    assert len(full) == len(train_rows)
    assert {row.rel_path for row in full} == {row.rel_path for row in train_rows}


def test_limit_rows_rejects_bad_limits_and_dataset_rejects_bad_rows(
    fake_dataset: dict
) -> None:
    """Check: unusable limits and empty row lists fail with a clear DatasetError."""
    train_rows = fake_dataset["rows"]["train"]
    with pytest.raises(DatasetError, match="limit must be >= 1"):
        limit_rows(train_rows, limit=0, seed=1)
    with pytest.raises(DatasetError, match="at least one split row"):
        LeafImageDataset([], fake_dataset["root"])


def test_dataset_reports_missing_and_undecodable_images(fake_dataset: dict) -> None:
    """Check: decode problems raise (never silently skipped) and name the file."""
    cfg, root = fake_dataset["cfg"], fake_dataset["root"]
    missing = SplitRow("Nope/424242.jpg", "Tomato___healthy", 9, "a" * 64, 999, "val")
    with pytest.raises(DatasetError, match="missing"):
        LeafImageDataset([missing], root, build_eval_transform(cfg))[0]
    broken = root / "Tomato___healthy" / "broken.jpg"
    broken.write_bytes(b"not an image")
    broken_row = SplitRow("Tomato___healthy/broken.jpg", "Tomato___healthy", 9,
                          "b" * 64, 998, "val")
    with pytest.raises(DatasetError, match="cannot be decoded"):
        LeafImageDataset([broken_row], root, build_eval_transform(cfg))[0]


def test_dataset_never_writes_to_the_image_tree(fake_dataset: dict) -> None:
    """Check: decoding every train row leaves the image tree byte-for-byte identical."""
    cfg, root = fake_dataset["cfg"], fake_dataset["root"]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
              for p in sorted(root.rglob("*.jpg"))}
    dataset = LeafImageDataset(fake_dataset["rows"]["train"], root, build_eval_transform(cfg))
    for i in range(len(dataset)):
        dataset[i]
    after = {p: (p.read_bytes(), p.stat().st_mtime_ns)
             for p in sorted(root.rglob("*.jpg"))}
    assert before == after, "reading images must not change bytes or mtime"
