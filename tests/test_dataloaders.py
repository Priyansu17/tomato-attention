"""Tests for the config-driven DataLoader factory (Phase 2C).

Batch sizes, worker counts, shuffling, drop-last, augmentation choice and the smoke
overrides must all come from configs/config.yaml, and repeated runs with the same
``seed.value`` must produce the same sample order. Synthetic images live in
``tmp_path``; the real dataset is only ever read through the persisted manifest.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from torch.utils.data import RandomSampler, SequentialSampler

from src.config import cfg_get, load_config
from src.data.dataloaders import SPLITS, DataLoaderBundle, build_dataloaders
from src.data.dataset import load_split_rows
from src.data.splitting import SPLITS_FILENAME
from tests.test_dataset import write_fake_dataset, write_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
REAL_CLASSES = load_config(CONFIG_PATH)["data"]["class_names"]


@pytest.fixture()
def loader_fixture(config_factory, real_config):
    """Synthetic 106-image dataset (70/12/24) wired to a temp config + manifest."""
    per_class = {name: count for name, count in zip(REAL_CLASSES, (14, 12, 10))}
    per_class.update({name: 10 for name in REAL_CLASSES[3:]})
    cfg = config_factory(REAL_CLASSES, images_per_class=0)
    root = Path(cfg["data"]["root"])
    records = write_fake_dataset(root, list(cfg["data"]["class_names"]), per_class)
    manifest = write_manifest(cfg, records, train_n=70, val_n=12)
    rows = {split: load_split_rows(manifest, split) for split in SPLITS}
    return {"cfg": cfg, "root": root, "manifest": manifest, "rows": rows}


def _train_order(bundle: DataLoaderBundle) -> list[str]:
    """Full shuffled epoch order of the train loader (sampler view, no decoding)."""
    sampler = bundle.loaders["train"].sampler
    return [bundle.datasets["train"].rows[i].rel_path for i in sampler]


def test_bundle_structure_sizes_and_batch_settings(loader_fixture: dict) -> None:
    """Check: loaders follow data.loader.* + training.batch_size per split profile."""
    cfg = loader_fixture["cfg"]
    bundle = build_dataloaders(cfg, rows=loader_fixture["rows"])
    assert isinstance(bundle, DataLoaderBundle)
    assert set(bundle.loaders) == set(bundle.datasets) == set(bundle.rows) == set(SPLITS)
    assert [len(bundle.datasets[s]) for s in SPLITS] == [70, 12, 24]

    assert bundle.loaders["train"].batch_size == cfg_get(cfg, "training.batch_size") == 16
    assert bundle.loaders["train"].drop_last is bool(cfg_get(cfg, "data.loader.drop_last"))
    assert bundle.loaders["val"].batch_size == cfg_get(cfg, "data.loader.eval_batch_size")
    assert bundle.loaders["test"].batch_size == cfg_get(cfg, "data.loader.eval_batch_size")
    assert bundle.loaders["val"].drop_last is False, "eval loaders must never drop_last"
    assert isinstance(bundle.loaders["train"].sampler, RandomSampler)
    assert isinstance(bundle.loaders["val"].sampler, SequentialSampler)
    assert isinstance(bundle.loaders["test"].sampler, SequentialSampler)

    train_steps = [type(t).__name__ for t in bundle.datasets["train"].transform.transforms]
    eval_steps = [type(t).__name__ for t in bundle.datasets["val"].transform.transforms]
    assert "RandomResizedCrop" in train_steps
    assert "RandomResizedCrop" not in eval_steps and "Resize" in eval_steps


def test_smoke_overrides_apply_from_config(loader_fixture: dict) -> None:
    """Check: smoke mode applies subset_size/batch_size/num_workers/shuffle overrides."""
    cfg = loader_fixture["cfg"]
    cfg["smoke_test"].update({"subset_size": 12, "batch_size": 4,
                              "num_workers": 0, "shuffle": False})
    cfg["data"]["loader"]["drop_last"] = False  # 9/13 eval rows are smaller than a batch
    bundle = build_dataloaders(cfg, smoke=True, rows=loader_fixture["rows"])
    assert [len(bundle.rows[s]) for s in SPLITS] == [12, 12, 12]
    # Train loader always follows training.batch_size (16); smoke_test.batch_size
    # only supplies the fallback default inside build_dataloaders.
    assert bundle.loaders["train"].batch_size == int(cfg_get(cfg, "training.batch_size"))
    assert bundle.loaders["train"].num_workers == 0
    # Class-balanced round-robin subset: 12 rows over 10 classes => at most 2 per class.
    counts: dict[str, int] = {}
    for row in bundle.rows["train"]:
        counts[row.class_name] = counts.get(row.class_name, 0) + 1
    assert len(counts) == len(REAL_CLASSES) and max(counts.values()) == 2


def test_workers_setting_is_honored(loader_fixture: dict) -> None:
    """Check: num_workers is config-driven, so blocked-worker machines can set 0."""
    cfg = loader_fixture["cfg"]
    configured = cfg_get(cfg, "data.loader.num_workers")
    production = build_dataloaders(cfg, rows=loader_fixture["rows"])
    assert production.loaders["train"].num_workers == configured
    assert production.loaders["val"].num_workers == configured
    cfg["data"]["loader"]["num_workers"] = 0
    cfg["smoke_test"]["num_workers"] = 0
    fallback = build_dataloaders(cfg, rows=loader_fixture["rows"])
    assert fallback.loaders["train"].num_workers == 0


def test_batches_iterate_with_configured_workers(loader_fixture: dict) -> None:
    """Check: batches decode end-to-end with the configured worker count (4/4)."""
    cfg = loader_fixture["cfg"]
    bundle = build_dataloaders(cfg, rows=loader_fixture["rows"])
    assert bundle.loaders["train"].num_workers == cfg_get(cfg, "data.loader.num_workers") == 4
    total, labels_seen = 0, []
    for images, labels in bundle.loaders["train"]:
        assert tuple(images.shape)[1:] == (3, 224, 224)
        assert images.dtype.is_floating_point
        total += int(images.shape[0])
        labels_seen.extend(int(label) for label in labels)
    batch_size = int(cfg_get(cfg, "training.batch_size"))
    assert total == 70 - (70 % batch_size) == len(labels_seen)
    assert set(labels_seen) <= {row.class_index for row in bundle.rows["train"]}


def test_same_seed_reproduces_train_order_and_other_seed_changes_it(
    loader_fixture: dict,
) -> None:
    """Check: the train epoch order is seeded by seed.value and fully reproducible."""
    cfg = loader_fixture["cfg"]
    assert cfg_get(cfg, "data.loader.shuffle_train") is True
    first = _train_order(build_dataloaders(cfg, rows=loader_fixture["rows"]))
    second = _train_order(build_dataloaders(cfg, rows=loader_fixture["rows"]))
    assert first == second, "same seed must reproduce the shuffled epoch order"
    sequential = [row.rel_path for row in
                  build_dataloaders(cfg, rows=loader_fixture["rows"]).datasets["train"].rows]
    assert sorted(first) == sorted(sequential)
    assert first != sequential, "shuffling must actually reorder the train rows"

    reseded = {**cfg, "seed": {**cfg["seed"], "value": int(cfg["seed"]["value"]) + 1}}
    assert _train_order(build_dataloaders(reseded, rows=loader_fixture["rows"])) != first


@pytest.mark.realdata
def test_dataloaders_over_the_real_read_only_manifest(real_config) -> None:
    """Check: smoke loaders build and decode over the real splits.csv, config-driven."""
    from src.utils.paths import manifests_dir

    manifest = manifests_dir(real_config) / SPLITS_FILENAME
    if not manifest.is_file():
        pytest.skip("real splits manifest not built yet")
    cfg = copy.deepcopy(real_config)
    cfg["data"]["loader"]["num_workers"] = 0
    cfg["smoke_test"]["num_workers"] = 0
    bundle = build_dataloaders(cfg, smoke=True)
    subset_size = int(cfg_get(cfg, "smoke_test.subset_size"))
    assert 0 < len(bundle.rows["train"]) <= subset_size
    # train batch always follows training.batch_size (16); smoke_test.batch_size (8) is
    # only its fallback default in build_dataloaders, so it does not change here.
    assert bundle.loaders["train"].batch_size == int(cfg_get(cfg, "training.batch_size"))
    assert bundle.loaders["val"].batch_size == int(cfg_get(cfg, "data.loader.eval_batch_size"))
    images, labels = next(iter(bundle.loaders["val"]))
    assert tuple(images.shape)[1:] == (3, 224, 224)
    assert int(labels.min()) >= 0
    assert int(labels.max()) < int(cfg_get(cfg, "data.num_classes"))


@pytest.mark.realdata
def test_cli_smoke_entry_point_returns_zero() -> None:
    """Check: the loaders CLI runs the real-data smoke path with workers=0 and exits 0."""
    from src.data.dataloaders import main as loaders_main
    from src.data.splitting import SPLITS_FILENAME
    from src.utils.paths import manifests_dir

    cfg = load_config(CONFIG_PATH)
    if not (manifests_dir(cfg) / SPLITS_FILENAME).is_file():
        pytest.skip("real splits manifest not built yet")
    assert loaders_main(["--config", str(CONFIG_PATH), "--smoke", "--workers", "0"]) == 0
