"""Tests for the config-driven transforms and global seeding (Phase 2C).

Every augmentation parameter must come from configs/config.yaml (no hardcoded
hyperparameters) and every run must be reproducible from ``seed.value``.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image

from src.config import cfg_get, load_config
from src.data.dataloaders import worker_init
from src.data.transforms import (
    TransformConfigError,
    build_eval_transform,
    build_train_transform,
    normalization,
)
from src.utils.seed import set_seed


@pytest.fixture()
def tcfg(config_factory) -> dict:
    """Validated config with the production class list, rooted in a temp dir."""
    names = cfg_get(load_config("configs/config.yaml"), "data.class_names")
    return config_factory(names, images_per_class=1)


def _noise_image(size: int = 96, seed: int = 0) -> Image.Image:
    data = np.random.default_rng(seed).integers(0, 256, size * size * 3, dtype=np.uint8)
    return Image.frombytes("RGB", (size, size), bytes(data))


def _index_of(pipeline: list, kind: type) -> int:
    kinds = [type(step) for step in pipeline]
    assert kind in kinds, f"missing {kind.__name__} in {[k.__name__ for k in kinds]}"
    return kinds.index(kind)


def test_train_transform_follows_augmentation_config(tcfg: dict) -> None:
    """Check: train pipeline steps and their parameters come from augmentation.train."""
    aug = cfg_get(tcfg, "augmentation.train")
    pipeline = build_train_transform(tcfg).transforms
    kinds = [type(step) for step in pipeline]

    # Default config: vertical flip + random erasing disabled, everything else on.
    assert kinds == [T.RandomResizedCrop, T.RandomRotation, T.RandomHorizontalFlip,
                     T.ColorJitter, T.ToTensor, T.Normalize]

    crop = pipeline[_index_of(pipeline, T.RandomResizedCrop)]
    assert max(crop.size) == min(crop.size) == cfg_get(tcfg, "data.image_size") == 224
    assert tuple(crop.scale) == pytest.approx(tuple(aug["random_resized_crop"]["scale"]))
    assert tuple(crop.ratio) == pytest.approx(tuple(aug["random_resized_crop"]["ratio"]))

    rot = pipeline[_index_of(pipeline, T.RandomRotation)]
    deg = float(aug["rotation"]["degrees"])
    assert tuple(float(v) for v in rot.degrees) == pytest.approx((-deg, deg))

    hflip = pipeline[_index_of(pipeline, T.RandomHorizontalFlip)]
    assert hflip.p == pytest.approx(aug["horizontal_flip"]["p"])

    jitter = pipeline[_index_of(pipeline, T.ColorJitter)]
    for attr in ("brightness", "contrast", "saturation"):
        value = float(aug["color_jitter"][attr])
        assert tuple(getattr(jitter, attr)) == pytest.approx((1.0 - value, 1.0 + value))
    hue = float(aug["color_jitter"]["hue"])
    assert tuple(jitter.hue) == pytest.approx((-hue, hue))

    mean, std = normalization(tcfg)
    normalize = pipeline[-1]
    assert normalize.mean.tolist() == pytest.approx(mean.tolist())
    assert normalize.std.tolist() == pytest.approx(std.tolist())


def test_disabled_augmentation_steps_are_skipped(tcfg: dict) -> None:
    """Check: toggling enabled flags in the config changes the pipeline, not code."""
    tcfg["augmentation"]["train"]["horizontal_flip"]["enabled"] = False
    tcfg["augmentation"]["train"]["vertical_flip"]["enabled"] = True
    tcfg["augmentation"]["train"]["random_resized_crop"]["enabled"] = False
    tcfg["augmentation"]["train"]["random_erasing"]["enabled"] = True
    pipeline = build_train_transform(tcfg).transforms
    kinds = [type(step) for step in pipeline]
    assert T.RandomHorizontalFlip not in kinds
    assert T.RandomVerticalFlip in kinds
    assert T.RandomResizedCrop not in kinds
    assert T.RandomErasing in kinds
    # Crop disabled => deterministic Resize+CenterCrop to image_size before ToTensor.
    assert kinds[:3] == [T.Resize, T.CenterCrop, T.RandomRotation]
    torch.manual_seed(3)
    tensor = build_train_transform(tcfg)(_noise_image(64, seed=1))
    assert tuple(tensor.shape) == (3, 224, 224)


def test_normalization_comes_from_model_section(tcfg: dict) -> None:
    """Check: model.normalization drives Normalize; bad vectors are rejected."""
    mean, std = normalization(tcfg)
    assert mean.tolist() == pytest.approx([0.485, 0.456, 0.406])
    assert std.tolist() == pytest.approx([0.229, 0.224, 0.225])

    broken = copy.deepcopy(tcfg)
    broken["model"]["normalization"]["mean"] = [0.5, 0.5]
    with pytest.raises(TransformConfigError, match="3-element"):
        normalization(broken)

    broken = copy.deepcopy(tcfg)
    broken["model"]["normalization"]["std"] = [1.0, 0.0, 1.0]
    with pytest.raises(TransformConfigError, match="zeros"):
        normalization(broken)

    # A custom normalization must actually change the produced tensors.
    custom = copy.deepcopy(tcfg)
    custom["model"]["normalization"] = {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
    image = Image.new("RGB", (224, 224), (128, 128, 128))
    raw = build_eval_transform(custom)(image)
    assert raw.mean().item() == pytest.approx(128 / 255, abs=1e-4)
    normed = build_eval_transform(tcfg)(image)
    value = 128 / 255
    for channel, (m, s) in enumerate(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
        assert normed[channel].mean().item() == pytest.approx((value - m) / s, abs=1e-4)


def test_eval_transform_is_deterministic_resize_crop_normalize(tcfg: dict) -> None:
    """Check: eval pipeline = Resize -> CenterCrop -> ToTensor -> Normalize, repeatable."""
    pipeline = build_eval_transform(tcfg).transforms
    assert [type(step) for step in pipeline] == [T.Resize, T.CenterCrop, T.ToTensor,
                                                 T.Normalize]
    assert pipeline[0].size == cfg_get(tcfg, "augmentation.eval.resize") == 256
    assert max(pipeline[1].size) == min(pipeline[1].size) == cfg_get(tcfg, "augmentation.eval.center_crop") == 224

    image = _noise_image(96, seed=0)
    first, second = build_eval_transform(tcfg)(image), build_eval_transform(tcfg)(image)
    assert torch.equal(first, second), "eval transform must be deterministic"
    assert tuple(first.shape) == (3, 224, 224)

    bad = copy.deepcopy(tcfg)
    bad["augmentation"]["eval"]["center_crop"] = 512
    with pytest.raises(TransformConfigError, match="cannot exceed resize"):
        build_eval_transform(bad)


def test_transform_is_seed_reproducible(tcfg: dict) -> None:
    """Check: same torch seed => identical augmented tensor (reproducibility)."""
    image = _noise_image(64, seed=7)
    transform = build_train_transform(tcfg)
    torch.manual_seed(123)
    first = transform(image)
    torch.manual_seed(123)
    assert torch.equal(first, transform(image))
    torch.manual_seed(124)
    assert not torch.equal(first, transform(image))


def test_set_seed_is_config_driven_and_reproducible(tcfg: dict) -> None:
    """Check: seed.value/deterministic/cudnn_benchmark are applied; draws repeat."""
    tcfg["seed"]["value"] = 777
    tcfg["seed"]["cudnn_benchmark"] = False
    assert set_seed(tcfg) == 777
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    first = torch.rand(4).tolist()

    set_seed(tcfg)
    assert torch.rand(4).tolist() == first, "same config seed must reproduce torch draws"

    tcfg["seed"]["value"] = 123
    set_seed(tcfg)
    assert torch.rand(4).tolist() != first, "a different config seed must change the stream"


def test_worker_init_helper_seeds_numpy_from_torch_seed() -> None:
    """Check: worker_init keeps python/numpy streams tied to torch's per-worker seed."""
    torch.manual_seed(42)
    base = torch.initial_seed()
    np.random.seed(base % (2**32))
    expected = float(np.random.rand())
    worker_init(0)  # outside workers: falls back to torch.initial_seed()
    assert float(np.random.rand()) == pytest.approx(expected)
