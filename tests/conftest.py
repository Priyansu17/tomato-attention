"""Shared pytest fixtures and marker registration.

Every synthetic dataset used by this suite is created inside pytest's ``tmp_path``, so the
read-only ``data/raw`` directory is never touched by tests.

Label ids always come from the order of ``config.data.class_names``. Some fixtures use class
names whose alphabetical (sorted) order DIFFERS from the configured order, which lets the
tests prove that filesystem ordering is never used for labels.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402  (needs sys.path set up first)
from src.utils.paths import dataset_root  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    """Register the project's custom pytest markers."""
    config.addinivalue_line(
        "markers", "realdata: needs the user-supplied PlantVillage dataset in data/raw"
    )
    config.addinivalue_line("markers", "slow: comparatively slow test")


def write_image(path: Path, size: tuple[int, int] = (8, 8), mode: str = "RGB") -> Path:
    """Write a tiny valid image file and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size).save(path)
    return path


def make_dataset(root: Path, spec: dict[str, int], size: tuple[int, int] = (8, 8),
                 extension: str = ".png") -> Path:
    """Create ``root/<class>/`` folders with ``spec[class]`` synthetic images each."""
    for class_name, count in spec.items():
        for index in range(count):
            write_image(root / class_name / f"{class_name}_{index}{extension}", size=size)
    return root


def dataset_inventory_hash(root: Path) -> str:
    """Hash the dataset inventory (relative path, size, mtime) to detect any change."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            digest.update(
                f"{path.relative_to(root).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8")
            )
    return digest.hexdigest()



@pytest.fixture
def project_root() -> Path:
    """Absolute path of the project root."""
    return PROJECT_ROOT


@pytest.fixture
def config_factory(tmp_path: Path):
    """Return a factory that builds a validated config on a temporary dataset."""

    def _factory(class_names, images_per_class: int = 2, extension: str = ".png", **overrides):
        raw = tmp_path / "raw"
        make_dataset(raw, {name: images_per_class for name in class_names}, extension=extension)
        settings = {
            "data.root": str(raw),
            "data.class_names": list(class_names),
            "data.num_classes": len(class_names),
            "data.extensions": [extension],
            "paths.manifests_dir": str(tmp_path / "manifests"),
        }
        settings.update(overrides)
        return load_config(PROJECT_ROOT / "configs" / "config.yaml", overrides=settings)

    return _factory


@pytest.fixture(scope="session")
def real_config():
    """The validated repository configuration (points at the real ``data/raw``)."""
    return load_config(PROJECT_ROOT / "configs" / "config.yaml")


@pytest.fixture(scope="session")
def real_dataset_root(real_config) -> Path:
    """The real dataset root; the test is skipped when no dataset is supplied."""
    root = dataset_root(real_config)
    if not root.is_dir() or not any(path.is_dir() for path in root.iterdir()):
        pytest.skip("user-supplied dataset not present in data/raw")
    return root
