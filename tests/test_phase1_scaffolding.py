"""Phase 1 scaffolding acceptance tests.

Encodes the Phase 1 acceptance criteria (checks A-F in ``README.md``):
directory layout, project rules, packaging files and the configuration schema.

These tests exercise **scaffolding only** — no dataset, model, training,
evaluation, Grad-CAM or Streamlit code exists yet, and none of it is imported
here (only the standard library plus PyYAML are required).

Run with::

    .\\venv\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import fnmatch
import os
import re
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRECTORIES = (
    "configs",
    "data/raw",
    "checkpoints",
    "notebooks",
    "src",
    "src/models",
    "app",
    "tests",
)

REQUIRED_CONFIG_SECTIONS = (
    "project",
    "seed",
    "data",
    "model",
    "attention",
    "training",
    "optimizer",
    "scheduler",
    "augmentation",
    "evaluation",
    "gradcam",
    "paths",
    "smoke_test",
)

REQUIRED_REQUIREMENTS = (
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "scikit-learn",
    "pyyaml",
    "pillow",
    "matplotlib",
    "seaborn",
    "streamlit",
    "tqdm",
    "jupyterlab",
    "ipykernel",
)

FORBIDDEN_REQUIREMENTS = ("tensorflow", "keras")

REQUIRED_CLINERULES_PHRASES = (
    "Python 3.10+",
    "Python 3.11",
    "PyTorch",
    "torchvision",
    "scikit-learn",
    "PyYAML",
    "Streamlit",
    "Do not create another virtual environment",
    "Do not hardcode hyperparameters in Python source files",
    "configs/config.yaml",
    "type hints",
    "docstrings",
    "reproducible random seeds",
    "Never download datasets automatically",
    "supplied manually by the user",
    "Never modify or delete the user's raw dataset",
    "Keep `data/` out of Git",
    "Keep `checkpoints/` out of Git",
    "Keep `.venv/` and `venv/` out of Git",
    "Never perform full model training automatically",
    "tiny subset and at most 1 epoch",
    "ask the user for explicit permission",
    "small, isolated phases",
    "Explain what each file does",
    "Never silently change the specified architecture",
    "Never silently change the specified dataset split requirements",
    "Stop and ask the user if a requirement is ambiguous or contradictory",
    "Do not start the next phase automatically",
    "always use `.\\venv\\python.exe` explicitly",
    "use `.\\venv\\python.exe -m pip`",
    "Do not use the base Conda Python",
    "Do not use a system Python installation",
    "Python-related verification, confirm that `.\\venv\\python.exe` reports",
    "clear, maintainable, testable code",
)

# Implementation files explicitly deferred to later phases (must not exist yet).
DEFERRED_IMPLEMENTATION_FILES = (
    "src/dataset.py",
    "src/models/cbam.py",
    "src/models/net.py",
    "src/train.py",
    "src/evaluate.py",
    "src/gradcam.py",
    "app/app.py",
)


# Files the project pipeline must never create inside the read-only dataset directory.
PIPELINE_ARTIFACT_PATTERNS = (
    "splits*.json",
    "splits*.csv",
    "manifest*.csv",
    "hashes*.json",
    "*.npy",
    "*.npz",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.pkl",
    "*.parquet",
)
SPLIT_DIR_NAMES = ("train", "val", "valid", "validation", "test")


def _top_level_class_dirs() -> set[str]:
    """Return the class folder names discovered under ``data/raw/`` (read-only).

    Supports both the flat layout (``data/raw/<class>/``) and the split-aware
    layout (``data/raw/<split>/<class>/``) described in the README.
    """
    data_raw = PROJECT_ROOT / "data" / "raw"
    top_level = {path.name for path in data_raw.iterdir() if path.is_dir()}
    if not top_level & set(SPLIT_DIR_NAMES):
        return top_level
    nested: set[str] = set()
    for split_name in SPLIT_DIR_NAMES:
        split_dir = data_raw / split_name
        if split_dir.is_dir():
            nested |= {path.name for path in split_dir.iterdir() if path.is_dir()}
    return nested


def _read_text(relative_path: str) -> str:
    """Return the UTF-8 contents of a project-relative file."""
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def _ignore_patterns() -> list[str]:
    """Return the non-comment, non-empty lines of ``.gitignore``."""
    return [
        line.strip()
        for line in _read_text(".gitignore").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _is_ignored(relative_path: str) -> bool:
    """Approximate Git ignore semantics for a single project-relative path."""
    normalized = relative_path.replace("\\", "/")
    parts = normalized.split("/")
    for pattern in _ignore_patterns():
        if pattern.endswith("/"):  # directory pattern
            if pattern.lstrip("/").rstrip("/") in parts:
                return True
        elif (
            fnmatch.fnmatch(normalized, pattern)
            or fnmatch.fnmatch(parts[-1], pattern)
            or any(fnmatch.fnmatch(part, pattern) for part in parts)
        ):
            return True
    return False


def _requirement_names() -> list[str]:
    """Return the normalized distribution names declared in requirements.txt."""
    names: list[str] = []
    for raw_line in _read_text("requirements.txt").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        names.append(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].strip().lower())
    return names


@pytest.fixture(scope="module")
def config() -> dict:
    """The parsed ``configs/config.yaml`` mapping."""
    return yaml.safe_load(_read_text("configs/config.yaml"))


def test_python_version_is_supported() -> None:
    """Check G: the interpreter satisfies the Python 3.10+ / 3.11 requirement."""
    assert sys.version_info >= (3, 10), f"unsupported Python {sys.version}"


def test_running_on_project_venv_interpreter() -> None:
    """Environment rule: the suite must run on `.\\venv\\python.exe` (Python 3.11.x).

    Guards the project rule "always use `.\\venv\\python.exe` explicitly" — running
    the tests from the base Conda Python or a system Python fails here on purpose.
    """
    expected = os.path.normcase(str((PROJECT_ROOT / "venv" / "python.exe").resolve()))
    actual = os.path.normcase(str(Path(sys.executable).resolve()))
    assert actual == expected, (
        f"wrong interpreter: {sys.executable}\n"
        r"run the suite with: .\venv\python.exe -m pytest tests -q"
    )
    assert sys.version_info[:2] == (3, 11), (
        f".\\venv\\python.exe must report Python 3.11.x, got {sys.version}"
    )


def test_required_directories_exist() -> None:
    """Check A: every scaffolded directory exists."""
    missing = [d for d in REQUIRED_DIRECTORIES if not (PROJECT_ROOT / d).is_dir()]
    assert not missing, f"missing directories: {missing}"


def test_clinerules_contains_required_rules() -> None:
    """Check B: ``.clinerules`` exists and states every required rule."""
    rules = _read_text(".clinerules")
    missing = [phrase for phrase in REQUIRED_CLINERULES_PHRASES if phrase not in rules]
    assert not missing, f"missing rules in .clinerules: {missing}"


def test_requirements_declare_core_dependencies() -> None:
    """Check C: core dependencies are declared and TensorFlow is absent."""
    names = _requirement_names()
    missing = [name for name in REQUIRED_REQUIREMENTS if name not in names]
    assert not missing, f"missing requirements: {missing}"
    forbidden = [name for name in FORBIDDEN_REQUIREMENTS if name in names]
    assert not forbidden, f"PyTorch-only project, remove: {forbidden}"


@pytest.mark.parametrize(
    "ignored_path",
    [
        "venv/python.exe",
        ".venv/pyvenv.cfg",
        "data/raw/Tomato___healthy/abc.JPG",
        "checkpoints/best_model.pt",
        "__pycache__/module.pyc",
        ".pytest_cache/CACHEDIR.TAG",
        "notebooks/.ipynb_checkpoints/demo.ipynb",
        "Thumbs.db",
        "desktop.ini",
        ".DS_Store",
        "results/metrics.json",
    ],
)
def test_gitignore_excludes_generated_paths(ignored_path: str) -> None:
    """Check D: environments, dataset, checkpoints and caches are ignored."""
    assert _is_ignored(ignored_path), f"not ignored: {ignored_path}"


@pytest.mark.parametrize(
    "tracked_path",
    [
        "configs/config.yaml",
        "requirements.txt",
        "README.md",
        "tests/test_phase1_scaffolding.py",
        "notebooks/.gitkeep",
        "src/models/__init__.py",
        "app/.gitkeep",
        ".clinerules",
    ],
)
def test_gitignore_keeps_sources_tracked(tracked_path: str) -> None:
    """Check D: source code, tests, notebooks and config must stay tracked."""
    assert not _is_ignored(tracked_path), f"wrongly ignored: {tracked_path}"


def test_config_parses_with_pyyaml_and_has_sections(config: dict) -> None:
    """Checks E + F: the YAML parses and holds all required sections."""
    assert isinstance(config, dict)
    missing = [section for section in REQUIRED_CONFIG_SECTIONS if section not in config]
    assert not missing, f"missing config sections: {missing}"


@pytest.mark.parametrize(
    ("dotted_key", "expected"),
    [
        ("data.image_size", 224),
        ("data.num_classes", 10),
        ("data.loader.batch_size", 16),
        ("training.max_epochs", 40),
        ("training.loss.label_smoothing", 0.1),
        ("optimizer.learning_rate", 1e-4),
        ("optimizer.weight_decay", 1e-4),
        ("model.backbone", "efficientnet_b0"),
        ("model.num_classes", 10),
        ("model.head.hidden_dim", 256),
        ("model.head.dropout", 0.4),
        ("attention.type", "cbam"),
        ("attention.insertion_stage", "last"),
        ("attention.cbam.channel_first", True),
        ("attention.cbam.channel.reduction_ratio", 16),
        ("attention.cbam.spatial.kernel_size", 7),
        ("seed.value", 42),
        ("seed.deterministic", True),
        ("training.early_stopping.enabled", True),
        ("training.freeze_backbone.enabled", True),
        ("smoke_test.enabled", True),
        ("smoke_test.max_epochs", 1),
    ],
)
def test_config_defaults(config: dict, dotted_key: str, expected: object) -> None:
    """Check F: important planned defaults are present and correctly typed."""
    node: object = config
    for key in dotted_key.split("."):
        assert isinstance(node, dict) and key in node, f"missing key: {dotted_key}"
        node = node[key]
    assert node == expected, f"{dotted_key}: expected {expected!r}, got {node!r}"


def test_config_split_ratios(config: dict) -> None:
    """Check F: the locked 0.70 / 0.15 / 0.15 stratified split is configured."""
    split = config["data"]["split"]
    assert (split["train"], split["val"], split["test"]) == (0.70, 0.15, 0.15)
    assert abs(sum((split["train"], split["val"], split["test"])) - 1.0) < 1e-9
    assert split["stratify"] is True
    assert config["data"]["duplicate_check"]["enabled"] is True


def test_backbone_finetuning_lr_is_lower(config: dict) -> None:
    """Check F: the backbone fine-tuning LR is lower than the head LR."""
    optimizer = config["optimizer"]
    assert optimizer["backbone_learning_rate"] < optimizer["learning_rate"]


def test_smoke_test_subset_is_small(config: dict) -> None:
    """Check F: the smoke test uses a tiny subset and at most one epoch."""
    smoke = config["smoke_test"]
    assert smoke["max_epochs"] <= 1
    assert 0 < smoke["subset_size"] <= 100
    assert smoke["num_workers"] == 0


def test_deferred_implementation_files_do_not_exist() -> None:
    """Checks I/J: Phase 1 created scaffolding only, no implementation."""
    present = [
        path for path in DEFERRED_IMPLEMENTATION_FILES if (PROJECT_ROOT / path).exists()
    ]
    assert not present, f"out-of-scope implementation files present: {present}"


def test_data_raw_is_read_only_no_pipeline_artifacts() -> None:
    """Check I: the dataset is user-supplied and ``data/raw/`` is never written to.

    The dataset is never downloaded by this project - it is placed in ``data/raw/``
    manually. This test enforces the read-only contract instead of requiring an
    empty directory: only class folders holding images may exist, and no split
    files, hashes, caches or tensors may ever be produced inside the dataset root.
    """
    data_raw = PROJECT_ROOT / "data" / "raw"
    assert data_raw.is_dir(), "data/raw/ is missing"

    violations: list[str] = []
    for path in data_raw.rglob("*"):
        if path.is_dir():
            continue
        if path.parent == data_raw:
            violations.append(f"{path.name} (stray file directly in data/raw/)")
        elif any(fnmatch.fnmatch(path.name.lower(), p) for p in PIPELINE_ARTIFACT_PATTERNS):
            violations.append(f"{path.name} (pipeline artifact)")
    assert not violations, (
        "data/raw/ must stay read-only; unexpected entries: " + str(violations[:10])
    )


def test_dataset_class_folders_match_config(config: dict) -> None:
    """Check I: the supplied class folders match ``data.class_names`` exactly.

    Skipped when the dataset has not been supplied yet, so the suite stays usable
    before the user adds PlantVillage to ``data/raw/``.
    """
    found = _top_level_class_dirs()
    if not found:
        pytest.skip("dataset not supplied yet - place PlantVillage folders in data/raw/")

    expected = set(config["data"]["class_names"])
    assert found == expected, (
        "class folder / config mismatch\n"
        f"  missing folders:    {sorted(expected - found)}\n"
        f"  unexpected folders: {sorted(found - expected)}"
    )
    assert len(found) == config["data"]["num_classes"] == 10


def test_no_trained_checkpoints_exist() -> None:
    """Check J: no training was performed, so no checkpoints exist."""
    leftovers = sorted(path.name for path in (PROJECT_ROOT / "checkpoints").iterdir())
    assert not leftovers, f"unexpected checkpoints: {leftovers}"

