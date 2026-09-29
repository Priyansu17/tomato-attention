"""Path helpers and the hard read-only guard for the dataset directory.

``data/raw`` is supplied by the user and must never be written to. Every output folder
used by the pipeline goes through :func:`ensure_dir` with the config, which refuses any
path inside the dataset root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


class DatasetWriteError(RuntimeError):
    """Raised when code attempts to write inside the read-only dataset root."""


def project_root() -> Path:
    """Return the repository root (the parent directory of the ``src`` package)."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """Resolve a (possibly relative) config path against the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root() / candidate


def dataset_root(cfg: Mapping[str, Any]) -> Path:
    """Return the configured dataset root."""
    return resolve_path(cfg["data"]["root"])


def assert_not_inside_dataset_root(path: str | Path, cfg: Mapping[str, Any]) -> Path:
    """Return the resolved ``path``, refusing anything inside the dataset root.

    Raises:
        DatasetWriteError: if the target is the dataset root itself or below it.
    """
    target = resolve_path(path).resolve()
    root = dataset_root(cfg).resolve()
    if target == root or root in target.parents:
        raise DatasetWriteError(
            f"refusing to write inside the read-only dataset root: {target}"
        )
    return target


def ensure_dir(path: str | Path, cfg: Mapping[str, Any] | None = None) -> Path:
    """Create a directory (parents included) and return it.

    When ``cfg`` is given the dataset guard is applied first, so output folders can
    never be created inside ``data/raw``.
    """
    target = Path(path)
    if cfg is not None:
        target = assert_not_inside_dataset_root(target, cfg)
    target.mkdir(parents=True, exist_ok=True)
    return target


def manifests_dir(cfg: Mapping[str, Any]) -> Path:
    """Return (and create) the directory holding dataset manifests."""
    return ensure_dir(cfg["paths"]["manifests_dir"], cfg)
