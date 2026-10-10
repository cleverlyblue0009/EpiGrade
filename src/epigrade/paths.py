"""Central path resolution.

Every module that needs a data/results path should go through here rather than
hardcoding a location - see config/paths.yaml for the actual values.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "config" / "paths.yaml"


@lru_cache(maxsize=1)
def _load() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_root() -> Path:
    return _REPO_ROOT


def data_root() -> Path:
    """Where downloads and caches live. config/paths.yaml by default; the EPIGRADE_DATA_ROOT
    environment variable overrides it, so another machine does not have to edit the config
    (the checked-in default is a D: drive path)."""
    override = os.environ.get("EPIGRADE_DATA_ROOT")
    if override:
        return Path(override)
    configured = Path(_load()["data_root"])
    if not _drive_exists(configured):
        # The checked-in config points at a drive this machine does not have. Fall back to
        # the repository's own gitignored data/ directory rather than failing on import.
        return _REPO_ROOT / "data"
    return configured


def _drive_exists(path: Path) -> bool:
    return not path.anchor or Path(path.anchor).exists()


def _data_dir(key: str) -> Path:
    configured = Path(_load()[key])
    if os.environ.get("EPIGRADE_DATA_ROOT") or not _drive_exists(configured):
        configured = data_root() / configured.name
    configured.mkdir(parents=True, exist_ok=True)
    return configured


def geo_cache_dir() -> Path:
    return _data_dir("geo_cache_dir")


def external_dir() -> Path:
    return _data_dir("external_dir")


def interim_dir() -> Path:
    return _data_dir("interim_dir")


def tables_dir() -> Path:
    p = _REPO_ROOT / _load()["tables_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def figures_dir() -> Path:
    p = _REPO_ROOT / _load()["figures_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def resource_dir() -> Path:
    p = _REPO_ROOT / _load()["resource_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p
