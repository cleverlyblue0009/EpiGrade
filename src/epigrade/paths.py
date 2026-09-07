"""Central path resolution.

Every module that needs a data/results path should go through here rather than
hardcoding a location - see config/paths.yaml for the actual values.
"""

from __future__ import annotations

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
    return Path(_load()["data_root"])


def geo_cache_dir() -> Path:
    p = Path(_load()["geo_cache_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def external_dir() -> Path:
    p = Path(_load()["external_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def interim_dir() -> Path:
    p = Path(_load()["interim_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


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
