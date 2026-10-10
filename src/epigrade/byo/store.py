"""Data the bring-your-own-signature engine reads. Two layers:

1. The checked-in demo resource (results/resource/byo/), which is all the app needs offline:
     corpus_index.npz    every probe ID known to the corpus or to a platform manifest, which
                         platforms and which corpus series carry it, and the mean case-control
                         beta difference in each of the 12 cohorts
     demo_betas.npz      sample-level betas (8-bit) for the corpus signature probes and the
                         test-fixture probes, for the 584 scored samples
     cohort_membership.tsv / demo_samples.tsv
2. An optional full local store (interim_dir()/byo_store/), built by
   scripts/build_byo_resource.py from the downloaded matrices: sample-level betas for EVERY
   probe. When present, an uploaded signature is evaluated on all of its probes; when absent,
   sample-level results use only the probes the demo resource carries, and say so.

Cohort-level effects (the identification statistic) are complete in the demo resource either
way - only AUC, the specificity row and attribution need sample-level betas.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from epigrade import paths

PLATFORMS = ("450K", "EPIC v1", "EPIC v2")
NAN_CODE = 255  # 8-bit betas: 0..254 = beta * 254, 255 = missing
FULL = "full_corpus"
DEMO = "demo_resource"


def resource_dir() -> Path:
    return paths.repo_root() / "results" / "resource" / "byo"


def full_store_dir() -> Path | None:
    """The optional local full store, or None if the data root is not available here."""
    try:
        path = paths.interim_dir() / "byo_store"
    except OSError:
        return None
    return path if (path / "betas.npy").exists() else None


@dataclass
class CorpusIndex:
    probe_ids: np.ndarray            # sorted
    platform_presence: np.ndarray    # bool, probes x len(platforms)
    platforms: list[str]
    platforms_available: list[str]   # manifests that could actually be sourced
    series: list[str]
    series_presence: np.ndarray      # bool, probes x len(series)
    cohorts: list[str]
    delta: np.ndarray                # float16, probes x len(cohorts); NaN = not in cohort

    def positions(self, probes) -> np.ndarray:
        """Row of each probe, or -1 if the corpus has never heard of it."""
        probes = np.asarray(probes, dtype=str)
        pos = np.searchsorted(self.probe_ids, probes)
        pos[pos >= len(self.probe_ids)] = 0
        return np.where(self.probe_ids[pos] == probes, pos, -1)

    def cohort_delta(self, cohort: str) -> np.ndarray:
        return self.delta[:, self.cohorts.index(cohort)].astype(float)


@lru_cache(maxsize=1)
def load_index() -> CorpusIndex:
    path = resource_dir() / "corpus_index.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run scripts/build_byo_resource.py (needs the downloaded "
            "matrices) or check out the repository's results/resource/byo/ directory"
        )
    z = np.load(path, allow_pickle=False)
    return CorpusIndex(
        probe_ids=z["probe_ids"], platform_presence=z["platform_presence"],
        platforms=list(z["platforms"]), platforms_available=list(z["platforms_available"]),
        series=list(z["series"]), series_presence=z["series_presence"],
        cohorts=list(z["cohorts"]), delta=z["delta"],
    )


@lru_cache(maxsize=1)
def cohort_membership() -> pd.DataFrame:
    """disorder, gsm_accession, role (case/control), series_id - who defines each cohort."""
    return pd.read_csv(resource_dir() / "cohort_membership.tsv", sep="\t")


@lru_cache(maxsize=1)
def sample_table() -> pd.DataFrame:
    """gsm_accession-indexed: series_id, group (disorder name / pooled_control / sibling)."""
    return pd.read_csv(resource_dir() / "demo_samples.tsv", sep="\t", index_col=0)


class BetaSource:
    """Sample-level betas for a probe list, from the full local store if it exists, else the
    demo resource."""

    def __init__(self, kind: str, probe_ids: np.ndarray, sample_ids: list[str], reader):
        self.kind = kind
        self.probe_ids = probe_ids
        self.sample_ids = sample_ids
        self._reader = reader
        self._order = np.argsort(probe_ids)
        self._sorted = probe_ids[self._order]

    def has(self, probes) -> np.ndarray:
        probes = np.asarray(probes, dtype=str)
        pos = np.searchsorted(self._sorted, probes)
        pos[pos >= len(self._sorted)] = 0
        return self._sorted[pos] == probes

    def get(self, probes) -> pd.DataFrame:
        """Probe x sample betas for the probes this source has (others are dropped)."""
        probes = np.asarray(probes, dtype=str)
        probes = probes[self.has(probes)]
        rows = self._order[np.searchsorted(self._sorted, probes)]
        return pd.DataFrame(self._reader(rows), index=probes, columns=self.sample_ids)


def _demo_source() -> BetaSource:
    z = np.load(resource_dir() / "demo_betas.npz", allow_pickle=False)
    codes = z["betas"]

    def read(rows: np.ndarray) -> np.ndarray:
        block = codes[rows].astype(float)
        block[block == NAN_CODE] = np.nan
        return block / (NAN_CODE - 1)

    return BetaSource(DEMO, z["probe_ids"], list(z["sample_ids"]), read)


def _full_source(store: Path) -> BetaSource:
    betas = np.load(store / "betas.npy", mmap_mode="r")
    probe_ids = np.load(store / "probe_ids.npy", allow_pickle=False)
    sample_ids = list(np.load(store / "sample_ids.npy", allow_pickle=False))

    def read(rows: np.ndarray) -> np.ndarray:
        order = np.argsort(rows)
        block = np.empty((len(rows), betas.shape[1]), dtype=float)
        block[order] = betas[rows[order]]  # sorted row access keeps the memmap read sequential
        return block

    return BetaSource(FULL, probe_ids, sample_ids, read)


@lru_cache(maxsize=2)
def load_beta_source(prefer_full: bool = True) -> BetaSource:
    store = full_store_dir() if prefer_full else None
    return _full_source(store) if store is not None else _demo_source()
