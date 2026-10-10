"""Beta-matrix and metadata loading for the whole corpus, in one place.

Every series is parsed once from its GEO processed file(s) and cached as a float32 parquet under
interim_dir(); nothing here touches raw IDATs. Paths come from config/paths.yaml.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from epigrade import paths
from epigrade.acquire.harmonize import harmonize
from epigrade.preprocess.series_matrix import (
    parse_beta_values_by_slide_position,
    parse_series_matrix,
)

GEO_MATRIX_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}nnn/{series}/matrix/{name}"

# series -> processed-beta file(s) under external_dir(). A series split across platforms by GEO
# (GSE116992: 7 samples on 450K, 22 on EPIC) lists one file per platform; they are joined on
# the probes present in both.
SERIES_FILES: dict[str, list[str]] = {
    "GSE74432": ["GSE74432_series_matrix.txt.gz"],
    "GSE104451": ["GSE104451_series_matrix.txt.gz"],
    "GSE55491": ["GSE55491_series_matrix.txt.gz"],
    "GSE97362": ["GSE97362_series_matrix.txt.gz"],
    # The GSE116300 series matrix carries no beta table; its depositors shipped betas as a
    # processed supplementary file keyed by slide position instead.
    "GSE116300": ["GSE116300_beta_values.txt.gz"],
    "GSE66552": ["GSE66552_series_matrix.txt.gz"],
    "GSE95040": ["GSE95040_series_matrix.txt.gz"],
    "GSE108423": ["GSE108423_series_matrix.txt.gz"],
    "GSE116992": [
        "GSE116992-GPL13534_series_matrix.txt.gz",
        "GSE116992-GPL21145_series_matrix.txt.gz",
    ],
    "GSE125367": ["GSE125367_series_matrix.txt.gz"],
    "GSE52588": ["GSE52588_series_matrix.txt.gz"],
}


def matrix_url(series_id: str, file_name: str) -> str:
    return GEO_MATRIX_URL.format(stub=series_id[:-3], series=series_id, name=file_name)


@lru_cache(maxsize=1)
def harmonized_samples() -> pd.DataFrame:
    """The harmonised sample table for every harvested accession, indexed by GSM."""
    samples = pd.read_parquet(paths.interim_dir() / "samples.parquet")
    return harmonize(samples).set_index("gsm_accession")


def series_meta(series_id: str) -> pd.DataFrame:
    meta = harmonized_samples()
    return meta[meta.series_id == series_id]


def is_downloaded(series_id: str) -> bool:
    cache = paths.interim_dir() / f"{series_id}_betas.parquet"
    return cache.exists() or all(
        (paths.external_dir() / f).exists() for f in SERIES_FILES[series_id]
    )


def _parse(series_id: str) -> pd.DataFrame:
    parts = []
    for name in SERIES_FILES[series_id]:
        gz = paths.external_dir() / name
        if not gz.exists():
            raise FileNotFoundError(
                f"{gz} not found - download it from {matrix_url(series_id, name)}"
            )
        if "series_matrix" in name:
            part = parse_series_matrix(gz)
        else:
            part = parse_beta_values_by_slide_position(gz, series_meta(series_id))
        parts.append(part)
    beta = parts[0] if len(parts) == 1 else pd.concat(parts, axis=1, join="inner")
    if beta.empty:
        raise ValueError(f"{series_id}: processed file(s) contain no beta table")
    return beta


@lru_cache(maxsize=3)
def load_beta(series_id: str) -> pd.DataFrame:
    """Probe x sample betas for one series, parsed on first use and cached as parquet. Only a
    few series are held in memory at once (the largest is ~0.9 GB as float64)."""
    cache = paths.interim_dir() / f"{series_id}_betas.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    beta = _parse(series_id)
    beta.to_parquet(cache)
    return beta


def load_probes(series_id: str, probes: pd.Index) -> pd.DataFrame:
    """Betas for a probe subset of one series (probes the series lacks are dropped)."""
    beta = load_beta(series_id)
    return beta.loc[beta.index.intersection(probes)]
