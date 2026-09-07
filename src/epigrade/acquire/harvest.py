"""GEO metadata harvest.

Uses GEOparse in `how="brief"` mode ONLY. Brief mode fetches the SOFT header/metadata for a
series and its samples without pulling each sample's probe table - full mode would pull
gigabytes per series (the per-probe data table for every sample). We never need per-probe data
here; beta matrices for the one series we actually score (GSE74432) are pulled separately in
`epigrade.preprocess` from the series matrix file, not from GEOparse's sample tables.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import GEOparse
import pandas as pd

from epigrade import paths

logger = logging.getLogger(__name__)

ACCESSIONS = [
    "GSE97362", "GSE74432", "GSE116300", "GSE116992", "GSE66552", "GSE95040",
    "GSE104451", "GSE125367", "GSE55491", "GSE108423", "GSE89353", "GSE52588",
    "GSE42861", "GSE85210", "GSE87571", "GSE87648", "GSE99863", "GSE35069",
]


@dataclass
class HarvestFailure:
    accession: str
    reason: str


def _characteristics_to_dict(char_list: list[str]) -> dict[str, str]:
    """Parse GEO's 'key: value' characteristics_ch1 list into a dict.

    Not every entry has a colon-separated key (some series just dump a free-text phrase); those
    are kept under a numbered fallback key rather than dropped.
    """
    out: dict[str, str] = {}
    for i, item in enumerate(char_list or []):
        if ":" in item:
            k, v = item.split(":", 1)
            k = k.strip().lower().replace(" ", "_")
            out[k] = v.strip()
        else:
            out[f"characteristic_{i}"] = item.strip()
    return out


def _gsm_to_row(gsm, gse_accession: str) -> dict:
    md = gsm.metadata
    row = {
        "gsm_accession": md.get("geo_accession", [None])[0],
        "series_id": gse_accession,
        "title": md.get("title", [None])[0],
        "source_name_ch1": md.get("source_name_ch1", [None])[0],
        "organism_ch1": md.get("organism_ch1", [None])[0],
        "platform_id": md.get("platform_id", [None])[0],
        "molecule_ch1": md.get("molecule_ch1", [None])[0],
        "submission_date": md.get("submission_date", [None])[0],
        "contact_institute": md.get("contact_institute", [None])[0],
        "characteristics_raw": " | ".join(md.get("characteristics_ch1", [])),
    }
    row.update(_characteristics_to_dict(md.get("characteristics_ch1", [])))
    return row


def harvest_one(accession: str, cache_dir) -> tuple[pd.DataFrame | None, HarvestFailure | None]:
    try:
        gse = GEOparse.get_GEO(geo=accession, destdir=str(cache_dir), how="brief", silent=True)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the whole harvest
        return None, HarvestFailure(accession, f"GEOparse fetch failed: {exc}")

    if not gse.gsms:
        return None, HarvestFailure(accession, "GEOparse returned zero samples")

    rows = [_gsm_to_row(gsm, accession) for gsm in gse.gsms.values()]
    return pd.DataFrame(rows), None


def harvest_all(accessions: list[str] | None = None) -> tuple[pd.DataFrame, list[HarvestFailure]]:
    accessions = accessions or ACCESSIONS
    cache_dir = paths.geo_cache_dir()
    frames, failures = [], []

    for acc in accessions:
        t0 = time.time()
        df, failure = harvest_one(acc, cache_dir)
        elapsed = time.time() - t0
        if failure:
            logger.warning("FAILED %s: %s", acc, failure.reason)
            failures.append(failure)
            continue
        logger.info("OK %s: %d samples (%.1fs)", acc, len(df), elapsed)
        frames.append(df)

    if not frames:
        return pd.DataFrame(), failures

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined, failures


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    combined, failures = harvest_all()

    out_path = paths.interim_dir() / "samples.parquet"
    combined.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(combined), out_path)

    per_series = combined.groupby("series_id").size().sort_index()
    print("\nSamples per series:")
    print(per_series.to_string())

    if failures:
        print(f"\n{len(failures)} accession(s) FAILED to harvest:")
        for f in failures:
            print(f"  {f.accession}: {f.reason}")
    else:
        print("\nAll accessions harvested successfully.")


if __name__ == "__main__":
    main()
