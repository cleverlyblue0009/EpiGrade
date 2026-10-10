"""Generate the five bring-your-own-signature test fixtures from the REAL corpus, so every probe
ID is a genuine Illumina identifier and none is invented.

  sig_sotos_true.csv     500 probes sampled from the published Choufani 7,085-probe list, with
                         their real directions
  sig_random.csv         500 probes sampled uniformly from the probes that passed filtering
  sig_sotos_flipped.csv  the same 500 Sotos probes with every direction inverted
  sig_tiny.csv           20 of those Sotos probes
  sig_low_coverage.csv   500 real probe IDs that no corpus cohort can use

Full copies go to external_dir()/test_signatures/; the copies committed to tests/fixtures/ carry
probe IDs and the submitted direction only - no beta values.
"""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd

from epigrade import paths
from epigrade.corpus import loader, registry
from epigrade.signature import choufani

SEED = 20261010
FIXTURE_DIR = paths.repo_root() / "tests" / "fixtures"
EPIC_V2_NAMES = "GPL33022_probe_names.txt"
EPIC_V1_MANIFEST = "GPL21145_MethylationEPIC_15073387_v-1-0.csv.gz"


def manifest_probe_ids(gz_name: str) -> list[str]:
    """Probe IDs from an Illumina manifest CSV as deposited on GEO: the data block starts at
    the IlmnID header row and ends at the [Controls] section."""
    ids, in_table = [], False
    with gzip.open(paths.external_dir() / gz_name, "rt", encoding="utf-8",
                   errors="replace") as f:
        for line in f:
            if line.startswith("IlmnID"):
                in_table = True
            elif line.startswith("[Controls]"):
                break
            elif in_table:
                ids.append(line.split(",", 1)[0])
    return ids


def epic_v2_probe_ids() -> set[str] | None:
    """Base probe IDs from the GEO GPL33022 platform table (its Name column, saved one per
    line), or None if that file has not been fetched. Control rows are dropped."""
    path = paths.external_dir() / EPIC_V2_NAMES
    if not path.exists():
        return None
    names = pd.read_csv(path, header=None)[0].dropna().astype(str)
    return set(names[names.str.match(r"^(cg|ch|rs)")])


def unusable_probe_ids(n: int, rng: np.random.Generator) -> tuple[list[str], str]:
    """Real probe IDs that are in no corpus beta matrix: EPIC v2-only probes when that manifest
    is available, otherwise EPIC v1 probes absent from every series."""
    corpus = set()
    for series_id in registry.BETA_SERIES:
        corpus.update(loader.load_beta(series_id).index)
    v2 = epic_v2_probe_ids()
    if v2 is not None:
        names = np.array(sorted(v2))
        source = "EPIC v2-only probes (GEO GPL33022), absent from every corpus matrix"
    else:
        names = np.array(manifest_probe_ids(EPIC_V1_MANIFEST))
        source = "EPIC v1 probes (GEO GPL21145) absent from every corpus matrix"
    pool = np.array(sorted(p for p in names if p.startswith("cg") and p not in corpus))
    if len(pool) < n:
        raise RuntimeError(f"only {len(pool)} probe IDs available outside the corpus")
    return list(rng.choice(pool, n, replace=False)), source


def main() -> None:
    rng = np.random.default_rng(SEED)
    published = choufani.load_signature()
    filtered = loader.load_beta("GSE74432").index

    sotos = published.loc[sorted(rng.choice(published.index, 500, replace=False))]
    direction = np.where(sotos["delta_beta"] < 0, "hypo", "hyper")
    true = pd.DataFrame({"probe_id": sotos.index, "direction": direction, "weight": 1.0,
                         "published_delta_beta": sotos["delta_beta"].to_numpy()})
    flipped = true.assign(direction=np.where(direction == "hypo", "hyper", "hypo"))
    tiny = true.iloc[np.sort(rng.choice(len(true), 20, replace=False))]
    random = pd.DataFrame({"probe_id": sorted(rng.choice(filtered, 500, replace=False))})
    low_ids, low_source = unusable_probe_ids(500, rng)
    low = pd.DataFrame({"probe_id": sorted(low_ids)})

    fixtures = {
        "sig_sotos_true.csv": true, "sig_random.csv": random,
        "sig_sotos_flipped.csv": flipped, "sig_tiny.csv": tiny, "sig_low_coverage.csv": low,
    }
    full_dir = paths.external_dir() / "test_signatures"
    full_dir.mkdir(parents=True, exist_ok=True)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in fixtures.items():
        df.to_csv(full_dir / name, index=False)
        committed = df[[c for c in ("probe_id", "direction") if c in df.columns]]
        committed.to_csv(FIXTURE_DIR / name, index=False, lineterminator="\n")
        print(f"  {name}: {len(df)} probes")
    print(f"sig_low_coverage source: {low_source}")
    print(f"Wrote full copies -> {full_dir}; probe-ID copies -> {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
