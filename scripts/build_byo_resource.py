"""Build what the bring-your-own-signature engine reads (see epigrade.byo.store).

Checked in, so the app runs offline with zero downloads:
  results/resource/byo/corpus_index.npz       probe index, platform and series presence,
                                              per-cohort mean case-control beta difference
  results/resource/byo/demo_betas.npz         8-bit sample betas for signature + fixture probes
  results/resource/byo/cohort_membership.tsv
  results/resource/byo/demo_samples.tsv

Local only (under interim_dir(), never committed):
  byo_store/betas.npy                         float16 sample betas for EVERY corpus probe

Platform manifests are read from GEO platform records (GPL13534, GPL21145, GPL33022). A
manifest that is not on disk is recorded as unavailable rather than guessed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from make_test_signatures import EPIC_V1_MANIFEST, epic_v2_probe_ids, manifest_probe_ids

from epigrade import paths
from epigrade.byo import store
from epigrade.corpus import build, loader, registry

MANIFEST_450K = "GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz"


def platform_ids() -> dict[str, set[str] | None]:
    ext = paths.external_dir()
    out: dict[str, set[str] | None] = {}
    out["450K"] = set(manifest_probe_ids(MANIFEST_450K)) if (ext / MANIFEST_450K).exists() \
        else None
    out["EPIC v1"] = set(manifest_probe_ids(EPIC_V1_MANIFEST)) \
        if (ext / EPIC_V1_MANIFEST).exists() else None
    out["EPIC v2"] = epic_v2_probe_ids()
    return out


def main() -> None:
    resolved = [build.resolve(d) for d in registry.DISORDERS]
    groups = build.corpus_groups(resolved)
    sample_ids = groups.index.tolist()
    column = {s: j for j, s in enumerate(sample_ids)}

    print("Indexing probes...")
    series_probes = {s: loader.load_beta(s).index for s in registry.BETA_SERIES}
    corpus_probes = np.array(sorted(set().union(*series_probes.values())), dtype=str)
    row = pd.Series(np.arange(len(corpus_probes)), index=corpus_probes)

    store_dir = paths.interim_dir() / "byo_store"
    store_dir.mkdir(parents=True, exist_ok=True)
    betas = np.lib.format.open_memmap(
        store_dir / "betas.npy", mode="w+", dtype=np.float16,
        shape=(len(corpus_probes), len(sample_ids)),
    )
    betas[:] = np.nan
    for series_id in registry.BETA_SERIES:
        beta = loader.load_beta(series_id)
        ids = groups.index[groups.series_id == series_id].tolist()
        betas[row.loc[beta.index].to_numpy()[:, None],
              np.array([column[s] for s in ids])[None, :]] = beta[ids].to_numpy(np.float16)
        print(f"  {series_id}: {len(ids)} samples x {len(beta)} probes", flush=True)
    betas.flush()
    np.save(store_dir / "probe_ids.npy", corpus_probes)
    np.save(store_dir / "sample_ids.npy", np.array(sample_ids, dtype=str))

    print("Cohort effects...")
    series_presence = np.column_stack(
        [pd.Index(corpus_probes).isin(series_probes[s]) for s in registry.BETA_SERIES]
    )
    series_col = {s: j for j, s in enumerate(registry.BETA_SERIES)}
    delta = np.full((len(corpus_probes), len(resolved)), np.nan, dtype=np.float16)
    membership = []
    for k, r in enumerate(resolved):
        controls = r.control_series_of()
        involved = set(r.case_series.values()) | set(controls.values())
        # A probe belongs to a cohort only if every series the cohort draws on carries it.
        usable = series_presence[:, [series_col[s] for s in involved]].all(axis=1)
        case_cols = [column[s] for s in r.case_ids]
        control_cols = [column[s] for s in controls]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            d = (np.nanmean(betas[:, case_cols].astype(float), axis=1)
                 - np.nanmean(betas[:, control_cols].astype(float), axis=1))
        delta[usable, k] = d[usable].astype(np.float16)
        membership += [{"disorder": r.name, "gsm_accession": s, "role": "case",
                        "series_id": r.case_series[s]} for s in r.case_ids]
        membership += [{"disorder": r.name, "gsm_accession": s, "role": "control",
                        "series_id": controls[s]} for s in controls]
        print(f"  {r.name}: {int(usable.sum())} usable probes", flush=True)

    print("Platform manifests...")
    manifests = platform_ids()
    available = [p for p in store.PLATFORMS if manifests[p] is not None]
    extra = set().union(*[manifests[p] for p in available]) - set(corpus_probes)
    all_probes = np.array(sorted(set(corpus_probes) | extra), dtype=str)
    at = np.searchsorted(all_probes, corpus_probes)
    full_delta = np.full((len(all_probes), len(resolved)), np.nan, dtype=np.float16)
    full_delta[at] = delta
    full_series = np.zeros((len(all_probes), len(registry.BETA_SERIES)), dtype=bool)
    full_series[at] = series_presence
    platform_presence = np.column_stack([
        pd.Index(all_probes).isin(manifests[p]) if manifests[p] is not None
        else np.zeros(len(all_probes), dtype=bool)
        for p in store.PLATFORMS
    ])
    for p in store.PLATFORMS:
        print(f"  {p}: " + (f"{len(manifests[p])} probe IDs" if manifests[p] is not None
                            else "manifest not available"))

    out = store.resource_dir()
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "corpus_index.npz", probe_ids=all_probes, platform_presence=platform_presence,
        platforms=np.array(store.PLATFORMS), platforms_available=np.array(available),
        series=np.array(registry.BETA_SERIES), series_presence=full_series,
        cohorts=np.array([r.name for r in resolved]), delta=full_delta,
    )
    pd.DataFrame(membership).to_csv(out / "cohort_membership.tsv", sep="\t", index=False)
    groups.rename_axis("gsm_accession").to_csv(out / "demo_samples.tsv", sep="\t")

    print("Demo betas...")
    demo = set()
    for path in sorted((paths.repo_root() / "results" / "signatures").glob("*.tsv")):
        if path.stem != "index":
            demo.update(pd.read_csv(path, sep="\t", usecols=["probe_id"])["probe_id"])
    for path in sorted((paths.repo_root() / "tests" / "fixtures").glob("sig_*.csv")):
        demo.update(pd.read_csv(path, usecols=["probe_id"])["probe_id"])
    demo_probes = np.array(sorted(demo & set(corpus_probes)), dtype=str)
    block = betas[np.sort(row.loc[demo_probes].to_numpy())].astype(float)
    codes = np.where(np.isnan(block), store.NAN_CODE,
                     np.rint(np.clip(block, 0, 1) * (store.NAN_CODE - 1))).astype(np.uint8)
    np.savez_compressed(out / "demo_betas.npz", probe_ids=demo_probes,
                        sample_ids=np.array(sample_ids, dtype=str), betas=codes)
    for f in sorted(out.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")
    print(f"Full store: {store_dir} ({len(corpus_probes)} probes x {len(sample_ids)} samples)")


if __name__ == "__main__":
    main()
