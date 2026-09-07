"""One-command reproduction of the Choufani et al. 2015 Sotos syndrome episignature classifier.

Usage: python scripts/demo_sotos.py

Checks the three published results from GSE74432:
  1. Discovery cohort separation (19 Sotos vs 53 controls)
  2. All 8 Weaver syndrome (EZH2) samples score negative
  3. The 16 NSD1 missense VOUS split ~9 positive / ~7 negative

Produces results/figures/sotos_heatmap.png and results/figures/sotos_scores.png, and
results/tables/sotos_scores.tsv / sotos_reproduction_summary.tsv. Never hand-edits a number to
match the paper - if a result doesn't reproduce, the summary table says so plainly.
"""

from __future__ import annotations

import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from epigrade import paths  # noqa: E402
from epigrade.acquire.harmonize import harmonize  # noqa: E402
from epigrade.preprocess.probe_filter import filter_probes  # noqa: E402
from epigrade.preprocess.series_matrix import parse_series_matrix  # noqa: E402
from epigrade.signature.choufani import classify_cohort, load_signature, score_cohort  # noqa: E402

SERIES_MATRIX_GZ = "GSE74432_series_matrix.txt.gz"


def get_beta_matrix() -> pd.DataFrame:
    """Parse (or load from cache) the GSE74432 beta matrix."""
    cache_path = paths.interim_dir() / "GSE74432_betas.parquet"
    if cache_path.exists():
        print(f"Loading cached beta matrix from {cache_path}")
        return pd.read_parquet(cache_path)

    gz_path = paths.external_dir() / SERIES_MATRIX_GZ
    print(f"Parsing {gz_path} (this can take a few minutes for a 450K series matrix)...")
    t0 = time.time()
    beta = parse_series_matrix(gz_path)
    print(f"Parsed {beta.shape[0]} probes x {beta.shape[1]} samples in {time.time()-t0:.0f}s")
    beta.to_parquet(cache_path)
    return beta


def main() -> None:
    # --- metadata: harmonized roles/cohorts for GSE74432 ---
    samples = pd.read_parquet(paths.interim_dir() / "samples.parquet")
    harmonized = harmonize(samples)
    meta = harmonized[harmonized.series_id == "GSE74432"].set_index("gsm_accession")
    cohort = classify_cohort(meta)
    print("\nCohort sizes:")
    print(cohort.value_counts().to_string())

    # --- beta matrix + probe filtering (reported, not forced) ---
    beta = get_beta_matrix()
    beta_filtered, filter_report = filter_probes(beta)
    print("\nProbe filtering report:")
    for k, v in filter_report.items():
        print(f"  {k}: {v}")
    pd.DataFrame([filter_report]).to_csv(
        paths.tables_dir() / "probe_filtering.tsv", sep="\t", index=False
    )

    # --- signature ---
    signature = load_signature()
    sig_probes = signature.index
    missing = sig_probes.difference(beta_filtered.index)
    present_probes = sig_probes.intersection(beta_filtered.index)
    print(f"\nSignature: {len(sig_probes)} probes ({len(signature)} expected 7085); "
          f"{len(present_probes)} present in the beta matrix after probe filtering "
          f"({len(missing)} missing/filtered out).")

    # --- score everyone ---
    scores = score_cohort(beta_filtered, cohort, present_probes)
    scores_path = paths.tables_dir() / "sotos_scores.tsv"
    scores.to_csv(scores_path, sep="\t", index=False)
    print(f"\nWrote {len(scores)} sample scores -> {scores_path}")

    # --- check the three published results ---
    summary_rows = []

    disc = scores[scores.cohort.isin(["discovery_case", "discovery_control"])]
    n_case = (disc.cohort == "discovery_case").sum()
    n_ctrl = (disc.cohort == "discovery_control").sum()
    case_scores = disc.loc[disc.cohort == "discovery_case", "score"]
    ctrl_scores = disc.loc[disc.cohort == "discovery_control", "score"]
    separated = case_scores.min() > ctrl_scores.max()
    summary_rows.append({
        "check": "discovery_separation",
        "expected": "19 Sotos cases separate cleanly from 53 controls (score > 0 vs < 0)",
        "observed": (
            f"n_case={n_case}, n_control={n_ctrl}; "
            f"case scores [{case_scores.min():.3f}, {case_scores.max():.3f}], "
            f"control scores [{ctrl_scores.min():.3f}, {ctrl_scores.max():.3f}]; "
            f"clean_separation={separated}; "
            f"case>0: {(case_scores > 0).sum()}/{n_case}; "
            f"control<0: {(ctrl_scores < 0).sum()}/{n_ctrl}"
        ),
        "reproduced": bool((case_scores > 0).all() and (ctrl_scores < 0).all()),
    })

    weaver = scores[scores.cohort == "weaver"]
    n_weaver_neg = (weaver["score"] < 0).sum()
    summary_rows.append({
        "check": "weaver_negative",
        "expected": "all 8 Weaver (EZH2) samples score negative",
        "observed": (
            f"{n_weaver_neg}/{len(weaver)} scored negative; "
            f"scores={weaver['score'].round(3).tolist()}"
        ),
        "reproduced": bool(len(weaver) == 8 and n_weaver_neg == 8),
    })

    missense = scores[scores.cohort == "missense_variant"]
    n_pos = (missense["score"] > 0).sum()
    n_neg = (missense["score"] < 0).sum()
    summary_rows.append({
        "check": "missense_split",
        "expected": (
            "16 NSD1 missense VOUS split 9 positive (pathogenic-like) / "
            "7 negative (benign-like)"
        ),
        "observed": f"n={len(missense)}, positive={n_pos}, negative={n_neg}",
        "reproduced": bool(len(missense) == 16 and n_pos == 9 and n_neg == 7),
    })

    replication = scores[scores.cohort == "replication_case"]
    n_repl_pos = (replication["score"] > 0).sum()
    summary_rows.append({
        "check": "replication_cohort (bonus, not in the original 3 checks)",
        "expected": "19 replication Sotos LOF cases (held out of discovery) mostly score positive",
        "observed": f"n={len(replication)}, positive={n_repl_pos}",
        "reproduced": None,
    })

    summary = pd.DataFrame(summary_rows)
    summary_path = paths.tables_dir() / "sotos_reproduction_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"\nWrote reproduction summary -> {summary_path}")
    print(summary.to_string())

    # --- figures ---
    make_figures(beta_filtered, present_probes, meta, cohort, scores)


def make_figures(beta, probes, meta, cohort, scores):
    # Strip/box plot of scores by cohort, zero line marked
    order = [
        "discovery_control", "discovery_case", "replication_case", "weaver", "missense_variant",
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = scores[scores.cohort.isin(order)].copy()
    plot_df["cohort"] = pd.Categorical(plot_df["cohort"], categories=order, ordered=True)
    positions = range(len(order))
    for pos, name in zip(positions, order):
        vals = plot_df.loc[plot_df.cohort == name, "score"]
        jitter = np.random.default_rng(0).normal(0, 0.06, size=len(vals))
        ax.scatter([pos] * len(vals) + jitter, vals, alpha=0.7, s=25)
        ax.hlines(vals.median(), pos - 0.2, pos + 0.2, color="black", linewidth=2)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("SS score (r_case - r_control)")
    ax.set_title("Choufani et al. Sotos classifier scores by cohort (GSE74432 reproduction)")
    fig.tight_layout()
    fig.savefig(paths.figures_dir() / "sotos_scores.png", dpi=150)
    plt.close(fig)

    # Heatmap of signature probes, samples clustered by cohort then score
    scored_ids = scores.sort_values(["cohort", "score"])["gsm_accession"].tolist()
    heat_probes = list(probes)[:300]  # cap for a legible/renderable figure
    mat = beta.loc[heat_probes, scored_ids]
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    im = ax2.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
    ax2.set_xlabel(f"{len(scored_ids)} samples (grouped by cohort, sorted by score)")
    ax2.set_ylabel(f"{len(heat_probes)} of {len(probes)} signature CpGs shown")
    ax2.set_title("Sotos NSD1+/- signature beta values (GSE74432 reproduction)")
    fig2.colorbar(im, ax=ax2, label="beta value")
    fig2.tight_layout()
    fig2.savefig(paths.figures_dir() / "sotos_heatmap.png", dpi=150)
    plt.close(fig2)
    print(f"\nWrote figures -> {paths.figures_dir()}")


if __name__ == "__main__":
    main()
