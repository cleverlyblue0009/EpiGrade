"""Calibration grid: evidence bands precomputed on a fixed grid of query scores, so the app's
"Grade a score" page can take any score and look up the nearest grid point instead of running
the calibration live. The app recomputes nothing; this is the offline step it relies on.

For each disorder with a classifier:
  - the calibration frame is the one evidence_bands.tsv uses (epigrade.corpus.calibration -
    the same helper scripts/build_corpus.py calls), built from the checked-in per-sample scores
    (results/tables/classifier_scores/<slug>.tsv) and cohort membership
    (results/resource/byo/cohort_membership.tsv), so this stage needs no downloaded data;
  - the grid is 41 evenly spaced points from the lowest to the highest score among its cases
    and controls, padded by 10% of that range on each side, rounded to 4 decimals, with the
    point nearest 0 replaced by exactly 0.0 so the borderline score is on the grid;
  - evaluate_score() runs once per grid point; the likelihood ratio, its intervals and the band
    do not depend on the prior, so each call is expanded to one row per prior in PRIOR_GRID
    with the posterior computed per prior (identical to calling it once per prior).

Writes results/tables/calibration_grid.tsv: the columns of evidence_bands.tsv, with
query_point = "grid". One row per (disorder, query_score, prior).

Runtime: 41 calls per disorder, 410 in all. Single-study cohorts skip the study-level bootstrap
(it is degenerate) and take well under a second per call; the two between-study cohorts run the
full 1,000-replicate study-level bootstrap and dominate: about 3 minutes each, 6.5 minutes in
all, measured on the development laptop.
Resumable: a disorder already in the output is skipped unless --force is given. --n-boot
lowers the study-level bootstrap for a quick look; the committed table uses the config value.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from epigrade import paths
from epigrade.corpus.calibration import PRIOR_GRID, calibration_frame, evidence_rows_at

N_POINTS = 41
PAD = 0.10
OUT_NAME = "calibration_grid.tsv"


def grid_points(scores: np.ndarray) -> list[float]:
    low, high = float(np.min(scores)), float(np.max(scores))
    span = high - low
    points = np.round(np.linspace(low - PAD * span, high + PAD * span, N_POINTS), 4)
    points[int(np.argmin(np.abs(points)))] = 0.0
    return sorted(float(x) for x in points)


def frames() -> dict[str, pd.DataFrame]:
    """Calibration frame per disorder with a classifier, in registry order."""
    index = pd.read_csv(paths.repo_root() / "results" / "signatures" / "index.tsv", sep="\t")
    members = pd.read_csv(paths.repo_root() / "results" / "resource" / "byo"
                          / "cohort_membership.tsv", sep="\t")
    out = {}
    for row in index.itertuples():
        scores = pd.read_csv(paths.tables_dir() / "classifier_scores" / f"{row.slug}.tsv",
                             sep="\t").set_index("gsm_accession")["score"]
        block = members[members.disorder == row.disorder]
        cases = block[block.role == "case"]
        controls = block[block.role == "control"]
        out[row.disorder] = calibration_frame(
            scores, dict(zip(cases.gsm_accession, cases.series_id)),
            dict(zip(controls.gsm_accession, controls.series_id)),
        )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute every disorder")
    parser.add_argument("--n-boot", type=int, default=None,
                        help="study-level bootstrap replicates (default: config value)")
    # parse_known_args: when run as a stage of scripts/reproduce.py the process arguments
    # are reproduce.py's own (--quick, --force); --force carries through, the rest is ignored.
    args, _ = parser.parse_known_args(argv)

    out_path = paths.tables_dir() / OUT_NAME
    existing = (pd.read_csv(out_path, sep="\t", keep_default_na=False, na_values=[""])
                if out_path.exists() and not args.force else pd.DataFrame())
    done = set(existing.disorder) if len(existing) else set()
    rows = existing.to_dict("records") if len(existing) else []

    for disorder, df in frames().items():
        if disorder in done:
            print(f"  {disorder}: already in {OUT_NAME}, skipped (--force to redo)")
            continue
        t0 = time.time()
        for score in grid_points(df["score"].to_numpy()):
            rows += evidence_rows_at(disorder, df, "grid", score, PRIOR_GRID, n_boot=args.n_boot)
        pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)  # checkpoint per disorder
        print(f"  {disorder}: {N_POINTS} grid points in {time.time() - t0:.0f}s", flush=True)

    table = pd.DataFrame(rows)
    order = {d: i for i, d in enumerate(frames())}
    table = table.sort_values(["disorder", "query_score", "prior"],
                              key=lambda c: c.map(order) if c.name == "disorder" else c)
    table.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(table)} rows -> results/tables/{OUT_NAME}")


if __name__ == "__main__":
    main()
