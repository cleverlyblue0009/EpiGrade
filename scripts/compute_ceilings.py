"""Attainable evidence ceiling: the strongest band a PERFECT classifier (fully non-overlapping
scores) could earn at a given cohort size, run through the identical bootstrap and banding
pipeline as real scores. It is a statement about cohort size and study structure, not about any
disorder's biology or any classifier's quality.

Writes:
  results/tables/attainable_ceiling.tsv        one row per disorder, at its real cohort shape
  results/tables/attainable_ceiling_curve.tsv  the ceiling against cohort size, for one study
                                               and for two studies on each side

Reads results/tables/evidence_bands.tsv and corpus_cohorts.tsv (scripts/build_corpus.py), so the
cohort shapes are the ones actually calibrated, not retyped here.
"""

from __future__ import annotations

import pandas as pd

from epigrade import paths
from epigrade.calibrate.calibrate import evidence_ceiling

CURVE_SIZES = [5, 6, 8, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200]


def _ceiling_fields(n_case: int, n_control: int, k_case: int, k_control: int) -> dict:
    result = evidence_ceiling(
        n_case=n_case, n_control=n_control, n_studies_case=k_case, n_studies_control=k_control,
    )
    return {
        "lr_point_estimate": result.lr_point_estimate,
        "points_conservative": result.points,
        "attainable_band": result.band,
        "confounded_by_design": result.confounded_by_design,
        "interpretation_scope": result.interpretation_scope,
        "between_study_bound_possible": not result.confounded_by_design,
    }


def disorder_table() -> pd.DataFrame:
    evidence = pd.read_csv(paths.tables_dir() / "evidence_bands.tsv", sep="\t")
    cohorts = pd.read_csv(paths.tables_dir() / "corpus_cohorts.tsv", sep="\t")
    typical = evidence[(evidence.query_point == "typical_case_score") & (evidence.prior == 0.10)]
    typical = typical.set_index("disorder")

    rows = []
    for disorder, arms in cohorts.groupby("disorder", sort=False):
        if disorder in typical.index:
            e = typical.loc[disorder]
            shape = (int(e.n_case), int(e.n_control), int(e.n_studies_case),
                     int(e.n_studies_control))
            observed, note = e.band, ""
        else:
            # No classifier (fewer than 10 cases): the ceiling is still computable from the
            # cohort's size, and shows what even a perfect classifier could claim there.
            controls = arms.drop_duplicates("control_series")
            shape = (int(arms.n_cases.sum()), int(controls.n_controls.sum()),
                     arms.series_id.nunique(), controls.control_series.nunique())
            observed, note = "NA", "no classifier built (fewer than 10 cases); ceiling only"
        n_case, n_control, k_case, k_control = shape
        rows.append({
            "disorder": disorder, "n_case": n_case, "n_control": n_control,
            "n_studies_case": k_case, "n_studies_control": k_control,
            **_ceiling_fields(n_case, n_control, k_case, k_control),
            "observed_band_at_typical_case_score": observed, "note": note,
        })
        print(f"  {disorder}: ceiling {rows[-1]['attainable_band']} "
              f"({rows[-1]['interpretation_scope']}), observed {observed}")
    return pd.DataFrame(rows)


def curve_table() -> pd.DataFrame:
    rows = []
    for label, k in (("one_study", 1), ("two_studies", 2)):
        for n in CURVE_SIZES:
            rows.append({
                "study_structure": label, "n_case": n, "n_control": n,
                "n_studies_case": k, "n_studies_control": k, **_ceiling_fields(n, n, k, k),
            })
            print(f"  curve {label} n={n}: {rows[-1]['attainable_band']}")
    return pd.DataFrame(rows)


def main() -> None:
    print("Per-disorder ceilings...")
    disorder_table().to_csv(paths.tables_dir() / "attainable_ceiling.tsv", sep="\t", index=False)
    print("Ceiling curve...")
    curve_table().to_csv(
        paths.tables_dir() / "attainable_ceiling_curve.tsv", sep="\t", index=False,
    )


if __name__ == "__main__":
    main()
