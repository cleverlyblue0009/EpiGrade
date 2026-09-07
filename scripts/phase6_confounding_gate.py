"""Phase 6: confounding gate. Structural single-study-vs-multi-study status is a metadata-only
fact (which series each disorder's cases come from) - it doesn't require downloading beta
values, so it's computed here for every in-scope disorder in the corpus, not just the two
(Sotos, Silver-Russell syndrome) with an actual derived/published classifier this session.

Per the spec: most disorders in this corpus come from a single study. Expect most cohorts to
fail this gate. That is the finding, not a bug - failing cohorts are kept and reported with the
reason, never silently dropped.
"""

from __future__ import annotations

import pandas as pd

from epigrade import paths
from epigrade.acquire.harmonize import harmonize

MIN_CASES_TO_BUILD_CLASSIFIER = 10


def main() -> None:
    samples = pd.read_parquet(paths.interim_dir() / "samples.parquet")
    harmonized = harmonize(samples)
    cases = harmonized[harmonized.role == "case"]

    rows = []
    for disorder, g in cases.groupby("disorder"):
        n_cases = len(g)
        studies = sorted(g["series_id"].unique())
        n_studies = len(studies)

        if n_cases < MIN_CASES_TO_BUILD_CLASSIFIER:
            status = "not-testable"
            reason = (
                f"only {n_cases} confirmed case(s) (< {MIN_CASES_TO_BUILD_CLASSIFIER} minimum "
                "to attempt building a classifier at all)"
            )
        elif n_studies == 1:
            status = "fail"
            reason = (
                f"all {n_cases} cases come from a single study ({studies[0]}) - disease status "
                "and study-of-origin/batch are fully confounded by design. A within-study "
                "case/control separation, however clean, cannot be attributed to the disorder "
                "rather than to batch. NOT REPORTABLE at face value regardless of how good the "
                "numbers look."
            )
        else:
            status = "pass-structural"  # multiple studies exist; doesn't mean a classifier
            reason = (
                f"{n_cases} cases span {n_studies} independent studies ({', '.join(studies)}) - "
                "a genuine leave-one-study-out test is structurally possible. See "
                "results/tables/srs_leave_one_study_out.tsv for whether a classifier could "
                "actually be derived and how it generalized, if this disorder was attempted "
                "this session."
            )

        rows.append({
            "disorder": disorder,
            "n_cases": n_cases,
            "n_studies": n_studies,
            "studies": ", ".join(studies),
            "confounded_by_design": n_studies <= 1,
            "status": status,
            "reason": reason,
        })

    df = pd.DataFrame(rows).sort_values(["status", "n_cases"], ascending=[True, False])
    out_path = paths.tables_dir() / "confounding_gate.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(df)} disorder confounding statuses -> {out_path}")
    print(df[["disorder", "n_cases", "n_studies", "status"]].to_string())
    print(f"\n{(df.status == 'fail').sum()}/{len(df)} disorders fail the gate "
          f"(single-study confound) - expected, per the project spec, to be most of them.")


if __name__ == "__main__":
    main()
