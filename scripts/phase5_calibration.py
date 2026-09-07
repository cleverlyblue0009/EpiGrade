"""Phase 5: apply calibration to the real Sotos classifier scores (from phase 3), plus the
evidence-ceiling table across a range of cohort sizes actually seen in this corpus.

Sotos is single-study (GSE74432 only), so a study-level bootstrap with exactly one case study
and one control study is degenerate by construction (every resample is identical to the
original data - see epigrade.calibrate.calibrate.bootstrap_lr_ci): no real between-study
confidence bound exists. That refusal stands. But the band is NOT blanket "NA" - it's computed
from the within-study (sample-level) bootstrap instead, and carries
`interpretation_scope="within_study_only"` plus a `reason` spelling out exactly why no
between-study claim is being made. The `within_study_ci_low/high` columns are what that band is
actually derived from in this case (see `reason`) - never silently treated as equivalent to a
validated between-study result. Every disorder in phase 4's cross-disorder matrix that also has
a real classifier and spans >=2 studies on both sides gets `interpretation_scope="between_study"`
instead, from the ordinary study-level bootstrap, as data becomes available.
"""

from __future__ import annotations

import pandas as pd

from epigrade import paths
from epigrade.calibrate.calibrate import evaluate_score, evidence_ceiling

PRIOR_GRID = [0.05, 0.10, 0.20]


def sotos_evidence_rows() -> list[dict]:
    scores = pd.read_csv(paths.tables_dir() / "sotos_scores.tsv", sep="\t")

    case_cohorts = {"discovery_case", "replication_case"}
    control_cohorts = {"discovery_control"}
    df = scores[scores.cohort.isin(case_cohorts | control_cohorts)].copy()
    df["label"] = df["cohort"].isin(case_cohorts).astype(int)
    df["study_id"] = "GSE74432"  # single study for this disorder in this corpus

    case_median = df.loc[df.label == 1, "score"].median()
    missense = scores[scores.cohort == "missense_variant"]
    missense_median = missense["score"].median() if len(missense) else float("nan")

    rows = []
    for query_name, query_score in [
        ("typical_case_score", case_median),
        ("typical_missense_vous_score", missense_median),
        ("borderline_score_zero", 0.0),
    ]:
        if pd.isna(query_score):
            continue
        for prior in PRIOR_GRID:
            result = evaluate_score("Sotos syndrome", df, query_score, prior=prior)
            rows.append({
                "disorder": result.disorder,
                "query_point": query_name,
                "query_score": round(query_score, 4),
                "prior": prior,
                "lr_point_estimate": result.lr_point_estimate,
                "lr_ci_low": result.lr_ci_low,
                "lr_ci_high": result.lr_ci_high,
                "points_conservative": result.points,
                "band": result.band,
                "posterior_prob": result.posterior_prob,
                "evidence_anchor": result.evidence_anchor,
                "n_case": result.n_case,
                "n_control": result.n_control,
                "n_studies_case": result.n_studies_case,
                "n_studies_control": result.n_studies_control,
                "confounded_by_design": result.confounded_by_design,
                "reason": result.reason,
                "within_study_ci_low": result.within_study_ci_low,
                "within_study_ci_high": result.within_study_ci_high,
                "interpretation_scope": result.interpretation_scope,
            })
    return rows


def ceiling_table() -> pd.DataFrame:
    """Attainable evidence ceiling at cohort sizes actually seen in this corpus (from the
    role=case counts in results/tables/sample_triage.tsv's companion harmonized data), plus a
    couple of illustrative smaller/larger sizes for the degradation curve."""
    sizes = [200, 100, 63, 38, 37, 29, 21, 20, 19, 16, 15, 10, 8, 6, 5]
    rows = []
    for n in sizes:
        n_studies = 2 if n >= 30 else 1  # optimistic assumption for the illustrative curve
        result = evidence_ceiling(n_case=n, n_control=max(n, 10), n_studies_case=n_studies)
        rows.append({
            "n_case": n,
            "n_studies_case_assumed": n_studies,
            "lr_point_estimate": result.lr_point_estimate,
            "points_conservative": result.points,
            "attainable_band": result.band,
            "confounded_by_design": result.confounded_by_design,
            "interpretation_scope": result.interpretation_scope,
        })
    return pd.DataFrame(rows)


def main() -> None:
    rows = sotos_evidence_rows()
    evidence_df = pd.DataFrame(rows)
    evidence_path = paths.tables_dir() / "evidence_bands.tsv"
    evidence_df.to_csv(evidence_path, sep="\t", index=False)
    print(f"Wrote {len(evidence_df)} evidence rows -> {evidence_path}")
    print(evidence_df.to_string())

    ceiling_df = ceiling_table()
    ceiling_path = paths.tables_dir() / "attainable_ceiling.tsv"
    ceiling_df.to_csv(ceiling_path, sep="\t", index=False)
    print(f"\nWrote evidence ceiling table -> {ceiling_path}")
    print(ceiling_df.to_string())


if __name__ == "__main__":
    main()
