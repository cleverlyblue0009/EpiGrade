"""The (score, label, study_id) frame a classifier is calibrated on, and the evidence rows built
from it. Shared by scripts/build_corpus.py (fixed reference query points, evidence_bands.tsv)
and scripts/build_calibration_grid.py (a grid of query points, calibration_grid.tsv), so both
tables come from exactly the same frame and the same calibration call.
"""

from __future__ import annotations

import pandas as pd

from epigrade.calibrate.calibrate import evaluate_score, posterior_probability

PRIOR_GRID = [0.05, 0.10, 0.20]


def calibration_frame(scores_by_gsm: pd.Series, case_series: dict[str, str],
                      control_series: dict[str, str]) -> pd.DataFrame:
    """A disorder's cases (label 1) and the controls it is defined against (label 0), each with
    its score and the study it came from. Order follows the two dicts: cases, then controls -
    the bootstrap draws studies in order of first appearance, so order is part of the result.
    Samples without a score are left out."""
    case_ids = [s for s in case_series if s in scores_by_gsm.index]
    control_ids = [s for s in control_series if s in scores_by_gsm.index]
    return pd.DataFrame({
        "score": scores_by_gsm.loc[case_ids + control_ids].to_numpy(),
        "label": [1] * len(case_ids) + [0] * len(control_ids),
        "study_id": [case_series[s] for s in case_ids]
                    + [control_series[s] for s in control_ids],
    })


def evidence_rows_at(disorder: str, df: pd.DataFrame, query_point: str, query_score: float,
                     priors=PRIOR_GRID, n_boot: int | None = None) -> list[dict]:
    """One evaluate_score call at `query_score`, expanded to one row per prior. The likelihood
    ratio, its intervals and the band do not depend on the prior (the bootstrap is seeded and
    never sees it); only the posterior does, and it is computed per prior."""
    result = evaluate_score(disorder, df, query_score, prior=priors[0], n_boot=n_boot)
    rows = []
    for prior in priors:
        rows.append({
            "disorder": disorder, "query_point": query_point,
            "query_score": round(query_score, 4), "prior": prior,
            "lr_point_estimate": result.lr_point_estimate,
            "lr_ci_low": result.lr_ci_low, "lr_ci_high": result.lr_ci_high,
            "points_conservative": result.points, "band": result.band,
            "posterior_prob": posterior_probability(result.lr_point_estimate, prior),
            "evidence_anchor": result.evidence_anchor,
            "n_case": result.n_case, "n_control": result.n_control,
            "n_studies_case": result.n_studies_case,
            "n_studies_control": result.n_studies_control,
            "confounded_by_design": result.confounded_by_design,
            "reason": result.reason,
            "within_study_ci_low": result.within_study_ci_low,
            "within_study_ci_high": result.within_study_ci_high,
            "interpretation_scope": result.interpretation_scope,
        })
    return rows
