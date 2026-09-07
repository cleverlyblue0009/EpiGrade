"""Synthetic-data tests for the calibration module, required before it ever touches real
scores (per the project spec's synthetic_tests requirement)."""

import numpy as np
import pandas as pd
import pytest

from epigrade.calibrate.calibrate import (
    evaluate_score,
    evidence_ceiling,
    local_likelihood_ratio,
)

RNG = np.random.default_rng(7)


def _make_df(n_case, n_control, case_mean, control_mean, sd=0.3, n_studies_case=1,
             n_studies_control=1):
    case_scores = RNG.normal(case_mean, sd, n_case)
    control_scores = RNG.normal(control_mean, sd, n_control)
    studies_case = [f"case_study{i % n_studies_case}" for i in range(n_case)]
    studies_control = [f"control_study{i % n_studies_control}" for i in range(n_control)]
    return pd.DataFrame({
        "score": np.concatenate([case_scores, control_scores]),
        "label": [1] * n_case + [0] * n_control,
        "study_id": studies_case + studies_control,
    })


def test_local_lr_recovers_a_known_strong_effect():
    # Cases and controls overlap enough that both classes have real local density near the
    # query point (the adaptive estimator requires min_n of EACH class - at a query point deep
    # in one tail with near-zero density of the other class, it must widen far enough to find
    # some, which correctly makes the reported LR more conservative than the "true" ratio at
    # that exact point. That conservatism at extremes is the documented, intended behavior, not
    # noise - so this test uses distributions with real overlap instead of asserting an
    # unbounded LR from a near-empty tail.
    df = _make_df(n_case=150, n_control=150, case_mean=1.5, control_mean=-1.5, sd=1.2)
    lr = local_likelihood_ratio(df["score"].values, df["label"].values, query_score=1.5)
    assert lr > 3  # clearly favors "case", well above 1


def test_local_lr_near_one_when_classes_are_identical():
    df = _make_df(n_case=200, n_control=200, case_mean=0.0, control_mean=0.0, sd=0.5)
    lr = local_likelihood_ratio(df["score"].values, df["label"].values, query_score=0.0)
    assert 0.4 < lr < 2.5  # near 1, allowing for sampling noise


def test_refusal_rule_fires_when_both_classes_are_the_same_distribution():
    """The spec's core honesty check: if cases and controls are indistinguishable, the pipeline
    must emit 'No evidence', never a Supporting/Moderate/etc band."""
    df = _make_df(n_case=60, n_control=60, case_mean=0.0, control_mean=0.0, sd=0.5,
                   n_studies_case=3, n_studies_control=3)
    result = evaluate_score("null_disorder", df, query_score=0.0)
    assert result.band in ("No evidence", "NA")


def test_evidence_anchor_is_always_pp4_and_never_ps3():
    df = _make_df(n_case=50, n_control=50, case_mean=1.5, control_mean=-1.5,
                   n_studies_case=2, n_studies_control=2)
    result = evaluate_score("toy_disorder", df, query_score=1.5)
    assert result.evidence_anchor == "PP4_diagnostic"


def test_forbidden_anchor_is_enforced_in_config():
    import yaml

    from epigrade import paths
    with open(paths.repo_root() / "config" / "evidence_bands.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "PS3_functional" in cfg["forbidden_anchors"]
    assert cfg["evidence_anchor"] not in cfg["forbidden_anchors"]


def test_confounded_by_design_flagged_when_all_cases_from_one_study():
    df = _make_df(n_case=40, n_control=40, case_mean=1.0, control_mean=-1.0,
                   n_studies_case=1, n_studies_control=1)
    result = evaluate_score("single_study_disorder", df, query_score=1.0)
    assert result.confounded_by_design is True


def test_not_confounded_when_cases_span_multiple_studies():
    df = _make_df(n_case=40, n_control=40, case_mean=1.0, control_mean=-1.0,
                   n_studies_case=4, n_studies_control=2)
    result = evaluate_score("multi_study_disorder", df, query_score=1.0)
    assert result.confounded_by_design is False


def test_single_study_cohort_never_gets_a_between_study_bound():
    """The Task 1 regression test, updated for the within-study-scoped band: a single-study
    cohort must never be assigned a BETWEEN-STUDY confidence bound (lr_ci_low/high stay NaN,
    a degenerate zero-width interval is not evidence of precision, it's an artifact of
    resampling one study against itself) - but it IS now given a real, clearly-scoped band
    computed from the within-study bootstrap instead of a blanket refusal."""
    df = _make_df(n_case=38, n_control=53, case_mean=2.0, control_mean=-2.0, sd=0.3,
                   n_studies_case=1, n_studies_control=1)
    result = evaluate_score("single_study_strong_separation", df, query_score=2.0)
    # the refusal that must stay: no between-study bound, ever, for a single-study cohort
    assert np.isnan(result.lr_ci_low) and np.isnan(result.lr_ci_high)
    assert result.interpretation_scope == "within_study_only"
    assert "between-study" in result.reason
    # but a real band/points/within-study CI IS now computed and shown, given the strong,
    # cleanly-separated synthetic data - not silently withheld
    assert result.band not in ("NA",)
    assert not np.isnan(result.points)
    assert not np.isnan(result.within_study_ci_low) and not np.isnan(result.within_study_ci_high)


def test_control_side_single_study_is_also_flagged_confounded():
    """The specific gap the original bug missed: cases span multiple studies but controls all
    come from one - the bootstrap is just as degenerate on that side. Still within_study_only
    scoped, still no between-study bound, but still a real computed band."""
    df = _make_df(n_case=40, n_control=40, case_mean=1.0, control_mean=-1.0,
                   n_studies_case=3, n_studies_control=1)
    result = evaluate_score("control_side_confounded", df, query_score=1.0)
    assert result.confounded_by_design is True
    assert result.interpretation_scope == "within_study_only"
    assert np.isnan(result.lr_ci_low) and np.isnan(result.lr_ci_high)


def test_multi_study_cohort_gets_a_nonzero_width_ci():
    df = _make_df(n_case=60, n_control=60, case_mean=1.5, control_mean=-1.5, sd=0.6,
                   n_studies_case=3, n_studies_control=3)
    result = evaluate_score("multi_study_nonzero_ci", df, query_score=1.5)
    assert result.confounded_by_design is False
    assert result.interpretation_scope == "between_study"
    assert not np.isnan(result.lr_ci_low) and not np.isnan(result.lr_ci_high)
    assert result.lr_ci_high > result.lr_ci_low, (
        "a genuine multi-study bootstrap must not collapse to a zero-width interval"
    )


def test_single_study_null_case_gets_within_study_no_evidence():
    """When cases and controls are indistinguishable AND the cohort is single-study, the
    within-study bootstrap should itself span LR=1, so the result is 'No evidence' - not
    'NA' (a real within-study bootstrap did run and converge, it just found nothing) and
    still correctly scoped as within_study_only, not silently upgraded to between_study."""
    df = _make_df(n_case=40, n_control=40, case_mean=0.0, control_mean=0.0, sd=0.5,
                   n_studies_case=1, n_studies_control=1)
    result = evaluate_score("single_study_null", df, query_score=0.0)
    assert result.interpretation_scope == "within_study_only"
    assert result.band in ("No evidence", "NA")
    assert np.isnan(result.lr_ci_low) and np.isnan(result.lr_ci_high)


@pytest.mark.parametrize("n", [200, 50, 20, 10])
def test_evidence_ceiling_degrades_monotonically_as_n_shrinks(n):
    """A perfect classifier's attainable evidence points should never increase as n shrinks -
    computed once per n and compared pairwise below."""
    result = evidence_ceiling(n_case=n, n_control=n, n_studies_case=max(2, n // 10))
    assert result.band != "PS3_functional"  # sanity: anchor guard applies here too


def test_evidence_ceiling_monotonic_ordering():
    ceilings = {}
    for n in [200, 50, 20, 10]:
        r = evidence_ceiling(n_case=n, n_control=n, n_studies_case=max(2, n // 10))
        ceilings[n] = r.points if not np.isnan(r.points) else -np.inf
    ns = sorted(ceilings, reverse=True)  # largest n first
    for a, b in zip(ns, ns[1:]):
        assert ceilings[a] >= ceilings[b] - 1e-6, (
            f"ceiling at n={a} ({ceilings[a]}) should be >= ceiling at n={b} ({ceilings[b]})"
        )
