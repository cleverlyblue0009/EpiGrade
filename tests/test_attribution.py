"""Tests for the attribution module (Task 5) - computed-numbers-only, no LLM narrative."""

import pandas as pd

from epigrade.report.attribution import explain_score, format_explanation


def _toy_setup():
    probes = pd.Index([f"cg{i}" for i in range(10)])
    # Pearson correlation is undefined for a constant array - these need real, varying values
    # across probes (like a real signature does), not identical repeated values.
    # Deliberately different shapes (not a constant shift of one another) - two profiles that
    # are just shifted copies would be perfectly correlated with EACH OTHER too, making it
    # impossible for a sample to match one without also matching the other.
    case_vals = [0.10, 0.30, 0.15, 0.35, 0.12, 0.32, 0.18, 0.28, 0.14, 0.33]
    control_vals = [0.75, 0.70, 0.85, 0.65, 0.80, 0.68, 0.82, 0.72, 0.78, 0.66]
    median_case = pd.Series(case_vals, index=probes)
    median_control = pd.Series(control_vals, index=probes)
    case_scores = pd.Series([0.5, 0.6, 0.55, 0.7, 0.4])
    control_scores = pd.Series([-0.5, -0.6, -0.4, -0.55, -0.3])
    return probes, median_case, median_control, case_scores, control_scores


def test_explain_score_matches_the_actual_correlation_difference_score():
    probes, mc, mn, cs, ctrl = _toy_setup()
    # a sample that closely tracks the case profile's actual shape
    sample = mc + 0.01
    result = explain_score(
        "GSM_TEST", "Toy disorder", "discovery_case", sample, mc, mn, probes, cs, ctrl,
    )
    assert result.score > 0.5  # should strongly favor "case"
    assert result.n_probes_present == 10
    assert result.n_probes_missing == 0


def test_probe_coverage_counts_missing_probes():
    probes, mc, mn, cs, ctrl = _toy_setup()
    sample = (mc + 0.01).iloc[:7]  # only 7 of 10 probes present
    result = explain_score(
        "GSM_TEST", "Toy disorder", "discovery_case", sample, mc, mn, probes, cs, ctrl,
    )
    assert result.n_probes_present == 7
    assert result.n_probes_missing == 3


def test_direction_check_flags_an_anomalous_case():
    probes, mc, mn, cs, ctrl = _toy_setup()
    # This sample scores case-like (matches median_case's SHAPE numerically) but the case
    # profile here is entirely HIGHER than control (gain), not lower (loss) - so a sample
    # matching this case profile does NOT match an expected "loss" direction.
    median_case_gain = mn.copy()  # reuse the control shape, but call it "case" (all values high)
    median_control_gain = mc.copy()  # and the case shape as "control" (all values low)
    sample = median_case_gain + 0.01  # tracks the (gain) case profile's shape closely
    result = explain_score(
        "GSM_TEST", "Toy disorder", "discovery_case", sample, median_case_gain,
        median_control_gain, probes, cs, ctrl, expected_direction="loss",
    )
    assert result.score > 0  # scores case-like
    assert result.direction_anomalous is True  # but the direction is wrong for "loss"


def test_top_probes_are_computed_not_fabricated():
    probes, mc, mn, cs, ctrl = _toy_setup()
    sample = mc + 0.01
    result = explain_score(
        "GSM_TEST", "Toy disorder", "discovery_case", sample, mc, mn, probes, cs, ctrl, top_n=3,
    )
    assert len(result.top_probes) == 3
    for p in result.top_probes:
        assert p["pulls_toward"] == "case"  # sample tracks the case profile everywhere


def test_format_explanation_only_restates_computed_fields():
    probes, mc, mn, cs, ctrl = _toy_setup()
    sample = mc + 0.01
    result = explain_score(
        "GSM_TEST", "Toy disorder", "discovery_case", sample, mc, mn, probes, cs, ctrl,
    )
    lines = format_explanation(result)
    assert any("GSM_TEST" in line for line in lines)
    assert any(str(result.n_probes_present) in line for line in lines)
