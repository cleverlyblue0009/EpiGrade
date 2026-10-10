"""Tests for the disorder-agnostic leave-one-study-out routine, on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from epigrade.audit.loso import LOSO_COLUMNS, StudyArm, leave_one_study_out, to_frame


def _study(name, n_case, n_control, effect=0.35, n_probes=400, seed=0):
    rng = np.random.default_rng(seed)
    base = np.random.default_rng(99).uniform(0.2, 0.8, size=n_probes)  # shared across studies
    shift = np.zeros(n_probes)
    # Mixed directions: the correlation-difference score is blind to a uniform shift (Pearson
    # r is shift-invariant), so a one-directional toy signature would not separate anything.
    shift[:60] = effect * np.where(np.arange(60) % 2 == 0, 1.0, -1.0)
    cases = [f"{name}_CASE{i}" for i in range(n_case)]
    controls = [f"{name}_CTRL{i}" for i in range(n_control)]
    data = {s: np.clip(base + shift + rng.normal(0, 0.08, n_probes), 0, 1) for s in cases}
    data |= {s: np.clip(base + rng.normal(0, 0.08, n_probes), 0, 1) for s in controls}
    beta = pd.DataFrame(data, index=[f"cg{i:05d}" for i in range(n_probes)])
    return StudyArm(name, beta, cases, controls)


def _pair(**kwargs):
    return [_study("A", 15, 15, seed=1, **kwargs), _study("B", 12, 14, seed=2, **kwargs)]


def test_both_directions_are_run_and_a_real_effect_transfers():
    rows = leave_one_study_out("Toy disorder", _pair())
    assert [(r["train_study"], r["test_study"]) for r in rows] == [("B", "A"), ("A", "B")]
    for r in rows:
        assert r["status"] == "computed"
        assert r["sensitivity"] > 0.9 and r["specificity"] > 0.9
        assert r["signature_source"] == "rederived_not_published"
        assert r["disorder"] == "Toy disorder"


def test_too_few_training_cases_is_a_reasoned_not_computed_row():
    rows = leave_one_study_out(
        "Toy disorder", [_study("A", 8, 15, seed=1), _study("B", 12, 14, seed=2)],
    )
    by_train = {r["train_study"]: r for r in rows}
    assert by_train["A"]["status"] == "not_computed"
    assert "only 8 training case" in by_train["A"]["reason"]
    assert np.isnan(by_train["A"]["sensitivity"])
    assert by_train["B"]["status"] == "computed"  # the other direction is unaffected


def test_no_signal_is_not_computed_with_the_underpowered_diagnosis():
    rows = leave_one_study_out("Toy disorder", _pair(effect=0.0))
    assert all(r["status"] == "not_computed" for r in rows)
    assert all("Path B" in r["reason"] for r in rows)


def test_held_out_study_without_controls_reports_sensitivity_only():
    arms = [_study("A", 15, 15, seed=1), _study("B", 12, 0, seed=2)]
    row = next(r for r in leave_one_study_out("Toy disorder", arms) if r["test_study"] == "B")
    assert row["status"] == "computed" and row["sensitivity"] > 0.9
    assert row["test_control_specificity"] == "NA"
    assert "no controls" in row["reason"]


def test_borrowed_training_controls_never_count_as_held_out_specificity():
    a = _study("A", 15, 15, seed=1)
    b = _study("B", 12, 0, seed=2)
    b.beta = pd.concat([b.beta, a.beta[a.control_ids]], axis=1)
    b.control_ids = list(a.control_ids)
    b.controls_borrowed_from = "A"
    # Both directions: A's controls are training controls whichever study is held out.
    for row in leave_one_study_out("Toy disorder", [a, b]):
        assert row["status"] == "computed"
        assert row["test_control_specificity"] == "NA" and row["n_test_controls"] == 0
        assert "training controls" in row["reason"]


def test_training_uses_the_strict_case_subset_but_testing_uses_all_cases():
    a = _study("A", 20, 15, seed=1)
    a.train_case_ids = a.train_case_ids[:12]
    rows = leave_one_study_out("Toy disorder", [a, _study("B", 12, 14, seed=2)])
    assert next(r for r in rows if r["test_study"] == "A")["n_test_cases"] == 20
    assert next(r for r in rows if r["train_study"] == "A")["n_train_cases"] == 12


def test_single_study_is_rejected():
    with pytest.raises(ValueError, match=">= 2 studies"):
        leave_one_study_out("Toy disorder", [_study("A", 15, 15)])


def test_frame_has_the_documented_columns():
    assert list(to_frame(leave_one_study_out("Toy disorder", _pair())).columns) == LOSO_COLUMNS
