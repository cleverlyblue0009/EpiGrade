"""Tests for the generalized (Path B) signature derivation and scoring, on synthetic data."""

import numpy as np
import pandas as pd

from epigrade.signature.generic import (
    MIN_CASES_TO_BUILD_CLASSIFIER,
    build_classifier,
    derive_signature,
    score_samples,
)

RNG = np.random.default_rng(1)


def _synthetic(n_probes=500, n_case=15, n_control=20, effect=0.35, noise=0.1):
    base = RNG.uniform(0.2, 0.8, size=n_probes)
    shift = np.zeros(n_probes)
    shift[:50] = effect  # only the first 50 probes carry real signal

    case_ids = [f"CASE{i}" for i in range(n_case)]
    control_ids = [f"CTRL{i}" for i in range(n_control)]
    data = {}
    for cid in case_ids:
        data[cid] = np.clip(base + shift + RNG.normal(0, noise, n_probes), 0, 1)
    for cid in control_ids:
        data[cid] = np.clip(base + RNG.normal(0, noise, n_probes), 0, 1)
    beta = pd.DataFrame(data, index=[f"cg{i:05d}" for i in range(n_probes)])
    return beta, case_ids, control_ids


def test_derive_signature_recovers_the_true_effect_probes():
    beta, case_ids, control_ids = _synthetic()
    sig = derive_signature(beta, case_ids, control_ids)
    assert len(sig) > 0
    recovered_true_positives = sig.index.str.replace("cg", "").astype(int) < 50
    # the vast majority of recovered probes should be from the true-effect block
    assert recovered_true_positives.mean() > 0.8


def test_build_classifier_returns_none_below_min_cases():
    beta, case_ids, control_ids = _synthetic(n_case=MIN_CASES_TO_BUILD_CLASSIFIER - 1)
    clf = build_classifier("Toy disorder", beta, case_ids, control_ids)
    assert clf is None


def test_build_classifier_and_score_separates_cases_from_controls():
    beta, case_ids, control_ids = _synthetic()
    clf = build_classifier("Toy disorder", beta, case_ids, control_ids)
    assert clf is not None
    assert clf.n_case == len(case_ids)

    scores = score_samples(clf, beta, case_ids + control_ids)
    case_scores = scores[case_ids]
    control_scores = scores[control_ids]
    assert case_scores.mean() > control_scores.mean()
