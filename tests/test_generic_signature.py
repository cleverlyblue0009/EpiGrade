"""Tests for the generalized (Path B) signature derivation and scoring, on synthetic data."""

import numpy as np
import pandas as pd

from epigrade.signature.generic import (
    MIN_CASES_TO_BUILD_CLASSIFIER,
    MIN_SIGNATURE_SIZE,
    build_classifier,
    derive_signature,
    diagnose_underpowered,
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


def test_build_classifier_refuses_a_fragile_few_probe_signature():
    """Regression test for a real bug found via Task 3: CHARGE syndrome derived exactly 3
    'significant' probes, which build_classifier previously accepted as a valid classifier -
    it then produced zero usable sample scores in score_samples (which requires >=10 valid
    probes per sample) and crashed the pipeline downstream. A handful of significant probes
    must be treated as equally underpowered as finding none, not silently accepted."""
    n_probes = 500
    base = RNG.uniform(0.2, 0.8, size=n_probes)
    shift = np.zeros(n_probes)
    shift[:3] = 0.35  # only 3 probes carry a real, strong effect - too few for a signature

    case_ids = [f"CASE{i}" for i in range(20)]
    control_ids = [f"CTRL{i}" for i in range(20)]
    data = {}
    for cid in case_ids:
        data[cid] = np.clip(base + shift + RNG.normal(0, 0.03, n_probes), 0, 1)
    for cid in control_ids:
        data[cid] = np.clip(base + RNG.normal(0, 0.03, n_probes), 0, 1)
    beta = pd.DataFrame(data, index=[f"cg{i:05d}" for i in range(n_probes)])

    sig = derive_signature(beta, case_ids, control_ids)
    assert 0 < len(sig) < MIN_SIGNATURE_SIZE, (
        "test setup should produce a nonzero but sub-minimum signature - adjust the synthetic "
        f"effect if this fails (got {len(sig)} probes)"
    )
    clf = build_classifier("Fragile disorder", beta, case_ids, control_ids)
    assert clf is None, "a signature below MIN_SIGNATURE_SIZE must not become a classifier"


def test_diagnose_underpowered_reports_the_actual_probe_count():
    beta, case_ids, control_ids = _synthetic(n_case=8, n_control=8, effect=0.15, noise=0.2)
    # MIN_CASES_TO_BUILD_CLASSIFIER would already refuse this, but diagnose_underpowered should
    # still run standalone and report a sane, non-crashing message either way.
    msg = diagnose_underpowered(beta, case_ids, control_ids)
    assert isinstance(msg, str) and len(msg) > 0
    assert str(len(case_ids)) in msg and str(len(control_ids)) in msg


def test_default_thresholds_are_fdr_bh_not_bonferroni():
    """The project-wide default (config/signature_thresholds.yaml), set once before retrying
    Silver-Russell syndrome, not loosened afterward based on outcome."""
    from epigrade.signature.generic import get_thresholds

    default = get_thresholds("Some disorder with no override")
    assert default["multiple_testing_method"] == "fdr_bh"
    assert default["effect_size_floor"] == 0.10


def test_sotos_thresholds_are_pinned_to_the_published_bonferroni_ones():
    """Sotos's production path never calls this module at all (it uses the published probe
    list - see epigrade.signature.choufani) - this only guards a from-scratch Path B
    re-derivation of Sotos as a validation exercise, so it reproduces what the paper actually
    did rather than picking up the new project-wide defaults."""
    from epigrade.signature.generic import get_thresholds

    sotos = get_thresholds("Sotos syndrome")
    assert sotos["multiple_testing_method"] == "bonferroni"
    assert sotos["effect_size_floor"] == 0.20


def test_configurable_thresholds_change_derive_signature_output():
    """A disorder-specific override actually takes effect, not just exists in the config."""
    beta, case_ids, control_ids = _synthetic(n_case=15, n_control=20, effect=0.35, noise=0.1)
    loose = derive_signature(
        beta, case_ids, control_ids, multiple_testing_method="fdr_bh", alpha=0.05,
        effect_size_floor=0.10,
    )
    strict = derive_signature(
        beta, case_ids, control_ids, multiple_testing_method="bonferroni", alpha=0.05,
        effect_size_floor=0.20,
    )
    assert len(loose) >= len(strict), (
        "FDR-BH + a lower effect floor should never recover FEWER probes than Bonferroni + a "
        "higher one on the same data"
    )
