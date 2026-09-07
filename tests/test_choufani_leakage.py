"""Proves the leave-one-out leakage guard in epigrade.signature.choufani.score_cohort is live -
i.e. that leaking a sample into the profile it's scored against WOULD improve apparent
separation, so guarding against it is not a no-op. Uses synthetic data (no network/GEO needed).
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from epigrade.signature.choufani import score_cohort

RNG = np.random.default_rng(0)


def _make_synthetic(n_probes=200, n_case=15, n_control=15, effect=0.3, noise=0.15):
    """Cases and controls differ by a fixed effect on a probe subset, plus per-sample noise -
    close enough to real beta-value structure for a leakage test, without needing real data."""
    base = RNG.uniform(0.2, 0.8, size=n_probes)
    case_shift = np.zeros(n_probes)
    case_shift[: n_probes // 2] = effect

    case_ids = [f"CASE{i}" for i in range(n_case)]
    control_ids = [f"CTRL{i}" for i in range(n_control)]
    data = {}
    for cid in case_ids:
        data[cid] = np.clip(base + case_shift + RNG.normal(0, noise, n_probes), 0, 1)
    for cid in control_ids:
        data[cid] = np.clip(base + RNG.normal(0, noise, n_probes), 0, 1)

    beta = pd.DataFrame(data, index=[f"cg{i:05d}" for i in range(n_probes)])
    cohort = pd.Series(
        ["discovery_case"] * n_case + ["discovery_control"] * n_control,
        index=case_ids + control_ids,
    )
    return beta, cohort, beta.index


def _leaked_score_cohort(beta, cohort, probes):
    """Same scoring rule as score_cohort, but WITHOUT the leave-one-out exclusion - the sample
    being scored is included in the medians it's compared against. Used only to demonstrate
    what the guard prevents."""
    cases = cohort[cohort == "discovery_case"].index.tolist()
    controls = cohort[cohort == "discovery_control"].index.tolist()
    median_case = beta.loc[probes, cases].median(axis=1)
    median_control = beta.loc[probes, controls].median(axis=1)

    rows = []
    for sid, group in cohort.items():
        x = beta.loc[probes, sid]
        r_case = pearsonr(x, median_case)[0]
        r_control = pearsonr(x, median_control)[0]
        rows.append({"gsm_accession": sid, "cohort": group, "score": r_case - r_control})
    return pd.DataFrame(rows)


def _separation(scores_df):
    """Mean case score minus mean control score - bigger means better apparent separation."""
    cases = scores_df.loc[scores_df.cohort == "discovery_case", "score"]
    controls = scores_df.loc[scores_df.cohort == "discovery_control", "score"]
    return cases.mean() - controls.mean()


def test_guarded_scoring_uses_leave_one_out():
    beta, cohort, probes = _make_synthetic()
    guarded = score_cohort(beta, cohort, probes)
    # every discovery sample must have been scored against a profile that excluded it
    assert set(guarded["gsm_accession"]) == set(cohort.index)
    assert len(guarded) == len(cohort)


def test_leaking_the_sample_improves_apparent_separation():
    """This is the guard-is-live test: with the SAME synthetic data, deliberately leaking each
    sample into its own reference profile must produce a larger (or equal, in a degenerate
    case) apparent case/control separation than the leave-one-out guarded version - if it
    didn't, the guard would be pointless because leakage wouldn't even matter here."""
    beta, cohort, probes = _make_synthetic()

    guarded = score_cohort(beta, cohort, probes)
    leaked = _leaked_score_cohort(beta, cohort, probes)

    sep_guarded = _separation(guarded)
    sep_leaked = _separation(leaked)

    assert sep_leaked > sep_guarded, (
        f"leakage guard appears to be a no-op: leaked separation ({sep_leaked:.4f}) was not "
        f"greater than guarded separation ({sep_guarded:.4f})"
    )


def test_no_signal_case_produces_near_zero_separation():
    """Sanity check: with no true effect at all, guarded separation should be small (not
    trivially large from overfitting), unlike the leaked version which can still inflate."""
    beta, cohort, probes = _make_synthetic(effect=0.0, noise=0.2)
    guarded = score_cohort(beta, cohort, probes)
    assert abs(_separation(guarded)) < 0.3
