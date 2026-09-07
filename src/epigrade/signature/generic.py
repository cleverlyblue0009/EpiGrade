"""Generalized episignature derivation and scoring for disorders without a published probe
list (i.e. anything other than Sotos syndrome, which uses the exact published signature -
see epigrade.signature.choufani).

This is the project spec's "Path B" procedure, simplified for the general case: Mann-Whitney U
per probe, Bonferroni-corrected, plus a mean-beta-difference effect-size filter. The published
Sotos derivation additionally ran three "family-swap" trials to avoid inflating significance
from related family members contributing near-duplicate methylation profiles; that step is
Sotos-specific (it depended on the paper's own family relationship annotations) and is not
generalized here - this is a real methodological simplification, not an oversight, and is
documented as such in docs/METHODS.md and in each classifier's metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr
from statsmodels.stats.multitest import multipletests

MIN_CASES_TO_BUILD_CLASSIFIER = 10
EFFECT_SIZE_THRESHOLD = 0.20  # >20% mean beta difference, matching the Sotos derivation


@dataclass
class Classifier:
    disorder: str
    signature_probes: pd.Index
    median_case: pd.Series
    median_control: pd.Series
    n_case: int
    n_control: int
    derivation: str  # "published" (Sotos) or "derived_path_b" (everything else)


def derive_signature(
    beta: pd.DataFrame, case_ids: list[str], control_ids: list[str], alpha: float = 0.05,
) -> pd.DataFrame:
    """Mann-Whitney U + Bonferroni + >20% mean-beta-difference effect-size filter.

    Returns a DataFrame indexed by probe_id with columns p_value, bonferroni_p, delta_beta,
    direction - restricted to probes passing both the significance and effect-size filters.
    """
    case_vals = beta[case_ids].values
    control_vals = beta[control_ids].values
    n_probes = beta.shape[0]

    # Vectorized across all probes at once (scipy's nan_policy='omit' handles per-row
    # missingness) - a plain Python loop over ~400K+ probes would take far too long.
    with np.errstate(invalid="ignore"):
        _, p_values = mannwhitneyu(
            case_vals, control_vals, axis=1, alternative="two-sided", nan_policy="omit",
        )
    p_values = np.asarray(p_values, dtype=float)
    # rows where either group had too few non-NaN values come back NaN from scipy already
    valid = ~np.isnan(p_values)
    bonferroni_p = np.full(n_probes, np.nan)
    bonferroni_p[valid] = multipletests(p_values[valid], alpha=alpha, method="bonferroni")[1]

    mean_case = np.nanmean(case_vals, axis=1)
    mean_control = np.nanmean(control_vals, axis=1)
    delta_beta = mean_case - mean_control

    result = pd.DataFrame({
        "p_value": p_values,
        "bonferroni_p": bonferroni_p,
        "delta_beta": delta_beta,
        "abs_delta_beta": np.abs(delta_beta),
        "direction": np.where(delta_beta < 0, "loss", "gain"),
    }, index=beta.index)

    sig = result[
        (result["bonferroni_p"] < alpha) & (result["abs_delta_beta"] > EFFECT_SIZE_THRESHOLD)
    ]
    return sig.sort_values("bonferroni_p")


def build_classifier(
    disorder: str, beta: pd.DataFrame, case_ids: list[str], control_ids: list[str],
) -> Classifier | None:
    """Returns None (not a fabricated classifier) if there aren't enough cases, or if the
    derivation finds zero significant probes - both are reported as NA upstream, not forced."""
    if len(case_ids) < MIN_CASES_TO_BUILD_CLASSIFIER:
        return None
    sig = derive_signature(beta, case_ids, control_ids)
    if len(sig) == 0:
        return None
    probes = sig.index
    return Classifier(
        disorder=disorder,
        signature_probes=probes,
        median_case=beta.loc[probes, case_ids].median(axis=1),
        median_control=beta.loc[probes, control_ids].median(axis=1),
        n_case=len(case_ids),
        n_control=len(control_ids),
        derivation="derived_path_b",
    )


def score_samples(clf: Classifier, beta: pd.DataFrame, sample_ids: list[str]) -> pd.Series:
    """Same correlation-difference rule as the published Sotos classifier, generalized."""
    probes = clf.signature_probes.intersection(beta.index)
    scores = {}
    for sid in sample_ids:
        if sid not in beta.columns:
            continue
        x = beta.loc[probes, sid]
        mc = clf.median_case.loc[probes]
        mn = clf.median_control.loc[probes]
        valid = x.notna() & mc.notna() & mn.notna()
        if valid.sum() < 10:
            continue
        r_case = pearsonr(x[valid], mc[valid])[0]
        r_control = pearsonr(x[valid], mn[valid])[0]
        scores[sid] = r_case - r_control
    return pd.Series(scores, name="score")
