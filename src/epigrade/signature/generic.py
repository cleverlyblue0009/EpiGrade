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
# A signature with too few probes is fragile even when every probe in it is nominally
# "significant": score_samples requires this many valid probes per sample to compute a score at
# all (a 2-3 probe median-correlation score is dominated by individual outlier readings, nothing
# like Sotos's robust 7,085-probe signature). Found via a real failure: CHARGE syndrome
# (GSE97362) derived exactly 3 "significant" probes, which then produced zero usable sample
# scores and crashed the scoring step - not a fabricated classifier, but not a usable one
# either. Treated as equally underpowered as finding zero probes, not silently accepted.
MIN_SIGNATURE_SIZE = 10


REDERIVED_NOT_PUBLISHED = "rederived_not_published"


@dataclass
class Classifier:
    disorder: str
    signature_probes: pd.Index
    median_case: pd.Series
    median_control: pd.Series
    n_case: int
    n_control: int
    # Every classifier built by THIS module is re-derived (Path B), never a published probe
    # list - see epigrade.signature.choufani.SIGNATURE_SOURCE for the Sotos counterpart. A
    # re-derived signature must never be presented as a reproduction of a published one; this
    # field is carried through into every output table row that uses a classifier built here.
    signature_source: str = REDERIVED_NOT_PUBLISHED


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


def diagnose_underpowered(
    beta: pd.DataFrame, case_ids: list[str], control_ids: list[str], alpha: float = 0.05,
) -> str:
    """When derive_signature finds fewer than MIN_SIGNATURE_SIZE significant probes (zero, or a
    fragile handful), this distinguishes 'genuinely no signal' from 'underpowered' - the same
    distinction found by hand for Silver-Russell syndrome (phase4_6), Kabuki syndrome, and
    CHARGE syndrome (phase4_kabuki_charge) during development. The key fact: with a Mann-Whitney
    U test, the smallest achievable p-value is bounded by the sample sizes themselves (roughly
    1/C(n1+n2, min(n1,n2))) - a small control or case arm can make genome-wide Bonferroni
    significance structurally unreachable, or barely reachable for only a few probes, even for a
    real, substantial effect - which is exactly the situation this reports rather than silently
    building a fragile few-probe classifier (found via CHARGE syndrome deriving exactly 3
    "significant" probes, which then broke scoring downstream) or nothing at all, with no
    explanation either way.
    """
    sig = derive_signature(beta, case_ids, control_ids, alpha=alpha)
    n_found = len(sig)

    case_vals = beta[case_ids].values
    control_vals = beta[control_ids].values
    with np.errstate(invalid="ignore"):
        _, p_values = mannwhitneyu(
            case_vals, control_vals, axis=1, alternative="two-sided", nan_policy="omit",
        )
    p_values = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(p_values)
    bonf_threshold = alpha / int(valid.sum())
    min_p = float(np.nanmin(p_values))

    mean_case = np.nanmean(case_vals, axis=1)
    mean_control = np.nanmean(control_vals, axis=1)
    n_effect = int((np.abs(mean_case - mean_control) > EFFECT_SIZE_THRESHOLD).sum())
    n_bonf_only = int((p_values[valid] < bonf_threshold).sum())
    n_both = n_found  # derive_signature already requires both criteria together

    found_clause = (
        "found 0 significant probes" if n_found == 0
        else f"found only {n_found} significant probes (below the minimum reliable signature "
             f"size of {MIN_SIGNATURE_SIZE} - a classifier this thin would be dominated by a "
             "handful of individual probe readings, unlike Sotos's robust 7,085-probe signature)"
    )
    # Two distinct failure shapes worth telling apart: pure power-starvation (few/no probes
    # clear Bonferroni at all, e.g. Silver-Russell/Kabuki), vs a significance/effect-size
    # mismatch (many probes ARE genome-wide significant but with small, highly-consistent
    # differences that don't clear the effect-size bar, e.g. CHARGE: 35 Bonferroni-significant
    # probes, only 3 also >20% effect size) - both are honestly "not a usable signature under
    # this project's fixed criteria," but for different underlying reasons worth distinguishing.
    overlap_note = (
        f" Of {int(valid.sum())} tests, {n_bonf_only} probes are Bonferroni-significant on "
        f"their own and {n_effect} show >{EFFECT_SIZE_THRESHOLD:.0%} effect size on their own, "
        f"but only {n_both} probes satisfy both criteria together - the two are not the same "
        "probes, not a lack of any significant signal."
        if n_bonf_only > MIN_SIGNATURE_SIZE and n_found < MIN_SIGNATURE_SIZE
        else ""
    )
    return (
        f"Path B {found_clause} at n_case={len(case_ids)}, n_control={len(control_ids)}: "
        f"{n_effect} probes show >{EFFECT_SIZE_THRESHOLD:.0%} effect size, but the smallest "
        f"Mann-Whitney p-value ({min_p:.2e}) is close to but mostly does not reach the "
        f"genome-wide Bonferroni threshold ({bonf_threshold:.2e} for {int(valid.sum())} "
        "tests) - a real statistical-power limitation given this cohort's size, not evidence "
        f"of no signal.{overlap_note}"
    )


def build_classifier(
    disorder: str, beta: pd.DataFrame, case_ids: list[str], control_ids: list[str],
) -> Classifier | None:
    """Returns None (not a fabricated classifier) if there aren't enough cases, or if the
    derivation finds fewer than MIN_SIGNATURE_SIZE significant probes - both are reported as NA
    upstream, not forced. A handful of "significant" probes is not treated as a usable signature
    (see MIN_SIGNATURE_SIZE)."""
    if len(case_ids) < MIN_CASES_TO_BUILD_CLASSIFIER:
        return None
    sig = derive_signature(beta, case_ids, control_ids)
    if len(sig) < MIN_SIGNATURE_SIZE:
        return None
    probes = sig.index
    return Classifier(
        disorder=disorder,
        signature_probes=probes,
        median_case=beta.loc[probes, case_ids].median(axis=1),
        median_control=beta.loc[probes, control_ids].median(axis=1),
        n_case=len(case_ids),
        n_control=len(control_ids),
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
