"""Generalized episignature derivation and scoring for disorders without a published probe
list (i.e. anything other than Sotos syndrome, which uses the exact published signature -
see epigrade.signature.choufani).

This is the project spec's "Path B" procedure, generalized: Mann-Whitney U per probe, a
multiple-testing correction, plus a mean-beta-difference effect-size filter. The published
Sotos derivation additionally ran three "family-swap" trials to avoid inflating significance
from related family members contributing near-duplicate methylation profiles; that step is
Sotos-specific (it depended on the paper's own family relationship annotations) and is not
generalized here - this is a real methodological simplification, not an oversight, and is
documented as such in docs/METHODS.md and in each classifier's metadata.

Thresholds (which correction method, alpha, effect-size floor) are configurable per disorder via
config/signature_thresholds.yaml, not hardcoded - see that file for why a single fixed threshold
set tuned to Sotos's unusually large effect does not generalize to quieter signatures.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from epigrade import paths

MIN_CASES_TO_BUILD_CLASSIFIER = 10
# A signature with too few probes is fragile even when every probe in it is nominally
# "significant": score_samples requires this many valid probes per sample to compute a score at
# all (a 2-3 probe median-correlation score is dominated by individual outlier readings, nothing
# like Sotos's robust 7,085-probe signature). Found via a real failure: CHARGE syndrome
# (GSE97362) derived exactly 3 "significant" probes, which then produced zero usable sample
# scores and crashed the scoring step - not a fabricated classifier, but not a usable one
# either. Treated as equally underpowered as finding zero probes, not silently accepted.
MIN_SIGNATURE_SIZE = 10
MIN_PROBES_TO_SCORE = 10


@lru_cache(maxsize=1)
def _load_thresholds_config() -> dict:
    with open(paths.repo_root() / "config" / "signature_thresholds.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_thresholds(disorder: str | None) -> dict:
    """Returns {multiple_testing_method, alpha, effect_size_floor} for this disorder - the
    config's override if one exists, else the project default. See
    config/signature_thresholds.yaml for what these are and why."""
    cfg = _load_thresholds_config()
    return cfg["overrides"].get(disorder, cfg["default"])


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
    # Training samples and their betas at the signature probes. Kept so that a training sample
    # can be scored against median profiles rebuilt WITHOUT it (see score_matrix) - the
    # leakage guard. Empty for a classifier constructed without them, which then cannot be
    # leave-one-out scored.
    train_case_ids: tuple[str, ...] = ()
    train_control_ids: tuple[str, ...] = ()
    train_beta: pd.DataFrame | None = None
    # delta_beta / direction / p-values per signature probe, when derived here.
    probe_table: pd.DataFrame | None = None


def mannwhitney_p(case_vals: np.ndarray, control_vals: np.ndarray) -> np.ndarray:
    """Two-sided Mann-Whitney U p-value per probe (row), NaNs omitted per row.

    Always the asymptotic (normal approximation, tie- and continuity-corrected) method. scipy's
    default 'auto' picks the exact method only when a group has <= 8 samples AND the data are
    tie-free, and makes that choice per call - so the same probe could get a different p-value
    depending on which other probes shared the call. Every derivation this project has reported
    was asymptotic already (both groups > 8, or ties present); pinning it makes that explicit
    and independent of how rows are batched.

    Rows with no NaN go through one vectorised call. Only rows that actually contain a NaN take
    scipy's per-row omit path, which is what made whole-matrix nan_policy='omit' calls take
    minutes on a series with any missing values.
    """
    case_vals = np.asarray(case_vals, dtype=float)
    control_vals = np.asarray(control_vals, dtype=float)
    p_values = np.full(case_vals.shape[0], np.nan)
    has_nan = np.isnan(case_vals).any(axis=1) | np.isnan(control_vals).any(axis=1)
    clean = ~has_nan
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # rows left with an empty group after omitting NaNs
        if clean.any():
            p_values[clean] = mannwhitneyu(
                case_vals[clean], control_vals[clean], axis=1, alternative="two-sided",
                method="asymptotic",
            )[1]
        if has_nan.any():
            p_values[has_nan] = mannwhitneyu(
                case_vals[has_nan], control_vals[has_nan], axis=1, alternative="two-sided",
                method="asymptotic", nan_policy="omit",
            )[1]
    return p_values


def probe_tests(
    beta: pd.DataFrame, case_ids: list[str], control_ids: list[str],
) -> pd.DataFrame:
    """The threshold-free part of a Path B derivation, for every probe: Mann-Whitney p-value
    and mean beta difference. Computed once and then passed to derive_signature /
    derivation_stats / diagnose_underpowered / build_classifier via `tests=`, so one derivation
    costs one pass over the matrix instead of four."""
    case_vals = beta[case_ids].to_numpy(dtype=float)
    control_vals = beta[control_ids].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows -> NaN mean
        delta_beta = np.nanmean(case_vals, axis=1) - np.nanmean(control_vals, axis=1)
    return pd.DataFrame(
        {"p_value": mannwhitney_p(case_vals, control_vals), "delta_beta": delta_beta},
        index=beta.index,
    )


def _corrected(tests: pd.DataFrame, alpha: float, method: str) -> np.ndarray:
    p_values = tests["p_value"].to_numpy()
    valid = ~np.isnan(p_values)
    corrected_p = np.full(len(p_values), np.nan)
    if valid.any():
        corrected_p[valid] = multipletests(p_values[valid], alpha=alpha, method=method)[1]
    return corrected_p


def derive_signature(
    beta: pd.DataFrame, case_ids: list[str], control_ids: list[str],
    disorder: str | None = None, alpha: float | None = None,
    effect_size_floor: float | None = None, multiple_testing_method: str | None = None,
    tests: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Mann-Whitney U + a multiple-testing correction + an effect-size filter.

    Thresholds come from config/signature_thresholds.yaml for `disorder` unless overridden by
    the explicit keyword arguments (used by callers that already resolved thresholds themselves,
    e.g. to log them once rather than re-reading the config per call).

    Returns a DataFrame indexed by probe_id with columns p_value, corrected_p, delta_beta,
    direction - restricted to probes passing both the significance and effect-size filters.
    """
    thresholds = get_thresholds(disorder)
    alpha = thresholds["alpha"] if alpha is None else alpha
    effect_size_floor = (
        thresholds["effect_size_floor"] if effect_size_floor is None else effect_size_floor
    )
    method = (
        thresholds["multiple_testing_method"]
        if multiple_testing_method is None else multiple_testing_method
    )
    if tests is None:
        tests = probe_tests(beta, case_ids, control_ids)

    delta_beta = tests["delta_beta"].to_numpy()
    result = pd.DataFrame({
        "p_value": tests["p_value"].to_numpy(),
        "corrected_p": _corrected(tests, alpha, method),
        "delta_beta": delta_beta,
        "abs_delta_beta": np.abs(delta_beta),
        "direction": np.where(delta_beta < 0, "loss", "gain"),
    }, index=tests.index)

    sig = result[
        (result["corrected_p"] < alpha) & (result["abs_delta_beta"] > effect_size_floor)
    ]
    return sig.sort_values("corrected_p")


def diagnose_underpowered(
    beta: pd.DataFrame, case_ids: list[str], control_ids: list[str],
    disorder: str | None = None, tests: pd.DataFrame | None = None,
) -> str:
    """When derive_signature finds fewer than MIN_SIGNATURE_SIZE significant probes (zero, or a
    fragile handful), this distinguishes 'genuinely no signal' from 'underpowered' - the same
    distinction found by hand for Silver-Russell syndrome, Kabuki syndrome, and CHARGE syndrome
    during development. The key fact: with a Mann-Whitney U test, the smallest achievable
    p-value is bounded by the sample sizes themselves (roughly 1/C(n1+n2, min(n1,n2))) - a small
    control or case arm can make significance structurally unreachable, or barely reachable for
    only a few probes, even for a real, substantial effect - which is exactly the situation this
    reports rather than silently building a fragile few-probe classifier or nothing at all, with
    no explanation either way.
    """
    thresholds = get_thresholds(disorder)
    alpha = thresholds["alpha"]
    effect_size_floor = thresholds["effect_size_floor"]
    method = thresholds["multiple_testing_method"]

    if tests is None:
        tests = probe_tests(beta, case_ids, control_ids)
    sig = derive_signature(beta, case_ids, control_ids, disorder=disorder, tests=tests)
    n_found = len(sig)

    p_values = tests["p_value"].to_numpy()
    valid = ~np.isnan(p_values)
    corrected_p = _corrected(tests, alpha, method)
    min_p = float(np.nanmin(p_values))
    n_effect = int((tests["delta_beta"].abs() > effect_size_floor).sum())
    n_sig_only = int((corrected_p[valid] < alpha).sum())
    n_both = n_found  # derive_signature already requires both criteria together

    found_clause = (
        "found 0 significant probes" if n_found == 0
        else f"found only {n_found} significant probes (below the minimum reliable signature "
             f"size of {MIN_SIGNATURE_SIZE} - a classifier this thin would be dominated by a "
             "handful of individual probe readings, unlike Sotos's robust 7,085-probe signature)"
    )
    # Two distinct failure shapes worth telling apart: pure power-starvation (few/no probes
    # clear the correction at all, e.g. Silver-Russell/Kabuki), vs a significance/effect-size
    # mismatch (many probes ARE significant but with small, highly-consistent differences that
    # don't clear the effect-size bar, e.g. CHARGE under the old Bonferroni/20% thresholds) -
    # both are honestly "not a usable signature under this disorder's fixed criteria," but for
    # different underlying reasons worth distinguishing.
    if n_sig_only > MIN_SIGNATURE_SIZE and n_found < MIN_SIGNATURE_SIZE:
        return (
            f"Path B ({method}, alpha={alpha}, effect_size_floor={effect_size_floor:.0%}) "
            f"{found_clause} at n_case={len(case_ids)}, n_control={len(control_ids)}: of "
            f"{int(valid.sum())} tests, {n_sig_only} probes clear {method} alpha={alpha} on "
            f"their own and {n_effect} show >{effect_size_floor:.0%} effect size on their own, "
            f"but only {n_both} probes satisfy both criteria together - the significant "
            "differences are real but small, below the effect-size floor."
        )
    return (
        f"Path B ({method}, alpha={alpha}, effect_size_floor={effect_size_floor:.0%}) "
        f"{found_clause} at n_case={len(case_ids)}, n_control={len(control_ids)}: "
        f"{n_effect} probes show >{effect_size_floor:.0%} effect size, but the smallest "
        f"Mann-Whitney p-value ({min_p:.2e}) is close to but mostly does not reach "
        f"significance under {method} at this cohort size - a real statistical-power "
        "limitation, not evidence of no signal."
    )


def derivation_stats(
    beta: pd.DataFrame, case_ids: list[str], control_ids: list[str], disorder: str | None = None,
    tests: pd.DataFrame | None = None,
) -> dict:
    """One row of what was actually tried, for results/tables/signature_derivation.tsv: the
    exact thresholds resolved for `disorder`, and the resulting probe counts at each filter
    stage. Always computed fresh from the same Mann-Whitney + correction procedure
    build_classifier uses - never a number carried over or guessed from a previous run."""
    thresholds = get_thresholds(disorder)
    alpha = thresholds["alpha"]
    effect_size_floor = thresholds["effect_size_floor"]
    method = thresholds["multiple_testing_method"]

    if tests is None:
        tests = probe_tests(beta, case_ids, control_ids)
    valid = ~np.isnan(tests["p_value"].to_numpy())
    corrected_p = _corrected(tests, alpha, method)
    delta = tests["delta_beta"].abs().to_numpy()

    n_significant = int((corrected_p[valid] < alpha).sum())
    n_after_effect = int(((corrected_p < alpha) & (delta > effect_size_floor) & valid).sum())

    return {
        "disorder": disorder,
        "multiple_testing_method": method,
        "alpha": alpha,
        "effect_size_floor": effect_size_floor,
        "n_probes_significant": n_significant,
        "n_probes_after_effect_filter": n_after_effect,
        "n_cases": len(case_ids),
        "n_controls": len(control_ids),
        "signature_source": REDERIVED_NOT_PUBLISHED,
    }


def classifier_from_probes(
    disorder: str, probes: pd.Index, beta: pd.DataFrame, case_ids: list[str],
    control_ids: list[str], signature_source: str = REDERIVED_NOT_PUBLISHED,
    probe_table: pd.DataFrame | None = None,
) -> Classifier:
    """A classifier from an already-chosen probe list: median case/control profiles over the
    given training samples. `probes` absent from `beta` are dropped."""
    probes = pd.Index(probes).intersection(beta.index)
    train_beta = beta.loc[probes, list(case_ids) + list(control_ids)].astype(float)
    return Classifier(
        disorder=disorder,
        signature_probes=probes,
        median_case=train_beta[list(case_ids)].median(axis=1),
        median_control=train_beta[list(control_ids)].median(axis=1),
        n_case=len(case_ids),
        n_control=len(control_ids),
        signature_source=signature_source,
        train_case_ids=tuple(case_ids),
        train_control_ids=tuple(control_ids),
        train_beta=train_beta,
        probe_table=probe_table.loc[probes] if probe_table is not None else None,
    )


def build_classifier(
    disorder: str, beta: pd.DataFrame, case_ids: list[str], control_ids: list[str],
    tests: pd.DataFrame | None = None,
) -> Classifier | None:
    """Returns None (not a fabricated classifier) if there aren't enough cases, or if the
    derivation finds fewer than MIN_SIGNATURE_SIZE significant probes - both are reported as NA
    upstream, not forced. A handful of "significant" probes is not treated as a usable signature
    (see MIN_SIGNATURE_SIZE). Thresholds come from config/signature_thresholds.yaml for
    `disorder` - see get_thresholds()."""
    if len(case_ids) < MIN_CASES_TO_BUILD_CLASSIFIER:
        return None
    sig = derive_signature(beta, case_ids, control_ids, disorder=disorder, tests=tests)
    if len(sig) < MIN_SIGNATURE_SIZE:
        return None
    return classifier_from_probes(
        disorder, sig.index, beta, case_ids, control_ids, probe_table=sig,
    )


def _correlation_difference(x: np.ndarray, mc: np.ndarray, mn: np.ndarray) -> tuple[float, int]:
    """r(x, median_case) - r(x, median_control) over the probes where all three are present.
    Returns (nan, n_valid) below MIN_PROBES_TO_SCORE or when a profile is constant."""
    valid = ~(np.isnan(x) | np.isnan(mc) | np.isnan(mn))
    n_valid = int(valid.sum())
    if n_valid < MIN_PROBES_TO_SCORE:
        return float("nan"), n_valid
    x, mc, mn = x[valid], mc[valid], mn[valid]
    if x.std() == 0 or mc.std() == 0 or mn.std() == 0:
        return float("nan"), n_valid
    return float(np.corrcoef(x, mc)[0, 1] - np.corrcoef(x, mn)[0, 1]), n_valid


def score_matrix(
    clf: Classifier, beta: pd.DataFrame, sample_ids: list[str], leave_one_out: bool = True,
) -> pd.DataFrame:
    """Score samples with the correlation-difference rule. Returns a frame indexed by sample id
    with columns score, n_probes_used, loo_guarded; samples absent from `beta` or with fewer
    than MIN_PROBES_TO_SCORE usable probes are omitted.

    LEAKAGE GUARD: a sample that was used to build the classifier's median profiles is scored
    against profiles rebuilt without it (leave-one-out). Everything else is scored against the
    stored training profiles, which never contained it. `leave_one_out=False` disables the
    guard and exists only so tests can demonstrate what it prevents.

    The guard covers the median profiles only. Which probes are in the signature was still
    decided using every training sample, so a training sample's score remains optimistic
    relative to a truly unseen sample - self-scores are flagged, never presented as held-out.
    """
    probes = clf.signature_probes
    sample_ids = [s for s in dict.fromkeys(sample_ids) if s in beta.columns]
    x_all = beta.reindex(index=probes, columns=sample_ids).to_numpy(dtype=float)
    mc_full = clf.median_case.reindex(probes).to_numpy(dtype=float)
    mn_full = clf.median_control.reindex(probes).to_numpy(dtype=float)

    train_cases = list(clf.train_case_ids)
    train_controls = list(clf.train_control_ids)
    can_guard = leave_one_out and clf.train_beta is not None
    if can_guard:
        case_block = clf.train_beta[train_cases].to_numpy(dtype=float)
        control_block = clf.train_beta[train_controls].to_numpy(dtype=float)

    rows = {}
    for j, sid in enumerate(sample_ids):
        mc, mn, guarded = mc_full, mn_full, False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN probe rows
            if can_guard and sid in train_cases:
                mc = np.nanmedian(np.delete(case_block, train_cases.index(sid), axis=1), axis=1)
                guarded = True
            elif can_guard and sid in train_controls:
                mn = np.nanmedian(
                    np.delete(control_block, train_controls.index(sid), axis=1), axis=1,
                )
                guarded = True
        score, n_valid = _correlation_difference(x_all[:, j], mc, mn)
        if np.isnan(score):
            continue
        rows[sid] = {"score": score, "n_probes_used": n_valid, "loo_guarded": guarded}
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["score", "n_probes_used", "loo_guarded"],
    )


def score_samples(clf: Classifier, beta: pd.DataFrame, sample_ids: list[str]) -> pd.Series:
    """Same correlation-difference rule as the published Sotos classifier, generalized. No
    leave-one-out: only use this for samples the classifier was not built from (a held-out
    study, another disorder). Use score_matrix for anything that may include training samples.
    """
    scores = score_matrix(clf, beta, sample_ids, leave_one_out=False)["score"]
    return scores.rename("score")
