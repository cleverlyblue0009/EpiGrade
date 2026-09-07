"""Score-to-evidence calibration. Knows nothing about methylation - takes only
(scores, labels, study_ids, prior). Local-likelihood-ratio calibration is Pejaver-style
(Pejaver et al. 2022, AJHG); the evidence-point conversion is Tavtigian et al. 2018's Bayesian
framework. Neither is claimed as novel here - see docs/METHODS.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml
from scipy.stats import gaussian_kde

from epigrade import paths

MIN_N_PER_CLASS = 5  # adaptive-window minimum; small because several disorders here are tiny


def _load_config() -> dict:
    with open(paths.repo_root() / "config" / "evidence_bands.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class EvidenceResult:
    disorder: str
    query_score: float
    lr_point_estimate: float
    lr_ci_low: float
    lr_ci_high: float
    points: float  # from the conservative (lower) bound
    band: str
    prior: float
    posterior_prob: float
    evidence_anchor: str
    n_case: int
    n_control: int
    n_studies_case: int
    n_studies_control: int
    confounded_by_design: bool
    reason: str
    # Within-study (sample-level, not study-level) bootstrap - a diagnostic only. Never used to
    # assign a band, and never a substitute for the between-study CI: it treats every sample as
    # independent, which is exactly the assumption study-level resampling exists to avoid.
    within_study_ci_low: float = float("nan")
    within_study_ci_high: float = float("nan")


def _knn_radius(sorted_scores: np.ndarray, query_score: float, k: int) -> float:
    """Smallest radius around query_score whose window contains >= k of sorted_scores."""
    distances = np.sort(np.abs(sorted_scores - query_score))
    k = min(k, len(distances))
    return float(distances[k - 1]) if k > 0 else float("nan")


def local_likelihood_ratio(
    scores: np.ndarray, labels: np.ndarray, query_score: float, min_n: int = MIN_N_PER_CLASS,
) -> float:
    """Adaptive-interval local LR, with an INDEPENDENT bandwidth per class: the radius needed
    to reach min_n of that class's own nearest scores to query_score. This matters whenever the
    two classes are cleanly separated with a gap (exactly what a *good* classifier produces): a
    single shared, symmetrically-widened window can jump straight from "0 of the far class" to
    "100% of both classes" in one step once it finally bridges the gap, which degenerates the
    ratio to a spurious ~1 (both classes then equally "fully included") regardless of how
    strong the true local separation is - the opposite of the intended behavior. Estimating
    each class's local density on its own natural scale avoids that: cases living in a tight
    cluster near the query get a small radius (high density), while a sparse or distant control
    tail gets a larger radius (low density), producing a large LR when local separation really
    is large. Dividing by each class's own total count makes this self-normalizing regardless
    of the raw case:control sampling ratio in the data (the "deposition artifact" the project
    spec warns about) - it is not corrected for separately, it is structurally absent here.
    """
    case_scores = scores[labels == 1]
    control_scores = scores[labels == 0]
    if len(case_scores) == 0 or len(control_scores) == 0:
        return float("nan")

    r_case = _knn_radius(case_scores, query_score, min_n)
    r_control = _knn_radius(control_scores, query_score, min_n)
    if np.isnan(r_case) or np.isnan(r_control):
        return float("nan")

    # a zero radius (>= min_n scores exactly at the query point) would divide by zero; floor it
    # at a small fraction of the overall score range instead of fabricating a result.
    floor = max((scores.max() - scores.min()) * 1e-6, 1e-9)
    r_case = max(r_case, floor)
    r_control = max(r_control, floor)

    case_density = min(min_n, len(case_scores)) / len(case_scores) / (2 * r_case)
    control_density = min(min_n, len(control_scores)) / len(control_scores) / (2 * r_control)
    if control_density == 0:
        return float("inf")
    return case_density / control_density


def kde_likelihood_ratio(scores: np.ndarray, labels: np.ndarray, query_score: float) -> float:
    """Alternative estimator: separate Gaussian KDE per class. More defensible in dense score
    regions; the adaptive interval estimator is preferred at score extremes where KDE tails are
    unreliable (this is exactly the tradeoff the project spec calls out)."""
    case_scores = scores[labels == 1]
    control_scores = scores[labels == 0]
    if len(case_scores) < 2 or len(control_scores) < 2:
        return float("nan")
    try:
        case_kde = gaussian_kde(case_scores)
        control_kde = gaussian_kde(control_scores)
    except np.linalg.LinAlgError:
        return float("nan")  # e.g. zero-variance class - do not fabricate a number
    control_density = control_kde(query_score)[0]
    if control_density == 0:
        return float("inf")
    return case_kde(query_score)[0] / control_density


def lr_to_points(lr: float, base: float) -> float:
    if lr <= 0 or np.isnan(lr):
        return float("nan")
    if np.isinf(lr):
        return float("inf")
    return float(np.log(lr) / np.log(base))


def points_to_band(points: float, bands: list[dict]) -> str:
    if np.isnan(points):
        return "NA"
    abs_points = abs(points)
    for band in bands:  # config is ordered highest threshold first
        if band["min_points"] <= 0:
            continue
        if abs_points >= band["min_points"]:
            return band["name"]
    return "No evidence"


def bootstrap_lr_ci(
    df: pd.DataFrame, query_score: float, estimator=local_likelihood_ratio,
    n_boot: int | None = None,
) -> tuple[float, float, float, bool]:
    """Bootstrap the LR's confidence interval, resampling at STUDY level - samples within a
    study are not independent. Returns (point_estimate, ci_low, ci_high, confounded_by_design).

    `df` must have columns: score, label (1=case, 0=control), study_id.

    If every case comes from a single study, study-level resampling degenerates to resampling
    one study against itself - it cannot test between-study generalization at all. This is
    flagged confounded_by_design=True (see epigrade.audit) rather than silently reported as if
    it were a real between-study confidence interval.
    """
    cfg = _load_config()["bootstrap"]
    n_boot = cfg["n_boot"] if n_boot is None else n_boot
    lo_pct, hi_pct = cfg["ci_percentiles"]

    case_studies = df.loc[df.label == 1, "study_id"].unique()
    control_studies = df.loc[df.label == 0, "study_id"].unique()
    # Confounded (degenerate) whenever EITHER side has fewer than 2 studies: rng.choice on a
    # size-1 array returns that same single study on every replicate, so every bootstrap
    # resample is identical to the original data and the "interval" has zero width by
    # construction - not because the estimate is precise, but because the resampling procedure
    # cannot vary at all. This was previously checked on the case side only; a single-study
    # control arm degenerates the same way and was missed.
    confounded = len(case_studies) < 2 or len(control_studies) < 2

    point = estimator(df["score"].values, df["label"].values, query_score)

    if confounded:
        # Do not run a bootstrap that can only ever reproduce the same point estimate 1000
        # times - that is not a confidence interval, it is the point estimate wearing a
        # disguise. Report it plainly as unavailable instead.
        return point, float("nan"), float("nan"), confounded

    rng = np.random.default_rng(0)
    boot_lrs = []
    for _ in range(n_boot):
        sampled_case_studies = rng.choice(case_studies, size=len(case_studies), replace=True)
        sampled_control_studies = rng.choice(
            control_studies, size=len(control_studies), replace=True
        )
        parts = []
        for i, s in enumerate(sampled_case_studies):
            sub = df[(df.study_id == s) & (df.label == 1)].copy()
            sub["study_id"] = f"case_resample_{i}"
            parts.append(sub)
        for i, s in enumerate(sampled_control_studies):
            sub = df[(df.study_id == s) & (df.label == 0)].copy()
            sub["study_id"] = f"control_resample_{i}"
            parts.append(sub)
        boot_df = pd.concat(parts, ignore_index=True)
        lr = estimator(boot_df["score"].values, boot_df["label"].values, query_score)
        if not np.isnan(lr) and not np.isinf(lr):
            boot_lrs.append(lr)

    if len(boot_lrs) < n_boot * 0.5:
        return point, float("nan"), float("nan"), confounded

    ci_low, ci_high = np.percentile(boot_lrs, [lo_pct, hi_pct])
    return point, float(ci_low), float(ci_high), confounded


def within_study_bootstrap_ci(
    df: pd.DataFrame, query_score: float, estimator=local_likelihood_ratio,
    n_boot: int = 500,
) -> tuple[float, float]:
    """Sample-level (not study-level) bootstrap - a DIAGNOSTIC ONLY, reported for reference
    alongside a degenerate between-study CI, never used to assign an evidence band. It treats
    every sample as independent, which is exactly the assumption study-level resampling exists
    to guard against - a tight within-study interval says only "this dataset is internally
    consistent," not "this would generalize to a new study."
    """
    rng = np.random.default_rng(1)
    scores = df["score"].values
    labels = df["label"].values
    case_idx = np.where(labels == 1)[0]
    control_idx = np.where(labels == 0)[0]
    if len(case_idx) == 0 or len(control_idx) == 0:
        return float("nan"), float("nan")

    boot_lrs = []
    for _ in range(n_boot):
        boot_case = rng.choice(case_idx, size=len(case_idx), replace=True)
        boot_control = rng.choice(control_idx, size=len(control_idx), replace=True)
        boot_scores = np.concatenate([scores[boot_case], scores[boot_control]])
        boot_labels = np.concatenate([labels[boot_case], labels[boot_control]])
        lr = estimator(boot_scores, boot_labels, query_score)
        if not np.isnan(lr) and not np.isinf(lr):
            boot_lrs.append(lr)

    if len(boot_lrs) < n_boot * 0.5:
        return float("nan"), float("nan")
    lo, hi = np.percentile(boot_lrs, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate_score(
    disorder: str, df: pd.DataFrame, query_score: float, prior: float | None = None,
    n_boot: int | None = None,
) -> EvidenceResult:
    """Full pipeline: bootstrap LR CI -> conservative-bound points -> band -> posterior at prior.

    The band is assigned from the CONSERVATIVE bound (the end of the CI closer to LR=1), never
    the point estimate. If the CI spans LR=1, the result is forced to "No evidence" regardless
    of the point estimate or the other bound.
    """
    cfg = _load_config()
    prior = cfg["default_prior"] if prior is None else prior
    base = cfg["odds_path_base"]

    point, ci_low, ci_high, confounded = bootstrap_lr_ci(df, query_score, n_boot=n_boot)
    within_lo, within_hi = within_study_bootstrap_ci(df, query_score)

    reason = ""
    if confounded:
        # Refuse to assign a band at all: with fewer than 2 studies on the case or control
        # side, every study-level bootstrap resample is identical to the original data (see
        # bootstrap_lr_ci), so the "interval" has zero width by construction, not because the
        # estimate is precise. Assigning a band from that would misrepresent a single-study
        # point estimate as a validated confidence bound. The within-study CI above is reported
        # for reference only and must never be used here.
        band = "NA"
        conservative_points = float("nan")
        reason = (
            "single-study cohort: study-level bootstrap is degenerate (all resamples "
            "identical), so no between-study confidence bound can be estimated"
        )
    elif np.isnan(ci_low) or np.isnan(ci_high):
        band = "NA"
        conservative_points = float("nan")
        reason = "bootstrap failed to converge (too few resamples produced a valid estimator)"
    elif ci_low <= 1.0 <= ci_high:
        band = "No evidence"
        conservative_points = 0.0
        reason = "bootstrap CI spans LR=1 (no evidence, regardless of the point estimate)"
    else:
        # conservative = whichever bound is closer to 1 (i.e. the weaker-evidence end)
        conservative_lr = ci_low if point >= 1 else ci_high
        conservative_points = lr_to_points(conservative_lr, base)
        band = points_to_band(conservative_points, cfg["bands"])
        reason = "band assigned from the bootstrap CI bound closer to LR=1 (conservative)"

    prior_odds = prior / (1 - prior)
    point_is_usable = not np.isnan(point) and not np.isinf(point)
    posterior_odds = prior_odds * point if point_is_usable else float("nan")
    posterior_prob = (
        posterior_odds / (1 + posterior_odds)
        if not np.isnan(posterior_odds) else float("nan")
    )

    anchor = cfg["evidence_anchor"]
    assert anchor not in cfg["forbidden_anchors"]

    return EvidenceResult(
        disorder=disorder,
        query_score=query_score,
        lr_point_estimate=point,
        lr_ci_low=ci_low,
        lr_ci_high=ci_high,
        points=conservative_points,
        band=band,
        prior=prior,
        posterior_prob=posterior_prob,
        evidence_anchor=anchor,
        n_case=int((df.label == 1).sum()),
        n_control=int((df.label == 0).sum()),
        n_studies_case=df.loc[df.label == 1, "study_id"].nunique(),
        n_studies_control=df.loc[df.label == 0, "study_id"].nunique(),
        confounded_by_design=confounded,
        reason=reason,
        within_study_ci_low=within_lo,
        within_study_ci_high=within_hi,
    )


def evidence_ceiling(n_case: int, n_control: int, n_studies_case: int = 1,
                      query_at: str = "case_median", n_repeats: int = 5,
                      n_boot: int = 200) -> EvidenceResult:
    """What's the best evidence strength ANY classifier could support at this cohort size,
    independent of classifier quality? Simulates a PERFECT classifier (completely
    non-overlapping score distributions) at the given n, runs it through the identical
    bootstrap+banding pipeline, and reports the resulting band as the ceiling. If even a
    perfect classifier can't clear "Supporting" at some n, no real classifier can either -
    that's a statement about cohort size, not about any particular disorder's biology.

    Averaged over `n_repeats` independent random draws (median of the point/CI estimates): a
    single draw's local-density estimate is noisy enough at small n that the ceiling can
    otherwise look non-monotonic in n purely from simulation luck, which would misrepresent
    what is fundamentally a smooth relationship between cohort size and attainable evidence.

    Uses a reduced bootstrap (`n_boot=200` vs. the spec-compliant 1000 used everywhere real
    scores are evaluated in epigrade.calibrate) since this is an illustrative simulation run
    `n_repeats` times, not a primary reported result - 5x200 gives a comparably stable estimate
    to 1x1000 at a fraction of the cost. Real disorder evidence (see scripts/phase5_calibration.py)
    always uses the full config-specified n_boot.
    """
    results = []
    for seed in range(n_repeats):
        rng = np.random.default_rng(1000 + seed)
        case_scores = rng.normal(1.0, 0.05, n_case)
        control_scores = rng.normal(-1.0, 0.05, n_control)
        studies_case = [f"case_study{i % n_studies_case}" for i in range(n_case)]
        # Controls must vary across at least as many studies as cases, or the bootstrap is
        # degenerate on the control side even when cases span multiple studies (bootstrap_lr_ci
        # now checks both sides - see Task 1's fix).
        studies_control = [f"control_study{i % n_studies_case}" for i in range(n_control)]

        df = pd.DataFrame({
            "score": np.concatenate([case_scores, control_scores]),
            "label": [1] * n_case + [0] * n_control,
            "study_id": studies_case + studies_control,
        })
        query = float(np.median(case_scores)) if query_at == "case_median" else 0.0
        results.append(
            evaluate_score(f"PERFECT_CLASSIFIER_n{n_case}", df, query, n_boot=n_boot)
        )

    def _median_field(name):
        vals = [getattr(r, name) for r in results if not np.isnan(getattr(r, name))]
        return float(np.median(vals)) if vals else float("nan")

    band_counts = pd.Series([r.band for r in results]).value_counts()
    representative_band = band_counts.idxmax()

    template = results[0]
    return EvidenceResult(
        disorder=template.disorder,
        query_score=_median_field("query_score"),
        lr_point_estimate=_median_field("lr_point_estimate"),
        lr_ci_low=_median_field("lr_ci_low"),
        lr_ci_high=_median_field("lr_ci_high"),
        points=_median_field("points"),
        band=representative_band,
        prior=template.prior,
        posterior_prob=_median_field("posterior_prob"),
        evidence_anchor=template.evidence_anchor,
        n_case=template.n_case,
        n_control=template.n_control,
        n_studies_case=template.n_studies_case,
        n_studies_control=template.n_studies_control,
        confounded_by_design=template.confounded_by_design,
        reason=f"median of {n_repeats} independent simulated draws",
        within_study_ci_low=_median_field("within_study_ci_low"),
        within_study_ci_high=_median_field("within_study_ci_high"),
    )
