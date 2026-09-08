"""Task 5: explains a score using ONLY quantities the pipeline already computed - no LLM
narrative, no biological inference. Every function here returns numbers and their direct
arithmetic derivations; the one text-producing function (`format_explanation`) is a plain
formatter that restates those numbers in sentences, labelled as such, and invents nothing.

Runs offline (like the rest of this pipeline) against a classifier + a real sample's beta
values - the app itself never recomputes anything, it only displays what this module already
wrote to the resource. See scripts/compute_attribution.py for the exemplar samples this is run
against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from scipy.stats import pearsonr


@dataclass
class Attribution:
    gsm_accession: str
    disorder: str
    score: float
    cohort_label: str  # what this sample actually is, e.g. "discovery_case"

    # --- probe coverage ---
    n_signature_probes: int
    n_probes_present: int
    n_probes_missing: int

    # --- direction check ---
    expected_direction: str | None  # e.g. "loss" for Sotos, or None if not established
    pct_probes_matching_expected_direction: float | None
    direction_anomalous: bool  # True if the score's sign disagrees with the expected direction

    # --- distribution position ---
    percentile_within_cases: float | None
    percentile_within_controls: float | None
    sd_from_control_mean: float | None
    sd_from_case_mean: float | None

    # --- top contributing probes (a distance-based proxy, not an exact decomposition of the
    # Pearson correlation difference - see contribution_method) ---
    contribution_method: str
    top_probes: list[dict] = field(default_factory=list)

    # --- band derivation (filled in by the caller from evidence_bands.tsv, if available) ---
    band: str | None = None
    lr_point_estimate: float | None = None
    interpretation_scope: str | None = None
    band_reason: str | None = None


def explain_score(
    gsm_accession: str, disorder: str, cohort_label: str,
    sample_beta: pd.Series, median_case: pd.Series, median_control: pd.Series,
    signature_probes: pd.Index, case_scores: pd.Series, control_scores: pd.Series,
    expected_direction: str | None = None, top_n: int = 10,
) -> Attribution:
    """`sample_beta` and the two medians must already be aligned to `signature_probes` (same
    index). `case_scores`/`control_scores` are the discovery cohort's own score distributions,
    used only for percentile/SD computations - never recomputed by the app."""
    x = sample_beta.reindex(signature_probes)
    mc = median_case.reindex(signature_probes)
    mn = median_control.reindex(signature_probes)

    present = x.notna() & mc.notna() & mn.notna()
    n_present = int(present.sum())
    n_missing = len(signature_probes) - n_present

    r_case = pearsonr(x[present], mc[present])[0]
    r_control = pearsonr(x[present], mn[present])[0]
    score = r_case - r_control

    # --- direction check: does this sample show the expected methylation direction (e.g.
    # Sotos's 99.3% loss) at the probes where it agrees with the case profile? ---
    direction_anomalous = False
    pct_matching = None
    if expected_direction is not None:
        # A probe "matches expected direction" if the sample's value moved from the control
        # median toward the case median in the direction Choufani et al. report for that probe
        # (loss = case median < control median, i.e. lower beta = more case-like).
        case_shows_loss = mc[present] < mn[present]
        sample_moved_toward_case = (
            (x[present] < mn[present]) if expected_direction == "loss"
            else (x[present] > mn[present])
        )
        matches = (case_shows_loss == (expected_direction == "loss")) & sample_moved_toward_case
        pct_matching = float(matches.mean()) if n_present else None
        # Anomalous specifically when the sample scores case-like (positive) but its own
        # probe-level movements mostly do NOT match the signature's expected direction.
        direction_anomalous = bool(score > 0 and pct_matching is not None and pct_matching < 0.5)

    # --- distribution position ---
    def _percentile(value, distribution):
        if len(distribution) == 0:
            return None
        return float((distribution < value).mean() * 100)

    def _sd_from(value, distribution):
        if len(distribution) < 2 or distribution.std(ddof=1) == 0:
            return None
        return float((value - distribution.mean()) / distribution.std(ddof=1))

    percentile_within_cases = _percentile(score, case_scores)
    percentile_within_controls = _percentile(score, control_scores)
    sd_from_control_mean = _sd_from(score, control_scores)
    sd_from_case_mean = _sd_from(score, case_scores)

    # --- top contributing probes: a distance-based proxy. At each probe, how much closer is
    # this sample to the case median than to the control median (in squared-distance terms)?
    # This is NOT an exact per-term decomposition of the Pearson correlation difference (that
    # doesn't decompose additively per-probe in a simple closed form) - it is a real, computed,
    # clearly-labelled approximation of which probes push the sample toward or away from the
    # case profile, useful for a "why" panel, not for re-deriving the exact score. ---
    dist_to_case = (x[present] - mc[present]) ** 2
    dist_to_control = (x[present] - mn[present]) ** 2
    contribution = dist_to_control - dist_to_case  # positive = closer to case than control
    top_idx = contribution.abs().sort_values(ascending=False).head(top_n).index
    top_probes = [
        {
            "probe_id": str(probe),
            "sample_beta": round(float(x[probe]), 4),
            "median_case_beta": round(float(mc[probe]), 4),
            "median_control_beta": round(float(mn[probe]), 4),
            "contribution": round(float(contribution[probe]), 5),
            "pulls_toward": "case" if contribution[probe] > 0 else "control",
        }
        for probe in top_idx
    ]

    return Attribution(
        gsm_accession=gsm_accession, disorder=disorder, score=float(score),
        cohort_label=cohort_label,
        n_signature_probes=len(signature_probes), n_probes_present=n_present,
        n_probes_missing=n_missing,
        expected_direction=expected_direction,
        pct_probes_matching_expected_direction=pct_matching,
        direction_anomalous=direction_anomalous,
        percentile_within_cases=percentile_within_cases,
        percentile_within_controls=percentile_within_controls,
        sd_from_control_mean=sd_from_control_mean, sd_from_case_mean=sd_from_case_mean,
        contribution_method=(
            "squared-distance-to-median proxy per probe (dist_to_control - dist_to_case); "
            "NOT an exact algebraic decomposition of the Pearson correlation difference"
        ),
        top_probes=top_probes,
    )


def format_explanation(a: Attribution) -> list[str]:
    """PLAIN FORMATTER ONLY: restates fields already on `a` as sentences. No inference, no
    biological claim beyond what was computed, no LLM call. Every number here traces back to a
    field set in explain_score()."""
    lines = [
        f"Score {a.score:+.3f} for {a.gsm_accession} ({a.cohort_label}), {a.disorder}.",
        f"Signature coverage: {a.n_probes_present}/{a.n_signature_probes} probes present "
        f"({a.n_probes_missing} missing on this sample's platform/array).",
    ]
    if a.percentile_within_cases is not None:
        lines.append(
            f"Falls at the {a.percentile_within_cases:.0f}th percentile of the discovery "
            f"case-score distribution and the {a.percentile_within_controls:.0f}th percentile "
            "of the control-score distribution."
        )
    if a.sd_from_control_mean is not None:
        lines.append(
            f"{a.sd_from_control_mean:+.2f} SD from the control mean, "
            f"{a.sd_from_case_mean:+.2f} SD from the case mean."
        )
    if a.expected_direction is not None:
        lines.append(
            f"{a.pct_probes_matching_expected_direction:.0%} of present signature probes show "
            f"the expected '{a.expected_direction}' direction at this sample."
        )
        if a.direction_anomalous:
            lines.append(
                "ANOMALOUS: this sample scores case-like but most of its probes do NOT move in "
                "the signature's expected direction - worth a second look before trusting the "
                "score at face value."
            )
    if a.band is not None:
        lines.append(
            f"Evidence band: {a.band} (LR={a.lr_point_estimate:.2f}, "
            f"scope={a.interpretation_scope}). {a.band_reason or ''}".strip()
        )
    return lines
