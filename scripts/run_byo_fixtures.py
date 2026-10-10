"""Run the five test signatures through the bring-your-own-signature engine and record what
actually happened next to what was expected, in results/tables/byo_fixture_results.tsv.

Uses the checked-in demo resource only (never the local full store), so this table is
reproducible from a clean clone with no downloads. An expectation that is not met is recorded
as not met, with the observed value - nothing is adjusted to make a fixture pass.
"""

from __future__ import annotations

import pandas as pd

from epigrade import paths
from epigrade.byo import store
from epigrade.byo.evaluate import BAND_ORDER, Evaluation, evaluate_file

FIXTURE_DIR = paths.repo_root() / "tests" / "fixtures"
SOTOS = "Sotos syndrome"


def _matched(e: Evaluation, disorder: str) -> bool:
    return e.status == "ok" and e.identification["top_match"] == disorder


# fixture -> (what the spec expects, [(component, check)])
EXPECTATIONS = {
    "sig_sotos_true.csv": (
        "identifies Sotos, AUC > 0.95, clear margin",
        [("identifies Sotos", lambda e: _matched(e, SOTOS)),
         ("AUC > 0.95", lambda e: _matched(e, SOTOS)
          and e.identification["top_match_auc"] > 0.95),
         ("clear margin over runner-up", lambda e: _matched(e, SOTOS)
          and e.identification["top_match_z"] > 2 * e.identification["runner_up_z"])],
    ),
    "sig_random.csv": (
        "no_match across 12 disorders searched (the spec also expected nothing above AUC 0.70)",
        [("no_match", lambda e: e.status == "ok" and e.identification["verdict"] == "no_match"),
         ("12 disorders searched", lambda e: e.identification["n_disorders_searched"] == 12),
         ("nothing above AUC 0.70", lambda e: e.status == "ok"
          and e.identification["n_cohorts_auc_at_or_above_threshold"] == 0)],
    ),
    "sig_sotos_flipped.csv": (
        "identifies Sotos but flags every probe direction-inconsistent",
        [("identifies Sotos", lambda e: _matched(e, SOTOS)),
         ("every direction flagged inconsistent", lambda e: e.attribution is not None
          and e.attribution["all_directions_inconsistent"])],
    ),
    "sig_tiny.csv": (
        "identifies Sotos with a wide CI and a low evidence ceiling",
        [("identifies Sotos", lambda e: _matched(e, SOTOS)),
         ("wider effect CI than the 500-probe signature", None),  # filled in below
         ("evidence ceiling lower than the 500-probe signature's", "ceiling"),
         ("evidence ceiling is Supporting or below", lambda e: e.evidence_ceiling is not None
          and e.evidence_ceiling["attainable_band"] in ("No evidence", "Supporting"))],
    ),
    "sig_low_coverage.csv": (
        "refused with insufficient_probe_coverage",
        [("refused: insufficient_probe_coverage",
          lambda e: e.status == "refused" and e.refusal_code == "insufficient_probe_coverage")],
    ),
}


def _relative_ci_width(e: Evaluation) -> float:
    top = e.cohorts[e.cohorts.disorder == e.identification["top_match"]].iloc[0]
    return float((top.effect_ci_high - top.effect_ci_low) / top.mean_abs_delta_beta)


def summarise(name: str, e: Evaluation, reference_width: float | None,
              reference_band: str | None = None) -> dict:
    expected, checks = EXPECTATIONS[name]
    ident = e.identification or {}
    outcomes = []
    for label, check in checks:
        if check == "ceiling":
            met = (e.evidence_ceiling is not None and reference_band is not None
                   and BAND_ORDER.index(e.evidence_ceiling["attainable_band"])
                   < BAND_ORDER.index(reference_band))
        elif check is None:
            met = (ident.get("top_match") is not None and reference_width is not None
                   and _relative_ci_width(e) > reference_width)
        else:
            met = bool(check(e))
        outcomes.append(f"{label}: {'met' if met else 'NOT met'}")
    top = None
    if ident.get("top_match"):
        top = e.cohorts[e.cohorts.disorder == ident["top_match"]].iloc[0]
    attribution = e.attribution or {}
    ceiling = e.evidence_ceiling or {}
    return {
        "fixture": name, "expected": expected, "status": e.status,
        "refusal_code": e.refusal_code or "", "verdict": ident.get("verdict", ""),
        "top_match": ident.get("top_match") or "",
        "top_match_auc": ident.get("top_match_auc"),
        "top_match_auc_ci_low": top.auc_ci_low if top is not None else None,
        "top_match_auc_ci_high": top.auc_ci_high if top is not None else None,
        "top_match_mean_abs_delta_beta": ident.get("top_match_mean_abs_delta_beta"),
        "effect_ci_low": top.effect_ci_low if top is not None else None,
        "effect_ci_high": top.effect_ci_high if top is not None else None,
        "top_match_z_vs_random_probes": ident.get("top_match_z"),
        "runner_up": ident.get("runner_up") or "", "runner_up_z": ident.get("runner_up_z"),
        "margin_z": ident.get("margin_z"),
        "highest_auc_disorder": ident.get("highest_auc_disorder", ""),
        "highest_auc": ident.get("highest_auc"),
        "n_cohorts_auc_at_or_above_0.70": ident.get("n_cohorts_auc_at_or_above_threshold"),
        "n_disorders_searched": ident.get("n_disorders_searched"),
        "n_with_own_signature": ident.get("n_with_own_signature"),
        "interpretation_scope": ident.get("interpretation_scope", ""),
        "n_submitted": e.coverage["n_submitted"], "n_found": e.coverage["n_found"],
        "n_missing": e.coverage["n_missing"],
        "n_directions_supplied": attribution.get("n_probes_with_direction"),
        "n_direction_inconsistent": attribution.get("n_direction_inconsistent"),
        "evidence_ceiling_band": ceiling.get("attainable_band", ""),
        "ceiling_limited_by": ceiling.get("limited_by", ""),
        "cohort_ceiling_band": ceiling.get("cohort_ceiling_band", ""),
        "signature_size_band": ceiling.get("signature_size_band", ""),
        "signature_size_conservative_lr": ceiling.get("signature_size_conservative_lr"),
        "between_study_bound_possible": ceiling.get("between_study_bound_possible"),
        "expectation_outcomes": "; ".join(outcomes),
        "all_expectations_met": all("NOT met" not in o for o in outcomes),
        "beta_source": e.coverage["beta_source"],
    }


def main() -> None:
    source = store.load_beta_source(prefer_full=False)
    evaluations = {name: evaluate_file(FIXTURE_DIR / name, source=source)
                   for name in EXPECTATIONS}
    reference = evaluations["sig_sotos_true.csv"]
    reference_width = (_relative_ci_width(reference)
                       if reference.identification.get("top_match") else None)
    reference_band = (reference.evidence_ceiling or {}).get("attainable_band")
    rows = [summarise(name, e, reference_width, reference_band)
            for name, e in evaluations.items()]
    df = pd.DataFrame(rows)
    out = paths.tables_dir() / "byo_fixture_results.tsv"
    df.to_csv(out, sep="\t", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(df[["fixture", "status", "verdict", "top_match", "top_match_auc",
              "top_match_z_vs_random_probes", "runner_up_z", "highest_auc"]].to_string())
    for r in rows:
        print(f"\n{r['fixture']}: {r['expectation_outcomes']}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
