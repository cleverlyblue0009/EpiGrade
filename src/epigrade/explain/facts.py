"""Fact sheets: the only input the explanation layer ever sees.

Each builder is deterministic and reads only a BYO evaluation object or the result tables. No
model is called here. Numbers are rounded once, here, and the explainer and the validator both
use these rounded forms, so prose and tables can never disagree on a decimal.

Rounding: AUC and Jaccard to 2 decimals; likelihood ratios, z-scores and percentages to 1
decimal; percentages are expressed as numbers out of 100 in keys ending in `_pct`.
"""

from __future__ import annotations

import math

import pandas as pd

from epigrade import paths

UNDER_MIN_N = 10


def _num(value, digits: int):
    """Round, keeping integers as integers and missing values as None."""
    if value is None:
        return None
    try:
        if isinstance(value, str) and not value.strip():
            return None
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    if digits == 0:
        return int(round(value))
    return round(value, digits)


def _pct(fraction, digits: int = 1):
    return _num(None if fraction is None else float(fraction) * 100, digits)


def _table(name: str) -> pd.DataFrame:
    return pd.read_csv(paths.tables_dir() / name, sep="\t", keep_default_na=False,
                       na_values=[""])


# --------------------------------------------------------------------------- BYO


def facts_for_byo(evaluation) -> dict:
    """From an epigrade.byo.evaluate.Evaluation."""
    cov, ident = evaluation.coverage, evaluation.identification or {}
    platforms = {
        name: (None if v.get("n_present") is None
               else {"n_present": int(v["n_present"]), "pct": _pct(v["fraction"])})
        for name, v in cov["platform_survival"].items()
    }
    facts = {
        "kind": "byo",
        "n_submitted": int(cov["n_submitted"]),
        "n_found": int(cov["n_found"]),
        "n_missing": int(cov["n_missing"]),
        "pct_missing": _pct(cov["fraction_missing"]),
        "platform_survival": platforms,
        "verdict": ident.get("verdict", evaluation.status),
        "refusal_code": evaluation.refusal_code,
        "n_disorders_searched": int(ident.get("n_disorders_searched", 0)),
        "n_with_own_signature": int(ident.get("n_with_own_signature", 0)),
        "top_match": ident.get("top_match"),
        "top_match_auc": _num(ident.get("top_match_auc"), 2),
        "auc_ci": ([_num(x, 2) for x in ident["top_match_auc_ci"]]
                   if ident.get("top_match_auc_ci") else None),
        "top_match_z": _num(ident.get("top_match_z"), 1),
        "runner_up": ident.get("runner_up"),
        "runner_up_z": _num(ident.get("runner_up_z"), 1),
        "margin_z": _num(ident.get("margin_z"), 1),
        "interpretation_scope": ident.get("interpretation_scope"),
        "highest_raw_auc": _num(ident.get("highest_auc"), 2),
        "highest_raw_auc_disorder": ident.get("highest_auc_disorder"),
        "n_cohorts_raw_auc_at_or_above_0_70": ident.get("n_cohorts_auc_at_or_above_threshold"),
        "evidence_ceiling_band": None, "cohort_ceiling": None, "signature_size_bound": None,
        "ceiling_limited_by": None, "between_study_bound_possible": None,
        "n_direction_supplied": None, "n_direction_inconsistent": None,
        "max_overlap_signature": None, "max_overlap_jaccard": None,
        "max_overlap_fraction_pct": None,
        "specificity_row": [],
        "notes": list(evaluation.messages) + [w["text"] for w in evaluation.warnings],
    }
    ceiling = evaluation.evidence_ceiling
    if ceiling:
        facts |= {
            "evidence_ceiling_band": ceiling["attainable_band"],
            "cohort_ceiling": ceiling["cohort_ceiling_band"],
            "signature_size_bound": ceiling["signature_size_band"],
            "ceiling_limited_by": ceiling["limited_by"],
            "between_study_bound_possible": bool(ceiling["between_study_bound_possible"]),
        }
    attribution = evaluation.attribution
    if attribution and attribution.get("direction_column_supplied"):
        facts["n_direction_supplied"] = int(attribution["n_probes_with_direction"])
        facts["n_direction_inconsistent"] = int(attribution["n_direction_inconsistent"])
    overlap = evaluation.overlap
    if overlap is not None and len(overlap):
        top = overlap.sort_values("jaccard", ascending=False).iloc[0]
        facts |= {
            "max_overlap_signature": top.corpus_signature,
            "max_overlap_jaccard": _num(top.jaccard, 2),
            "max_overlap_fraction_pct": _pct(top.fraction_of_submitted_in_corpus_signature),
        }
    if evaluation.specificity is not None:
        for r in evaluation.specificity.itertuples():
            if r.status != "computed":
                continue
            facts["specificity_row"].append({
                "cohort": r.scored_group, "n": int(r.n),
                "fraction_positive_pct": _pct(r.fraction_positive),
                "below_min_n": int(r.n) < UNDER_MIN_N,
            })
    return facts


# --------------------------------------------------------------------------- disorder


def facts_for_disorder(disorder: str) -> dict:
    """From disorder_verdicts.tsv, confounding_gate.tsv, signature_derivation.tsv and
    leave_one_study_out.tsv."""
    verdict = _table("disorder_verdicts.tsv").set_index("disorder").loc[disorder]
    gate = _table("confounding_gate.tsv").set_index("disorder").loc[disorder]
    derivation = _table("signature_derivation.tsv")
    canonical = derivation[(derivation.disorder == disorder)
                           & (derivation.derivation_scope == "canonical")].iloc[0]
    loso = _table("leave_one_study_out.tsv")
    loso = loso[loso.disorder == disorder]
    directions = [{
        "train_study": r.train_study, "test_study": r.test_study,
        "computed": r.status == "computed",
        "sensitivity": r.test_case_sensitivity if r.status == "computed" else None,
        "specificity": r.test_control_specificity if r.status == "computed" else None,
    } for r in loso.itertuples()]
    built = str(verdict.signature_built).lower() == "true"
    return {
        "kind": "disorder",
        "disorder": disorder,
        "n_cases": int(verdict.n_cases),
        "n_studies": int(verdict.n_studies),
        "studies": verdict.studies,
        "verdict": verdict.verdict,
        "confounding_gate": gate.status,
        "signature_built": built,
        "n_signature_probes": int(verdict.n_signature_probes) if built else 0,
        "signature_source": verdict.signature_source if built else None,
        "controls_source": verdict.controls_source,
        "n_training_controls": _num(canonical.n_controls, 0),
        "loso_directions_total": len(directions),
        "loso_directions_computed": sum(d["computed"] for d in directions),
        "loso_directions": directions,
        "interpretation_scope": (verdict.interpretation_scope
                                 if verdict.interpretation_scope != "NA" else None),
        "band_at_typical_case_score": (verdict.band_at_typical_case_score
                                       if verdict.band_at_typical_case_score != "NA" else None),
    }


# --------------------------------------------------------------------------- grade


def facts_for_grade(disorder: str, score: float, prior: float) -> dict:
    """For one computed row: a reference point in evidence_bands.tsv, else a grid point in
    calibration_grid.tsv. `score` must be a query score the pipeline actually evaluated - this
    layer never computes a new likelihood ratio."""
    rows = pd.DataFrame()
    for name in ("evidence_bands.tsv", "calibration_grid.tsv"):
        if not (paths.tables_dir() / name).exists():
            continue
        bands = _table(name)
        rows = bands[(bands.disorder == disorder) & (bands.prior.astype(float) == float(prior))]
        rows = rows[(rows.query_score.astype(float) - float(score)).abs() < 5e-5]
        if not rows.empty:
            break
    if rows.empty:
        raise ValueError(
            f"no computed evidence row for {disorder} at score {score} and prior {prior}"
        )
    r = rows.iloc[0]
    between = r.interpretation_scope == "between_study"
    return {
        "kind": "grade",
        "disorder": disorder,
        "query_point": r.query_point,
        "score": _num(r.query_score, 4),
        "prior_pct": _pct(r.prior, 0),
        "likelihood_ratio": _num(r.lr_point_estimate, 1),
        "lr_ci_low": _num(r.lr_ci_low if between else r.within_study_ci_low, 1),
        "lr_ci_high": _num(r.lr_ci_high if between else r.within_study_ci_high, 1),
        "ci_type": "between-study" if between else "within-study",
        "band": r.band,
        "posterior_pct": _pct(r.posterior_prob),
        "n_case": int(r.n_case),
        "n_control": int(r.n_control),
        "n_studies_case": int(r.n_studies_case),
        "n_studies_control": int(r.n_studies_control),
        "interpretation_scope": r.interpretation_scope,
        "reason": r.reason,
    }
