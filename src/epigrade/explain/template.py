"""Deterministic plain-language explanations from a fact sheet: no model, no network.

Used whenever the model is unavailable or its output fails validation, and it is what the app
shows with no API key. Every sentence is a fixed frame filled from the fact sheet.
"""

from __future__ import annotations


def _fmt(x, digits: int = 1) -> str:
    """Fixed decimals, matching how the fact sheet rounded the value."""
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    if isinstance(x, int):
        return f"{x:,}"
    return str(x)


def byo(f: dict) -> str:
    if f["verdict"] == "refused":
        return (
            f"This probe list was not scored. Only {f['n_found']} of the {f['n_submitted']} "
            f"submitted probes are present in any of the {f['n_disorders_searched']} disorder "
            "cohorts in the corpus, and at least 10 are needed. The refusal code is "
            f"{f['refusal_code']}."
        )
    head = (f"{f['n_found']} of the {f['n_submitted']} submitted probes were found in the "
            f"corpus, and the signature was compared against {f['n_disorders_searched']} "
            f"disorder cohorts, {f['n_with_own_signature']} of which have a signature of their "
            "own.")
    if f["verdict"] == "no_match":
        text = (head + " No cohort passed both tests, so no disorder is named.")
        if f.get("highest_raw_auc") is not None:
            text += (
                f" The highest raw AUC was {_fmt(f['highest_raw_auc'], 2)} in "
                f"{f['highest_raw_auc_disorder']}, but random probe sets of the same size did "
                "as well, which is why AUC alone does not decide a match."
            )
        return text
    text = (
        head + f" It identifies {f['top_match']}, with an AUC of {_fmt(f['top_match_auc'], 2)} "
        f"and an excess over random probes of z = {_fmt(f['top_match_z'])}, against "
        f"{_fmt(f['runner_up_z'])} for the runner-up, {f['runner_up']}."
    )
    if f["interpretation_scope"] == "within_study_only":
        text += (" This match is within one study only: it does not show that the signature "
                 "would transfer to another laboratory.")
    if f.get("evidence_ceiling_band"):
        text += (f" The strongest evidence band it could support is "
                 f"{f['evidence_ceiling_band']}, limited by {f['ceiling_limited_by']}.")
    if f.get("n_direction_supplied") and f["n_direction_inconsistent"] == f["n_direction_supplied"]:
        text += (f" All {f['n_direction_supplied']} submitted directions are the opposite of "
                 "those observed in the matched cohort.")
    return text


def disorder(f: dict) -> str:
    studies = "study" if f["n_studies"] == 1 else "studies"
    text = (f"{f['disorder']} has {f['n_cases']} cases in this corpus, from {f['n_studies']} "
            f"{studies}, and its verdict is {f['verdict']}.")
    if not f["signature_built"]:
        return text + (" There are fewer than 10 cases, so no signature was built; its cases "
                       "are only scored by the other classifiers.")
    source = ("the published probe list" if f["signature_source"] == "published_probe_list"
              else "a signature re-derived here")
    text += f" Its classifier uses {source}, with {_fmt(f['n_signature_probes'])} probes."
    if f["controls_source"] == "cross_series":
        text += (" Its controls were taken from another series, so it detects series "
                 "membership rather than disease.")
    if f["loso_directions_total"]:
        text += (f" Leave-one-study-out was computed in {f['loso_directions_computed']} of "
                 f"{f['loso_directions_total']} directions.")
    else:
        text += " With one study, no cross-study test is possible."
    if f.get("band_at_typical_case_score"):
        scope = ("within one study only" if f["interpretation_scope"] == "within_study_only"
                 else "across studies")
        text += (f" At a typical case score the evidence band is "
                 f"{f['band_at_typical_case_score']}, {scope}.")
    return text


def grade(f: dict) -> str:
    text = (f"At a score of {_fmt(f['score'], 4)} for {f['disorder']}, the likelihood ratio is "
            f"{_fmt(f['likelihood_ratio'])}")
    if f["lr_ci_low"] is not None and f["lr_ci_high"] is not None:
        text += (f", with a {f['ci_type']} interval of {_fmt(f['lr_ci_low'])} to "
                 f"{_fmt(f['lr_ci_high'])}")
    text += f". The evidence band is {f['band']}."
    if f.get("posterior_pct") is not None:
        text += (f" With an assumed prior of {_fmt(f['prior_pct'], 0)}%, the posterior probability "
                 f"is {_fmt(f['posterior_pct'])}%.")
    text += (f" The comparison uses {f['n_case']} cases and {f['n_control']} controls.")
    if f["interpretation_scope"] == "within_study_only":
        text += (" All of them come from a single study, so this describes stability inside "
                 "that study, not a result that has been shown to transfer.")
    return text


TEMPLATES = {"byo": byo, "disorder": disorder, "grade": grade}


def explain(facts: dict) -> str:
    return TEMPLATES[facts["kind"]](facts)
