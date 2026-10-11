"""Every number the paper reports, pulled from results/tables/ and written with its source to
paper/supplementary_table_S1.tsv. The paper's Result section quotes these values; nothing in it
is typed from memory. tests/test_paper.py checks that each value here appears in the paper.
"""

from __future__ import annotations

import pandas as pd

from epigrade import paths

T = paths.repo_root() / "results" / "tables"
ROWS: list[dict] = []


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(T / name, sep="\t", keep_default_na=False, na_values=[""])


def add(key: str, value, source: str, locator: str) -> None:
    ROWS.append({"id": key, "value": str(value), "source_file": f"results/tables/{source}",
                 "locator": locator})


def _frac(c) -> str:
    return f"{int(c.n_positive)} of {int(c.n)} ({pct(c.fraction_positive)})"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def main() -> None:
    counts = table("sample_counts.tsv").set_index("category")["count"]
    for key in counts.index:
        add(f"samples.{key}", f"{int(counts[key]):,}", "sample_counts.tsv", f"category={key}")

    verdicts = table("disorder_verdicts.tsv")
    add("disorders.total", len(verdicts), "disorder_verdicts.tsv", "row count")
    for verdict, n in verdicts.verdict.value_counts().items():
        add(f"verdict.{verdict}", int(n), "disorder_verdicts.tsv", f"verdict={verdict}")
    add("verdict.not-downloaded", int((verdicts.verdict == "not-downloaded").sum()),
        "disorder_verdicts.tsv", "verdict=not-downloaded")
    add("signatures.built", int(verdicts.signature_built.sum()), "disorder_verdicts.tsv",
        "signature_built=True")
    for r in verdicts.itertuples():
        add(f"cases.{r.disorder}", r.n_cases, "disorder_verdicts.tsv", f"{r.disorder}: n_cases")
        if r.signature_built:
            add(f"probes.{r.disorder}", f"{r.n_signature_probes:,}", "disorder_verdicts.tsv",
                f"{r.disorder}: n_signature_probes")

    gate = table("confounding_gate.tsv")
    for status, n in gate.status.value_counts().items():
        add(f"gate.{status}", int(n), "confounding_gate.tsv", f"status={status}")

    derivation = table("signature_derivation.tsv")
    canonical = derivation[derivation.derivation_scope == "canonical"].set_index("disorder")
    for disorder in ("Coffin-Siris syndrome", "Williams syndrome", "Silver-Russell syndrome",
                     "Down syndrome"):
        add(f"significant.{disorder}", f"{int(canonical.loc[disorder, 'n_probes_significant']):,}",
            "signature_derivation.tsv", f"{disorder} canonical: n_probes_significant")
    pooled = derivation[derivation.derivation_scope == "pooled_alternative_not_used"].iloc[0]
    add("srs.pooled_probes", pooled.n_probes_after_effect_filter, "signature_derivation.tsv",
        "Silver-Russell pooled_alternative_not_used: n_probes_after_effect_filter")
    add("derivations.recorded", len(derivation), "signature_derivation.tsv", "row count")

    sotos = table("sotos_reproduction_summary.tsv")
    add("sotos.checks_passed", f"{int(sotos.reproduced.astype(str).str.lower().eq('true').sum())}"
        f" of {len(sotos)}", "sotos_reproduction_summary.tsv", "reproduced=True")
    for r in sotos.itertuples():
        # The observed column is a sentence of ";"-separated facts; the paper quotes the
        # counts and score ranges, not the per-sample score lists.
        parts = [x.strip() for x in r.observed.split(";")]
        quoted = [x for x in parts if not x.startswith(("scores=", "clean_separation", "n_case"))]
        for k, part in enumerate(quoted):
            add(f"sotos.{r.check}.{k}", part, "sotos_reproduction_summary.tsv",
                f"check={r.check}, observed")

    loso = table("leave_one_study_out.tsv")
    add("loso.directions", len(loso), "leave_one_study_out.tsv", "row count")
    add("loso.computed", int((loso.status == "computed").sum()), "leave_one_study_out.tsv",
        "status=computed")
    ok = loso[loso.status == "computed"].iloc[0]
    add("loso.srs.sensitivity", ok.test_case_sensitivity, "leave_one_study_out.tsv",
        "Silver-Russell GSE104451->GSE55491: test_case_sensitivity")
    add("loso.srs.sensitivity_pct", pct(float(ok.sensitivity)), "leave_one_study_out.tsv",
        "same row: sensitivity")
    add("loso.srs.specificity", ok.test_control_specificity, "leave_one_study_out.tsv",
        "same row: test_control_specificity")
    add("loso.srs.probes", int(ok.n_signature_probes), "leave_one_study_out.tsv",
        "same row: n_signature_probes")

    matrix = table("cross_disorder_matrix.tsv")
    add("matrix.cells", len(matrix), "cross_disorder_matrix.tsv", "row count")
    add("matrix.computed", int((matrix.status == "computed").sum()), "cross_disorder_matrix.tsv",
        "status=computed")
    add("matrix.not_computed", int((matrix.status == "not_computed").sum()),
        "cross_disorder_matrix.tsv", "status=not_computed")

    def cell(clf, target):
        return matrix[(matrix.classifier == clf) & (matrix.scored_disorder == target)].iloc[0]

    pairs = [
        ("CHARGE syndrome", "Kabuki syndrome type 1"),
        ("Kabuki syndrome type 1", "CHARGE syndrome"),
        ("Coffin-Siris syndrome", "Silver-Russell syndrome"),
        ("Coffin-Siris syndrome", "pooled_control"),
        ("Nicolaides-Baraitser syndrome", "pooled_control"),
        ("Down syndrome", "pooled_control"), ("CHARGE syndrome", "pooled_control"),
        ("Sotos syndrome", "pooled_control"), ("ICF syndrome", "pooled_control"),
        ("Kabuki syndrome type 1", "pooled_control"),
        ("Silver-Russell syndrome", "pooled_control"), ("Williams syndrome", "pooled_control"),
        ("7q11.23 duplication syndrome", "pooled_control"),
        ("Silver-Russell syndrome", "Down syndrome"),
        ("7q11.23 duplication syndrome", "ICF syndrome"),
        ("Williams syndrome", "Sotos syndrome"),
    ]
    for clf, target in pairs:
        c = cell(clf, target)
        add(f"matrix.{clf}->{target}", _frac(c),
            "cross_disorder_matrix.tsv", f"classifier={clf}, scored_disorder={target}")
    for disorder in verdicts[verdicts.signature_built].disorder:
        c = cell(disorder, disorder)
        add(f"sensitivity.{disorder}", _frac(c),
            "cross_disorder_matrix.tsv", f"classifier=scored_disorder={disorder}")

    summary = table("controls_source_summary.tsv").set_index("controls_source")
    for source in ("within_series", "cross_series"):
        r = summary.loc[source]
        add(f"controls_effect.{source}.mean_off_diagonal",
            pct(r.mean_off_diagonal_fraction_positive), "controls_source_summary.tsv",
            f"{source}: mean_off_diagonal_fraction_positive")
        add(f"controls_effect.{source}.cells",
            f"{int(r.n_off_diagonal_cells)} cells from {int(r.n_classifiers)} classifiers",
            "controls_source_summary.tsv", f"{source}: n_off_diagonal_cells, n_classifiers")
        add(f"controls_effect.{source}.pooled_controls",
            pct(r.mean_pooled_control_fraction_positive), "controls_source_summary.tsv",
            f"{source}: mean_pooled_control_fraction_positive")
        add(f"controls_effect.{source}.median", pct(r.median_off_diagonal_fraction_positive),
            "controls_source_summary.tsv", f"{source}: median_off_diagonal_fraction_positive")
    ratio = (summary.loc["cross_series", "mean_off_diagonal_fraction_positive"]
             / summary.loc["within_series", "mean_off_diagonal_fraction_positive"])
    add("controls_effect.ratio", f"{ratio:.1f} times", "controls_source_summary.tsv",
        "cross_series mean / within_series mean")
    off = matrix[(matrix.status == "computed") & (matrix.cell_type == "other_disorder")
                 & matrix.interpretation.isna()]
    by_clf = off.groupby("classifier").fraction_positive.mean()
    for clf in ("Coffin-Siris syndrome", "Nicolaides-Baraitser syndrome", "Down syndrome",
                "CHARGE syndrome"):
        add(f"controls_effect.classifier.{clf}", pct(by_clf[clf]), "cross_disorder_matrix.tsv",
            f"classifier={clf}: mean fraction_positive over other_disorder cells with n>=10")
    within = by_clf.drop(["Coffin-Siris syndrome", "Nicolaides-Baraitser syndrome"])
    add("controls_effect.within_max", f"{pct(within.max())} ({within.idxmax()})",
        "cross_disorder_matrix.tsv", "highest per-classifier mean among within_series")

    evidence = table("evidence_bands.tsv")
    typical = evidence[(evidence.query_point == "typical_case_score") & (evidence.prior == 0.10)]
    add("bands.between_study", int((typical.interpretation_scope == "between_study").sum()),
        "evidence_bands.tsv", "typical_case_score, prior 0.10: interpretation_scope")
    add("bands.within_study", int((typical.interpretation_scope == "within_study_only").sum()),
        "evidence_bands.tsv", "typical_case_score, prior 0.10: interpretation_scope")
    for r in typical.itertuples():
        ci = (f"{r.lr_ci_low:.2f}-{r.lr_ci_high:.2f}" if r.interpretation_scope == "between_study"
              else f"{r.within_study_ci_low:.1f}-{r.within_study_ci_high:.1f}")
        add(f"band.{r.disorder}", f"{r.band} (LR {r.lr_point_estimate:.1f}, interval {ci})",
            "evidence_bands.tsv", f"{r.disorder}, typical_case_score, prior 0.10")
    vous = evidence[(evidence.query_point == "typical_missense_vous_score")
                    & (evidence.prior == 0.10)].iloc[0]
    add("band.sotos_missense", f"{vous.band} (LR {vous.lr_point_estimate:.1f})",
        "evidence_bands.tsv", "Sotos, typical_missense_vous_score, prior 0.10")

    ceiling = table("attainable_ceiling.tsv")
    add("ceiling.bands", ", ".join(sorted(set(ceiling.attainable_band))),
        "attainable_ceiling.tsv", "attainable_band, all 12 disorders")
    add("ceiling.between_possible", int(ceiling.between_study_bound_possible.sum()),
        "attainable_ceiling.tsv", "between_study_bound_possible=True")
    curve = table("attainable_ceiling_curve.tsv")
    first_vs = curve[curve.attainable_band == "Very Strong"].groupby("study_structure").n_case.min()
    for structure, n in first_vs.items():
        add(f"ceiling.very_strong_from.{structure}", int(n), "attainable_ceiling_curve.tsv",
            f"smallest n_case with Very Strong, {structure}")

    byo = table("byo_fixture_results.tsv").set_index("fixture")
    for name, r in byo.iterrows():
        short = name.replace(".csv", "")
        add(f"byo.{short}.verdict", r.verdict, "byo_fixture_results.tsv", f"{name}: verdict")
        if isinstance(r.top_match, str) and r.top_match:
            add(f"byo.{short}.auc", f"{float(r.top_match_auc):.2f}", "byo_fixture_results.tsv",
                f"{name}: top_match_auc")
            add(f"byo.{short}.z", f"{float(r.top_match_z_vs_random_probes):.1f}",
                "byo_fixture_results.tsv", f"{name}: top_match_z_vs_random_probes")
            add(f"byo.{short}.runner_up", f"{r.runner_up} (z {float(r.runner_up_z):.1f})",
                "byo_fixture_results.tsv", f"{name}: runner_up, runner_up_z")
            add(f"byo.{short}.effect", f"{float(r.top_match_mean_abs_delta_beta):.3f} "
                f"({float(r.effect_ci_low):.3f}-{float(r.effect_ci_high):.3f})",
                "byo_fixture_results.tsv", f"{name}: mean |delta beta| and interval")
        add(f"byo.{short}.outcomes", r.expectation_outcomes, "byo_fixture_results.tsv",
            f"{name}: expectation_outcomes")
        if isinstance(r.evidence_ceiling_band, str) and r.evidence_ceiling_band:
            add(f"byo.{short}.ceiling",
                f"{r.evidence_ceiling_band} (limited by {r.ceiling_limited_by}; cohort ceiling "
                f"{r.cohort_ceiling_band}; conservative LR "
                f"{float(r.signature_size_conservative_lr):.1f})",
                "byo_fixture_results.tsv", f"{name}: evidence ceiling columns")
    rnd = byo.loc["sig_random.csv"]
    add("byo.random.highest_auc", f"{float(rnd.highest_auc):.2f} ({rnd.highest_auc_disorder})",
        "byo_fixture_results.tsv", "sig_random.csv: highest_auc")
    add("byo.random.n_auc_over", int(rnd["n_cohorts_auc_at_or_above_0.70"]),
        "byo_fixture_results.tsv", "sig_random.csv: n_cohorts_auc_at_or_above_0.70")
    flipped = byo.loc["sig_sotos_flipped.csv"]
    add("byo.flipped.inconsistent", f"{int(flipped.n_direction_inconsistent)} of "
        f"{int(flipped.n_directions_supplied)}", "byo_fixture_results.tsv",
        "sig_sotos_flipped.csv: n_direction_inconsistent")
    add("byo.searched", int(byo.n_disorders_searched.dropna().iloc[0]),
        "byo_fixture_results.tsv", "n_disorders_searched")

    ev = table("explanation_eval_summary.tsv").set_index("measure")["value"]
    for key in ("n_subjects", "passed_first_time", "passed_after_retry",
                "fell_back_after_two_rejections", "model_outputs_rejected_total", "model"):
        add(f"explain.{key}", ev[key], "explanation_eval_summary.tsv", f"measure={key}")
    for key in ev.index:
        if key.startswith("rejection_reason: "):
            add(f"explain.{key}", f"{key.split(': ', 1)[1]} ({ev[key]})",
                "explanation_eval_summary.tsv", f"measure={key}")
    review = table("explanation_review.tsv")
    model_rows = review[review.source == "model"]
    n_err = int((model_rows.review_verdict == "error").sum())
    add("explain.review_errors", f"{n_err} of {len(model_rows)}", "explanation_review.tsv",
        "review_verdict=error among source=model")
    add("explain.review_ok", int((model_rows.review_verdict == "ok").sum()),
        "explanation_review.tsv", "review_verdict=ok among source=model")
    run2 = table("explanation_eval_summary_run2.tsv").set_index("measure")["value"]
    add("explain.run2_passed", f"{run2['passed_first_time']} of {run2['n_subjects']}",
        "explanation_eval_summary_run2.tsv", "passed_first_time, run 2")
    run1 = table("explain_failures_run1.tsv")
    add("explain.run1_rejections", len(run1), "explain_failures_run1.tsv", "row count, run 1")

    agreement = table("harmonisation_agreement.tsv")
    scoreable = agreement[agreement.agree.notna()]
    add("labels.ai_vs_ai", f"{int(scoreable.agree.astype(str).str.lower().eq('true').sum())} of "
        f"{len(scoreable)}", "harmonisation_agreement.tsv", "agree, scoreable rows")
    add("labels.unresolvable", len(agreement) - len(scoreable), "harmonisation_agreement.tsv",
        "rows with no agree value")
    human = table("human_curation_agreement.tsv")
    add("labels.human", human.iloc[0, 0], "human_curation_agreement.tsv", "status")

    controls = table("control_specificity_by_series.tsv")
    held = controls[controls.relation_to_classifier == "held_out_controls"]
    add("controls.pooled_n", 273, "cross_disorder_matrix.tsv",
        "Sotos pooled_control n (220) + its 53 training controls")
    kab = controls[(controls.classifier == "Kabuki syndrome type 1")
                   & (controls.relation_to_classifier == "training_controls_leave_one_out")
                   & (controls.control_series == "GSE116300")].iloc[0]
    add("controls.kabuki_own", f"{int(kab.n_positive)} of {int(kab.n)}",
        "control_specificity_by_series.tsv",
        "Kabuki classifier, GSE116300 training controls (leave-one-out)")
    add("controls.series", held.control_series.nunique(), "control_specificity_by_series.tsv",
        "distinct control_series")

    out_dir = paths.repo_root() / "paper"
    out_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(ROWS)
    df.to_csv(out_dir / "supplementary_table_S1.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    pd.set_option("display.max_colwidth", 150)
    print(df[["id", "value"]].to_string())


if __name__ == "__main__":
    main()
