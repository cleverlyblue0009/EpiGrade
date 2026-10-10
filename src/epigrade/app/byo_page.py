"""'Test your own signature' page: upload a probe list, see which corpus cohort (if any) it
identifies. Runs entirely from the checked-in demo resource - no downloads, no API keys. If the
full local corpus store exists it is used instead, so an arbitrary upload is evaluated on all of
its probes.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from epigrade import paths
from epigrade.app import summary
from epigrade.byo import store
from epigrade.byo.evaluate import (
    AUC_THRESHOLD,
    EXPECTED_COLUMNS,
    N_NULL,
    Evaluation,
    SignatureInputError,
    evaluate,
    parse_signature,
)
from epigrade.explain.facts import facts_for_byo

FIXTURE_DIR = paths.repo_root() / "tests" / "fixtures"
EXAMPLES = {
    "500 published Sotos probes (true signature)": "sig_sotos_true.csv",
    "500 random probes": "sig_random.csv",
    "Sotos probes with every direction inverted": "sig_sotos_flipped.csv",
    "20 Sotos probes (tiny signature)": "sig_tiny.csv",
    "500 probes absent from the corpus": "sig_low_coverage.csv",
}
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#8a8984"


@st.cache_data(show_spinner="Scoring against all 12 cohorts...")
def _evaluate(raw: bytes, name: str) -> Evaluation:
    return evaluate(parse_signature(raw, name))


def _coverage_card(result: Evaluation) -> None:
    cov, summary = result.coverage, result.input_summary
    st.subheader("Coverage")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probes submitted", f"{cov['n_submitted']:,}")
    c2.metric("Usable in the corpus", f"{cov['n_found']:,}")
    c3.metric("Missing", f"{cov['n_missing']:,}",
              f"{cov['fraction_missing']:.1%} of submitted", delta_color="off")
    c4.metric("Rows rejected / duplicates",
              f"{summary['n_rejected_bad_format']} / {summary['n_duplicates_removed']}")
    platform = pd.DataFrame([
        {"platform": name,
         "probes present": "manifest not available" if v["n_present"] is None
         else f"{v['n_present']:,} of {cov['n_submitted']:,} ({v['fraction']:.0%})"}
        for name, v in cov["platform_survival"].items()
    ])
    left, right = st.columns(2)
    left.caption("Survival across array platforms")
    left.dataframe(platform, hide_index=True, use_container_width=True)
    right.caption("Present in each corpus beta matrix")
    right.dataframe(
        pd.DataFrame({"series": list(cov["corpus_series_presence"]),
                      "probes present": list(cov["corpus_series_presence"].values())}),
        hide_index=True, use_container_width=True, height=212,
    )
    if cov["beta_source"] == store.DEMO:
        st.caption(
            "Running from the checked-in demo resource: sample-level betas cover the corpus "
            f"signature probes and the example files ({cov['n_with_sample_level_betas']:,} of "
            "this signature's usable probes)."
        )


def _verdict(result: Evaluation) -> None:
    ident = result.identification
    st.subheader("Identification")
    searched = (f"Searched **{ident['n_disorders_searched']} disorder cohorts** "
                f"({ident['n_with_own_signature']} of which have a signature of their own).")
    if ident["verdict"] == "no_match":
        st.info(
            f"**No match.** {searched} None passed both criteria, so no best-of-set guess is "
            "made.", icon="🔍",
        )
        if ident.get("highest_auc", 0) >= AUC_THRESHOLD:
            st.caption(
                f"The highest raw AUC was {ident['highest_auc']:.2f} "
                f"({ident['highest_auc_disorder']}), in "
                f"{ident['n_cohorts_auc_at_or_above_threshold']} cohort(s) at or above "
                f"{AUC_THRESHOLD:.2f}. That is not a match: random probe sets of the same size "
                "separate those cohorts just as well, because the disorder shifts methylation "
                "genome-wide within a single study."
            )
        return
    scope = ident["interpretation_scope"]
    st.success(
        f"**This signature identifies {ident['top_match']}.** {searched}", icon="✅",
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC, cases vs that study's controls", f"{ident['top_match_auc']:.2f}",
              "95% CI {:.2f}-{:.2f}".format(*ident["top_match_auc_ci"]), delta_color="off")
    c2.metric("Excess over random probes (z)", f"{ident['top_match_z']:.1f}")
    c3.metric("Runner-up", ident["runner_up"].replace(" syndrome", ""),
              f"z {ident['runner_up_z']:.1f}", delta_color="off")
    c4.metric("Margin over runner-up (z)", f"{ident['margin_z']:.1f}")
    if scope == "within_study_only":
        st.warning(
            f"**Within-study only.** Every {ident['top_match']} case in this corpus comes "
            "from a design that cannot give a between-study bound. This match shows the "
            "signature separates that study's cases from its controls. It does not show the "
            "signature would transfer to another laboratory.", icon="⚠️",
        )
    if ident["top_match_too_small_to_interpret"]:
        st.warning("The matched cohort has fewer than 10 cases: too small to interpret.",
                   icon="⚠️")
    if ident["other_candidates"]:
        st.caption("Also passed both criteria, ranked lower: "
                   + ", ".join(ident["other_candidates"]) + ".")


def _cohort_table(result: Evaluation) -> None:
    with st.expander("All 12 cohorts searched"):
        st.caption(
            f"A cohort is a candidate only if AUC is at least {AUC_THRESHOLD:.2f} and the "
            "submitted probes show a larger mean case-control difference than every one of "
            f"{N_NULL:,} random probe sets of the same size."
        )
        cols = ["disorder", "n_cases", "n_controls", "n_probes_used", "auc",
                "mean_abs_delta_beta", "z_vs_random_probes", "exceeds_all_random_sets",
                "is_candidate", "has_own_signature", "interpretation_scope", "interpretation",
                "reason"]
        table = result.cohorts[[c for c in cols if c in result.cohorts.columns]]
        st.dataframe(table.round(3), hide_index=True, use_container_width=True)


def _specificity(result: Evaluation) -> None:
    st.subheader("Specificity row")
    match = result.identification["top_match"]
    st.caption(
        "Fraction scoring positive in every cohort, with profiles anchored on the matched "
        f"cohort ({match}). High in the matched cohort is sensitivity. High anywhere else is "
        "cross-reactivity."
    )
    spec = result.specificity[result.specificity.status == "computed"].copy()
    spec["label"] = spec["scored_group"].str.replace(" syndrome", "").str.replace(
        "pooled_control", "Pooled controls")
    spec["kind"] = spec["scored_group"].map(
        lambda g: "matched cohort" if g == match
        else "pooled controls" if g == "pooled_control" else "other disorder")
    spec["text"] = spec.apply(
        lambda r: f"{r.fraction_positive:.0%}  ({int(r.n_positive)}/{int(r.n)})"
        + ("  n<10" if r.interpretation else ""), axis=1)
    base = alt.Chart(spec).encode(
        y=alt.Y("label:N", sort=spec["label"].tolist(), title=None),
        x=alt.X("fraction_positive:Q", title="Fraction scoring positive",
                scale=alt.Scale(domain=[0, 1.18]),
                axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1])),
    )
    bars = base.mark_bar(size=16, cornerRadiusEnd=3).encode(
        color=alt.Color("kind:N", title=None, scale=alt.Scale(
            domain=["matched cohort", "other disorder", "pooled controls"],
            range=[BLUE, ORANGE, NEUTRAL])),
        tooltip=["scored_group", "n", "n_positive",
                 alt.Tooltip("fraction_positive:Q", format=".1%"), "reason"],
    )
    labels = base.mark_text(align="left", dx=4).encode(text="text:N")
    st.altair_chart((bars + labels).properties(height=26 * len(spec) + 30),
                    use_container_width=True)


def _ceiling(result: Evaluation) -> None:
    ceiling = result.evidence_ceiling
    st.subheader("Evidence ceiling")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strongest band supportable", ceiling["attainable_band"],
              f"limited by {ceiling['limited_by']}", delta_color="off")
    c2.metric("Cohort ceiling",
              ceiling["cohort_ceiling_band"],
              f"{ceiling['n_case']} case / {ceiling['n_control']} control", delta_color="off")
    c3.metric("Signature-size bound", ceiling["signature_size_band"],
              f"{ceiling['n_probes_scored']} probes", delta_color="off")
    c4.metric("Between-study bound possible",
              "yes" if ceiling["between_study_bound_possible"] else "no")
    st.caption(
        "The ceiling is the lower of two bounds. The cohort ceiling is what a perfect "
        "classifier could earn at the matched cohort's size and study structure. The "
        "signature-size bound resamples this signature's probes together with the cohort's "
        "samples: fewer probes give a lower bound. " + ceiling["signature_size_reason"]
    )


def _overlap(result: Evaluation) -> None:
    st.subheader("Overlap with the corpus signatures")
    table = result.overlap.rename(columns={
        "fraction_of_submitted_in_corpus_signature": "fraction_of_submitted",
    })
    st.dataframe(table.round(3), hide_index=True, use_container_width=True)


def _why(result: Evaluation) -> None:
    a = result.attribution
    with st.expander("Why this result?"):
        st.caption("Computed numbers only. No language model and no biological inference.")
        if a["direction_column_supplied"]:
            st.write(
                f"- Direction check: {a['n_direction_consistent']} of "
                f"{a['n_probes_with_direction']} submitted directions agree with the "
                f"direction observed in {a['matched_disorder']}."
            )
            if a["all_directions_inconsistent"]:
                st.error("Every submitted direction is inverted relative to the matched "
                         "cohort.", icon="🚩")
        else:
            st.write("- No direction column was supplied, so direction was not checked.")
        st.write(
            f"- Observed directions among the scored probes: "
            f"{a['observed_direction_counts']['hypo']} lower in cases, "
            f"{a['observed_direction_counts']['hyper']} higher."
        )
        for line in a["explanation"]:
            st.write(f"- {line}")
        st.caption(f"Top contributing probes for {a['exemplar_sample']} "
                   f"({a['contribution_method']}):")
        st.dataframe(pd.DataFrame(a["top_probes"]), hide_index=True, use_container_width=True)


def render() -> None:
    st.header("Test your own signature")
    st.caption(
        "Upload a probe list and see which of the 12 disorder cohorts it identifies, if any. "
        "This page takes probe IDs only - never sample or patient data."
    )
    left, right = st.columns(2)
    uploaded = left.file_uploader("Probe list (CSV or TSV)", type=["csv", "tsv", "txt"],
                                  help=f"Columns: {EXPECTED_COLUMNS}.")
    options = ["(none)", *EXAMPLES]
    # ?example=<file name> preselects an example (with ?page=Test your own signature).
    linked = {v: k for k, v in EXAMPLES.items()}.get(st.query_params.get("example", ""))
    example = right.selectbox("...or use an example", options,
                              index=options.index(linked) if linked else 0)

    if uploaded is not None:
        raw, name = uploaded.getvalue(), uploaded.name
    elif example != "(none)":
        name = EXAMPLES[example]
        raw = (FIXTURE_DIR / name).read_bytes()
    else:
        st.info(f"Choose a file or an example. Expected columns: {EXPECTED_COLUMNS}.")
        return

    try:
        result = _evaluate(raw, name)
    except SignatureInputError as exc:
        st.error(f"**Could not read {name}.** {exc}", icon="🚫")
        return
    except FileNotFoundError as exc:
        st.error(str(exc), icon="🚫")
        return

    _coverage_card(result)
    for warning in result.warnings:
        st.warning(f"**{warning['code']}**: {warning['text']}", icon="⚠️")
    if result.status == "refused":
        st.error(f"**Refused: {result.refusal_code}.** {result.messages[-1]}", icon="🚫")
        summary.render(facts_for_byo(result), key="byo-refused")
    else:
        _verdict(result)
        _cohort_table(result)
        if result.identification["top_match"] is not None:
            _specificity(result)
            _ceiling(result)
            _why(result)
        _overlap(result)
    summary.render(facts_for_byo(result), key="byo")
    notes = [m for m in result.messages if not m.startswith(("Refused", "No match"))]
    if notes:
        with st.expander(f"Notes ({len(notes)})", expanded=result.status == "ok"):
            for note in notes:
                st.write(f"- {note}")
    st.download_button(
        "Download the full evaluation (TSV)",
        result.to_long_table().to_csv(sep="\t", index=False).encode("utf-8"),
        file_name=f"epigrade_evaluation_{name.rsplit('.', 1)[0]}.tsv",
        mime="text/tab-separated-values",
    )
