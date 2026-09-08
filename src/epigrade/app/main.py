"""EpiGrade Streamlit app. Reads ONLY from results/resource/epigrade_v1.json - never
recomputes anything. Run with: streamlit run src/epigrade/app/main.py
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from epigrade import paths

st.set_page_config(page_title="EpiGrade", layout="wide")

RESOURCE_PATH = paths.resource_dir() / "epigrade_v1.json"


@st.cache_data
def load_resource() -> dict:
    with open(RESOURCE_PATH, encoding="utf-8") as f:
        return json.load(f)


def disclaimer_banner() -> None:
    st.warning(
        "**Not for patient use.** EpiGrade is a research and benchmarking tool for "
        "laboratories and researchers. It reports evidence strength for a classifier score "
        "against this public-data corpus - it is not a diagnostic device, never accepts "
        "patient data or raw arrays, and issues no diagnosis.",
        icon="⚠️",
    )


def confounding_status(resource: dict, disorder: str) -> dict | None:
    for row in resource["confounding_gate"]:
        if row["disorder"] == disorder:
            return row
    return None


def page_grade(resource: dict) -> None:
    st.header("Grade a score")
    st.caption(
        "Enter a classifier score for a disorder in this corpus and see what evidence "
        "strength it can honestly support - or why grading is refused."
    )

    # Cohorts that pass the gate first (so the default selection - index 0 - is a disorder that
    # can show a genuine between-study result, not a confounded one), then single-study
    # (fail) cohorts, then not-testable ones last.
    status_priority = {"pass-structural": 0, "fail": 1, "not-testable": 2}
    gate_rows = sorted(
        resource["confounding_gate"],
        key=lambda r: (status_priority.get(r["status"], 9), r["disorder"]),
    )
    disorders = [r["disorder"] for r in gate_rows]
    status_by_disorder = {r["disorder"]: r["status"] for r in gate_rows}
    status_icon = {"pass-structural": "✅", "fail": "⚠️", "not-testable": "🚫"}
    disorder = st.selectbox(
        "Disorder", disorders,
        format_func=lambda d: f"{status_icon.get(status_by_disorder[d], '')} {d}",
    )
    gate = confounding_status(resource, disorder)

    if gate is None:
        st.error("No confounding-gate record for this disorder - cannot grade.")
        return

    if gate["status"] == "not-testable":
        # Genuinely nothing to show: too few cases even to attempt building a classifier, so
        # there is no evidence_bands data for this disorder at all - unlike "fail" below, this
        # isn't a scoping question, there's no number to scope.
        st.error(
            f"**Grading refused.** {disorder}: {gate['reason']}", icon="🚫",
        )
        return

    if gate["status"] == "fail":
        # Confounded by design (single study) - this used to hard-stop grading entirely. It no
        # longer does: the confounding explanation is shown here, alongside (not instead of)
        # whatever within-study-scoped result is available below. The refusal that DOES still
        # hold is a between-study confidence bound - see each row's interpretation_scope.
        st.warning(
            f"**{disorder} fails the confounding gate**: {gate['reason']}", icon="⚠️",
        )
        st.caption(
            "This applies even to Sotos syndrome, whose reproduction is otherwise verified "
            "exactly against the published paper (see the Cohort audit page) - a clean "
            "within-study separation still cannot be told apart from a batch effect when "
            "every case comes from one study. What follows below, if anything, is scoped "
            "accordingly - within this one study only, not a validated cross-study result."
        )
    else:
        st.success(f"{disorder} passes the confounding gate (structural): {gate['reason']}")

    evidence_rows = [r for r in resource["evidence_bands"] if r["disorder"] == disorder]
    if not evidence_rows:
        st.info(
            "No evidence-band results have been computed for this disorder yet in this "
            "resource (no classifier was successfully built/scored this session - see the "
            "Cross-disorder matrix page)."
        )
        return

    prior = st.select_slider(
        "Assumed disease prior", options=sorted({r["prior"] for r in evidence_rows}),
        value=0.10,
    )
    for row in evidence_rows:
        if row["prior"] != prior:
            continue
        st.subheader(row["query_point"].replace("_", " "))

        if row["band"] == "NA":
            # A per-row refusal: even the within-study bootstrap failed to converge (or, for a
            # non-confounded cohort, the between-study one did) - there is truly no band here,
            # not just a scoping caveat on one.
            st.error(f"**No band assigned.** {row['reason']}", icon="🚫")
            continue

        scope = row.get("interpretation_scope", "between_study")
        if scope == "within_study_only":
            st.warning(
                "**Within-study only** - no between-study confidence bound exists for this "
                "single-study cohort. The band below describes how stable the estimate is "
                "inside this one dataset, not whether it would generalize to an independent "
                "study.",
                icon="⚠️",
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LR (point estimate)", f"{row['lr_point_estimate']:.2f}")
        c2.metric(
            "Evidence band" + (" (within-study)" if scope == "within_study_only" else ""),
            row["band"],
        )
        c3.metric("Posterior probability", f"{row['posterior_prob']:.1%}"
                  if row["posterior_prob"] is not None else "NA")
        c4.metric("Cohort size", f"{row['n_case']} case / {row['n_control']} control")
        st.caption(row["reason"])

        render_why_this_result(resource, disorder, row["query_point"])


# Which attribution exemplar cohort(s) a given query point corresponds to - matches how
# scripts/compute_attribution.py assigned query points to cohorts.
_QUERY_POINT_COHORTS = {
    "typical_case_score": ("discovery_case", "discovery_control", "weaver"),
    "typical_missense_vous_score": ("missense_variant",),
}


def render_why_this_result(resource: dict, disorder: str, query_point: str) -> None:
    """Task 5: an expandable panel explaining a real, computed exemplar sample's score using
    only quantities the pipeline already worked out (epigrade.report.attribution) - no LLM
    narrative, no biological inference beyond what was computed."""
    attribution = resource.get("attribution")
    if not attribution or attribution.get("disorder") != disorder:
        return
    wanted_cohorts = _QUERY_POINT_COHORTS.get(query_point, ())
    examples = [e for e in attribution["examples"] if e["cohort_label"] in wanted_cohorts]
    if not examples:
        return

    with st.expander(f"Why this result? ({len(examples)} example sample(s))"):
        st.caption(
            "Computed numbers only - no language model, no biological inference beyond what "
            "was directly derived from the classifier and the beta values below."
        )
        for ex in examples:
            st.markdown(f"**{ex['gsm_accession']}** ({ex['cohort_label']}) - "
                        f"score {ex['score']:+.3f}")
            for line in ex["explanation"]:
                st.write(f"- {line}")
            if ex.get("direction_anomalous"):
                st.error(
                    "Flagged anomalous: scores case-like but most signature probes do not "
                    "move in the expected direction.", icon="🚩",
                )
            top_df = pd.DataFrame(ex["top_probes"])
            if not top_df.empty:
                st.caption(f"Top contributing probes ({ex['contribution_method']}):")
                st.dataframe(top_df, use_container_width=True, hide_index=True)


def page_calibration(resource: dict) -> None:
    st.header("Calibration curves")
    st.caption("Every evidence-band result computed this session, with the bootstrap CI.")
    df = pd.DataFrame(resource["evidence_bands"])
    if df.empty:
        st.info("No evidence-band results in this resource yet.")
        return
    st.dataframe(df, use_container_width=True)

    st.subheader("Attainable evidence ceiling vs. cohort size")
    ceiling_df = pd.DataFrame(resource["attainable_ceiling"])
    if not ceiling_df.empty:
        st.line_chart(ceiling_df.set_index("n_case")["points_conservative"])
        st.caption(
            "A PERFECT classifier's attainable evidence points at each cohort size, "
            "independent of any real classifier's quality - see docs/METHODS.md. Rows with "
            "interpretation_scope='within_study_only' (typically n<30, single assumed study) "
            "show what a perfect classifier's within-dataset stability looks like, not a "
            "validated between-study bound - a perfectly-separable toy classifier can look "
            "'Strong' even at very small n for exactly this reason, which is itself the "
            "point: within-study numbers alone can be misleadingly reassuring."
        )
        st.dataframe(ceiling_df, use_container_width=True)


def page_cohort_audit(resource: dict) -> None:
    st.header("Cohort audit")
    st.caption(
        "Confounding gate status per disorder. Failing cohorts are shown, not hidden - "
        "most disorders in this corpus come from a single study, which is the expected, "
        "reportable finding per the project design, not a bug."
    )
    df = pd.DataFrame(resource["confounding_gate"])

    def _style_status(val):
        color = {"fail": "#ffcccc", "not-testable": "#fff3cd",
                 "pass-structural": "#d4edda"}.get(val, "")
        return f"background-color: {color}"

    st.dataframe(
        df.style.map(_style_status, subset=["status"]), use_container_width=True,
    )

    st.subheader("Sotos syndrome reproduction (phase 3)")
    summary_df = pd.DataFrame(resource["demo"]["reproduction_summary"])
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Leave-one-study-out: Silver-Russell syndrome")
    loso_df = pd.DataFrame(resource["leave_one_study_out"]["silver_russell_syndrome"])
    if not loso_df.empty:
        st.dataframe(loso_df, use_container_width=True)

    st.subheader("Label harmonization")
    h = resource["harmonisation"]
    counts = {r["category"]: r["count"] for r in h.get("sample_counts", [])}
    if counts:
        c1, c2, c3 = st.columns(3)
        c1.metric("Samples harvested", counts.get("harvested", "NA"))
        c2.metric("Retained for analysis", counts.get("retained_for_analysis", "NA"))
        c3.metric("Excluded", counts.get("excluded_total", "NA"))
        st.caption(
            "Retained + excluded sum to harvested exactly (asserted when this table is "
            "built - see scripts/phase1_report.py). Excluded, by reason:"
        )
        reason_rows = [
            {"reason": r["category"].replace("_", " "), "n": r["count"]}
            for r in h["sample_counts"] if r.get("kind") == "excluded_reason"
        ]
        st.dataframe(pd.DataFrame(reason_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No sample_counts data in this resource yet.")

    rate = h["hand_curation_agreement_rate"]
    st.metric("AI cross-check agreement", f"{rate:.0%}" if rate is not None else "NA")
    st.caption(
        "This is an AI-performed cross-check against a second, independent implementation "
        "of the same rule-based judgment - NOT independent human clinical curation. See "
        "'Human-curated agreement' below for the (separate, not yet available) human number, "
        "and docs/METHODS.md for the full distinction. " + h["hand_curation_note"]
    )

    st.subheader("Human-curated agreement")
    human = resource.get("human_curation")
    if human is None:
        st.info(
            "No human-curated agreement data in this resource yet - a curator hasn't filled "
            "in data/external/human_curated.csv. Never substituted with the AI number above."
        )
    elif human.get("status") == "pending":
        st.warning(f"**Human curation pending**: {human.get('reason', 'not yet filled in')}")
    else:
        st.metric("Human-vs-pipeline agreement", f"{human['agreement_rate']:.0%}")
        st.caption(
            f"{human.get('n_scoreable', '?')} human-scored comparisons - reported separately "
            "from the AI cross-check above, never merged into one figure."
        )


def page_cross_disorder(resource: dict) -> None:
    st.header("Cross-disorder specificity matrix")
    computed = pd.DataFrame(resource["cross_disorder_matrix"]["computed"])
    if not computed.empty:
        st.subheader("Computed this session")
        st.dataframe(computed, use_container_width=True)
    not_computed = pd.DataFrame(resource["cross_disorder_matrix"]["not_computed"])
    if not not_computed.empty:
        st.subheader("Not computed (honest NA, not a silent omission)")
        st.dataframe(not_computed, use_container_width=True)
        st.caption(
            "These disorders' series matrices were not downloaded in this session due to "
            "GEO FTP bandwidth/time constraints - not because they were excluded on purpose."
        )


def main() -> None:
    st.title("EpiGrade")
    st.caption(
        "An open, reproducible benchmark for DNA methylation episignature classifiers."
    )
    disclaimer_banner()

    if not RESOURCE_PATH.exists():
        st.error(f"Resource file not found at {RESOURCE_PATH}. Run "
                 "`python -m epigrade.report.resource` first.")
        return

    resource = load_resource()
    st.caption(
        f"Resource `{resource['resource_version']}` generated "
        f"{resource['generated_from']['generated_at_utc']} from git "
        f"`{resource['generated_from']['git_sha'][:10]}`."
    )

    page = st.sidebar.radio(
        "Page", ["Grade a score", "Calibration curves", "Cohort audit", "Cross-disorder matrix"],
    )
    if page == "Grade a score":
        page_grade(resource)
    elif page == "Calibration curves":
        page_calibration(resource)
    elif page == "Cohort audit":
        page_cohort_audit(resource)
    elif page == "Cross-disorder matrix":
        page_cross_disorder(resource)


if __name__ == "__main__":
    main()
