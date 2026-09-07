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

    disorders = sorted({row["disorder"] for row in resource["confounding_gate"]})
    disorder = st.selectbox("Disorder", disorders)
    gate = confounding_status(resource, disorder)

    if gate is None:
        st.error("No confounding-gate record for this disorder - cannot grade.")
        return

    if gate["status"] == "fail":
        st.error(
            f"**Grading refused.** {disorder} fails the confounding gate: {gate['reason']}",
            icon="🚫",
        )
        st.info(
            "This applies even to Sotos syndrome, whose reproduction is otherwise verified "
            "exactly against the published paper (see the Cohort audit page) - a clean "
            "within-study separation still cannot be told apart from a batch effect when "
            "every case comes from one study. That is the point of this gate."
        )
        return
    if gate["status"] == "not-testable":
        st.error(
            f"**Grading refused.** {disorder}: {gate['reason']}", icon="🚫",
        )
        return

    st.success(f"{disorder} passes the confounding gate (structural): {gate['reason']}")

    evidence_rows = [r for r in resource["evidence_bands"] if r["disorder"] == disorder]
    if not evidence_rows:
        st.warning(
            "This disorder passes the structural confounding gate, but no evidence-band "
            "results have been computed for it yet in this resource (no classifier was "
            "successfully built/scored this session - see the Cross-disorder matrix page)."
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
            # A per-row refusal, distinct from (and in addition to) the confounding-gate
            # refusal above: even a disorder that passes the structural gate can still hit
            # this if, say, a bootstrap failed to converge for one specific query point.
            st.error(f"**No band assigned.** {row['reason']}", icon="🚫")
            wl, wh = row.get("within_study_ci_low"), row.get("within_study_ci_high")
            if wl is not None and wh is not None:
                st.caption(
                    f"For reference only (never used to assign a band): a within-study "
                    f"sample-level bootstrap gives LR ∈ [{wl:.2f}, {wh:.2f}]. This ignores "
                    "study structure entirely and says nothing about between-study "
                    "generalization - it is not a substitute for the refused band above."
                )
            continue

        c1, c2, c3 = st.columns(3)
        c1.metric("LR (point estimate)", f"{row['lr_point_estimate']:.2f}")
        c2.metric("Evidence band (conservative)", row["band"])
        c3.metric("Posterior probability", f"{row['posterior_prob']:.1%}"
                  if row["posterior_prob"] is not None else "NA")
        st.caption(row["reason"])


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
            "independent of any real classifier's quality - see docs/METHODS.md."
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
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples harvested", h["n_samples_total"] or "NA")
    c2.metric("Flagged/excluded", h["n_flagged_or_excluded"])
    rate = h["hand_curation_agreement_rate"]
    c3.metric("Hand-curation agreement",
              f"{rate:.0%}" if rate is not None else "NA")
    st.caption(h["hand_curation_note"])


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
