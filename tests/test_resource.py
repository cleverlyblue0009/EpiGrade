"""Smoke tests for the app's resource export - the one file the Streamlit app is allowed to
read from."""

import json

from epigrade import paths
from epigrade.report.resource import build_resource

RESOURCE_PATH = paths.resource_dir() / "epigrade_v1.json"


def test_build_resource_has_required_top_level_sections():
    resource = build_resource()
    for key in ("resource_version", "generated_from", "disclaimer", "confounding_gate",
                "demo", "evidence_bands", "cross_disorder_matrix"):
        assert key in resource


def test_disclaimer_mentions_not_for_patient_use():
    resource = build_resource()
    assert "not" in resource["disclaimer"].lower()
    assert "diagnos" in resource["disclaimer"].lower()


def test_checked_in_resource_file_is_valid_json_and_matches_shape():
    """This is the file the app actually reads - must exist and be well-formed on a fresh
    clone (it's tracked in git, unlike everything under data/)."""
    assert RESOURCE_PATH.exists(), (
        "results/resource/epigrade_v1.json is missing - run `python -m epigrade.report.resource`"
    )
    with open(RESOURCE_PATH, encoding="utf-8") as f:
        resource = json.load(f)
    assert resource["resource_version"] == "epigrade_v1"
    assert isinstance(resource["confounding_gate"], list) and len(resource["confounding_gate"]) > 0


def test_resource_json_is_strictly_valid_no_nan_tokens():
    """Regression test for a real bug caught during Task 2's clean-clone check: pandas'
    default read_csv NA-string handling silently turned our literal band="NA" text into a
    float NaN, which json.dump then wrote as the non-standard `NaN` token - not valid JSON per
    spec, and would fail to parse in most non-Python JSON consumers."""
    text = RESOURCE_PATH.read_text(encoding="utf-8")
    assert "NaN" not in text, "resource JSON contains a non-standard NaN token"


def test_sotos_evidence_rows_are_within_study_scoped_not_between_study():
    """Sotos is single-study, so its evidence rows must never claim a between-study confidence
    bound (lr_ci_low/high stay null - a degenerate zero-width interval is not real precision).
    They DO now get a real, computed band instead of a blanket 'NA' refusal, explicitly labeled
    interpretation_scope='within_study_only' so the app can show the number and the caveat
    together rather than one instead of the other."""
    with open(RESOURCE_PATH, encoding="utf-8") as f:
        resource = json.load(f)
    sotos_rows = [r for r in resource["evidence_bands"] if r["disorder"] == "Sotos syndrome"]
    assert sotos_rows, "expected at least one Sotos evidence row"
    for r in sotos_rows:
        assert r["interpretation_scope"] == "within_study_only"
        assert r["lr_ci_low"] is None and r["lr_ci_high"] is None, (
            "a single-study cohort must never be given a between-study confidence bound"
        )
        assert r["band"] != "NA", "a real within-study-scoped band should be computed here"
        assert r["within_study_ci_low"] is not None and r["within_study_ci_high"] is not None
        assert "between-study" in r["reason"]


def test_harmonisation_scoreable_count_excludes_unresolvable_rows():
    """Regression test for a real bug: r.get('agree') is not None doesn't catch a float NaN
    (the 19 UNRESOLVABLE hand-curation rows read back from the TSV as NaN, not None), so they
    were being silently counted as both scoreable and 'agreed' (bool(nan) is True in Python),
    inflating n_scoreable from 81 to 100. Fixed with pd.notna()."""
    resource = build_resource()
    h = resource["harmonisation"]
    assert h["hand_curation_n_scoreable"] == 81, (
        f"expected exactly 81 scoreable rows (100 sampled - 19 UNRESOLVABLE), "
        f"got {h['hand_curation_n_scoreable']}"
    )


def test_sample_counts_add_up_and_are_not_the_same_number_twice():
    """Regression test for a real bug: the Cohort audit page reported 'Samples harvested 871'
    and 'Flagged/excluded 871' as the identical number, because both were being read from
    sample_triage.tsv - which only ever contains EXCLUDED samples. sample_counts.tsv
    (scripts/phase1_report.py) now provides real, mutually-exclusive categories."""
    resource = build_resource()
    counts = {r["category"]: r["count"] for r in resource["harmonisation"]["sample_counts"]}
    assert counts["harvested"] > 0
    assert counts["harvested"] != counts["excluded_total"], (
        "harvested and excluded must not be the same number - that was the original bug"
    )
    assert counts["retained_for_analysis"] + counts["excluded_total"] == counts["harvested"]
    reasons = [
        r["count"] for r in resource["harmonisation"]["sample_counts"]
        if r["kind"] == "excluded_reason"
    ]
    assert sum(reasons) == counts["excluded_total"]


def test_sotos_fails_its_own_confounding_gate_in_the_resource():
    """The app must refuse to grade Sotos even though its reproduction is fully verified -
    that's the whole point of the gate. Locking this in as a regression test."""
    with open(RESOURCE_PATH, encoding="utf-8") as f:
        resource = json.load(f)
    sotos_gate = next(
        r for r in resource["confounding_gate"] if r["disorder"] == "Sotos syndrome"
    )
    assert sotos_gate["status"] == "fail"
    assert sotos_gate["confounded_by_design"] is True
