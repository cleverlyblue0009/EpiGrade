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


def test_na_band_survives_as_the_string_na_not_null_or_nan():
    with open(RESOURCE_PATH, encoding="utf-8") as f:
        resource = json.load(f)
    na_rows = [r for r in resource["evidence_bands"] if r["disorder"] == "Sotos syndrome"]
    assert na_rows, "expected at least one Sotos evidence row"
    assert all(r["band"] == "NA" for r in na_rows)


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
