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
