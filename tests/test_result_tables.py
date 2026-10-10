"""Invariants on the checked-in result tables. These guard against provenance and
completeness bugs that were found in real output, not hypothetical ones."""

import pandas as pd
import pytest

from epigrade import paths
from epigrade.signature.choufani import SIGNATURE_SOURCE as PUBLISHED
from epigrade.signature.generic import REDERIVED_NOT_PUBLISHED

SOTOS = "Sotos syndrome"


def _table(name: str) -> pd.DataFrame:
    path = paths.repo_root() / "results" / "tables" / name
    if not path.exists():
        pytest.skip(f"{name} not generated yet")
    return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""])


def test_only_sotos_may_claim_a_published_probe_list():
    """Regression: three Silver-Russell rows were once tagged published_probe_list. Silver-
    Russell has no published probe list here - it is a Path B re-derivation."""
    for name in ("cross_disorder_matrix.tsv", "signature_derivation.tsv",
                 "leave_one_study_out.tsv"):
        df = _table(name)
        key = "classifier" if "classifier" in df.columns else "disorder"
        published = df[df["signature_source"] == PUBLISHED]
        assert set(published[key]) <= {SOTOS}, f"{name}: non-Sotos row claims {PUBLISHED}"


def test_every_computed_matrix_row_names_its_signature_source():
    df = _table("cross_disorder_matrix.tsv")
    computed = df[df["status"] == "computed"]
    assert computed["signature_source"].isin([PUBLISHED, REDERIVED_NOT_PUBLISHED]).all()
    assert (computed.loc[computed["classifier"] == SOTOS, "signature_source"] == PUBLISHED).all()
    assert (
        computed.loc[computed["classifier"] != SOTOS, "signature_source"]
        == REDERIVED_NOT_PUBLISHED
    ).all()


def test_computed_matrix_rows_carry_a_value_and_not_computed_rows_carry_a_reason():
    """Regression: the Silver-Russell self row was status='computed' with a blank
    fraction_positive."""
    df = _table("cross_disorder_matrix.tsv")
    assert set(df["status"]) <= {"computed", "not_computed"}
    computed = df[df["status"] == "computed"]
    assert computed["fraction_positive"].notna().all()
    assert computed["n"].notna().all()
    not_computed = df[df["status"] == "not_computed"]
    assert not_computed["fraction_positive"].isna().all()
    assert (not_computed["reason"].fillna("").str.len() > 0).all()


def test_leave_one_study_out_is_disorder_agnostic():
    df = _table("leave_one_study_out.tsv")
    assert "disorder" in df.columns and df["disorder"].notna().all()
    assert not (paths.repo_root() / "results" / "tables" / "srs_leave_one_study_out.tsv").exists()
    not_computed = df[df["status"] == "not_computed"]
    assert (not_computed["reason"].fillna("").str.len() > 0).all()
    assert (not_computed["test_case_sensitivity"] == "NA").all()
