"""Regression tests for the label harmonizer against the three accessions with published
ground truth (see the project spec's `ground_truth` block). Skipped if the metadata hasn't
been harvested yet (data/ is gitignored - a fresh clone needs to run the harvest once)."""

import pandas as pd
import pytest

from epigrade import paths
from epigrade.acquire.harmonize import harmonize

SAMPLES_PATH = paths.interim_dir() / "samples.parquet"

pytestmark = pytest.mark.skipif(
    not SAMPLES_PATH.exists(),
    reason="samples.parquet not harvested yet - run `python -m epigrade.acquire.harvest`",
)


@pytest.fixture(scope="module")
def harmonized():
    df = pd.read_parquet(SAMPLES_PATH)
    return harmonize(df)


def _counts(harmonized, series_id, group_cols):
    g = harmonized[harmonized.series_id == series_id]
    return g.groupby(group_cols, dropna=False).size()


def test_gse74432_sotos_case_control_split(harmonized):
    g = harmonized[harmonized.series_id == "GSE74432"]
    c = _counts(harmonized, "GSE74432", ["role", "disorder"])
    assert c[("case", "Sotos syndrome")] == 38  # 41 total minus 3 fibroblast
    assert c[("case", "Weaver syndrome")] == 8
    assert c[("under_test", "Sotos syndrome")] == 16  # the NSD1-variant VOUS cohort
    assert (g["role"] == "matched_control").sum() == 53  # 57 total minus 4 fibroblast


def test_gse74432_fibroblast_flagged_wrong_tissue(harmonized):
    g = harmonized[harmonized.series_id == "GSE74432"]
    fibro = g[g["source_name_ch1"].str.contains("fibroblast", case=False, na=False)]
    assert len(fibro) == 7
    assert (fibro["role"] == "wrong_tissue").all()


def test_gse97362_discovery_cohorts(harmonized):
    c = _counts(harmonized, "GSE97362", ["role", "sample_type"])
    assert c[("case", "CHD7 LOF discovery cohort")] == 19
    assert c[("case", "KMT2D LOF discovery cohort")] == 11


def test_gse97362_validation_and_sequence_variant_excluded_from_case(harmonized):
    g = harmonized[harmonized.series_id == "GSE97362"]
    validation_like = g[g["sample_type"].isin(["Validation cohort", "CHD7 sequence variant",
                                                "KMT2D sequence variant"])]
    assert (validation_like["role"] != "case").all()


def test_gse116300_role_split(harmonized):
    c = _counts(harmonized, "GSE116300", ["role"])
    assert c["matched_control"] == 9
    assert c["unaffected_relative"] == 6  # the 6 "case parent" samples
    assert c["case"] + c.get("under_test", 0) == 29  # 26 confirmed + 3 VUS


def test_every_role_is_from_the_enum(harmonized):
    valid = {"case", "matched_control", "population_control", "unaffected_relative",
             "under_test", "cell_line", "wrong_tissue", "exclude_other"}
    assert set(harmonized["role"].unique()) <= valid
