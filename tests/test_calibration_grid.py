"""The precomputed calibration grid behind the Grade a score input."""

import numpy as np
import pandas as pd
import pytest

from epigrade import paths
from epigrade.corpus import registry
from epigrade.explain import facts as F
from epigrade.explain.explain import validate
from epigrade.explain.template import explain as template_explain

TABLES = paths.repo_root() / "results" / "tables"
BANDS = {"No evidence", "Supporting", "Moderate", "Strong", "Very Strong", "NA"}
WITH = [d.name for d in registry.DISORDERS if d.signature != registry.NO_SIGNATURE]
WITHOUT = [d.name for d in registry.DISORDERS if d.signature == registry.NO_SIGNATURE]


@pytest.fixture(scope="module")
def grid():
    path = TABLES / "calibration_grid.tsv"
    if not path.exists():
        pytest.skip("calibration_grid.tsv not generated yet")
    return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""])


@pytest.fixture(scope="module")
def evidence():
    return pd.read_csv(TABLES / "evidence_bands.tsv", sep="\t", keep_default_na=False,
                       na_values=[""])


def test_41_scores_by_3_priors_for_each_disorder_with_a_classifier(grid):
    assert sorted(grid.disorder.unique()) == sorted(WITH)
    assert not set(WITHOUT) & set(grid.disorder)
    for disorder, block in grid.groupby("disorder"):
        assert block.query_score.nunique() == 41, disorder
        assert sorted(block.prior.unique()) == [0.05, 0.1, 0.2], disorder
        assert len(block) == 41 * 3
        assert not block.duplicated(["query_score", "prior"]).any()


def test_same_columns_as_evidence_bands(grid, evidence):
    assert list(grid.columns) == list(evidence.columns)
    assert set(grid.query_point) == {"grid"}


def test_every_band_is_an_allowed_name(grid):
    assert set(grid.band) <= BANDS


def test_zero_is_an_exact_grid_point_for_every_disorder(grid):
    for disorder, block in grid.groupby("disorder"):
        assert (block.query_score == 0.0).any(), disorder


def test_grid_covers_the_cohort_and_agrees_with_evidence_bands_at_the_typical_score(
    grid, evidence,
):
    typical = evidence[evidence.query_point == "typical_case_score"]
    for row in typical.itertuples():
        block = grid[(grid.disorder == row.disorder) & (grid.prior == row.prior)]
        points = np.sort(block.query_score.unique())
        assert points[0] < row.query_score < points[-1]
        nearest = block.iloc[(block.query_score - row.query_score).abs().argmin()]
        assert nearest.interpretation_scope == row.interpretation_scope, row.disorder
        if nearest.band != row.band:
            # Allowed only within the grid's resolution: the band must change between the
            # two grid points that bracket the typical score.
            below = block[block.query_score <= row.query_score].iloc[-1]
            above = block[block.query_score >= row.query_score].iloc[0]
            assert row.band in {below.band, above.band}, row.disorder


def test_no_forbidden_evidence_anchor(grid):
    text = grid.astype(str).to_csv(sep="\t")
    assert "PS3_functional" not in text
    assert set(grid.evidence_anchor) == {"PP4_diagnostic"}


def test_grade_fact_sheet_for_a_grid_point_carries_the_grid_numbers(grid):
    block = grid[(grid.disorder == "Silver-Russell syndrome") & (grid.prior == 0.1)]
    row = block.iloc[5]  # a grid point away from both reference scores
    f = F.facts_for_grade(row.disorder, row.query_score, row.prior)
    assert f["query_point"] == "grid" and f["score"] == round(row.query_score, 4)
    assert f["band"] == row.band
    if pd.notna(row.lr_point_estimate):
        assert f["likelihood_ratio"] == round(row.lr_point_estimate, 1)
    assert validate(template_explain(f), f) == []
