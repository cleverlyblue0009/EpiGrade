"""Phase 1 corpus tests: the leakage guard on synthetic data, the registry's shape, and the
asserted corpus facts on the checked-in tables."""

import numpy as np
import pandas as pd
import pytest

from epigrade import paths
from epigrade.corpus import registry
from epigrade.signature.choufani import SIGNATURE_SOURCE as PUBLISHED
from epigrade.signature.generic import (
    REDERIVED_NOT_PUBLISHED,
    classifier_from_probes,
    get_thresholds,
    mannwhitney_p,
    score_matrix,
)

SOTOS = "Sotos syndrome"
MULTI_STUDY = ["Silver-Russell syndrome", "Kabuki syndrome type 1",
               "Nicolaides-Baraitser syndrome"]
NO_SIGNATURE = ["Weaver syndrome", "Claes-Jensen syndrome"]


def _table(name: str) -> pd.DataFrame:
    path = paths.repo_root() / "results" / name
    if not path.exists():
        pytest.skip(f"{name} not generated yet")
    return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""])


# --------------------------------------------------------------------------- leakage guard


def _noisy_cohort(n_probes=300, n_case=12, n_control=12, effect=0.12, noise=0.15, seed=3):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.2, 0.8, n_probes)
    shift = effect * np.where(np.arange(n_probes) % 2 == 0, 1.0, -1.0)
    cases = [f"CASE{i}" for i in range(n_case)]
    controls = [f"CTRL{i}" for i in range(n_control)]
    data = {s: np.clip(base + shift + rng.normal(0, noise, n_probes), 0, 1) for s in cases}
    data |= {s: np.clip(base + rng.normal(0, noise, n_probes), 0, 1) for s in controls}
    return pd.DataFrame(data, index=[f"cg{i:08d}" for i in range(n_probes)]), cases, controls


def _separation(scores: pd.DataFrame, cases, controls) -> float:
    return scores.loc[cases, "score"].mean() - scores.loc[controls, "score"].mean()


def test_leaking_training_samples_into_their_own_profile_inflates_separation():
    """The guard-is-live test: disable leave-one-out and separation must get better. If it
    did not, the guard would be decoration."""
    beta, cases, controls = _noisy_cohort()
    clf = classifier_from_probes("Toy", beta.index, beta, cases, controls)
    guarded = score_matrix(clf, beta, cases + controls)
    leaked = score_matrix(clf, beta, cases + controls, leave_one_out=False)
    assert guarded["loo_guarded"].all() and not leaked["loo_guarded"].any()
    assert _separation(leaked, cases, controls) > _separation(guarded, cases, controls)


def test_guard_only_touches_training_samples():
    beta, cases, controls = _noisy_cohort()
    clf = classifier_from_probes("Toy", beta.index, beta, cases[:8], controls[:8])
    held_out = cases[8:] + controls[8:]
    guarded = score_matrix(clf, beta, cases + controls)
    leaked = score_matrix(clf, beta, cases + controls, leave_one_out=False)
    assert not guarded.loc[held_out, "loo_guarded"].any()
    np.testing.assert_allclose(guarded.loc[held_out, "score"], leaked.loc[held_out, "score"])
    assert guarded.loc[cases[:8], "loo_guarded"].all()


def test_with_no_real_effect_guarded_scores_do_not_separate_but_leaked_ones_do():
    beta, cases, controls = _noisy_cohort(effect=0.0, noise=0.2)
    clf = classifier_from_probes("Toy", beta.index, beta, cases, controls)
    guarded = score_matrix(clf, beta, cases + controls)
    leaked = score_matrix(clf, beta, cases + controls, leave_one_out=False)
    assert _separation(leaked, cases, controls) > 0.05
    assert _separation(guarded, cases, controls) < _separation(leaked, cases, controls) / 2


def test_samples_with_too_few_signature_probes_get_no_score():
    beta, cases, controls = _noisy_cohort()
    clf = classifier_from_probes("Toy", beta.index, beta, cases, controls)
    other_platform = beta.iloc[:5].rename(columns={cases[0]: "NEW1"})[["NEW1"]]
    assert score_matrix(clf, other_platform, ["NEW1"]).empty


def test_mannwhitney_rows_with_missing_values_match_scipy_row_by_row():
    from scipy.stats import mannwhitneyu

    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, (40, 12)), rng.normal(0.5, 1, (40, 10))
    a[3, 2] = b[7, 1] = a[9, 0] = np.nan
    got = mannwhitney_p(a, b)
    for i in (0, 3, 7, 9, 20):
        x, y = a[i][~np.isnan(a[i])], b[i][~np.isnan(b[i])]
        expected = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")[1]
        assert got[i] == pytest.approx(expected)


# --------------------------------------------------------------------------- registry


def test_registry_is_twelve_disorders_ten_with_a_signature():
    assert len(registry.DISORDERS) == 12
    without = [d.name for d in registry.DISORDERS if d.signature == registry.NO_SIGNATURE]
    assert sorted(without) == sorted(NO_SIGNATURE)
    published = [d.name for d in registry.DISORDERS if d.signature == registry.PUBLISHED]
    assert published == [SOTOS]
    multi = [d.name for d in registry.DISORDERS if len(d.arms) >= 2]
    assert sorted(multi) == sorted(MULTI_STUDY)


def test_only_sotos_overrides_the_default_thresholds():
    default = get_thresholds(None)
    assert default == {"multiple_testing_method": "fdr_bh", "alpha": 0.05,
                       "effect_size_floor": 0.10}
    for d in registry.DISORDERS:
        if d.name != SOTOS:
            assert get_thresholds(d.name) == default, f"{d.name} has a threshold override"
    assert get_thresholds(SOTOS)["multiple_testing_method"] == "bonferroni"


# --------------------------------------------------------------------------- derivations


def test_ten_canonical_signatures_are_recorded_and_two_are_marked_too_small():
    df = _table("tables/signature_derivation.tsv")
    canonical = df[df.derivation_scope == "canonical"].set_index("disorder")
    assert len(canonical) == 12
    assert (canonical.status == "built").sum() == 10
    assert (canonical.loc[NO_SIGNATURE, "status"] == "not_attempted_too_few_cases").all()
    for col in ("multiple_testing_method", "alpha", "effect_size_floor", "n_probes_significant",
                "n_probes_after_effect_filter", "n_cases", "n_controls", "signature_source"):
        assert col in df.columns


def test_signature_source_is_published_for_sotos_only_everywhere():
    derivation = _table("tables/signature_derivation.tsv")
    built = derivation[derivation.status == "built"]
    assert set(built.loc[built.disorder == SOTOS, "signature_source"]) == {PUBLISHED}
    assert set(built.loc[built.disorder != SOTOS, "signature_source"]) == {
        REDERIVED_NOT_PUBLISHED
    }
    index = _table("signatures/index.tsv").set_index("disorder")
    assert len(index) == 10
    assert index.loc[SOTOS, "signature_source"] == PUBLISHED and index.loc[SOTOS,
                                                                           "n_probes"] == 7085
    assert set(index.drop(SOTOS)["signature_source"]) == {REDERIVED_NOT_PUBLISHED}


def test_no_rederived_derivation_used_anything_but_the_default_thresholds():
    """One documented parameter change (Bonferroni/20% -> FDR-BH/10%), then stop. Any row at
    other thresholds would be evidence of threshold-hunting."""
    df = _table("tables/signature_derivation.tsv")
    rederived = df[df.disorder != SOTOS]
    assert set(rederived.multiple_testing_method) == {"fdr_bh"}
    assert set(rederived.alpha.astype(float)) == {0.05}
    assert set(rederived.effect_size_floor.astype(float)) == {0.1}


def test_borrowed_controls_are_flagged():
    df = _table("tables/signature_derivation.tsv")
    canonical = df[df.derivation_scope == "canonical"].set_index("disorder")
    cross = set(canonical.index[canonical.controls_source == "cross_series"])
    assert cross == {"Coffin-Siris syndrome", "Nicolaides-Baraitser syndrome"}


# --------------------------------------------------------------------------- LOSO


def test_leave_one_study_out_ran_both_directions_for_all_three_eligible_disorders():
    df = _table("tables/leave_one_study_out.tsv")
    assert sorted(df.disorder.unique()) == sorted(MULTI_STUDY)
    assert (df.groupby("disorder").size() == 2).all()
    for _, pair in df.groupby("disorder"):
        assert set(pair.train_study) == set(pair.test_study)


def test_silver_russell_cross_study_result_is_the_asserted_one():
    df = _table("tables/leave_one_study_out.tsv")
    forward = df[(df.train_study == "GSE104451") & (df.test_study == "GSE55491")].iloc[0]
    assert forward.status == "computed"
    assert forward.test_case_sensitivity == "5/18"
    assert forward.test_control_specificity == "6/6"
    assert int(forward.n_signature_probes) == 64
    reverse = df[(df.train_study == "GSE55491") & (df.test_study == "GSE104451")].iloc[0]
    assert reverse.status == "not_computed" and "n_control=6" in reverse.reason


# --------------------------------------------------------------------------- matrix


def test_cross_disorder_matrix_is_a_complete_grid_with_no_unexplained_gap():
    df = _table("tables/cross_disorder_matrix.tsv")
    names = [d.name for d in registry.DISORDERS]
    assert sorted(df.classifier.unique()) == sorted(names)
    assert sorted(df.scored_disorder.unique()) == sorted(names + ["pooled_control"])
    assert len(df) == 12 * 13 and not df.duplicated(["classifier", "scored_disorder"]).any()
    computed = df[df.status == "computed"]
    assert len(computed) == 10 * 13
    assert set(df[df.status == "not_computed"].classifier) == set(NO_SIGNATURE)
    small = computed[computed.n.astype(int) < 10]
    assert set(small.scored_disorder) == set(NO_SIGNATURE)
    assert (small.interpretation == "too small to interpret").all()
    assert computed[computed.n.astype(int) >= 10].interpretation.isna().all()


def test_charge_fires_on_kabuki_at_the_asserted_rate():
    df = _table("tables/cross_disorder_matrix.tsv")
    cell = df[(df.classifier == "CHARGE syndrome")
              & (df.scored_disorder == "Kabuki syndrome type 1")].iloc[0]
    assert int(cell.n) == 37 and int(cell.n_positive) == 14
    assert float(cell.fraction_positive) == pytest.approx(0.378)


def test_matrix_case_counts_match_the_corpus():
    df = _table("tables/cross_disorder_matrix.tsv")
    sotos_row = df[df.classifier == SOTOS].set_index("scored_disorder")
    expected = {"Silver-Russell syndrome": 63, SOTOS: 38, "Kabuki syndrome type 1": 37,
                "Down syndrome": 29, "Coffin-Siris syndrome": 21, "Williams syndrome": 20,
                "CHARGE syndrome": 19, "Nicolaides-Baraitser syndrome": 16, "ICF syndrome": 15,
                "7q11.23 duplication syndrome": 10, "Weaver syndrome": 8,
                "Claes-Jensen syndrome": 6}
    for disorder, n in expected.items():
        assert int(sotos_row.loc[disorder, "n"]) == n, disorder


# --------------------------------------------------------------------------- calibration


def test_a_single_study_side_never_gets_a_between_study_bound():
    df = _table("tables/evidence_bands.tsv")
    assert df.disorder.nunique() == 10
    single = df[(df.n_studies_case < 2) | (df.n_studies_control < 2)]
    assert (single.interpretation_scope == "within_study_only").all()
    assert single.lr_ci_low.isna().all() and single.lr_ci_high.isna().all()
    assert single.within_study_ci_low.notna().all()
    multi = df[(df.n_studies_case >= 2) & (df.n_studies_control >= 2)]
    assert (multi.interpretation_scope == "between_study").all()
    assert set(multi.disorder) == {"Silver-Russell syndrome", "Kabuki syndrome type 1"}


def test_every_disorder_has_a_ceiling_from_its_real_cohort_size():
    ceiling = _table("tables/attainable_ceiling.tsv").set_index("disorder")
    assert len(ceiling) == 12
    assert int(ceiling.loc[SOTOS, "n_case"]) == 38
    assert not ceiling.loc[SOTOS, "between_study_bound_possible"]
    assert ceiling.loc["Silver-Russell syndrome", "between_study_bound_possible"]


def test_sotos_reproduction_passes_all_four_checks():
    df = _table("tables/sotos_reproduction_summary.tsv")
    assert list(df.check) == ["discovery_separation", "weaver_negative", "missense_split",
                              "replication"]
    assert df.reproduced.astype(str).str.lower().eq("true").all()


def test_verdicts_cover_twelve_disorders_in_four_categories():
    df = _table("tables/disorder_verdicts.tsv").set_index("disorder")
    assert len(df) == 12
    assert set(df.verdict) <= {"graded", "single-study", "too-few-cases", "not-downloaded"}
    assert sorted(df.index[df.verdict == "graded"]) == sorted(MULTI_STUDY)
    assert sorted(df.index[df.verdict == "too-few-cases"]) == sorted(NO_SIGNATURE)


@pytest.mark.parametrize("name", [
    "cross_disorder_matrix.png", "leave_one_study_out.png", "signature_sizes.png",
    "attainable_ceiling.png", "disorder_verdict_summary.png",
    *[f"{d.slug}_scores.png" for d in registry.DISORDERS
      if d.signature != registry.NO_SIGNATURE],
])
def test_required_figures_exist_in_both_variants(name):
    figures = paths.repo_root() / "results" / "figures"
    assert (figures / name).exists() and (figures / "slides" / name).exists()
