"""Bring-your-own-signature engine: every refusal rule, the five fixtures, and Form 2.

Everything here runs against the checked-in demo resource (never the local full store), so it
behaves the same on a clean clone as on a machine with the corpus downloaded.
"""

import io

import numpy as np
import pandas as pd
import pytest

from epigrade import paths
from epigrade.byo import store
from epigrade.byo.evaluate import (
    EXPECTED_COLUMNS,
    REFUSED_COVERAGE,
    WARN_LOW_COVERAGE,
    SignatureInputError,
    evaluate,
    evaluate_file,
    parse_signature,
    signature_from_labelled_samples,
)

FIXTURES = paths.repo_root() / "tests" / "fixtures"
SOTOS = "Sotos syndrome"


@pytest.fixture(scope="module")
def source():
    return store.load_beta_source(prefer_full=False)


@pytest.fixture(scope="module")
def sotos_probes():
    return pd.read_csv(FIXTURES / "sig_sotos_true.csv")


@pytest.fixture(scope="module")
def absent_probes():
    return pd.read_csv(FIXTURES / "sig_low_coverage.csv")["probe_id"].tolist()


@pytest.fixture(scope="module")
def results(source):
    return {name: evaluate_file(FIXTURES / name, source=source) for name in [
        "sig_sotos_true.csv", "sig_random.csv", "sig_sotos_flipped.csv", "sig_tiny.csv",
        "sig_low_coverage.csv",
    ]}


def _csv(text: str) -> io.BytesIO:
    return io.BytesIO(text.encode("utf-8"))


# --------------------------------------------------------------------------- parsing


def test_empty_file_is_a_clear_error_naming_the_expected_columns():
    with pytest.raises(SignatureInputError) as exc:
        parse_signature(_csv("   \n"))
    assert exc.value.code == "empty_file" and EXPECTED_COLUMNS in str(exc.value)


def test_missing_probe_id_column_names_the_expected_columns():
    with pytest.raises(SignatureInputError) as exc:
        parse_signature(_csv("cpg,dir\ncg00000029,hypo\n"))
    assert exc.value.code == "missing_probe_id_column"
    assert "probe_id" in str(exc.value) and "cpg" in str(exc.value)


def test_headerless_file_gets_a_hint_about_the_missing_header():
    with pytest.raises(SignatureInputError) as exc:
        parse_signature(_csv("cg00000029\ncg00000108\n"))
    assert "header row may be missing" in str(exc.value)


def test_header_only_and_binary_files_are_errors_not_tracebacks():
    with pytest.raises(SignatureInputError) as exc:
        parse_signature(_csv("probe_id,direction\n"))
    assert exc.value.code == "no_rows"
    with pytest.raises(SignatureInputError) as exc:
        parse_signature(io.BytesIO(b"\xff\xfe\x00\x81\x8d"))
    assert exc.value.code == "not_text"


def test_malformed_rows_and_bad_ids_are_rejected_and_counted_and_the_rest_continue():
    parsed = parse_signature(_csv(
        "probe_id,direction,weight\ncg00000029,hypo,2\nnot_a_probe,hyper,1\n"
        "12345,hypo,1\nch.1.1234567R,HYPER,abc\nrs1234567,sideways,\n"
    ))
    assert parsed.n_rows == 5 and parsed.n_invalid_format == 2 and parsed.n_submitted == 3
    assert parsed.invalid_examples == ["not_a_probe", "12345"]
    assert parsed.probes.loc["ch.1.1234567R", "direction"] == "hyper"
    assert pd.isna(parsed.probes.loc["rs1234567", "direction"])
    assert parsed.n_invalid_direction == 1 and parsed.n_invalid_weight == 1
    assert parsed.probes.loc["cg00000029", "weight"] == 2.0
    assert parsed.probes.loc["ch.1.1234567R", "weight"] == 1.0


def test_duplicates_are_removed_and_counted():
    parsed = parse_signature(_csv("probe_id\ncg00000029\ncg00000029\ncg00000108\ncg00000029\n"))
    assert parsed.n_duplicates == 2 and parsed.n_submitted == 2


def test_tsv_and_epic_v2_suffixes_are_accepted():
    parsed = parse_signature(_csv("probe_id\tdirection\ncg00000029_TC21\thypo\n"))
    assert parsed.probes.index.tolist() == ["cg00000029"]


# --------------------------------------------------------------------------- refusal rules


def test_fewer_than_ten_probes_found_is_refused(source, sotos_probes, absent_probes):
    table = pd.DataFrame({"probe_id": sotos_probes.probe_id.tolist()[:9] + absent_probes[:40]})
    result = evaluate(parse_signature(table), source=source)
    assert result.status == "refused" and result.refusal_code == REFUSED_COVERAGE
    assert result.coverage["n_found"] == 9 and result.cohorts is None
    assert result.identification["n_disorders_searched"] == 12


def test_more_than_half_missing_is_computed_with_the_exact_percentage(
    source, sotos_probes, absent_probes,
):
    table = pd.DataFrame({"probe_id": sotos_probes.probe_id.tolist()[:20] + absent_probes[:30]})
    result = evaluate(parse_signature(table), source=source)
    assert result.status == "ok"
    warning = next(w for w in result.warnings if w["code"] == WARN_LOW_COVERAGE)
    assert warning["percent_missing"] == 60.0 and "60.0%" in warning["text"]
    assert result.identification["top_match"] == SOTOS


def test_exactly_half_missing_does_not_warn(source, sotos_probes, absent_probes):
    table = pd.DataFrame({"probe_id": sotos_probes.probe_id.tolist()[:20] + absent_probes[:20]})
    assert evaluate(parse_signature(table), source=source).warnings == []


def test_duplicates_and_bad_ids_are_reported_in_the_result(source, sotos_probes):
    ids = sotos_probes.probe_id.tolist()[:30]
    table = pd.DataFrame({"probe_id": ids + ids[:5] + ["bogus1", "bogus2"]})
    result = evaluate(parse_signature(table), source=source)
    assert result.input_summary["n_duplicates_removed"] == 5
    assert result.input_summary["n_rejected_bad_format"] == 2
    assert result.input_summary["n_submitted"] == 30
    assert any("2 row(s) rejected" in m for m in result.messages)
    assert any("5 duplicate" in m for m in result.messages)


def test_a_match_to_a_gate_failing_disorder_is_within_study_only(results):
    result = results["sig_sotos_true.csv"]
    assert result.identification["interpretation_scope"] == "within_study_only"
    assert any("within_study_only" in m for m in result.messages)
    sotos = result.cohorts.set_index("disorder").loc[SOTOS]
    assert sotos.interpretation_scope == "within_study_only"


# --------------------------------------------------------------------------- always reported


def test_every_result_reports_how_much_was_searched(results):
    for name, result in results.items():
        assert result.identification["n_disorders_searched"] == 12, name
        assert result.identification["n_with_own_signature"] == 10, name
    for name in ("sig_sotos_true.csv", "sig_random.csv"):
        cohorts = results[name].cohorts
        assert len(cohorts) == 12 and cohorts.interpretation_scope.notna().all()


def test_coverage_reports_platforms_and_each_corpus_matrix(results):
    cov = results["sig_sotos_true.csv"].coverage
    assert cov["n_submitted"] == 500 and cov["n_found"] == 500 and cov["n_missing"] == 0
    assert set(cov["platform_survival"]) == set(store.PLATFORMS)
    assert cov["platform_survival"]["450K"]["n_present"] == 500
    assert len(cov["corpus_series_presence"]) == 11
    assert cov["corpus_series_presence"]["GSE74432"] == 500


# --------------------------------------------------------------------------- fixtures


def test_true_sotos_signature_identifies_sotos_with_a_clear_margin(results):
    ident = results["sig_sotos_true.csv"].identification
    assert ident["verdict"] == "match" and ident["top_match"] == SOTOS
    assert ident["top_match_auc"] > 0.95
    assert ident["top_match_z"] > 2 * ident["runner_up_z"]


def test_random_probes_give_no_match_even_though_their_raw_auc_is_high(results):
    """The reason the match rule is null-calibrated: random probes reach a high AUC in several
    single-study cohorts, and must still not be called a match."""
    result = results["sig_random.csv"]
    assert result.status == "ok" and result.identification["verdict"] == "no_match"
    assert result.identification["top_match"] is None
    assert result.identification["highest_auc"] > 0.9
    assert result.specificity is None and result.attribution is None
    assert not result.cohorts.is_candidate.any()


def test_flipped_directions_still_identify_sotos_and_every_direction_is_flagged(results):
    result = results["sig_sotos_flipped.csv"]
    assert result.identification["top_match"] == SOTOS
    attribution = result.attribution
    assert attribution["all_directions_inconsistent"]
    assert attribution["n_direction_inconsistent"] == attribution["n_probes_with_direction"] == 500
    true = results["sig_sotos_true.csv"].attribution
    assert true["n_direction_consistent"] == 500


def test_tiny_signature_identifies_sotos_with_a_wider_interval(results):
    tiny, full = results["sig_tiny.csv"], results["sig_sotos_true.csv"]
    assert tiny.identification["top_match"] == SOTOS

    def width(result):
        row = result.cohorts.set_index("disorder").loc[SOTOS]
        return (row.effect_ci_high - row.effect_ci_low) / row.mean_abs_delta_beta

    assert width(tiny) > 2 * width(full)
    assert tiny.identification["top_match_z"] < full.identification["top_match_z"]


def test_low_coverage_fixture_is_refused(results):
    result = results["sig_low_coverage.csv"]
    assert result.status == "refused" and result.refusal_code == REFUSED_COVERAGE
    assert result.coverage["n_found"] == 0 and result.coverage["n_submitted"] == 500


def test_matched_results_carry_specificity_ceiling_overlap_and_attribution(results):
    result = results["sig_sotos_true.csv"]
    spec = result.specificity.set_index("scored_group")
    assert len(spec) == 13 and spec.loc[SOTOS, "fraction_positive"] == 1.0
    assert spec.loc["pooled_control", "n"] == 273 - 53
    assert result.evidence_ceiling["between_study_bound_possible"] is False
    overlap = result.overlap.set_index("corpus_signature")
    assert len(overlap) == 10 and overlap.loc[SOTOS, "n_shared"] == 500
    assert any("subset of a known signature" in m for m in result.messages)
    assert len(result.attribution["top_probes"]) == 10


def test_checked_in_fixture_table_matches_a_fresh_run(results):
    path = paths.repo_root() / "results" / "tables" / "byo_fixture_results.tsv"
    if not path.exists():
        pytest.skip("byo_fixture_results.tsv not generated yet")
    table = pd.read_csv(path, sep="\t", keep_default_na=False).set_index("fixture")
    for name, result in results.items():
        assert table.loc[name, "status"] == result.status
        assert table.loc[name, "top_match"] == (result.identification["top_match"] or "")


def test_long_table_export_has_every_section(results):
    long = results["sig_sotos_true.csv"].to_long_table()
    assert list(long.columns) == ["section", "item", "field", "value"]
    for section in ("input", "coverage", "identification", "cohort_search", "specificity",
                    "overlap", "evidence_ceiling", "attribution"):
        assert (long.section.str.startswith(section)).any(), section
    assert results["sig_low_coverage.csv"].to_long_table().section.eq("result").any()


# --------------------------------------------------------------------------- form 2


def _labelled(n_probes=400, n_case=14, n_control=14, effect=0.3, seed=5):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.2, 0.8, n_probes)
    shift = np.zeros(n_probes)
    shift[:40] = effect * np.where(np.arange(40) % 2 == 0, 1.0, -1.0)
    cases = [f"S{i}" for i in range(n_case)]
    controls = [f"S{i}" for i in range(n_case, n_case + n_control)]
    data = {s: np.clip(base + shift + rng.normal(0, 0.05, n_probes), 0, 1) for s in cases}
    data |= {s: np.clip(base + rng.normal(0, 0.05, n_probes), 0, 1) for s in controls}
    beta = pd.DataFrame(data, index=[f"cg{i:08d}" for i in range(n_probes)])
    labels = pd.DataFrame({"sample_id": cases + controls,
                           "label": ["case"] * n_case + ["control"] * n_control})
    return labels, beta


def test_form_two_derives_a_signature_with_directions():
    labels, beta = _labelled()
    parsed = signature_from_labelled_samples(labels, beta)
    assert 30 <= parsed.n_submitted <= 45
    assert parsed.source.startswith("derived_from_labelled_samples")
    assert parsed.probes.loc["cg00000000", "direction"] == "hyper"
    assert parsed.probes.loc["cg00000001", "direction"] == "hypo"


def test_form_two_refuses_too_few_cases_and_no_signal_with_clear_messages():
    labels, beta = _labelled(n_case=6)
    with pytest.raises(SignatureInputError) as exc:
        signature_from_labelled_samples(labels, beta)
    assert exc.value.code == "too_few_labelled_samples"
    labels, beta = _labelled(effect=0.0)
    with pytest.raises(SignatureInputError) as exc:
        signature_from_labelled_samples(labels, beta)
    assert exc.value.code == "no_signature_derived" and "Path B" in str(exc.value)
    with pytest.raises(SignatureInputError) as exc:
        signature_from_labelled_samples(pd.DataFrame({"id": ["a"]}), beta)
    assert exc.value.code == "missing_label_columns"
