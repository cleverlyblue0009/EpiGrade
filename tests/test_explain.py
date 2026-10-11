"""The explanation layer: fact sheets come from real results, the validator rejects anything
the facts do not contain, and every failure path falls back to the template."""

import json

import pandas as pd
import pytest
import requests

from epigrade import paths
from epigrade.byo import store
from epigrade.byo.evaluate import evaluate_file
from epigrade.corpus import registry
from epigrade.explain import explain as E
from epigrade.explain import facts as F
from epigrade.explain import groq_client, template

FIXTURES = paths.repo_root() / "tests" / "fixtures"
DISORDERS = [d.name for d in registry.DISORDERS]


@pytest.fixture(scope="module")
def byo_facts():
    source = store.load_beta_source(prefer_full=False)
    return {name: F.facts_for_byo(evaluate_file(FIXTURES / name, source=source))
            for name in ("sig_sotos_true.csv", "sig_random.csv", "sig_sotos_flipped.csv",
                         "sig_tiny.csv", "sig_low_coverage.csv")}


@pytest.fixture(autouse=True)
def isolated_outputs(tmp_path, monkeypatch):
    """Keep tests from writing to the real cache or failure log."""
    monkeypatch.setattr(E, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(E, "failures_path", lambda: tmp_path / "explain_failures.tsv")
    # Never let a developer's real .env reach a test.
    monkeypatch.setattr(groq_client, "ENV_FILE", tmp_path / "missing.env")
    return tmp_path



# --------------------------------------------------------------------------- fact sheets


def test_byo_facts_are_populated_from_the_evaluation_not_defaults(byo_facts):
    table = pd.read_csv(paths.repo_root() / "results" / "tables" / "byo_fixture_results.tsv",
                        sep="\t", keep_default_na=False).set_index("fixture")
    f = byo_facts["sig_sotos_true.csv"]
    row = table.loc["sig_sotos_true.csv"]
    assert f["top_match"] == row.top_match == "Sotos syndrome"
    assert f["top_match_z"] == round(float(row.top_match_z_vs_random_probes), 1)
    assert f["runner_up"] == row.runner_up
    assert f["evidence_ceiling_band"] == row.evidence_ceiling_band
    assert f["n_disorders_searched"] == 12 and f["n_with_own_signature"] == 10
    required = ["n_submitted", "n_found", "n_missing", "pct_missing", "platform_survival",
                "verdict", "top_match_auc", "auc_ci", "runner_up_z", "margin_z",
                "interpretation_scope", "cohort_ceiling", "signature_size_bound",
                "between_study_bound_possible", "n_direction_supplied",
                "n_direction_inconsistent", "max_overlap_signature", "max_overlap_jaccard",
                "max_overlap_fraction_pct", "specificity_row", "notes"]
    for key in required:
        assert f[key] is not None, key
    assert len(f["specificity_row"]) == 13
    assert byo_facts["sig_low_coverage.csv"]["refusal_code"] == "insufficient_probe_coverage"
    assert byo_facts["sig_sotos_flipped.csv"]["n_direction_inconsistent"] == 500


@pytest.mark.parametrize("disorder", DISORDERS)
def test_disorder_facts_match_the_verdict_table(disorder):
    verdicts = pd.read_csv(paths.repo_root() / "results" / "tables" / "disorder_verdicts.tsv",
                           sep="\t", keep_default_na=False).set_index("disorder")
    f = F.facts_for_disorder(disorder)
    row = verdicts.loc[disorder]
    assert f["n_cases"] == int(row.n_cases) and f["verdict"] == row.verdict
    assert f["n_signature_probes"] == int(row.n_signature_probes)
    assert f["controls_source"] == row.controls_source
    loso = pd.read_csv(paths.repo_root() / "results" / "tables" / "leave_one_study_out.tsv",
                       sep="\t")
    assert f["loso_directions_total"] == int((loso.disorder == disorder).sum())


def test_grade_facts_only_exist_for_computed_rows():
    bands = pd.read_csv(paths.repo_root() / "results" / "tables" / "evidence_bands.tsv",
                        sep="\t")
    r = bands.iloc[0]
    f = F.facts_for_grade(r.disorder, r.query_score, r.prior)
    assert f["band"] == r.band and f["likelihood_ratio"] == round(r.lr_point_estimate, 1)
    with pytest.raises(ValueError):
        F.facts_for_grade(r.disorder, 123.456, r.prior)


# --------------------------------------------------------------------------- validator


def test_every_template_passes_the_validator(byo_facts):
    sheets = list(byo_facts.values()) + [F.facts_for_disorder(d) for d in DISORDERS]
    for f in sheets:
        text = template.explain(f)
        assert E.validate(text, f) == [], (f.get("disorder") or f.get("verdict"), text)


def test_validator_rejects_invented_numbers_and_unsupported_claims():
    f = F.facts_for_disorder("Sotos syndrome")
    bad = ("Sotos syndrome is diagnosed by a mutation that alters a histone enzyme. The "
           "classifier reaches an AUC of 0.97 using 450 probes. Weaver syndrome looks similar.")
    problems = " | ".join(E.validate(bad, f))
    for expected in ("0.97", "450", "diagnostic", "mutation", "gene-function", "Weaver"):
        assert expected in problems, expected


def test_validator_accepts_percentages_both_ways_and_small_counts(byo_facts):
    f = byo_facts["sig_sotos_true.csv"]
    text = ("Every one of the 500 probes was found, so 0.0% were missing. "
            "It was compared against 12 cohorts and two studies.")
    assert E.validate(text, f) == []


def test_validator_enforces_length():
    f = F.facts_for_disorder("Sotos syndrome")
    assert any("sentences" in p for p in E.validate("Sotos syndrome has 38 cases.", f))


# --------------------------------------------------------------------------- client + fallback


def test_no_key_means_no_call_and_the_template(monkeypatch, byo_facts):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    called = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: called.append(1))
    result = E.explain(byo_facts["sig_random.csv"])
    assert result.source == E.SOURCE_TEMPLATE and result.fallback_reason == "no_api_key"
    assert not called


class _Reply:
    def __init__(self, status, text="", headers=None):
        self.status_code, self._text, self.headers = status, text, headers or {}

    def json(self):
        return {"choices": [{"message": {"content": self._text}}]}


def test_a_valid_model_reply_is_shown_and_cached(monkeypatch, isolated_outputs, byo_facts):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    f = byo_facts["sig_tiny.csv"]
    good = template.explain(f)
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Reply(200, good))
    result = E.explain(f)
    assert result.source == E.SOURCE_MODEL and result.attempts == 1
    cached = json.loads((isolated_outputs / f"{result.fact_hash}.json").read_text())
    assert cached["source"] == "model" and "test-key" not in json.dumps(cached)


def test_two_rejections_fall_back_and_are_logged(monkeypatch, isolated_outputs, byo_facts):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    replies = iter([_Reply(200, "This is a diagnosis. The AUC is 0.42 for everyone."),
                    _Reply(200, "The mutation causes it. It scored 77 points.")])
    monkeypatch.setattr(requests, "post", lambda *a, **k: next(replies))
    result = E.explain(byo_facts["sig_sotos_true.csv"])
    assert result.source == E.SOURCE_TEMPLATE and result.attempts == 2
    assert len(result.rejections) == 2
    log = pd.read_csv(isolated_outputs / "explain_failures.tsv", sep="\t")
    assert len(log) == 2 and set(log.attempt) == {1, 2}
    assert "0.42" in log.reason.iloc[0]


def test_rate_limit_is_retried_once_then_gives_up(monkeypatch, byo_facts):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setattr(groq_client.time, "sleep", lambda s: None)
    calls = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append(1) or _Reply(429))
    result = E.explain(byo_facts["sig_random.csv"])
    assert len(calls) == 2 and result.fallback_reason == "rate_limited"
    assert result.source == E.SOURCE_TEMPLATE


@pytest.mark.parametrize("error", [requests.Timeout, requests.ConnectionError])
def test_network_failures_fall_back(monkeypatch, error, byo_facts):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")

    def boom(*a, **k):
        raise error()

    monkeypatch.setattr(requests, "post", boom)
    assert E.explain(byo_facts["sig_random.csv"]).source == E.SOURCE_TEMPLATE


def test_key_never_reaches_the_repository():
    for path in (paths.repo_root() / "results" / "cache").rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "gsk_" not in text and "Bearer" not in text


def test_key_is_read_from_the_env_file_when_the_variable_is_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("# comment\nOTHER=1\nGROQ_API_KEY = \"from-file\"\n", encoding="utf-8")
    monkeypatch.setattr(groq_client, "ENV_FILE", env)
    assert groq_client.api_key() == "from-file" and groq_client.has_key()
    monkeypatch.setenv("GROQ_API_KEY", "from-shell")
    assert groq_client.api_key() == "from-shell"
    env.write_text("GROQ_API_KEY=\n", encoding="utf-8")
    monkeypatch.delenv("GROQ_API_KEY")
    assert not groq_client.has_key()


def test_env_file_is_gitignored():
    import subprocess

    result = subprocess.run(["git", "check-ignore", ".env"], cwd=paths.repo_root(),
                            capture_output=True, text=True)
    assert result.returncode == 0

