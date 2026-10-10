"""The paper's structure, and that the numbers it reports are the ones in the result tables."""

import re

import pandas as pd
import pytest

from epigrade import paths

PAPER = paths.repo_root() / "paper" / "EpiGrade_paper.md"
S1 = paths.repo_root() / "paper" / "supplementary_table_S1.tsv"
SECTIONS = ["Abstract", "Introduction", "Literature Review", "Methodology", "Analysis",
            "Result", "Discussion", "Conclusion", "References"]

pytestmark = pytest.mark.skipif(not PAPER.exists(), reason="paper not written yet")


def _text() -> str:
    return PAPER.read_text(encoding="utf-8")


def _section(name: str) -> str:
    match = re.search(rf"^## (?:\d+\. )?{name}\n(.*?)(?=^## |\Z)", _text(), re.S | re.M)
    assert match, f"section {name} not found"
    return match.group(1)


def test_the_nine_sections_appear_in_the_required_order():
    headings = [re.sub(r"^\d+\. ", "", h) for h in re.findall(r"^## (.+)$", _text(), re.M)]
    required = [h for h in headings if h in SECTIONS]
    assert required == SECTIONS


def test_length_is_within_the_required_range():
    body = _text().split("## References")[0]
    n_words = len(re.findall(r"\b[\w'-]+\b", body))
    assert 5000 <= n_words <= 7000, n_words


def test_abstract_length():
    n_words = len(re.findall(r"\b[\w'%.,-]+\b", _section("Abstract").split("**Keywords")[0]))
    assert 200 <= n_words <= 250, n_words


def test_every_supplementary_number_is_the_one_printed_in_the_paper():
    """Each S1 value was read from a result table by scripts/paper_numbers.py. If a table
    changes and the paper is not updated, the old value disappears from S1 and this fails."""
    table = pd.read_csv(S1, sep="\t", keep_default_na=False, dtype=str)
    text = _text()
    missing = [f"{r.id} = {r.value}" for r in table.itertuples() if r.value not in text]
    assert not missing, "not found in the paper:\n  " + "\n  ".join(missing)


def test_supplementary_table_points_at_real_files():
    table = pd.read_csv(S1, sep="\t", keep_default_na=False, dtype=str)
    for source in table.source_file.unique():
        assert (paths.repo_root() / source).exists(), source


def test_references_are_numbered_consecutively_and_all_cited():
    refs = [int(n) for n in re.findall(r"^\[(\d+)\] ", _section("References"), re.M)]
    assert refs == list(range(1, len(refs) + 1)) and len(refs) >= 15
    body = _text().split("## References")[0]
    cited = {int(n) for group in re.findall(r"\[(\d+(?:, ?\d+)*)\]", body)
             for n in group.split(",")}
    assert set(refs) <= cited, sorted(set(refs) - cited)


def test_honesty_constraints_are_stated():
    text = _text()
    assert "We do not claim the calibration method is new" in text
    assert "AI-versus-AI" in text and "human curation pending" in text
    assert "not a diagnostic device" in text
