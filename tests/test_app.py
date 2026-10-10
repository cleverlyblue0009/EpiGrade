"""Smoke tests: every app page renders from the checked-in resource alone, with the
not-for-patient-use banner, and the bring-your-own page handles every example and bad input."""

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from epigrade import paths  # noqa: E402
from epigrade.app import byo_page  # noqa: E402

APP = str(paths.repo_root() / "src" / "epigrade" / "app" / "main.py")
BYO = "Test your own signature"


def _run(page: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120).run()
    if page is not None:
        at.sidebar.radio[0].set_value(page).run()
    return at


def _text(at: AppTest) -> str:
    blocks = [*at.markdown, *at.success, *at.info, *at.warning, *at.error, *at.caption]
    return " ".join(str(b.value) for b in blocks)


def test_bring_your_own_page_is_second_in_the_sidebar():
    options = _run().sidebar.radio[0].options
    assert options[0] == "Grade a score" and options[1] == BYO


@pytest.mark.parametrize("page", ["Grade a score", BYO, "Calibration curves", "Cohort audit",
                                  "Cross-disorder matrix"])
def test_every_page_renders_with_the_disclaimer(page):
    at = _run(page)
    assert not at.exception, at.exception
    assert any("Not for patient use" in str(w.value) for w in at.warning)


@pytest.mark.parametrize("example", list(byo_page.EXAMPLES))
def test_every_example_runs_without_an_exception(example):
    at = _run(BYO)
    at.selectbox[0].set_value(example).run()
    assert not at.exception, at.exception
    text = _text(at)
    name = byo_page.EXAMPLES[example]
    if name == "sig_low_coverage.csv":
        assert "insufficient_probe_coverage" in text
    elif name == "sig_random.csv":
        assert "No match" in text and "12 disorder cohorts" in text
    else:
        assert "identifies Sotos syndrome" in text and "12 disorder cohorts" in text
        assert "Within-study only" in text
