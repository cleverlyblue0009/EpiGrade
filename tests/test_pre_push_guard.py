"""Tests for scripts/pre_push_guard.py's violation rules."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "pre_push_guard", Path(__file__).resolve().parents[1] / "scripts" / "pre_push_guard.py",
)
guard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(guard)


def test_clean_files_pass():
    assert guard.find_violations({"src/epigrade/paths.py": 2_000, "README.md": 11_000}) == []


def test_file_over_50mb_is_rejected():
    problems = guard.find_violations({"results/big.tsv": guard.MAX_BYTES + 1})
    assert len(problems) == 1 and "exceeds" in problems[0]


def test_file_exactly_at_limit_passes():
    assert guard.find_violations({"results/edge.tsv": guard.MAX_BYTES}) == []


def test_anything_under_data_is_rejected_regardless_of_size():
    problems = guard.find_violations({"data/external/tiny.csv": 10})
    assert len(problems) == 1 and "data/" in problems[0]


def test_windows_separators_are_normalised():
    assert guard.find_violations({"data\\external\\tiny.csv": 10})


def test_data_prefix_does_not_catch_lookalike_directories():
    assert guard.find_violations({"database/schema.sql": 10}) == []
