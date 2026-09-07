"""Smoke tests for centralized path resolution (config/paths.yaml)."""

from epigrade import paths


def test_repo_root_contains_pyproject():
    assert (paths.repo_root() / "pyproject.toml").exists()


def test_data_root_is_configured_off_repo():
    # data must never live inside the git-tracked repo tree
    assert not str(paths.data_root()).startswith(str(paths.repo_root()))


def test_output_dirs_are_created_on_demand():
    for factory in (paths.tables_dir, paths.figures_dir, paths.resource_dir):
        d = factory()
        assert d.exists() and d.is_dir()
