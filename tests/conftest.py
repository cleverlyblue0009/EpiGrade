"""Every test runs without a model key and without network access to the model, whatever is in
the developer's .env: the app tests render pages that would otherwise call Groq live and write
to the committed cache. Tests that exercise the client set a fake key and mock requests."""

import pytest

from epigrade.explain import groq_client


@pytest.fixture(autouse=True)
def no_model_key(monkeypatch, tmp_path_factory):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(groq_client, "ENV_FILE",
                        tmp_path_factory.getbasetemp() / "no-such.env")
