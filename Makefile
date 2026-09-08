# Convenience wrapper around scripts/reproduce.py - see README.md for the full explanation
# (expected runtime, download size) and what each stage does.

PYTHON ?= python

.PHONY: reproduce quick force test lint app

reproduce:
	$(PYTHON) scripts/reproduce.py

quick:
	$(PYTHON) scripts/reproduce.py --quick

force:
	$(PYTHON) scripts/reproduce.py --force

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

app:
	streamlit run src/epigrade/app/main.py
