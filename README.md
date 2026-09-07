# EpiGrade

An open, reproducible benchmark for DNA methylation episignature classifiers.

Rebuilds published episignature classifiers from public GEO data alone, cross-checks them
against each other, and reports how much evidence a given score can honestly support -
with every number traceable to a script and a public accession.

## Not for patient use

**EpiGrade is a research and benchmarking tool for laboratories and researchers. It is not a
diagnostic device.** It takes a classifier score and reports the evidence strength that score
can support against the accessions in this corpus - it never accepts patient data, raw
methylation arrays, or issues a diagnosis. Nothing here should inform a clinical decision on
its own.

## What this is (and isn't)

- The local-likelihood-ratio calibration used here is Pejaver-style calibration (Pejaver et al.
  2022), already published; a generic CRAN implementation (`acmgscaler`) exists. We do not claim
  this method is novel.
- Cross-disorder specificity testing is routine inside the private EpiSign Knowledge Database.
  The claim here is that our version is reproducible from public data, not that the idea is new.
- "Benign" labels in GSE74432's missense cohort were assigned by the classifier under evaluation,
  not by an independent truth source. This is reported as a limitation, not resolved away.
- If a number can't be computed honestly, the output says `NA` with a reason. Nothing is
  fabricated to fill a table.

See [docs/METHODS.md](docs/METHODS.md), [docs/LIMITATIONS.md](docs/LIMITATIONS.md), and
[docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md) for the full picture.

## Reproducing the demo

```
# one-time setup (see docs/METHODS.md for the venv/data-root layout)
python scripts/demo_sotos.py
```

This reproduces the Choufani et al. 2015 Sotos syndrome classifier from GSE74432: discovery
cohort separation, the 8 Weaver-syndrome negative controls, and the 9/7 split of the 16 NSD1
missense variants of uncertain significance.

## Layout

- `src/epigrade/` - library code (acquire, preprocess, signature, calibrate, audit, report, app)
- `config/` - paths and evidence-band tables (never hardcoded in source)
- `scripts/` - one-command entry points (e.g. the Sotos demo)
- `results/` - tracked tables and figures produced by scripts (never hand-edited)
- `data/` - gitignored; see `config/paths.yaml` for where bulk data actually lives on disk
