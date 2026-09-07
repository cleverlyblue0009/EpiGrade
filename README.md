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

## Run the app (zero downloads)

A demo resource ships in the repo (`results/resource/epigrade_v1.json`), so the app runs on a
fresh clone with no setup beyond installing dependencies:

```
pip install -r requirements.txt && pip install -e .
streamlit run src/epigrade/app/main.py
```

Four pages: **Grade a score**, **Calibration curves**, **Cohort audit**, **Cross-disorder
matrix**. The app never recomputes anything - if you want fresher numbers, rerun the pipeline
below and then `python -m epigrade.report.resource` to rebuild the resource it reads.

## Reproducing the full pipeline

Requires network access to NCBI GEO (see `config/paths.yaml` for where downloads land - large
files never go in the repo). Each step is idempotent and caches its output:

```
python -m epigrade.acquire.harvest              # phase 1: metadata for all 18 accessions
python scripts/phase1_report.py                 # triage log + hand-curation cross-check
python scripts/demo_sotos.py                    # phase 3: THE reproduction (downloads GSE74432)
python scripts/phase5_calibration.py            # evidence bands + attainable ceiling
python scripts/phase4_6_srs_and_matrix.py       # cross-disorder matrix + leave-one-study-out
python scripts/phase6_confounding_gate.py       # confounding gate, all 12 in-scope disorders
python scripts/generate_provenance.py           # docs/DATA_PROVENANCE.md
python -m epigrade.report.resource              # rebuild results/resource/epigrade_v1.json
```

This reproduces the Choufani et al. 2015 Sotos syndrome classifier from GSE74432 exactly:
19/19 discovery cases positive, 53/53 controls negative, 8/8 Weaver-syndrome samples negative,
and the 16 NSD1 missense VOUS splitting 9 positive / 7 negative - all three of the paper's
published results, verified before anything else in this pipeline was built.

## Layout

- `src/epigrade/` - library code (acquire, preprocess, signature, calibrate, audit, report, app)
- `config/` - paths, label vocabulary, and evidence-band tables (never hardcoded in source)
- `scripts/` - one-command entry points, one per pipeline phase
- `results/` - tracked tables, figures, and the app's resource JSON (never hand-edited)
- `data/` - gitignored; see `config/paths.yaml` for where bulk data actually lives on disk
