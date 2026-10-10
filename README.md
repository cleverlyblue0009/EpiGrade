# EpiGrade

An open, reproducible benchmark for DNA methylation episignature classifiers.

```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # macOS/Linux: .venv/bin/pip
.venv/Scripts/python scripts/reproduce.py
```

That one command takes a clean clone to every table and figure in `results/`.

| | |
|---|---|
| Download | about 2.0 GB kept on disk (1.77 GB of processed beta matrices for 11 series, 0.23 GB of array manifests). The EPIC v2 probe list has no manifest file on GEO, so its 2.5 GB platform table is streamed and only the probe names are kept. |
| Disk | about 6 GB under the data root (downloads plus parsed caches). Nothing is written into the repository. |
| Runtime | about 15 minutes of computation on a laptop. Download time is on top and depends on NCBI GEO: 20 minutes on a good day, well over an hour on a bad one. |
| Data root | `D:/epigrade_data` by default (`config/paths.yaml`). Set `EPIGRADE_DATA_ROOT` to use another location. On a machine with no D: drive it falls back to `data/` inside the clone, which is gitignored. |
| Re-running | Safe to interrupt and re-run: finished stages are skipped. `--force` reruns everything. `--quick` uses no network and rebuilds whatever can be rebuilt from what is on disk. |

**To look at the results without downloading anything**, the app runs from files checked into
the repository:

```
.venv/Scripts/streamlit run src/epigrade/app/main.py
```

## Not for patient use

**EpiGrade is a research and benchmarking tool for laboratories and researchers. It is not a
diagnostic device.** It never accepts patient data or raw methylation arrays and never issues a
diagnosis. Nothing here should inform a clinical decision on its own.

## What it does

Episignature classifiers are validated against a private database that outside laboratories
cannot inspect. EpiGrade rebuilds what can be rebuilt from public GEO data alone and reports how
far each result can be trusted:

1. **Corpus.** 18 GEO accessions, 3,732 samples harvested, 1,622 retained, 2,110 excluded with a
   recorded reason for each. 12 disorders.
2. **Classifiers.** A signature for each of the 10 disorders with at least 10 cases. Sotos uses
   the published 7,085-probe list; the other nine are re-derived here and labelled as such.
3. **Audit.** A confounding gate, leave-one-study-out, and every classifier scored against every
   other disorder and against pooled controls.
4. **Calibration.** Scores converted to evidence bands, with the strongest band each cohort's
   size could ever support.
5. **Identification.** Upload your own probe list and see which cohort, if any, it identifies.

## What it found

Every number here comes from a file in `results/tables/`. None is typed by hand.

| | |
|---|---|
| Sotos reproduction | All four checks pass against Choufani et al. 2015: 19 discovery cases and 53 controls separate cleanly at score 0; 8 of 8 Weaver samples negative; 16 NSD1 missense variants split 9 positive, 7 negative; 19 of 19 held-out replication cases positive. `sotos_reproduction_summary.tsv` |
| Twelve disorders, four verdicts | 3 graded (cases in two studies), 7 single-study, 2 with too few cases, 0 not downloaded. `disorder_verdicts.tsv` |
| Leave-one-study-out | **1 of 6 directions could be tested.** Silver-Russell trained on GSE104451 and tested on GSE55491: 5 of 18 cases positive, 6 of 6 controls negative. The reverse direction and both Kabuki directions find no significant probes. Neither Nicolaides-Baraitser study has 10 cases. `leave_one_study_out.tsv` |
| Cross-disorder matrix | A complete 12 by 13 grid: 130 computed cells and 26 reasoned gaps. CHARGE scores 14 of 37 Kabuki type 1 cases positive (37.8%). `cross_disorder_matrix.tsv` |
| The controls-source effect | Classifiers trained against out-of-series controls detect series membership, not disease. Coffin-Siris and Nicolaides-Baraitser are both in that category: on other disorders' cases they score a mean of 68.1% positive, against 11.9% for the eight classifiers with in-series controls. `controls_source_summary.tsv` |
| Evidence bands | Only Silver-Russell and Kabuki type 1 have a between-study bound, and both read "No evidence" at a typical case score. The other eight bands are within-study only. `evidence_bands.tsv` |
| AUC alone is unsafe for identification | 500 random probes reach AUC 1.00 on Williams syndrome and at least 0.70 on 7 of 12 cohorts. The z-score against random probe sets is what makes no_match work. `byo_fixture_results.tsv` |
| Label check | Two AI-written implementations of the labelling rules agree on 81 of 81 scoreable samples. This is an AI-vs-AI consistency check. Human curation is still pending. |

One threshold change was made in the whole project. Choufani's Bonferroni correction with a
20% effect floor, tuned to the unusually large Sotos effect, found nothing for any other
disorder. It was replaced once by Benjamini-Hochberg FDR at 0.05 with a 10% floor, before any
retry, and has not been touched since. All nine re-derived signatures use it
(`config/signature_thresholds.yaml`, `signature_derivation.tsv`).

## The app

Five pages, each carrying the not-for-patient-use banner:

- **Grade a score**: enter a classifier score for a disorder and see the evidence band it
  supports, or why grading is limited.
- **Test your own signature**: upload a probe list (or pick one of five examples). Reports
  platform and corpus coverage, the identification verdict across all 12 cohorts, the
  signature's specificity row, an evidence ceiling that accounts for cohort size and signature
  size, overlap with the corpus
  signatures, and a "Why this result" panel. Download the evaluation as TSV.
- **Calibration curves**: evidence bands and attainable ceilings.
- **Cohort audit**: the 12-disorder verdict table, leave-one-study-out, the confounding gate,
  the Sotos reproduction, and label harmonisation.
- **Cross-disorder matrix**: the heatmap and every cell behind it.

On a clean clone the "Test your own signature" page has sample-level data for the corpus
signature probes and the five examples. After `scripts/reproduce.py` it covers every probe.

## What is ours and what is not

- The calibration is Pejaver-style local likelihood-ratio calibration (Pejaver et al. 2022). A
  generic implementation exists on CRAN (`acmgscaler`). The method is not new.
- Cross-disorder specificity testing is routine inside the private EpiSign Knowledge Database.
  The claim here is that this version is reproducible from public data.
- Using a language model to help curate GEO metadata is an application of published work, not
  a contribution.
- "Benign" labels in the GSE74432 missense cohort were assigned by the classifier under
  evaluation, not by an independent truth source.
- A result that cannot be computed is reported as not computed, with a reason.

See [docs/METHODS.md](docs/METHODS.md), [docs/LIMITATIONS.md](docs/LIMITATIONS.md) and
[docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md).

## Pipeline stages

`scripts/reproduce.py` runs these in order. Each can be run on its own.

```
python -m epigrade.acquire.harvest         # metadata for all 18 accessions
python scripts/phase1_report.py            # triage log, sample counts, label cross-check
python scripts/download_matrices.py        # beta matrices, array manifests, Sotos signature
python scripts/demo_sotos.py               # the Sotos reproduction and its four checks
python scripts/build_corpus.py             # all signatures, matrix, leave-one-study-out, bands
python scripts/compute_ceilings.py         # attainable evidence ceilings
python scripts/make_figures.py             # every figure, plus slides/ versions
python scripts/make_test_signatures.py     # the five test signatures, from the real corpus
python scripts/build_byo_resource.py       # data behind "Test your own signature"
python scripts/run_byo_fixtures.py         # the five test signatures through the engine
python scripts/compute_attribution.py      # "Why this result" examples
python scripts/phase6_confounding_gate.py  # confounding gate
python scripts/generate_provenance.py      # docs/DATA_PROVENANCE.md
python -m epigrade.report.resource         # the resource file the app reads
```

`make reproduce`, `make quick`, `make test`, `make lint` and `make app` wrap the common ones.

## Layout

- `src/epigrade/`: library code (`acquire`, `preprocess`, `signature`, `corpus`, `calibrate`,
  `audit`, `byo`, `report`, `app`)
- `config/`: paths, label vocabulary, signature thresholds, evidence bands
- `scripts/`: one entry point per pipeline stage, plus `pre_push_guard.py`
- `results/tables/`, `results/figures/`, `results/signatures/`, `results/resource/`: tracked
  outputs, never hand-edited. `results/figures/slides/` holds the same figures at 200 dpi with
  larger fonts.
- `paper/`: the paper (Markdown source, Word build, and a table mapping every reported number to its
  source file); `presentation/`: the slide deck and its build script
- `tests/`: unit tests and invariants on the checked-in tables
- `data/`: gitignored. Nothing under it, and no file over 50 MB, is ever committed
  (`python scripts/pre_push_guard.py --install` enforces this before every push).
