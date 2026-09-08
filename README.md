# EpiGrade

```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt -e .   # or source .venv/bin/activate on macOS/Linux
python scripts/reproduce.py
```

One command, from a clean clone to every figure and table. Network-bound and variable - NCBI
GEO's FTP was anywhere from fast to very slow in development (a single ~215MB file once took
~45 minutes); budget **30-90 minutes** and **~1.1 GB** downloaded to `D:/epigrade_data` (never
into the repo - see `config/paths.yaml`). Safe to re-run or interrupt: each stage is skipped
once it's already succeeded (`--force` reruns everything). For a fast, network-free pass that
just regenerates reports and the app's resource file from whatever is already cached, use
`python scripts/reproduce.py --quick` (seconds, not minutes).

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

## What `scripts/reproduce.py` actually runs

Each stage is idempotent (a marker under `D:/epigrade_data/interim/.reproduce_markers/` skips it
on the next run) and independently invocable, if you want to run or inspect one on its own:

```
python -m epigrade.acquire.harvest              # phase 1: metadata for all 18 accessions
python scripts/phase1_report.py                 # triage log + hand-curation cross-check
python scripts/demo_sotos.py                    # phase 3: THE reproduction (downloads GSE74432)
python scripts/phase4_6_srs_and_matrix.py       # cross-disorder matrix + leave-one-study-out
python scripts/phase4_kabuki_charge.py          # Kabuki/CHARGE classifier attempts (see below)
python scripts/phase5_calibration.py            # evidence bands + attainable ceiling
python scripts/phase6_confounding_gate.py       # confounding gate, all 12 in-scope disorders
python scripts/make_summary_figure.py           # case-count summary bar chart (+ slides/)
python scripts/generate_provenance.py           # docs/DATA_PROVENANCE.md
python -m epigrade.report.resource              # rebuild results/resource/epigrade_v1.json
```

## What it reproduces

**The Sotos syndrome reproduction (phase 3) is exact and fully verified** - the paper's three
published results, checked programmatically, not eyeballed: 19/19 discovery cases score
positive, 53/53 discovery controls score negative (clean separation), all 8 Weaver-syndrome
samples score negative, and the 16 NSD1 missense VOUS split exactly 9 positive / 7 negative.

**Silver-Russell syndrome, Kabuki syndrome, and CHARGE syndrome are re-derived** (no published
probe list exists for any of them - `signature_source=rederived_not_published`), each checked
for specificity before being trusted, not just accepted because a classifier built at all:

- **Silver-Russell syndrome** is the project's headline beyond Sotos: the only disorder with a
  real between-study result. A classifier built on GSE104451 alone, with 0% cross-reactivity
  against Sotos, tested on the fully independent GSE55491 study: 5/18 sensitivity, 6/6
  specificity - modest but genuine. A pooled-studies version was tried and explicitly *rejected*
  after it failed its own specificity check (94.7% false-positive rate against Sotos).
- **CHARGE syndrome** builds (907 probes, 89.5% self-consistency) but shows a real, moderate
  37.8% cross-reactivity against Kabuki cases - reported plainly, not smoothed over.
- **Kabuki syndrome** fails from its own single study, builds when pooled with a second
  independent study's own matched cohort (278 probes, 0% cross-reactivity against CHARGE).

This took one documented threshold change (`config/signature_thresholds.yaml`: the original
Bonferroni/20%-effect-size thresholds were Choufani et al.'s own, calibrated to Sotos's
unusually large effect, and found nothing for any of these three disorders when applied
uniformly). The new default - Benjamini-Hochberg FDR at 0.05, 10% effect floor, a standard
choice for genome-wide CpG testing - was set once, before retrying anything, and not adjusted
per outcome. Full diagnostics for every attempt, including five real bugs found and fixed along the way (a
degenerate-CI bug, a JSON-serialization bug, a sample-count bug, a matrix-merge corruption bug,
and a calibration data-loss bug), are in
[docs/LIMITATIONS.md](docs/LIMITATIONS.md#finish-today-session-fixes-and-a-genuine-negative-result)
and its two continuations below it.

## Current results at a glance

Every number below is produced by a script and lands in `results/tables/` or
`results/resource/epigrade_v1.json` - none are typed by hand (see "no_hand_typed_numbers" in
the project's own working rules).

| | |
|---|---|
| Sotos reproduction (phase 3) | 19/19 discovery cases positive, 53/53 controls negative, 8/8 Weaver negative, missense VOUS split 9 positive/7 negative - all exact matches to the paper. Published probe list (`signature_source=published_probe_list`) |
| Silver-Russell syndrome | The one disorder with a **genuine between-study result**: a 64-probe classifier built from GSE104451 alone (0% cross-reactivity against Sotos) tested on the fully-independent GSE55491 study - 5/18 sensitivity, 6/6 specificity. `evidence_bands.tsv` carries a real `interpretation_scope="between_study"` band for it - and it honestly reads **"No evidence"** at every prior, reflecting that modest cross-study sensitivity rather than being forced to look better. A pooled-studies attempt (328 probes) was tried and explicitly rejected after failing its own specificity check (94.7% false-positive rate against Sotos) |
| CHARGE syndrome | Builds: 907 probes, 89.5% self-consistency, but a real 37.8% cross-reactivity against Kabuki cases - reported as a genuine specificity concern, not smoothed over |
| Kabuki syndrome | Fails from GSE116300 alone; builds when pooled with GSE97362's own KMT2D-LOF cohort (278 probes, 97.3% self-consistency, 0% cross-reactivity against CHARGE) |
| Signature derivation thresholds | One documented change (`config/signature_thresholds.yaml`): Benjamini-Hochberg FDR 0.05 / 10% effect floor as the project default (Sotos's original Bonferroni/20% stays pinned for its own from-scratch validation path), set once before retrying anything, not tuned per outcome - see `results/tables/signature_derivation.tsv` for every attempt's exact parameters and probe counts |
| Label harmonization (AI cross-check) | 81/81 (100%) agreement on scoreable samples (19 of the 100 sampled marked `UNRESOLVABLE` from public metadata, correctly excluded rather than guessed) - `results/tables/harmonisation_agreement.tsv` |
| Label harmonization (human curation) | **pending** - a 30-row template weighted toward hard cases is generated (`data/external/human_curated.csv`) but not yet filled in by a person; `scripts/score_human_curation.py` reports this status honestly rather than substituting the AI number |
| Sample counts | 3,732 harvested = 1,622 retained for analysis + 2,110 excluded (198 unaffected relatives, 108 under-test, 7 wrong-tissue, 558 non-target-disease controls, 1,239 unresolvable labels) - `results/tables/sample_counts.tsv` |
| Confounding gate (12 in-scope disorders) | 7 fail (single-study, confounded by design - including Sotos itself), 2 not-testable (n<10), 3 pass-structural (span >=2 studies) - `results/tables/confounding_gate.tsv` |
| Cross-disorder matrix | **One** unified table (`status`: computed/not_computed) with **14 real cells** beyond nothing (Sotos 3, Silver-Russell 3, Kabuki 4, CHARGE 4) plus 8 honest not-computed rows (series matrices not downloaded this session) - `results/tables/cross_disorder_matrix.tsv`, rendered as `results/figures/cross_disorder_matrix.png` |
| "Why this result" | A real, computed attribution for one representative sample per Sotos cohort - probe coverage, direction check, distribution position, top contributing probes - no LLM narrative, `results/tables/attribution.json`, shown as an expander on the Grade a score page |

One correction to the original project brief worth stating plainly: its note that naively
counting GSE97362's validation/sequence-variant samples as cases "inflates cohort sizes to 59
and 26" only reconciles for 26 (the KMT2D-variant total). No combination of the harvested
fields produces 59 for the CHD7 side - the actual CHD7-related totals in the data are 19
confirmed discovery cases, 13 sequence-variant, and 40+ validation-cohort samples (72 total if
naively pooled, not 59). Reported here rather than silently forced to match - see
[docs/METHODS.md](docs/METHODS.md) for the full reconciliation.

## Layout

- `src/epigrade/` - library code (acquire, preprocess, signature, calibrate, audit, report, app)
- `config/` - paths, label vocabulary, and evidence-band tables (never hardcoded in source)
- `scripts/` - one-command entry points, one per pipeline phase
- `results/` - tracked tables, figures, and the app's resource JSON (never hand-edited).
  `results/figures/slides/` mirrors the same figures at 200dpi with projector-legible fonts,
  for a talk/slide deck - the originals in `results/figures/` are untouched.
- `data/` - gitignored; see `config/paths.yaml` for where bulk data actually lives on disk
