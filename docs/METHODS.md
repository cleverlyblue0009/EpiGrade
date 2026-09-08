# Methods

This document is updated as each phase lands; it is the reference for exactly how each number
in `results/` was produced and where each external input came from.

## Positioning against prior work

EpiGrade does not claim novelty for its individual components - see the project's honesty
rules. It combines them into one reproducible-from-public-data pipeline:

- **Calibration**: Pejaver-style local-likelihood-ratio calibration (Pejaver et al. 2022, AJHG)
  and the Bayesian evidence-point framework (Tavtigian et al. 2018), with Brnich et al. 2020's
  OddsPath framing for how evidence strength maps to ACMG-style categories. A generic CRAN
  implementation of the calibration idea already exists (`acmgscaler`, Badonyi & Marsh 2025).
- **Cross-disorder specificity testing**: routine practice inside the private EpiSign Knowledge
  Database. The contribution here is that it's reproducible from public GEO accessions alone.
- **Episignature catalogue / accession list**: Aref-Eshghi et al. 2020, AJHG (42 syndromes
  screened, 34 robust episignatures).
- **Closest prior art**: Husson et al. 2024, EJHG - independent evaluation of 10 episignatures
  with covariate adjustment.
- **Clinical reporting context**: Kerkhof/Levy et al. 2024, Genet Med.
- **Private-data gap**: the EpiSign Knowledge Database cannot be deposited publicly for
  institutional/ethics reasons - this is the gap EpiGrade's public-data-only approach fills.

## Phase 1: metadata harvest and label harmonization

- GEOparse `how="brief"` only, never `how="full"` (full mode pulls each sample's per-probe
  data table - gigabytes per series for no benefit at the metadata stage).
- All 18 accessions harvested cleanly (3,732 samples, zero fetch failures); see
  `results/tables/sample_triage.tsv` for the harmonizer's full role assignment.
- `config/vocabulary.yaml` was built **empirically** from the actual harvested metadata (title,
  characteristics_ch1-derived fields, sample_type, source_name_ch1) for all 18 accessions -
  not from assumed prior knowledge of what each accession "should" contain. Three accessions
  (GSE74432, GSE97362, GSE116300) have externally published ground truth in the project spec;
  the harmonizer's output matches all three exactly (see `tests/test_harmonize.py`), including
  one refinement not explicitly required by the spec (GSE116300's 3 VUS-classified Kabuki
  samples are separated from the 26 pathogenic/likely-pathogenic cases into `under_test`, since
  `variant_classification` explicitly says "VUS").
- One spec figure did not reconcile: the spec's illustrative note that naively counting
  GSE97362's validation/sequence-variant samples as cases "inflates cohort sizes to 59 and 26"
  matches 26 (KMT2D variant total = 10 sequence-variant + 16 validation-cohort) but not 59
  (CHD7 variant total sequence-variant + validation-cohort = 13 + 40 = 53, and 19+53=72, neither
  of which is 59). Reported here rather than forced.
- Label harmonizer confidence router: rule-based matcher (canonical vocabulary substring/regex
  match, checked against "clean" categorical fields first, then freer-text fields) computes a
  confidence score; rows below 0.7 are escalated. Escalation uses **Gemini** (`google-genai`,
  default model `gemini-2.5-flash-lite`), not Anthropic (an explicit project decision, since the
  session's ANTHROPIC key was swapped for a Gemini key mid-project). Escalation rows are
  deduplicated by a canonicalized metadata signature (numeric IDs stripped) so, e.g., 586
  near-identical bare "Proband###" rows in GSE89353 cost one LLM call, not 586. With no
  `GEMINI_API_KEY` set, rows are left `needs_review` - never guessed.
- The "100-sample hand curation" (`scripts/phase1_report.py`) is an AI-performed cross-check
  against a second, structurally independent implementation of the same judgment (an explicit
  per-series answer key), not independent human clinical curation - see that script's docstring
  for the full honesty framing. It caught three real bugs during development (fibroblast
  samples wrongly going to `cell_line` instead of `wrong_tissue`; GSE74432's 16 "NSD1 variant"
  VOUS samples falling through to no disorder/role at all; and `"normal"/"control"` text
  matching `matched_control` even in series with no in-scope disorder, e.g. GSE87571's aging
  cohort). After fixes, cross-check agreement is 100% on the 81 scoreable samples (19 of the
  100 sampled rows were honestly marked `UNRESOLVABLE` from public metadata alone, not scored).

## Phase 2/3: GSE74432 beta values and the Sotos reproduction

### Data sources and exact retrieval

- **Series matrix**: `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE74nnn/GSE74432/matrix/GSE74432_series_matrix.txt.gz`
  (processed beta values, 215,281,941 bytes) - the processed-data file, not raw IDATs.
- **Supplementary Data 3 (7,085-probe signature) and Data 7 (missense variant clinical table)**:
  Choufani et al. 2015 is open access (PMC4703864). Retrieved via the Europe PMC supplementary-
  files API: `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4703864/supplementaryFiles`
  (a zip of all `ncomms10207-s*` files; `s4.xlsx` = Supplementary Data 3, `s8.xlsx` = Data 7).
  Verified against the spec's ground truth: 7,085 probes exactly, 99.3% loss-of-methylation
  direction exactly.
- **Chen et al. 2013 cross-reactive probes**: the original Weksberg-lab file (29,233 probes,
  cg + ch sheets), retrieved via the `Jfortin1/funnorm_repro` GitHub mirror:
  `https://github.com/Jfortin1/funnorm_repro/raw/master/bad_probes/48639-non-specific-probes-Illumina450k.xlsx`.
  **Not obtained**: Chen et al.'s separate SNP-containing-probe list - no direct machine-
  readable mirror of that specific list was found from a public source in this session (see
  Limitations). Note this turned out not to matter for reproducing the paper's 424,586 figure:
  the downloaded series matrix already arrives with exactly 424,586 probes and zero of the
  known cross-reactive probes present, meaning the original depositors filtered it before
  upload. This module's filtering step is a documented no-op on this particular file, and the
  match to the paper's number is a property of the deposited data, not of this code.

### Discovery vs. replication cohort reconstruction

GEO's `disease_state` field alone does not distinguish the paper's discovery cohort (used to
build the classifier's reference profiles) from its replication cohort (held out). This was
recovered by cross-referencing GEO sample **titles** against the original patient IDs listed in
Supplementary Data 1 (19 discovery Sotos LOF, blood-only after excluding 3 fibroblast), Data 2
(53 discovery controls), and Data 5 (19 replication Sotos LOF - all titled with an "HK-" prefix,
with zero "HK-" titles among the controls or discovery cases). This "HK- prefix = replication"
rule was verified exhaustively against every GSE74432 title, not assumed.

### Scoring rule

Exact rule from the paper - a correlation-difference score, not a trained classifier (SVM,
logistic regression, etc.):

```
SS_score(sample) = pearson(sample, median_case_profile) - pearson(sample, median_control_profile)
```

computed over the 7,085 signature probes. **Leakage guard**: median profiles are built from the
discovery cohort (19 cases + 53 controls) only; every discovery-cohort sample is scored against
a leave-one-out profile that excludes itself. Every other cohort (replication cases, Weaver,
missense VOUS) is scored against the full discovery-derived profile, since they were never part
of building it. `tests/test_choufani_leakage.py` proves this guard is live on synthetic data:
deliberately disabling the leave-one-out exclusion is asserted to *improve* apparent separation,
confirming the guard isn't a no-op.

### Reproduction result

All three published results reproduced exactly on the first fully-wired run (no numbers were
adjusted to fit): 19/19 discovery Sotos cases score positive and 53/53 discovery controls score
negative (clean separation); 8/8 Weaver syndrome samples score negative; the 16 NSD1 missense
VOUS split exactly 9 positive / 7 negative. As a bonus check not in the original three, all 19
held-out replication Sotos cases also scored positive. See `scripts/demo_sotos.py`,
`results/tables/sotos_reproduction_summary.tsv`, and `results/figures/sotos_*.png`.

## Phase 4/6: Path B thresholds, and the Silver-Russell/Kabuki/CHARGE classifiers

Path B (deriving a signature with no published probe list: Mann-Whitney U + a multiple-testing
correction + an effect-size floor) needs thresholds, and one fixed set does not generalize.
Applying Choufani et al.'s own Sotos thresholds (Bonferroni alpha=0.05, >20% effect size -
calibrated to NSD1's unusually large effect, where 7,085 probes survived even Bonferroni)
uniformly to every other disorder found nothing for Silver-Russell syndrome, Kabuki, or CHARGE.

`config/signature_thresholds.yaml` now holds a project-wide default - Benjamini-Hochberg FDR at
alpha=0.05, 10% effect-size floor - with Sotos's original thresholds pinned as a per-disorder
override (used only if Sotos is ever re-derived from scratch as a validation exercise; the
production path uses the published probe list directly, see `epigrade.signature.choufani`). FDR
control, not a loosened Bonferroni, is the standard alternative for genome-wide CpG testing
specifically because Bonferroni is notoriously over-conservative for the correlated probes
methylation arrays produce. This default was set once, before retrying any disorder under it,
and was not adjusted afterward based on outcome - see `results/tables/signature_derivation.tsv`
for the exact parameters and probe counts behind every attempt, and `tests/test_generic_signature.py`
for the tests locking in that Sotos's own thresholds stay pinned regardless.

Under this default: Silver-Russell syndrome (GSE104451 alone, 21 confirmed 11p15-LOM cases vs
16 controls) builds a 64-probe classifier with clean specificity (0% cross-reactivity against
Sotos) and a genuine, if modest, leave-one-study-out result on GSE55491 (5/18 sensitivity, 6/6
specificity) - the only disorder in this corpus with a real between-study evidence band (see
`results/tables/evidence_bands.tsv`; it reads "No evidence" at every prior, honestly reflecting
that modest cross-study sensitivity). CHARGE syndrome (GSE97362, 19 cases vs 29 controls) builds
907 probes with 89.5% self-consistency but a real, moderate specificity concern - 37.8%
cross-reactivity against Kabuki cases. Kabuki does not build from GSE116300 alone, but does when
pooled with GSE97362's own KMT2D-LOF cohort (278 probes, 97.3% self-consistency, 0%
cross-reactivity against CHARGE). See `docs/LIMITATIONS.md` for the full writeup, including a
pooling attempt for SRS that was tried and explicitly rejected after failing a specificity check
(94.7% false-positive rate against Sotos), and a real data-corruption bug found and fixed while
building the merged cross-disorder matrix.
