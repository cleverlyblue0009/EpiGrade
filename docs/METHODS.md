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
specificity) - one of two disorders (with Kabuki type 1) that have a between-study evidence band (see
`results/tables/evidence_bands.tsv`; it reads "No evidence" at every prior, honestly reflecting
that modest cross-study sensitivity). CHARGE syndrome (GSE97362, 19 cases vs 29 controls) builds
907 probes with 17 of 19 cases positive but a real, moderate specificity concern - 37.8%
cross-reactivity against Kabuki cases. Kabuki does not build from GSE116300 alone, but does when
pooled with GSE97362's own KMT2D-LOF cohort (278 probes, 35 of 37 cases positive under leave-one-out, 0%
cross-reactivity against CHARGE). See `docs/LIMITATIONS.md` for the full writeup, including a
pooling attempt for SRS that was tried and explicitly rejected after failing a specificity check
(92.1% (35 of 38) false-positive rate against Sotos), and a real data-corruption bug found and fixed while
building the merged cross-disorder matrix.

## Complete corpus (all 12 disorders)

`scripts/build_corpus.py` builds every classifier in one pass from the cohort definitions in
`src/epigrade/corpus/registry.py`. That file is the only place case and control membership is
defined.

- **Derivation.** Two-sided Mann-Whitney U per probe (asymptotic method throughout),
  Benjamini-Hochberg FDR at 0.05, mean beta difference above 10%. Sotos uses the published
  list. Every derivation, including each leave-one-study-out direction and the rejected pooled
  Silver-Russell alternative, is a row in `results/tables/signature_derivation.tsv`.
- **Controls.** A study's own matched controls where it has them. GSE116992 has none, so
  Coffin-Siris and its Nicolaides-Baraitser cases use the GSE125367 controls (a different
  laboratory). Down syndrome uses its 29 unaffected siblings. Both are tagged on every row.
- **Leakage guard.** `score_matrix` rebuilds the median profile without a training sample
  before scoring it. Probe selection is not nested. `tests/test_corpus.py` disables the guard
  and asserts separation increases.
- **Leave-one-study-out.** `epigrade.audit.loso`, both directions for each two-study disorder.
  A control used in training never counts toward held-out specificity.
- **Cross-disorder matrix.** Every classifier against all 12 case cohorts and 273 pooled
  matched controls, minus the classifier's own training controls.
- **Calibration.** `epigrade.calibrate`: study-level bootstrap where cases and controls each
  span at least two studies, sample-level otherwise, with the scope recorded.
- **Ceiling.** `scripts/compute_ceilings.py` simulates a perfect classifier at each disorder's
  real numbers of cases, controls and studies.

## Named finding: the controls-source effect

**Classifiers trained against out-of-series controls detect series membership, not disease.
Coffin-Siris and Nicolaides-Baraitser are both in that category.**

Every classifier carries a `controls_source` tag in `results/tables/cross_disorder_matrix.tsv`:
`within_series` if each study it trains on supplied its own controls, `cross_series` if any of
them took controls from another series. `results/tables/controls_source_summary.tsv` reduces
the matrix to the comparison that matters:

| controls_source | classifiers | off-diagonal cells (n >= 10) | mean fraction positive | on pooled controls |
|---|---|---|---|---|
| within_series | 8 | 72 | 11.9% | 11.9% |
| cross_series | 2 | 18 | 68.1% | 72.2% |

"Off-diagonal" means cases of a different disorder from the one the classifier was built for,
so every positive is a false positive. The mean is over cells; `tests/test_corpus.py` recomputes
it from the matrix. The heatmap outlines cross-series rows and prints the note above, and the
app's Cross-disorder page shows both.

Caveats: the two cross-series classifiers share one set of borrowed controls (GSE125367, a
different laboratory from GSE116992), so this is one confound seen twice. Within-series is not
a guarantee either: Down syndrome, compared against siblings from its own series, has the
highest within-series mean (42.2%).

## Identification engine

`src/epigrade/byo/evaluate.py`. For a submitted probe list:

1. Identifiers are validated (cg/ch/rs; EPIC v2 suffixes stripped), duplicates removed, and
   coverage reported per platform and per corpus matrix.
2. For each of the 12 cohorts, the weighted mean absolute case-control beta difference at the
   submitted probes is compared with 1,000 random probe sets of the same size drawn from that
   cohort's probes. AUC of cases against that cohort's controls is computed from
   leave-one-out correlation scores.
3. A cohort is a candidate if AUC is at least 0.70 and the submitted probes exceed all 1,000
   random sets. Candidates are ranked by z against the random-probe distribution. No candidate
   means no_match.
4. For a match: the signature's specificity row anchored on the matched cohort, an evidence
   ceiling (below), Jaccard overlap and containment against the ten corpus signatures, and direction
   agreement between the submitted column and the observed sign.

The random-probe comparison replaced an AUC-only rule after random probes were found to reach
AUC 1.00 in several single-study cohorts. See `docs/LIMITATIONS.md`.

**Evidence ceiling for a submitted signature.** The lower of two bounds. The cohort ceiling is
the band a perfect classifier earns at the matched cohort's size and study structure
(`attainable_ceiling.tsv`). The signature-size bound resamples the signature's probes and the
cohort's samples together 200 times, rescores the cohort each time, and reads the band from the
2.5th percentile of the local likelihood ratio at the typical case score. Fewer probes mean
more probe-sampling uncertainty and a lower bound: 20 Sotos probes are capped at Moderate,
500 at Strong. The result names which bound was limiting.

**AUC alone is unsafe for identification in this corpus.** Five hundred random probes reach an
AUC of 1.00 on Williams syndrome and at least 0.70 on 7 of 12 cohorts. The z-score against
random probe sets is what makes `no_match` work; AUC is reported but never decides a match by
itself.

## Plain-language summaries (explanation layer)

`src/epigrade/explain/`. The language model is a formatter, never an analyst.

1. **Fact sheet** (`facts.py`): built deterministically from the result tables or the BYO
   evaluation, rounded once. No model call.
2. **Model** (`groq_client.py`): `openai/gpt-oss-120b` on Groq (`config/explain.yaml`; the id is
   checked against the live model list for the key in use). Key from `GROQ_API_KEY` or the
   gitignored `.env`. Timeout 15 s, one retry on HTTP 429; any failure returns nothing.
3. **Validator** (`explain.py`): rejects a number not in the fact sheet (digits or words),
   clinical or mechanistic language, an unnamed disorder, a wrong data type or method name, an
   interpretive or quality claim, or a length outside 2-8 sentences. One retry with the
   reasons, then the template. Rejections are logged to `results/tables/explain_failures.tsv`.
4. **Template** (`template.py`): a fixed text from the same fact sheet; every template passes
   the validator. Used with no key, on any network failure, and after two rejections.
5. **Review gate**: outputs that passed the validator but were found wrong on review
   (`results/tables/explanation_review.tsv`) are replaced by the template in the app.

Measured (`scripts/eval_explanations.py`, final run): of 17 subjects, 12 passed first time, 4
after the retry, 1 fell back. On review, 7 of the 16 accepted outputs contained an error of
meaning and are blocked. The review was done by an AI assistant, not a person. Earlier runs,
including one that exposed two validator false positives, are kept as `*_run1.tsv` and
`*_run2.tsv`.

