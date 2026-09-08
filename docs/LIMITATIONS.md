# Limitations

This document is updated as each phase lands. It exists so that a reader doesn't have to dig
through code comments to find out where this project's numbers stop being fully independent.

## Not for patient use

EpiGrade takes a classifier score and reports the evidence strength it can support against this
corpus. It is not a diagnostic device, does not accept patient data or raw arrays, and nothing
here should inform a clinical decision on its own.

## Phase 1: label harmonization

- The "100-sample hand curation" agreement check (`results/tables/harmonisation_agreement.tsv`)
  was performed by this project's AI assistant, reading raw GEO metadata directly and writing an
  explicit, structurally independent second implementation of the same judgment to cross-check
  the generic rule engine against. **It is not independent human clinical curation.** The only
  genuinely external validation is the exact match, on the three accessions with published
  ground truth in the project spec (GSE74432, GSE97362, GSE116300), to numbers drawn from the
  original papers - see `tests/test_harmonize.py`.
- 19 of the 100 hand-checked samples were honestly marked `UNRESOLVABLE` from public GEO
  metadata alone (e.g. GSE89353's bare "Proband###" titles carry no diagnosis at all; GSE87648's
  "HL"/"HS" simplified-diagnosis codes could not be confidently resolved to a specific meaning).
  These are excluded from the agreement-rate denominator rather than guessed either way.
- GSE89353 (620 samples) has no per-sample diagnosis anywhere in its public GEO metadata beyond
  a bare "Proband###" or "Proband###_mother/_father" title. It cannot feed any single-disorder
  classifier in this project and is excluded from phase 4's cross-disorder matrix for that
  reason, not because it was inconvenient.
- The spec's illustrative claim that naively counting GSE97362's validation/sequence-variant
  samples as cases "inflates cohort sizes to 59 and 26" only reconciles for 26 (KMT2D variant);
  no combination of the harvested fields produces 59 for CHD7. Reported, not forced - see
  `docs/METHODS.md`.

## Phase 2/3: GSE74432 preprocessing and the Sotos reproduction

- **Chen et al. 2013 SNP-containing probe list**: could not be sourced as a direct, citable
  machine-readable file from a public mirror in this session. The cross-reactive probe list (the
  other half of Chen et al.'s filter) was obtained as the exact original file. In practice this
  didn't block reproducing the paper's 424,586-probe figure, because the downloaded series
  matrix already arrives pre-filtered by the original depositors to exactly that count with zero
  of the known cross-reactive probes present - so the match reflects the deposited data, not an
  independent re-derivation of the filter on this project's part. A from-scratch re-derivation
  (Path B in the spec: Mann-Whitney U / Bonferroni / family-swap trials / effect-size filter)
  was not needed since the published signature list (Path A) was successfully retrieved, and was
  not separately attempted - if the exact reproduction had failed, Path B would be the next step.
- The GSE74432 "benign" missense classification is not independently established: the paper (and
  this reproduction) determines the 7 "benign-scoring" NSD1 missense variants by running the
  classifier itself, not from an independent clinical truth source. Treat the missense-split
  result as "the classifier's own self-consistent behavior on VOUS," not as external validation
  that those 7 variants are truly benign.
- Sex estimation (X/Y probe-based) and Houseman blood-cell-composition covariates, called for in
  phase 2, are not yet implemented as of this writing - the core reproduction (phase 3) does not
  require them (the original Choufani scoring rule uses raw signature-probe betas directly, with
  no covariate adjustment), so they were deprioritized behind getting the three published results
  reproduced. GSE35069 (the Reinius et al. blood-cell-type reference panel) is present in this
  corpus and is the natural reference for a Houseman implementation if/when it is built.

## Finish-today session: fixes and a genuine negative result

- **A real bug, found and fixed**: every Sotos row in `evidence_bands.tsv` had
  `lr_ci_low == lr_ci_high == point_estimate`, because a single-study cohort makes study-level
  bootstrap resampling degenerate (every resample is identical to the original data). A "Strong"
  band was still being assigned from that zero-width, non-informative interval. Fixed: a cohort
  with fewer than 2 studies on either the case or control side now gets `band="NA"` with an
  explicit reason, never a band from a degenerate CI. A within-study (sample-level) bootstrap is
  reported alongside for reference only, explicitly labeled as never substituting for the
  refused band. See `tests/test_calibrate.py` for the regression tests.
- **A second real bug, found via a clean-clone check**: pandas' default `read_csv` NA-string
  handling treats the literal text `"NA"` as null. The `band="NA"` refusal above was therefore
  silently becoming a float `NaN` on the TSV->JSON path, which `json.dump` then wrote as the
  non-standard `NaN` token - not valid JSON per spec. Fixed by reading with
  `keep_default_na=False` and sanitizing any residual NaN to `null` before writing, with
  `allow_nan=False` so a future regression fails loudly instead of silently re-corrupting the
  file. Neither bug would have been visible from a quick read of the numbers - both were only
  caught by writing tests that check the *shape* of a refusal, not just that one exists.
- **Kabuki syndrome (GSE116300) and CHARGE syndrome (GSE97362) do not clear this project's own
  reliability bar for a re-derived classifier**, despite both being genuinely real, well-
  established episignatures in the literature. This was investigated thoroughly, not assumed:
  - Kabuki (GSE116300 alone: 26 confirmed cases, 9 controls): 44 probes show >20% effect size,
    but the smallest Mann-Whitney p-value (1.84e-05) is nowhere near the genome-wide Bonferroni
    threshold (1.04e-07). Pooling with GSE97362's own 11 KMT2D-LOF-discovery Kabuki cases and 11
    matched controls (the same legitimate technique already used for Silver-Russell syndrome)
    raises the cohort to 37 cases/20 controls and gets dramatically closer - smallest p=5.25e-08
    vs a 1.04e-07 threshold, off by less than 2x - but still does not clear it. This is a
    genuinely close miss, illustrating concretely how much cohort size, not biology, is the
    limiting factor here.
  - CHARGE (GSE97362, 19 discovery-LOF cases, 29 matched controls): a *different* failure shape.
    35 probes ARE genome-wide Bonferroni-significant, and 32 separately show >20% effect size,
    but only 3 probes satisfy both criteria at once - the smallest p-values belong to probes
    with small, highly consistent differences, not the probes with the largest raw effect size.
    A 3-probe signature was initially built (technically "passing" the original criteria) and
    then produced zero usable sample scores downstream (`score_samples` requires 10 valid probes
    per sample), crashing the pipeline. Fixed by adding `MIN_SIGNATURE_SIZE=10` to
    `build_classifier` itself: a handful of significant probes is now treated as equally
    underpowered as finding none, consistently, rather than silently accepted and breaking later.
  - Neither result is a workaround-and-retry situation: no threshold was loosened, no
    alternative statistical test was substituted, and no correction method was switched, in
    either direction, specifically to make either disorder pass. The same Mann-Whitney U /
    Bonferroni / >20% effect-size procedure, and the same `MIN_SIGNATURE_SIZE=10` bar, were
    applied uniformly to Sotos (worked, via the published list, not this procedure), Silver-
    Russell (failed), Kabuki (failed even pooled), and CHARGE (failed). The cross-disorder
    matrix (`results/tables/cross_disorder_matrix.tsv`) therefore still contains only the three
    Sotos-classifier rows established in phase 4/6; `cross_disorder_matrix_not_computed.tsv`
    carries the full diagnosis for every disorder that didn't make it in, including these two.
  - This is arguably a more informative result for a reproducibility benchmark than a working
    3x3 matrix would have been: it demonstrates, with a specific near-miss margin, that small
    public GEO cohorts can hold back even disorders with genuinely robust real signatures -
    which is precisely the private-vs-public-data gap this project exists to make visible.
- **A third real bug, found during the documentation truth pass**: `resource.py`'s harmonisation
  agreement aggregation used `r.get("agree") is not None` to filter out the 19 `UNRESOLVABLE`
  hand-curation rows - but those rows read back from the TSV as a float `NaN` (pandas' `to_dict`
  conversion), not a Python `None`, and `NaN is not None` evaluates to `True`. Worse, `bool(nan)`
  is *also* `True` in Python, so all 19 were silently counted as both "scoreable" and "agreed",
  inflating the denominator from 81 to 100. The reported agreement rate still came out as 100%
  either way (all 81 genuinely scoreable rows did agree), which is exactly why it went unnoticed
  until checked explicitly rather than trusted at face value - a dataset with any real
  disagreement would have silently under-reported it. Fixed with `pd.notna()`; regression test
  in `tests/test_resource.py` locks the correct denominator (81) in place.

## Second finish-today session: configurable thresholds, a real SRS result, and two more bugs

- **One documented threshold change, applied uniformly, not tuned per outcome.** The original
  Path B thresholds (Mann-Whitney U + Bonferroni + >20% effect-size floor) were Choufani et
  al.'s own, calibrated to Sotos's unusually large NSD1 effect (7,085 probes survived even
  Bonferroni). Applied uniformly to Silver-Russell syndrome, Kabuki, and CHARGE, they found
  nothing for any of the three. `config/signature_thresholds.yaml` now holds a project-wide
  default (Benjamini-Hochberg FDR at 0.05, 10% effect-size floor - a standard, defensible choice
  for genome-wide CpG testing, not a loosened Bonferroni) plus a pinned override for Sotos so a
  from-scratch Path B re-derivation of it (a validation exercise only - production Sotos uses
  the published probe list, see `epigrade.signature.choufani`) still reproduces what the paper
  did. This default was set once, before retrying any disorder, and was not adjusted afterward
  based on which disorders it did or didn't unlock - see `tests/test_generic_signature.py`.
- **Silver-Russell syndrome now has a real classifier and a genuine between-study result.**
  GSE104451 alone (21 molecularly-confirmed 11p15-LOM cases, 16 controls) builds a 64-probe
  classifier under the new default. Checked for specificity before trusting it: 19/21 (90%)
  self-sensitivity, 16/16 (100%) self-specificity, and 0% cross-reactivity against 38 unrelated
  Sotos cases. Tested on the held-out GSE55491 study (the actual leave-one-study-out test):
  5/18 (28%) sensitivity, 6/6 (100%) specificity - modest but genuine, not fabricated.
  `results/tables/evidence_bands.tsv` now carries a real `interpretation_scope="between_study"`
  row for SRS (the only one in the corpus) - and it honestly reads **"No evidence"** at every
  prior, because the between-study bootstrap CI spans LR=1. This is the correct, non-forced
  outcome given the modest cross-study sensitivity, and is itself the headline result: SRS is
  the one disorder here where a real (non-degenerate) between-study bound could be computed at
  all, and that bound says this particular classifier doesn't yet reliably generalize.
  - Also tried and explicitly rejected: pooling GSE104451 + GSE55491 for derivation (328 probes,
    more than GSE104451 alone). Checked the same way before accepting or rejecting it: only
    69.8%-equivalent (19/37) self-consistency and a **94.7% false-positive rate scored against
    Sotos cases** - a near-total specificity failure, evidently from GSE55491's molecularly-
    unconfirmed "clinical SRS"/UPD(7) cases introducing batch/non-specific signal into the case
    pool. More probes passing the same filters is not the same as a better signature; the
    pooled classifier was NOT used as canonical anywhere downstream once this was found.
  - The reverse LOSO direction (GSE55491 training GSE104451) still fails - only 6 controls,
    0 significant probes even under FDR-BH/10%. Reported as insufficient controls, not
    re-attempted with a further-loosened threshold.
- **CHARGE syndrome now builds**: 907 probes from 19 discovery-LOF cases vs 29 matched controls
  (previously 3 fragile probes under the old thresholds - see the first finish-today session's
  writeup above). Self-consistency 89.5%. Cross-reactivity: 0% against 20 of Kabuki's controls,
  6.9% against its own 29 training controls, but **37.8% against 37 Kabuki cases** - a real,
  moderate specificity concern, reported plainly rather than smoothed over. Kabuki's own
  cross-reactivity against CHARGE cases is 0%, so this asymmetry (CHARGE calls many Kabuki
  cases positive; Kabuki calls no CHARGE cases positive) is itself worth further investigation,
  not resolved here.
- **Kabuki syndrome still does not build**, even under the new thresholds, from GSE116300 alone
  (26 confirmed cases, 9 controls: 758 probes show >10% effect size but the smallest p-value,
  1.84e-05, is far from FDR-BH significance at this cohort size). Pooled with GSE97362's own 11
  KMT2D-LOF-discovery cases and 11 matched controls, it DOES build - 278 probes, 97.3%
  self-consistency, 0% cross-reactivity against CHARGE and against GSE97362's controls, 15%
  against its own GSE116300 control subset. This pooled classifier is used as canonical for
  Kabuki (unlike the SRS pooling case, this one checked out on specificity).
- **A real bug, found and fixed while building `scripts/merge_cross_disorder_matrix.py`**: the
  first run correctly merged `cross_disorder_matrix.tsv` and `..._not_computed.tsv` into one
  table with a `status` column. Re-running the same script a second time (to apply a heatmap
  rendering fix) fed the script's OWN already-merged output back in as the "computed" input and
  unconditionally set `status="computed"` on all of it - silently relabeling 8 genuinely
  not-computed rows as computed (their actual value cells stayed empty, but the status label
  itself became false). Caught by inspecting the output before moving on, not by a pre-written
  test. Fixed by only defaulting missing status to "computed", never overwriting an existing
  one; the affected table was regenerated from the original source scripts (which were never
  corrupted) rather than patched in place. This is exactly the class of silent-mislabeling bug
  the project's own honesty rules exist to catch - recorded here per those same rules.
- **Cross-disorder matrix, current real state**: 14 computed cells (Sotos: 3, Silver-Russell: 3,
  Kabuki: 4, CHARGE: 4) plus 8 honestly `not_computed` rows (series matrices for Coffin-Siris,
  Nicolaides-Baraitser, Down, Williams, 7q11.23 duplication, ICF, Claes-Jensen, and Kabuki type 2
  were not downloaded this session - a time/bandwidth constraint, not a decision to exclude
  them). `results/figures/cross_disorder_matrix.png` renders this as a heatmap with diagonal
  (self/sensitivity) cells outlined in black - a plain shared red-is-high color scale would
  otherwise make a *good* 100% self-sensitivity cell look identical to a *bad* 100%
  cross-reactivity cell, which was caught by looking at the rendered figure, not assumed correct
  from the code.
