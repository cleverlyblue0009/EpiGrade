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
