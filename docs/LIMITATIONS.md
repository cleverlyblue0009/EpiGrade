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
