"""Reproduction of the Choufani et al. 2015 (Nature Communications, ncomms10207) Sotos syndrome
episignature classifier on GSE74432.

Scoring rule (exact, from the paper - this is NOT an SVM or any other learned classifier):

    SS_score(sample) = pearson(sample, median_case_profile)
                        - pearson(sample, median_control_profile)

computed over the signature probes only. Positive => classified Sotos-like.

Cohort structure (recovered from cross-referencing GEO sample titles against the paper's own
Supplementary Data 1/2/5/6/7 - see docs/METHODS.md for the exact reconciliation):
  - Discovery cohort (Supplementary Data 1 + 2): 19 blood Sotos with confirmed NSD1 LOF
    (GEO titles NOT prefixed "HK-") + 53 blood controls. ONLY this cohort may be used to build
    the median_case/median_control profiles - see the leakage guard in `score_cohort`.
  - Replication cohort (Supplementary Data 5): 19 more blood Sotos LOF cases, GEO titles
    prefixed "HK-". Scored against the discovery-derived profiles, never used to build them.
  - Weaver syndrome (EZH2): 8 samples, expected to score negative (not Sotos-like).
  - NSD1 missense VOUS (Supplementary Data 7): 16 samples (disease_state "NSD1 variant"),
    6 from the paper's own discovery cohort + 10 from its validation cohort. Scored, not used
    to build profiles. This is the missense_split reproduction target (paper: 9 positive / 7
    negative).
  - 7 fibroblast samples (3 Sotos + 4 control): excluded entirely (tissue mismatch, flagged
    wrong_tissue by the harmonizer) - never scored, per the project's exclusion rules.
"""

from __future__ import annotations

import openpyxl
import pandas as pd
from scipy.stats import pearsonr

from epigrade import paths

SIGNATURE_PATH = "choufani_2015/SupplementaryData3_signature_probes.xlsx"
EXPECTED_SIGNATURE_SIZE = 7085


def load_signature() -> pd.DataFrame:
    """The 7,085-probe NSD1+/- signature from Supplementary Data 3."""
    path = paths.external_dir() / SIGNATURE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - see docs/METHODS.md for the retrieval command "
            "(Europe PMC supplementary-files API for PMC4703864)."
        )
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Suppl. Data 3"]
    rows = [
        r for r in ws.iter_rows(min_row=4, values_only=True)
        if isinstance(r[0], str) and r[0].startswith("cg")
    ]
    wb.close()
    df = pd.DataFrame(rows, columns=[
        "illumina_id", "p_value", "bonferroni_p", "delta_beta", "abs_delta_beta",
        "methylation_effect", "mean_not_ss", "mean_ss", "regression_abs_delta_beta",
        "regression_p", "regression_bonferroni_p", "genome_build", "chromosome",
        "genomic_location_hg19", "strand", "relation_to_cpg_island", "gene_symbol",
    ])
    return df.set_index("illumina_id")


def classify_cohort(meta: pd.DataFrame) -> pd.Series:
    """Assign each GSE74432 sample to a cohort role for scoring, from harmonized metadata.

    Returns a Series aligned to `meta`'s index with values in:
    discovery_case, discovery_control, replication_case, weaver, missense_variant, excluded
    """
    is_fibroblast = meta["source_name_ch1"].str.contains("fibroblast", case=False, na=False)
    is_hk = meta["title"].str.contains(r"^HK-?\d", case=False, na=False, regex=True)

    cohort = pd.Series("excluded", index=meta.index)
    cohort[is_fibroblast] = "excluded"  # tissue mismatch - never scored

    sotos = (meta["disease_state"] == "Sotos") & ~is_fibroblast
    cohort[sotos & ~is_hk] = "discovery_case"
    cohort[sotos & is_hk] = "replication_case"

    control = (meta["disease_state"] == "Control") & ~is_fibroblast
    cohort[control] = "discovery_control"  # no HK-prefixed controls exist in this series

    cohort[meta["disease_state"] == "Weaver"] = "weaver"
    cohort[meta["disease_state"] == "NSD1 variant"] = "missense_variant"
    return cohort


def _median_profile(beta: pd.DataFrame, sample_ids: list[str], probes: pd.Index) -> pd.Series:
    return beta.loc[probes, sample_ids].median(axis=1)


def score_cohort(beta: pd.DataFrame, cohort: pd.Series, signature_probes: pd.Index) -> pd.DataFrame:
    """Score every non-excluded GSE74432 sample.

    LEAKAGE GUARD: for discovery_case/discovery_control samples (the ones used to BUILD the
    median profiles), each sample is scored against profiles built with itself excluded
    (leave-one-out). Every other cohort is scored against the full discovery-derived profiles,
    which never included them in the first place. See tests/test_choufani_leakage.py for a test
    that deliberately disables this guard and asserts separation improves - proving the guard is
    live, not a no-op.
    """
    discovery_cases = cohort[cohort == "discovery_case"].index.tolist()
    discovery_controls = cohort[cohort == "discovery_control"].index.tolist()

    full_median_case = _median_profile(beta, discovery_cases, signature_probes)
    full_median_control = _median_profile(beta, discovery_controls, signature_probes)

    rows = []
    for sample_id, group in cohort.items():
        if group == "excluded" or sample_id not in beta.columns:
            continue

        if group in ("discovery_case", "discovery_control"):
            loo_cases = [s for s in discovery_cases if s != sample_id]
            loo_controls = [s for s in discovery_controls if s != sample_id]
            median_case = _median_profile(beta, loo_cases, signature_probes)
            median_control = _median_profile(beta, loo_controls, signature_probes)
        else:
            median_case, median_control = full_median_case, full_median_control

        x = beta.loc[signature_probes, sample_id]
        valid = x.notna() & median_case.notna() & median_control.notna()
        r_case = pearsonr(x[valid], median_case[valid])[0]
        r_control = pearsonr(x[valid], median_control[valid])[0]

        rows.append({
            "gsm_accession": sample_id,
            "cohort": group,
            "score": r_case - r_control,
            "r_case": r_case,
            "r_control": r_control,
            "n_probes_used": int(valid.sum()),
        })

    return pd.DataFrame(rows)
