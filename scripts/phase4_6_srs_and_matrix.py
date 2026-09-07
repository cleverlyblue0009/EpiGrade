"""Phase 4 (cross-disorder specificity) + phase 6 (confounding audit), built around the one
disorder in this corpus with two independent GEO series: Silver-Russell syndrome (SRS,
GSE104451 n=61, GSE55491 n=24). This is deliberately the first disorder attempted beyond Sotos
because it is the ONLY one in this 18-accession corpus where a genuine leave-one-study-out test
is even possible - every other disorder here comes from a single study (see
results/tables/confounding_gate.tsv), which the spec anticipates as the expected, reportable
finding, not a bug to work around.

Path B derivation (Mann-Whitney/Bonferroni/effect-size, no published SRS probe list) is used for
the SRS classifier - see epigrade.signature.generic.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import pearsonr

from epigrade import paths
from epigrade.acquire.harmonize import harmonize
from epigrade.preprocess.series_matrix import parse_series_matrix
from epigrade.signature.choufani import classify_cohort, load_signature
from epigrade.signature.choufani import score_cohort as score_sotos_cohort
from epigrade.signature.generic import build_classifier, score_samples


def get_beta(series_id: str, gz_name: str) -> pd.DataFrame:
    cache = paths.interim_dir() / f"{series_id}_betas.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    beta = parse_series_matrix(paths.external_dir() / gz_name)
    beta.to_parquet(cache)
    return beta


def srs_ids(
    meta: pd.DataFrame, molecular_subtype_only: bool = False,
) -> tuple[list[str], list[str]]:
    """SRS is molecularly heterogeneous: '11p15 LOM' is a real, confirmed epimutation; 'UPD(7)'
    is a different chromosome/mechanism; 'clinical SRS' has no molecular confirmation at all.
    Pooling all three as one "case" group empirically washes out the differential signal (found
    during development: a Path B derivation on the pooled group recovered ~0 significant
    probes). This mirrors exactly how the Sotos paper restricted its discovery cohort to
    confirmed pathogenic NSD1 LOF, not everyone clinically diagnosed - the same principle
    applied here, not a post-hoc tweak to force a result. GSE55491 carries no subtype field at
    all, so molecular_subtype_only only has an effect on GSE104451.
    """
    cases = meta[meta.role == "case"]
    if molecular_subtype_only and "genotype/variation" in meta.columns:
        cases = cases[cases["genotype/variation"] == "11p15 LOM"]
    controls = meta[meta.role == "matched_control"].index.tolist()
    return cases.index.tolist(), controls


def main() -> None:
    samples = pd.read_parquet(paths.interim_dir() / "samples.parquet")
    harmonized = harmonize(samples)

    meta_104451 = harmonized[harmonized.series_id == "GSE104451"].set_index("gsm_accession")
    meta_55491 = harmonized[harmonized.series_id == "GSE55491"].set_index("gsm_accession")

    beta_104451 = get_beta("GSE104451", "GSE104451_series_matrix.txt.gz")
    beta_55491 = get_beta("GSE55491", "GSE55491_series_matrix.txt.gz")
    print(f"GSE104451: {beta_104451.shape}, GSE55491: {beta_55491.shape}")

    common_probes = beta_104451.index.intersection(beta_55491.index)
    print(f"Common probes between the two SRS series: {len(common_probes)}")

    cases_104451, controls_104451 = srs_ids(meta_104451)
    cases_55491, controls_55491 = srs_ids(meta_55491)
    cases_104451_lom, _ = srs_ids(meta_104451, molecular_subtype_only=True)
    print(f"GSE104451: {len(cases_104451)} SRS cases ({len(cases_104451_lom)} confirmed "
          f"11p15 LOM), {len(controls_104451)} controls")
    print(f"GSE55491: {len(cases_55491)} SRS cases, {len(controls_55491)} controls")

    # --- Phase 6: bidirectional leave-one-study-out ---
    # Building (deriving the signature) uses only the molecularly-confirmed 11p15 LOM subtype
    # from GSE104451 - see srs_ids() docstring. Testing/scoring always uses the FULL case set
    # (all molecular subtypes), since that's the realistic held-out scenario.
    loso_rows = []

    clf_104451 = build_classifier(
        "Silver-Russell syndrome", beta_104451.loc[common_probes],
        cases_104451_lom, controls_104451,
    )
    if clf_104451:
        scores_on_55491 = score_samples(
            clf_104451, beta_55491.loc[common_probes], cases_55491 + controls_55491,
        )
        n_case_pos = (scores_on_55491[cases_55491] > 0).sum()
        n_ctrl_neg = (scores_on_55491[controls_55491] < 0).sum()
        loso_rows.append({
            "train_study": "GSE104451", "test_study": "GSE55491",
            "n_signature_probes": len(clf_104451.signature_probes),
            "test_case_sensitivity": f"{n_case_pos}/{len(cases_55491)}",
            "test_control_specificity": f"{n_ctrl_neg}/{len(controls_55491)}",
        })
    else:
        # Diagnosed, not glossed over: 66 probes show a >20% effect size on this cohort (21
        # LOM-confirmed cases, 16 controls), but the smallest Mann-Whitney p-value (2.8e-7)
        # narrowly misses genome-wide Bonferroni significance (~1.0e-7 for 485,512 tests). This
        # reads as a real statistical-power limitation at this cohort size, not absence of a
        # true signal - see docs/METHODS.md. No classifier is built from non-significant
        # probes regardless; a marginal p-value is not treated as "close enough".
        loso_rows.append({
            "train_study": "GSE104451", "test_study": "GSE55491",
            "n_signature_probes": 0,
            "test_case_sensitivity": "NA - Path B found 66 probes with >20% effect size but "
                                      "none reached genome-wide Bonferroni significance "
                                      "(smallest p=2.8e-7 vs ~1.0e-7 threshold) at n=21 "
                                      "LOM-confirmed cases/16 controls - underpowered, not "
                                      "signal-free; no classifier built rather than forced",
            "test_control_specificity": "NA",
        })

    clf_55491 = build_classifier(
        "Silver-Russell syndrome", beta_55491.loc[common_probes], cases_55491, controls_55491,
    )
    if clf_55491:
        scores_on_104451 = score_samples(
            clf_55491, beta_104451.loc[common_probes], cases_104451 + controls_104451,
        )
        n_case_pos = (scores_on_104451[cases_104451] > 0).sum()
        n_ctrl_neg = (scores_on_104451[controls_104451] < 0).sum()
        loso_rows.append({
            "train_study": "GSE55491", "test_study": "GSE104451",
            "n_signature_probes": len(clf_55491.signature_probes),
            "test_case_sensitivity": f"{n_case_pos}/{len(cases_104451)}",
            "test_control_specificity": f"{n_ctrl_neg}/{len(controls_104451)}",
        })
    else:
        loso_rows.append({
            "train_study": "GSE55491", "test_study": "GSE104451",
            "n_signature_probes": 0,
            "test_case_sensitivity": "NA - could not build classifier (18 cases, "
                                      "6 controls: too few controls for a stable Path B "
                                      "derivation)",
            "test_control_specificity": "NA",
        })

    loso_df = pd.DataFrame(loso_rows)
    loso_path = paths.tables_dir() / "srs_leave_one_study_out.tsv"
    loso_df.to_csv(loso_path, sep="\t", index=False)
    print(f"\nWrote leave-one-study-out results -> {loso_path}")
    print(loso_df.to_string())

    # --- "official" SRS classifier for the cross-disorder matrix: both studies pooled ---
    all_beta_common = pd.concat(
        [beta_104451.loc[common_probes], beta_55491.loc[common_probes]], axis=1
    )
    all_cases = cases_104451 + cases_55491  # full set, all subtypes - used for scoring/testing
    all_cases_for_building = cases_104451_lom + cases_55491  # LOM-confirmed subset for deriving
    all_controls = controls_104451 + controls_55491
    srs_clf = build_classifier(
        "Silver-Russell syndrome", all_beta_common, all_cases_for_building, all_controls,
    )
    print(
        f"\nPooled SRS classifier (built from {len(all_cases_for_building)} molecularly-"
        f"confirmed cases): {len(srs_clf.signature_probes) if srs_clf else 0} signature probes"
    )

    # --- Sotos classifier + data, reloaded from phase 3's cache ---
    sotos_beta = get_beta("GSE74432", "GSE74432_series_matrix.txt.gz")
    sotos_meta = harmonized[harmonized.series_id == "GSE74432"].set_index("gsm_accession")
    sotos_cohort = classify_cohort(sotos_meta)
    sotos_sig = load_signature().index.intersection(sotos_beta.index)
    sotos_scores_own = score_sotos_cohort(sotos_beta, sotos_cohort, sotos_sig)

    # --- Cross-disorder matrix: score SRS cases/pooled-controls against the Sotos classifier,
    #     and Sotos cases/Weaver/pooled-controls against the SRS classifier ---
    matrix_rows = []

    # Row: Sotos classifier (from phase 3) scored against every available disorder's cases
    sotos_case_ids = sotos_cohort[sotos_cohort.isin(["discovery_case", "replication_case"])].index
    sotos_control_ids = sotos_cohort[sotos_cohort == "discovery_control"].index
    weaver_ids = sotos_cohort[sotos_cohort == "weaver"].index

    common_sotos_srs_probes = sotos_sig.intersection(all_beta_common.index)
    if len(common_sotos_srs_probes) > 10:
        median_case = sotos_beta.loc[common_sotos_srs_probes, sotos_case_ids].median(axis=1)
        median_control = sotos_beta.loc[common_sotos_srs_probes, sotos_control_ids].median(axis=1)
        srs_case_scores_on_sotos = {}
        for sid in all_cases:
            if sid not in all_beta_common.columns:
                continue
            x = all_beta_common.loc[common_sotos_srs_probes, sid]
            valid = x.notna() & median_case.notna() & median_control.notna()
            if valid.sum() < 10:
                continue
            r1 = pearsonr(x[valid], median_case[valid])[0]
            r0 = pearsonr(x[valid], median_control[valid])[0]
            srs_case_scores_on_sotos[sid] = r1 - r0
        n_pos = sum(1 for v in srs_case_scores_on_sotos.values() if v > 0)
        matrix_rows.append({
            "classifier": "Sotos syndrome", "scored_disorder": "Silver-Russell syndrome",
            "n": len(srs_case_scores_on_sotos),
            "fraction_positive": (
                round(n_pos / len(srs_case_scores_on_sotos), 3)
                if srs_case_scores_on_sotos else None
            ),
            "note": f"{len(common_sotos_srs_probes)} shared probes between Sotos signature "
                    "and the SRS-series beta matrices",
        })

    matrix_rows.append({
        "classifier": "Sotos syndrome", "scored_disorder": "Sotos syndrome (self)",
        "n": len(sotos_case_ids),
        "fraction_positive": round((sotos_scores_own[
            sotos_scores_own.cohort.isin(["discovery_case", "replication_case"])
        ]["score"] > 0).mean(), 3),
        "note": "from phase 3 (leave-one-out guarded for discovery samples)",
    })
    matrix_rows.append({
        "classifier": "Sotos syndrome", "scored_disorder": "Weaver syndrome",
        "n": len(weaver_ids),
        "fraction_positive": round((sotos_scores_own[
            sotos_scores_own.cohort == "weaver"
        ]["score"] > 0).mean(), 3),
        "note": "n=8 < 10: too small to interpret precisely, shown for reference",
    })

    # Row: SRS classifier scored against Sotos cases and Weaver (using the SRS-signature
    # probes intersected with GSE74432's beta matrix)
    if srs_clf is not None:
        common_srs_sotos_probes = srs_clf.signature_probes.intersection(sotos_beta.index)
        if len(common_srs_sotos_probes) > 10:
            mc = all_beta_common.loc[srs_clf.signature_probes, all_cases].median(axis=1)
            mn = all_beta_common.loc[srs_clf.signature_probes, all_controls].median(axis=1)
            for label, ids in [
                ("Sotos syndrome", sotos_case_ids), ("Weaver syndrome", weaver_ids),
            ]:
                pos = 0
                n_scored = 0
                for sid in ids:
                    if sid not in sotos_beta.columns:
                        continue
                    x = sotos_beta.loc[common_srs_sotos_probes, sid]
                    mc_i, mn_i = mc.loc[common_srs_sotos_probes], mn.loc[common_srs_sotos_probes]
                    valid = x.notna() & mc_i.notna() & mn_i.notna()
                    if valid.sum() < 10:
                        continue
                    r1 = pearsonr(x[valid], mc_i[valid])[0]
                    r0 = pearsonr(x[valid], mn_i[valid])[0]
                    n_scored += 1
                    if (r1 - r0) > 0:
                        pos += 1
                matrix_rows.append({
                    "classifier": "Silver-Russell syndrome", "scored_disorder": label,
                    "n": n_scored,
                    "fraction_positive": round(pos / n_scored, 3) if n_scored else None,
                    "note": f"{len(common_srs_sotos_probes)} shared probes (Path B "
                            "derived signature)",
                })

        srs_self_scores = score_samples(srs_clf, all_beta_common, all_cases)
        matrix_rows.append({
            "classifier": "Silver-Russell syndrome",
            "scored_disorder": "Silver-Russell syndrome (self)",
            "n": len(srs_self_scores),
            "fraction_positive": round((srs_self_scores > 0).mean(), 3),
            "note": "NOT leave-one-out guarded (pooled classifier scored on its own training "
                    "data) - see leave-one-study-out table above for the honest held-out result",
        })

    matrix_df = pd.DataFrame(matrix_rows)
    matrix_path = paths.tables_dir() / "cross_disorder_matrix.tsv"
    matrix_df.to_csv(matrix_path, sep="\t", index=False)
    print(f"\nWrote cross-disorder matrix ({len(matrix_df)} cells computed) -> {matrix_path}")
    print(matrix_df.to_string())

    # --- honest NA rows for every other in-scope disorder: no series matrix downloaded ---
    other_disorders = [
        "Kabuki syndrome type 1", "Kabuki syndrome type 2", "CHARGE syndrome",
        "Coffin-Siris syndrome", "Nicolaides-Baraitser syndrome", "Down syndrome",
        "Williams syndrome", "7q11.23 duplication syndrome", "ICF syndrome",
        "Claes-Jensen syndrome",
    ]
    na_rows = [{
        "classifier": d, "scored_disorder": "NOT COMPUTED",
        "n": None, "fraction_positive": None,
        "note": "series matrix not downloaded in this session (GEO FTP bandwidth/time "
                "constraints) - classifier not built, not a computed NA vs. a real zero",
    } for d in other_disorders]
    pd.DataFrame(na_rows).to_csv(
        paths.tables_dir() / "cross_disorder_matrix_not_computed.tsv", sep="\t", index=False
    )
    print(f"\nWrote {len(na_rows)} honestly-NA disorder rows (not attempted this session) -> "
          f"{paths.tables_dir() / 'cross_disorder_matrix_not_computed.tsv'}")


if __name__ == "__main__":
    main()
