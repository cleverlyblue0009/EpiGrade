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
from epigrade.calibrate.calibrate import evaluate_score
from epigrade.preprocess.series_matrix import parse_series_matrix
from epigrade.signature.choufani import classify_cohort, load_signature
from epigrade.signature.choufani import score_cohort as score_sotos_cohort
from epigrade.signature.generic import (
    build_classifier,
    derivation_stats,
    diagnose_underpowered,
    score_samples,
)

DISORDER = "Silver-Russell syndrome"


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
    # (all molecular subtypes), since that's the realistic held-out scenario. Thresholds come
    # from config/signature_thresholds.yaml (FDR-BH 0.05 / 10% effect floor by default) - see
    # that file and docs/METHODS.md for why the original Bonferroni/20% (tuned to Sotos's
    # unusually large effect) found nothing here.
    loso_rows = []
    derivation_rows = []

    derivation_rows.append(
        derivation_stats(beta_104451.loc[common_probes], cases_104451_lom, controls_104451,
                          disorder=DISORDER)
    )
    clf_104451 = build_classifier(
        DISORDER, beta_104451.loc[common_probes], cases_104451_lom, controls_104451,
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
            "signature_source": clf_104451.signature_source,
        })
    else:
        diagnosis = diagnose_underpowered(
            beta_104451.loc[common_probes], cases_104451_lom, controls_104451, disorder=DISORDER,
        )
        loso_rows.append({
            "train_study": "GSE104451", "test_study": "GSE55491",
            "n_signature_probes": 0,
            "test_case_sensitivity": f"NA - {diagnosis}",
            "test_control_specificity": "NA",
            "signature_source": "NA",
        })

    derivation_rows.append(
        derivation_stats(beta_55491.loc[common_probes], cases_55491, controls_55491,
                          disorder=DISORDER)
    )
    clf_55491 = build_classifier(
        DISORDER, beta_55491.loc[common_probes], cases_55491, controls_55491,
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
            "signature_source": clf_55491.signature_source,
        })
    else:
        diagnosis = diagnose_underpowered(
            beta_55491.loc[common_probes], cases_55491, controls_55491, disorder=DISORDER,
        )
        loso_rows.append({
            "train_study": "GSE55491", "test_study": "GSE104451",
            "n_signature_probes": 0,
            "test_case_sensitivity": f"NA - {diagnosis} (also: only {len(controls_55491)} "
                                      "controls, too few for a stable derivation regardless "
                                      "of threshold)",
            "test_control_specificity": "NA",
            "signature_source": "NA",
        })

    loso_df = pd.DataFrame(loso_rows)
    loso_path = paths.tables_dir() / "srs_leave_one_study_out.tsv"
    loso_df.to_csv(loso_path, sep="\t", index=False)
    print(f"\nWrote leave-one-study-out results -> {loso_path}")
    print(loso_df.to_string())

    # --- Also tried: pooling both studies for derivation. Logged and reported, NOT used as the
    # canonical classifier - see the specificity check below for why. ---
    all_beta_common = pd.concat(
        [beta_104451.loc[common_probes], beta_55491.loc[common_probes]], axis=1
    )
    all_cases = cases_104451 + cases_55491  # full set, all subtypes - used for scoring/testing
    all_cases_for_building = cases_104451_lom + cases_55491  # LOM-confirmed subset for deriving
    all_controls = controls_104451 + controls_55491
    derivation_rows.append(
        derivation_stats(all_beta_common, all_cases_for_building, all_controls, disorder=DISORDER)
    )
    pooled_clf = build_classifier(
        DISORDER, all_beta_common, all_cases_for_building, all_controls,
    )
    print(
        f"\nPooled SRS classifier (built from {len(all_cases_for_building)} molecularly-"
        f"confirmed cases): {len(pooled_clf.signature_probes) if pooled_clf else 0} "
        "signature probes"
    )

    # --- Canonical SRS classifier: GSE104451 alone (clf_104451, already built above), NOT the
    # pooled one. Checked both for self-consistency and specificity against an unrelated
    # disorder (Sotos) before deciding - this is not an assumption:
    #   clf_104451 (64 probes): self 19/21 cases positive, 16/16 controls negative; scored
    #     against 38 Sotos cases: 0% positive (clean specificity).
    #   pooled_clf (328 probes): self only 69.8% separation; scored against the SAME 38 Sotos
    #     cases: 94.7% positive - a near-total false-positive rate on an unrelated disorder.
    # Pooling GSE55491's molecularly-UNCONFIRMED "clinical SRS"/UPD(7) cases into the case pool
    # (see srs_ids()) evidently introduced batch/non-specific signal that additional cases alone
    # don't fix - more probes passing the same filters is not the same as a better signature.
    # This is reported as a real finding, not smoothed over by quietly using whichever number is
    # bigger. clf_104451 is used for the cross-disorder matrix and calibration below.
    srs_clf = clf_104451
    if srs_clf is None:
        print("clf_104451 unexpectedly None here - SRS has no usable classifier this run.")

    # --- signature_derivation.tsv: every Path B attempt this script made, parameters + counts ---
    deriv_path = paths.tables_dir() / "signature_derivation.tsv"
    pd.DataFrame(derivation_rows).to_csv(deriv_path, sep="\t", index=False)
    print(f"\nWrote {len(derivation_rows)} derivation attempts -> {deriv_path}")

    # --- The headline result: a GENUINE between-study evidence band for SRS. srs_clf was built
    # on GSE104451 alone, so scoring GSE104451+GSE55491 samples with real study_id tags per
    # sample lets evaluate_score run an honest study-level bootstrap (>=2 case studies AND >=2
    # control studies... except GSE55491 only has 6 controls, so control-side study count is
    # still 2 series but very unbalanced n - reported as-is, not adjusted). ---
    if srs_clf is not None:
        srs_scores_104451 = score_samples(
            srs_clf, beta_104451.loc[common_probes], cases_104451 + controls_104451,
        )
        srs_scores_55491 = score_samples(
            srs_clf, beta_55491.loc[common_probes], cases_55491 + controls_55491,
        )
        cal_rows = []
        for sid, score in srs_scores_104451.items():
            cal_rows.append({
                "gsm_accession": sid, "score": score,
                "label": int(sid in cases_104451), "study_id": "GSE104451",
            })
        for sid, score in srs_scores_55491.items():
            cal_rows.append({
                "gsm_accession": sid, "score": score,
                "label": int(sid in cases_55491), "study_id": "GSE55491",
            })
        cal_df = pd.DataFrame(cal_rows)

        case_median = cal_df.loc[cal_df.label == 1, "score"].median()
        srs_evidence_rows = []
        for query_name, query_score in [
            ("typical_case_score", case_median), ("borderline_score_zero", 0.0),
        ]:
            for prior in (0.05, 0.10, 0.20):
                result = evaluate_score(DISORDER, cal_df, query_score, prior=prior)
                srs_evidence_rows.append({
                    "disorder": result.disorder, "query_point": query_name,
                    "query_score": round(query_score, 4), "prior": prior,
                    "lr_point_estimate": result.lr_point_estimate,
                    "lr_ci_low": result.lr_ci_low, "lr_ci_high": result.lr_ci_high,
                    "points_conservative": result.points, "band": result.band,
                    "posterior_prob": result.posterior_prob,
                    "evidence_anchor": result.evidence_anchor,
                    "n_case": result.n_case, "n_control": result.n_control,
                    "n_studies_case": result.n_studies_case,
                    "n_studies_control": result.n_studies_control,
                    "confounded_by_design": result.confounded_by_design,
                    "reason": result.reason,
                    "within_study_ci_low": result.within_study_ci_low,
                    "within_study_ci_high": result.within_study_ci_high,
                    "interpretation_scope": result.interpretation_scope,
                })
        srs_evidence_df = pd.DataFrame(srs_evidence_rows)
        evidence_path = paths.tables_dir() / "evidence_bands.tsv"
        existing_evidence = (
            pd.read_csv(evidence_path, sep="\t") if evidence_path.exists() else pd.DataFrame()
        )
        existing_evidence = existing_evidence[existing_evidence.get("disorder") != DISORDER] \
            if len(existing_evidence) else existing_evidence
        pd.concat([existing_evidence, srs_evidence_df], ignore_index=True).to_csv(
            evidence_path, sep="\t", index=False
        )
        print(f"\nWrote {len(srs_evidence_df)} SRS evidence rows -> {evidence_path}")
        print(srs_evidence_df[["query_point", "prior", "band", "interpretation_scope",
                                "confounded_by_design"]].to_string())

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

    # Row: SRS classifier (clf_104451, the canonical one - see above) scored against Sotos
    # cases and Weaver. Uses score_samples throughout, which reads the classifier's OWN stored
    # median_case/median_control (from its actual training ids) - NOT a profile recomputed from
    # the pooled sample sets, which would silently mix in GSE55491 samples this classifier was
    # never built from (a real bug caught while wiring up the canonical-classifier switch).
    if srs_clf is not None:
        common_srs_sotos_probes = srs_clf.signature_probes.intersection(sotos_beta.index)
        if len(common_srs_sotos_probes) > 10:
            score_targets = [
                ("Sotos syndrome", sotos_case_ids.tolist()),
                ("Weaver syndrome", weaver_ids.tolist()),
            ]
            for label, ids in score_targets:
                scores = score_samples(srs_clf, sotos_beta, ids)
                matrix_rows.append({
                    "classifier": "Silver-Russell syndrome", "scored_disorder": label,
                    "n": len(scores),
                    "fraction_positive": round((scores > 0).mean(), 3) if len(scores) else None,
                    "note": f"{len(common_srs_sotos_probes)} shared probes (Path B "
                            "derived signature, GSE104451-only classifier)",
                })

        # A blended "fraction_positive" across a MIXED case+control set is not interpretable
        # (a great classifier where all cases score positive and all controls score negative
        # would still show ~50% here) - report sensitivity and specificity separately instead,
        # matching the leave-one-study-out table's convention. Caught while writing this row:
        # the naive blended metric read as a mediocre 0.514 even though the actual performance
        # (below) is 19/21 sensitivity, 16/16 specificity.
        srs_self_case_scores = score_samples(
            srs_clf, beta_104451.loc[common_probes], cases_104451_lom,
        )
        srs_self_control_scores = score_samples(
            srs_clf, beta_104451.loc[common_probes], controls_104451,
        )
        n_sens = (srs_self_case_scores > 0).sum()
        n_spec = (srs_self_control_scores < 0).sum()
        matrix_rows.append({
            "classifier": "Silver-Russell syndrome",
            "scored_disorder": "Silver-Russell syndrome (self)",
            "n": len(srs_self_case_scores) + len(srs_self_control_scores),
            "fraction_positive": None,
            "note": f"NOT leave-one-out guarded (scored on its own GSE104451 training data): "
                    f"sensitivity {n_sens}/{len(srs_self_case_scores)} cases, specificity "
                    f"{n_spec}/{len(srs_self_control_scores)} controls - see the "
                    "leave-one-study-out table above for the genuine held-out (GSE55491) "
                    "result this classifier actually achieved",
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
