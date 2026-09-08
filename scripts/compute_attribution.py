"""Task 5: computes real "why this result" attribution for one representative (median-scoring)
sample per Sotos cohort, using the already-cached beta matrix and the classifier's own stored
profiles - never recomputed by the app, which only displays results/tables/attribution.json.
"""

from __future__ import annotations

import json

import pandas as pd

from epigrade import paths
from epigrade.acquire.harmonize import harmonize
from epigrade.report.attribution import explain_score, format_explanation
from epigrade.signature.choufani import classify_cohort, load_signature

DISORDER = "Sotos syndrome"
EXPECTED_DIRECTION = "loss"  # Choufani et al.: 99.3% of signature CpGs show loss in cases


def main() -> None:
    beta_path = paths.interim_dir() / "GSE74432_betas.parquet"
    if not beta_path.exists():
        print(f"{beta_path} not cached yet - run scripts/demo_sotos.py first. Skipping.")
        return
    beta = pd.read_parquet(beta_path)

    samples = pd.read_parquet(paths.interim_dir() / "samples.parquet")
    harmonized = harmonize(samples)
    meta = harmonized[harmonized.series_id == "GSE74432"].set_index("gsm_accession")
    cohort = classify_cohort(meta)
    signature = load_signature()
    sig_probes = signature.index.intersection(beta.index)

    scores = pd.read_csv(paths.tables_dir() / "sotos_scores.tsv", sep="\t")
    evidence = pd.read_csv(paths.tables_dir() / "evidence_bands.tsv", sep="\t")
    sotos_evidence = evidence[evidence.disorder == DISORDER]

    discovery_case_ids = cohort[cohort == "discovery_case"].index
    discovery_control_ids = cohort[cohort == "discovery_control"].index
    median_case = beta.loc[sig_probes, discovery_case_ids].median(axis=1)
    median_control = beta.loc[sig_probes, discovery_control_ids].median(axis=1)

    case_score_dist = scores.loc[scores.cohort == "discovery_case", "score"]
    control_score_dist = scores.loc[scores.cohort == "discovery_control", "score"]

    query_point_for_cohort = {
        "discovery_case": "typical_case_score", "discovery_control": "typical_case_score",
        "weaver": "typical_case_score", "missense_variant": "typical_missense_vous_score",
    }

    examples = []
    for cohort_label in ["discovery_case", "discovery_control", "weaver", "missense_variant"]:
        cohort_scores = scores[scores.cohort == cohort_label]
        if cohort_scores.empty:
            continue
        # the median-scoring sample in this cohort - the "typical" example, not cherry-picked
        median_score = cohort_scores["score"].median()
        exemplar_row = cohort_scores.iloc[
            (cohort_scores["score"] - median_score).abs().argsort().iloc[0]
        ]
        gsm = exemplar_row["gsm_accession"]

        result = explain_score(
            gsm_accession=gsm, disorder=DISORDER, cohort_label=cohort_label,
            sample_beta=beta[gsm], median_case=median_case, median_control=median_control,
            signature_probes=sig_probes, case_scores=case_score_dist,
            control_scores=control_score_dist, expected_direction=EXPECTED_DIRECTION,
        )

        qp = query_point_for_cohort[cohort_label]
        band_row = sotos_evidence[
            (sotos_evidence.query_point == qp) & (sotos_evidence.prior == 0.10)
        ]
        if len(band_row):
            b = band_row.iloc[0]
            result.band = b["band"]
            result.lr_point_estimate = float(b["lr_point_estimate"])
            result.interpretation_scope = b["interpretation_scope"]
            result.band_reason = b["reason"]

        examples.append({
            **result.__dict__,
            "explanation": format_explanation(result),
        })
        print(f"\n=== {cohort_label}: {gsm} (score={result.score:+.3f}) ===")
        for line in format_explanation(result):
            print(f"  {line}")

    out_path = paths.tables_dir() / "attribution.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"disorder": DISORDER, "examples": examples}, f, indent=2, default=str)
    print(f"\nWrote {len(examples)} attribution examples -> {out_path}")


if __name__ == "__main__":
    main()
