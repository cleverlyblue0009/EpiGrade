"""Phase 1 reporting: the exclusion/triage log, and a hand-curated agreement check.

IMPORTANT HONESTY NOTE on the "hand curation": this was performed by the project's AI assistant
reading each sampled row's raw GEO fields directly and writing an explicit per-series answer key
(HAND_LABELS below), independently of the generic substring-matching code path in
epigrade.acquire.harmonize - a second, structurally different implementation of the same
underlying judgment, used here as a cross-check that catches generalization bugs (this is how
the wrong_tissue/NSD1-variant/validation-cohort bugs fixed earlier were originally found). It is
NOT independent human clinical curation, and the resulting agreement rate should be read as an
internal-consistency and bug-finding check, not as external validation of labeling accuracy.
The only genuinely external check in this project is the exact match, on GSE74432/GSE97362/
GSE116300, against the published ground truth in the project spec - see tests/test_harmonize.py.
"""

from __future__ import annotations

import pandas as pd

from epigrade import paths
from epigrade.acquire.harmonize import harmonize

RANDOM_SEED = 42
N_SAMPLE = 100

# Explicit per-series hand-derived answer key, keyed by the field(s) that actually carry the
# diagnostic signal for that series (see docs/METHODS.md for how each was determined from the
# raw metadata). A lambda receives the row and returns (role, disorder) or None if genuinely
# unresolvable from public metadata alone (in which case it is excluded from agreement scoring,
# not scored as a "miss").
HAND_LABELS = {
    "GSE74432": lambda r: (
        ("wrong_tissue", "Sotos syndrome" if r["disease_state"] == "Sotos" else None)
        if "fibroblast" in str(r["source_name_ch1"]).lower()
        else ("under_test", "Sotos syndrome") if r["disease_state"] == "NSD1 variant"
        else ("case", "Sotos syndrome") if r["disease_state"] == "Sotos"
        else ("case", "Weaver syndrome") if r["disease_state"] == "Weaver"
        else ("matched_control", None) if r["disease_state"] == "Control"
        else None
    ),
    "GSE97362": lambda r: (
        ("unaffected_relative", None)
        if "relative" in str(r.get("characteristics_raw", "")).lower()
        else ("case", "CHARGE syndrome")
        if r.get("sample_type") == "CHD7 LOF discovery cohort"
        else ("case", "Kabuki syndrome type 1")
        if r.get("sample_type") == "KMT2D LOF discovery cohort"
        else ("matched_control", None)
        if str(r.get("sample_type", "")).startswith("Control for")
        else ("under_test", "CHARGE syndrome") if "CHD7" in str(r.get("disease_state", ""))
        else ("under_test", "Kabuki syndrome type 1")
        if "KMT2D" in str(r.get("disease_state", ""))
        else ("under_test", "Kabuki syndrome type 2")
        if "KDM6A" in str(r.get("disease_state", ""))
        else None
    ),
    "GSE116300": lambda r: (
        ("unaffected_relative", None) if r.get("case_status") == "case parent"
        else ("matched_control", None) if r.get("case_status") == "control"
        else ("under_test", "Kabuki syndrome type 1")
        if "VUS" in str(r.get("variant_classification", ""))
        else ("case", "Kabuki syndrome type 1") if r.get("case_status") == "case"
        else None
    ),
    "GSE104451": lambda r: (
        ("matched_control", None) if r.get("genotype/variation") == "control"
        else ("case", "Silver-Russell syndrome")
    ),
    "GSE108423": lambda r: (
        ("matched_control", None) if r.get("source_name_ch1") == "male control"
        else ("case", "Claes-Jensen syndrome")
        if "with KDM5C mutation" in str(r.get("source_name_ch1", ""))
        else ("unaffected_relative", "Claes-Jensen syndrome")
        if "relative" in str(r.get("source_name_ch1", ""))
        else None
    ),
    "GSE116992": lambda r: ("case", r.get("disease_state")),
    "GSE125367": lambda r: (
        ("matched_control", None) if r.get("title") == "Control sample"
        else ("case", "Nicolaides-Baraitser syndrome") if r.get("title") == "NCBRS case"
        else ("under_test", "Nicolaides-Baraitser syndrome")
        if r.get("title") == "SMARCA2 test variant"
        else None
    ),
    "GSE35069": lambda r: None,  # cell-type reference panel; no per-sample disorder to hand-label
    "GSE42861": lambda r: (
        ("population_control", None) if r.get("disease_state") == "Normal"
        else ("exclude_other", "Rheumatoid arthritis")
    ),
    "GSE52588": lambda r: (
        ("case", "Down syndrome") if r.get("disease_state") == "Down syndrome"
        else ("unaffected_relative", "Down syndrome")
    ),
    "GSE55491": lambda r: (
        ("case", "Silver-Russell syndrome") if r.get("disease_state") == "SRS"
        else ("matched_control", None)
    ),
    "GSE66552": lambda r: (
        ("matched_control", None) if r.get("group") == "TD control"
        else ("case", "Williams syndrome") if r.get("group") == "WS"
        else ("case", "7q11.23 duplication syndrome") if r.get("group") == "Dup7"
        else None
    ),
    "GSE85210": lambda r: ("population_control", None),  # smoking cohort, no disorder in scope
    "GSE87571": lambda r: ("population_control", None),
    "GSE87648": lambda r: (
        ("exclude_other", "Inflammatory bowel disease (Crohn's)")
        if r.get("simplified_diagnosis") == "CD"
        else ("exclude_other", "Inflammatory bowel disease (ulcerative colitis)")
        if r.get("simplified_diagnosis") == "UC"
        else None  # HL/HS codes: genuinely not resolvable from public metadata alone
    ),
    "GSE89353": lambda r: (
        ("unaffected_relative", None)
        if str(r.get("title", "")).endswith(("_mother", "_father"))
        else None  # bare "Proband": no diagnosis published in GEO metadata
    ),
    "GSE95040": lambda r: (
        ("matched_control", None) if r.get("disease_state") == "normal"
        else ("case", "ICF syndrome")
    ),
    "GSE99863": lambda r: ("population_control", None),
}


def hand_label(row: pd.Series) -> tuple[str, str | None] | None:
    fn = HAND_LABELS.get(row["series_id"])
    if fn is None:
        return None
    return fn(row)


def main() -> None:
    samples_path = paths.interim_dir() / "samples.parquet"
    raw = pd.read_parquet(samples_path)
    harmonized = harmonize(raw)

    # --- sample_triage.tsv: every sample NOT pooled as a plain case/control, with the rule ---
    excluded_roles = {"unaffected_relative", "under_test", "cell_line", "wrong_tissue",
                       "exclude_other"}
    triage = harmonized[harmonized["role"].isin(excluded_roles)][
        ["gsm_accession", "series_id", "role", "disorder", "gene", "confidence",
         "resolution_status", "rule_rationale"]
    ].sort_values(["series_id", "role"])
    triage_path = paths.tables_dir() / "sample_triage.tsv"
    triage.to_csv(triage_path, sep="\t", index=False)
    print(f"Wrote {len(triage)} excluded/flagged samples -> {triage_path}")

    # --- sample_counts.tsv: mutually-exclusive categories that sum to the harvested total.
    # Fixes a real bug: the app's Cohort audit page was reporting "Samples harvested 871" AND
    # "Flagged/excluded 871" as the SAME number, because it was reading both from this triage
    # table (which only ever contains excluded samples) instead of the full harvested set.
    # needs_review only ever occurs within role=population_control in this pipeline (verified:
    # every other role is always auto_accepted) - checked before writing this, not assumed. ---
    is_unresolved = harmonized["resolution_status"] == "needs_review"
    retained_roles = {"case", "matched_control", "population_control"}
    retained_mask = harmonized["role"].isin(retained_roles) & ~is_unresolved

    def _role_count(role: str) -> int:
        return int((harmonized.role == role).sum())

    count_rows = [
        {"category": "harvested", "count": len(harmonized), "kind": "total"},
        {"category": "retained_for_analysis", "count": int(retained_mask.sum()),
         "kind": "total"},
        {"category": "excluded_total", "count": int((~retained_mask).sum()), "kind": "total"},
        {"category": "unaffected_relatives", "count": _role_count("unaffected_relative"),
         "kind": "excluded_reason"},
        {"category": "under_test_samples", "count": _role_count("under_test"),
         "kind": "excluded_reason"},
        {"category": "wrong_tissue", "count": _role_count("wrong_tissue"),
         "kind": "excluded_reason"},
        {"category": "non_target_disease_controls", "count": _role_count("exclude_other"),
         "kind": "excluded_reason"},
        {"category": "unresolvable_labels", "count": int(is_unresolved.sum()),
         "kind": "excluded_reason"},
    ]
    n_reasons_sum = sum(r["count"] for r in count_rows if r["kind"] == "excluded_reason")
    assert n_reasons_sum == count_rows[2]["count"], (
        f"excluded reasons ({n_reasons_sum}) must sum to excluded_total "
        f"({count_rows[2]['count']}) - categories are not mutually exclusive"
    )
    assert count_rows[1]["count"] + count_rows[2]["count"] == count_rows[0]["count"], (
        "retained + excluded must sum to harvested"
    )
    counts_path = paths.tables_dir() / "sample_counts.tsv"
    pd.DataFrame(count_rows).to_csv(counts_path, sep="\t", index=False)
    print(f"\nWrote sample count breakdown -> {counts_path}")
    print(pd.DataFrame(count_rows).to_string())

    # --- 100-sample hand-curation agreement check ---
    per_series_samples = [
        g.sample(n=min(len(g), 6), random_state=RANDOM_SEED)
        for _, g in harmonized.groupby("series_id")
    ]
    stratified = pd.concat(per_series_samples, ignore_index=True)
    sample = stratified.sample(n=min(N_SAMPLE, len(stratified)), random_state=RANDOM_SEED)

    rows = []
    for _, row in sample.iterrows():
        hl = hand_label(row)
        if hl is None:
            rows.append({
                "gsm_accession": row["gsm_accession"], "series_id": row["series_id"],
                "hand_role": "UNRESOLVABLE", "hand_disorder": None,
                "auto_role": row["role"], "auto_disorder": row["disorder"],
                "agree": None,
            })
            continue
        hand_role, hand_disorder = hl
        agree = (hand_role == row["role"]) and (
            hand_disorder is None or hand_disorder == row["disorder"]
        )
        rows.append({
            "gsm_accession": row["gsm_accession"], "series_id": row["series_id"],
            "hand_role": hand_role, "hand_disorder": hand_disorder,
            "auto_role": row["role"], "auto_disorder": row["disorder"],
            "agree": agree,
        })

    agreement_df = pd.DataFrame(rows)
    agreement_df["agree"] = agreement_df["agree"].astype("boolean")  # pandas nullable bool
    scoreable = agreement_df[agreement_df["agree"].notna()]
    overall_rate = scoreable["agree"].mean() if len(scoreable) else float("nan")

    per_role = (
        scoreable.groupby("hand_role")["agree"].agg(["mean", "count"])
        .rename(columns={"mean": "agreement_rate", "count": "n"})
    )

    agreement_path = paths.tables_dir() / "harmonisation_agreement.tsv"
    agreement_df.to_csv(agreement_path, sep="\t", index=False)
    print(f"\nWrote {len(agreement_df)} hand-vs-auto comparisons -> {agreement_path}")
    print(f"  scoreable: {len(scoreable)} ({len(agreement_df) - len(scoreable)} marked "
          f"UNRESOLVABLE from public metadata, excluded from scoring)")
    print(f"  overall agreement rate: {overall_rate:.1%}")
    print("\nPer-role agreement:")
    print(per_role.to_string())

    disagreements = scoreable[~scoreable["agree"]]
    if len(disagreements):
        print(f"\n{len(disagreements)} disagreement(s):")
        print(disagreements.to_string())


if __name__ == "__main__":
    main()
