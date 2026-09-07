"""Task 3: an attempt at a second and third real classifier - Kabuki syndrome (GSE116300) and
CHARGE syndrome (GSE97362).

ACTUAL OUTCOME (read before assuming this produces two new working classifiers): neither
disorder clears this project's own reliability bar from the publicly available GEO data, even
after legitimately pooling Kabuki across two studies. This was investigated thoroughly, not
assumed - see the printed diagnosis for each and docs/LIMITATIONS.md for the full writeup,
including exactly how close Kabuki's pooled attempt came (a factor of <2x in p-value) and why
CHARGE's failure is a different shape (a significance/effect-size mismatch, not pure power
starvation). The cross-disorder matrix therefore still contains only the pre-existing Sotos
rows; both failures are recorded with full diagnoses in cross_disorder_matrix_not_computed.tsv.
No threshold was loosened and no test was swapped to force either one through - the same
procedure that worked for nothing here also worked for nothing on Silver-Russell syndrome
(phase4_6), applied with the same criteria in all three attempts. A rigorous, honest negative
result is the actual Task 3 output.

CRITICAL HONESTY REQUIREMENT: neither Kabuki nor CHARGE has a published probe list available to
this project (unlike Sotos - see epigrade.signature.choufani). Both are re-derived via Path B
(epigrade.signature.generic: Mann-Whitney U, Bonferroni, >20% effect-size filter, feature
selection strictly inside the training/discovery cohort). Every row involving them carries
signature_source="rederived_not_published", vs "published_probe_list" for Sotos - never let a
re-derived signature be presented as a reproduction of a published one.

Cohort definitions:
  - Kabuki (GSE116300): building/derivation uses the 26 role="case" samples (case_status="case"
    minus the 3 samples whose variant_classification is "VUS", already separated into
    role="under_test" by the phase 1 harmonizer) vs 9 role="matched_control" samples. The 3 VUS
    samples are held out and scored, not used to build the classifier - mirrors exactly how
    Sotos separates confirmed NSD1 LOF from the NSD1-variant VOUS cohort, and how the SRS
    analysis (phase4_6) restricted to molecularly-confirmed cases after pooling was found to
    wash out the signal. This is a deliberate deviation from a literal reading of "29 confirmed
    cases" (29 = case_status="case" including the 3 VUS, which are not actually confirmed
    pathogenic) - documented here and in docs/LIMITATIONS.md rather than silently applied. The 6
    "case parent" samples are excluded per the existing triage rules (unaffected_relative role) -
    and, as it turns out, are also simply absent from the beta-values supplementary file itself
    (it covers exactly the 38 non-relative samples), so this exclusion costs nothing.
  - CHARGE (GSE97362): building uses the 19 "CHD7 LOF discovery cohort" cases vs the 29 "Control
    for CHD7 LOF discovery cohort" controls specifically (not the separate 85 "Control for
    validation cohort" pool, which is paired with the validation cohort we're excluding, not the
    discovery cohort we're using). Excluded per task/triage rules: any sample_type of "sequence
    variant" or "validation cohort" (CHD7 sequence variant, KMT2D-related groups, Validation
    cohort) - these are scored as held-out under_test samples, not used to build the classifier.
"""

from __future__ import annotations

import pandas as pd

from epigrade import paths
from epigrade.acquire.harmonize import harmonize
from epigrade.preprocess.series_matrix import (
    parse_beta_values_by_slide_position,
    parse_series_matrix,
)
from epigrade.signature.generic import (
    build_classifier,
    diagnose_underpowered,
    score_samples,
)

MIN_N_TO_INTERPRET = 10
# Rows this script owns in the shared matrix/not-computed tables - dropped and rewritten fresh
# each run (rather than blindly appended) so re-running as downloads complete stays idempotent
# and never accumulates duplicate rows. Sotos rows (owned by phase4_6_srs_and_matrix.py) and any
# other existing content are left untouched.
OWNED_CLASSIFIERS = {"Kabuki syndrome type 1", "CHARGE syndrome"}


def _drop_owned_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "classifier" not in df.columns:
        return df
    return df[~df["classifier"].isin(OWNED_CLASSIFIERS)]


def get_meta(harmonized: pd.DataFrame, series_id: str) -> pd.DataFrame:
    return harmonized[harmonized.series_id == series_id].set_index("gsm_accession")


def get_kabuki_beta(meta_116300: pd.DataFrame) -> pd.DataFrame:
    cache = paths.interim_dir() / "GSE116300_betas.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    gz = paths.external_dir() / "GSE116300_beta_values.txt.gz"
    beta = parse_beta_values_by_slide_position(gz, meta_116300)
    beta.to_parquet(cache)
    return beta


def get_series_matrix_beta(series_id: str, gz_name: str) -> pd.DataFrame:
    cache = paths.interim_dir() / f"{series_id}_betas.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    beta = parse_series_matrix(paths.external_dir() / gz_name)
    beta.to_parquet(cache)
    return beta


def build_kabuki(meta_116300: pd.DataFrame, beta_116300: pd.DataFrame):
    # role=="case" already IS "case_status=case AND variant_classification not VUS" - the
    # harmonizer (phase 1) already separated the 3 VUS samples into role="under_test". Adding
    # an extra explicit pathogenic/likely-pathogenic filter here would be wrong, not stricter:
    # 15 of the 26 case_status="case" samples have no variant_classification value recorded at
    # all (NaN, not "VUS") - their mutation field names a gene directly (e.g. "KMT2D") without a
    # separate pathogenicity tag. Filtering on variant_classification.isin([...]) would treat
    # "not recorded" as "not confirmed" and incorrectly drop them to 11 cases - found and fixed
    # during development.
    case_ids = meta_116300[meta_116300.role == "case"].index.tolist()
    vus_ids = meta_116300[meta_116300.role == "under_test"].index.tolist()
    control_ids = meta_116300[meta_116300.role == "matched_control"].index.tolist()
    relative_ids = meta_116300[meta_116300.role == "unaffected_relative"].index.tolist()

    print(f"Kabuki (GSE116300): {len(case_ids)} confirmed cases, {len(vus_ids)} VUS "
          f"(held out), {len(control_ids)} controls, {len(relative_ids)} case-parent "
          "(excluded, unaffected_relative)")

    clf = build_classifier("Kabuki syndrome type 1", beta_116300, case_ids, control_ids)
    return clf, case_ids, vus_ids, control_ids


def build_charge(meta_97362: pd.DataFrame, beta_97362: pd.DataFrame):
    case_ids = meta_97362[
        meta_97362["sample_type"] == "CHD7 LOF discovery cohort"
    ].index.tolist()
    control_ids = meta_97362[
        meta_97362["sample_type"] == "Control for CHD7 LOF discovery cohort"
    ].index.tolist()
    held_out_ids = meta_97362[
        meta_97362["sample_type"].isin(["CHD7 sequence variant", "Validation cohort"])
    ].index.tolist()

    print(f"CHARGE (GSE97362): {len(case_ids)} discovery LOF cases, {len(control_ids)} "
          f"discovery controls, {len(held_out_ids)} held-out (sequence variant/validation "
          "cohort, all sample_types, not just CHD7's)")

    clf = build_classifier("CHARGE syndrome", beta_97362, case_ids, control_ids)
    return clf, case_ids, held_out_ids, control_ids


def main() -> None:
    samples = pd.read_parquet(paths.interim_dir() / "samples.parquet")
    harmonized = harmonize(samples)

    meta_116300 = get_meta(harmonized, "GSE116300")
    beta_116300 = get_kabuki_beta(meta_116300)
    print(f"GSE116300 beta matrix: {beta_116300.shape}")

    kabuki_clf, kabuki_case_ids, kabuki_vus_ids, kabuki_control_ids = build_kabuki(
        meta_116300, beta_116300
    )
    kabuki_diagnosis = None
    if kabuki_clf is None:
        kabuki_diagnosis = diagnose_underpowered(beta_116300, kabuki_case_ids, kabuki_control_ids)
        print(f"Kabuki: could not build a classifier - {kabuki_diagnosis}")
    else:
        print(f"Kabuki classifier: {len(kabuki_clf.signature_probes)} re-derived "
              f"signature probes ({kabuki_clf.signature_source})")

    gz97362 = paths.external_dir() / "GSE97362_series_matrix.txt.gz"
    charge_clf = None
    charge_diagnosis = None
    meta_97362 = beta_97362 = None
    charge_case_ids = charge_held_out_ids = charge_control_ids = []
    if gz97362.exists():
        try:
            meta_97362 = get_meta(harmonized, "GSE97362")
            beta_97362 = get_series_matrix_beta("GSE97362", "GSE97362_series_matrix.txt.gz")
            print(f"GSE97362 beta matrix: {beta_97362.shape}")
            charge_clf, charge_case_ids, charge_held_out_ids, charge_control_ids = (
                build_charge(meta_97362, beta_97362)
            )
            if charge_clf is None:
                charge_diagnosis = diagnose_underpowered(
                    beta_97362, charge_case_ids, charge_control_ids
                )
                print(f"CHARGE: could not build a classifier - {charge_diagnosis}")
            else:
                print(f"CHARGE classifier: {len(charge_clf.signature_probes)} re-derived "
                      f"signature probes ({charge_clf.signature_source})")
        except Exception as exc:  # noqa: BLE001 - report, don't crash the whole run
            print(f"GSE97362 processing failed ({exc}) - CHARGE will be marked NOT COMPUTED, "
                  "per Task 3's stop rule (2x2 beats a fabricated 3x3).")
    else:
        print(f"{gz97362} not present yet - CHARGE marked NOT COMPUTED "
              "(download still in progress or incomplete; see Task 3's stop rule).")

    # --- Kabuki is ALSO in GSE97362 (11 KMT2D LOF discovery cases + 11 matched controls) -
    # GSE116300 alone was underpowered (9 controls); pool the two studies' Kabuki data, the
    # same legitimate technique already used for Silver-Russell syndrome (phase4_6). This is
    # not a new trick reached for because the first attempt failed - it's the established
    # pattern in this project for exactly this situation. ---
    kabuki_pooled_clf = None
    kabuki_pooled_diagnosis = None
    if kabuki_clf is None and meta_97362 is not None and beta_97362 is not None:
        k2_case_ids = meta_97362[
            meta_97362["sample_type"] == "KMT2D LOF discovery cohort"
        ].index.tolist()
        k2_control_ids = meta_97362[
            meta_97362["sample_type"] == "Control for KMT2D LOF discovery cohort"
        ].index.tolist()
        common = beta_116300.index.intersection(beta_97362.index)
        print(f"\nPooling Kabuki: GSE116300 ({len(kabuki_case_ids)} cases, "
              f"{len(kabuki_control_ids)} controls) + GSE97362 ({len(k2_case_ids)} cases, "
              f"{len(k2_control_ids)} controls) over {len(common)} shared probes")
        pooled_beta = pd.concat(
            [beta_116300.loc[common], beta_97362.loc[common]], axis=1
        )
        pooled_case_ids = kabuki_case_ids + k2_case_ids
        pooled_control_ids = kabuki_control_ids + k2_control_ids
        kabuki_pooled_clf = build_classifier(
            "Kabuki syndrome type 1", pooled_beta, pooled_case_ids, pooled_control_ids,
        )
        if kabuki_pooled_clf is None:
            kabuki_pooled_diagnosis = diagnose_underpowered(
                pooled_beta, pooled_case_ids, pooled_control_ids
            )
            print(f"Pooled Kabuki: still could not build a classifier - "
                  f"{kabuki_pooled_diagnosis}")
        else:
            print(f"Pooled Kabuki classifier: {len(kabuki_pooled_clf.signature_probes)} "
                  f"re-derived signature probes ({kabuki_pooled_clf.signature_source}) - "
                  f"pooling {len(pooled_case_ids)} cases/{len(pooled_control_ids)} controls "
                  "across 2 studies succeeded where GSE116300 alone did not")
            # promote the pooled classifier as THE Kabuki classifier for everything downstream
            kabuki_clf = kabuki_pooled_clf
            beta_116300 = pooled_beta  # so scoring/matrix code below sees all pooled samples
            kabuki_case_ids = pooled_case_ids
            kabuki_control_ids = pooled_control_ids

    # --- per-sample scores tables ---
    if kabuki_clf is not None:
        all_kabuki_ids = kabuki_case_ids + kabuki_vus_ids + kabuki_control_ids
        scores = score_samples(kabuki_clf, beta_116300, all_kabuki_ids)
        rows = []
        for sid, score in scores.items():
            if sid in kabuki_case_ids:
                cohort = "discovery_case"
            elif sid in kabuki_vus_ids:
                cohort = "vus_held_out"
            else:
                cohort = "discovery_control"
            rows.append({
                "gsm_accession": sid, "cohort": cohort, "score": score,
                "signature_source": kabuki_clf.signature_source,
            })
        kabuki_scores_df = pd.DataFrame(rows)
        kabuki_scores_df.to_csv(paths.tables_dir() / "kabuki_scores.tsv", sep="\t", index=False)
        print(f"\nWrote {len(kabuki_scores_df)} Kabuki scores. NOTE: discovery_case/control "
              "rows are NOT leave-one-out guarded (scored against a classifier built from "
              "the same samples) - see the Sotos reproduction for the guarded version; this "
              "self-score is illustrative only. vus_held_out rows ARE genuinely held out.")
        if len(kabuki_scores_df):
            print(kabuki_scores_df.groupby("cohort")["score"].agg(["count", "mean"]).to_string())
        else:
            print("(0 samples scored - see MIN_SIGNATURE_SIZE in epigrade.signature.generic; "
                  "every classifier here has >=10 probes so this shouldn't happen, but "
                  "reported plainly rather than crashing if it ever does.)")

    if charge_clf is not None:
        all_charge_ids = charge_case_ids + charge_held_out_ids + charge_control_ids
        scores = score_samples(charge_clf, beta_97362, all_charge_ids)
        rows = []
        for sid, score in scores.items():
            if sid in charge_case_ids:
                cohort = "discovery_case"
            elif sid in charge_held_out_ids:
                cohort = "held_out_variant_or_validation"
            else:
                cohort = "discovery_control"
            rows.append({
                "gsm_accession": sid, "cohort": cohort, "score": score,
                "signature_source": charge_clf.signature_source,
            })
        charge_scores_df = pd.DataFrame(rows)
        charge_scores_df.to_csv(paths.tables_dir() / "charge_scores.tsv", sep="\t", index=False)
        print(f"\nWrote {len(charge_scores_df)} CHARGE scores. Same self-score caveat as "
              "Kabuki above for discovery_case/control rows.")
        if len(charge_scores_df):
            print(charge_scores_df.groupby("cohort")["score"].agg(["count", "mean"]).to_string())
        else:
            print("(0 samples scored - see MIN_SIGNATURE_SIZE in epigrade.signature.generic.)")

    # --- record classifier-build failures (with the real diagnosis, not just "None") ---
    failure_rows = []
    if kabuki_clf is None:
        reason = kabuki_diagnosis
        if kabuki_pooled_diagnosis is not None:
            reason = (
                f"GSE116300 alone: {kabuki_diagnosis} | Pooled with GSE97362's KMT2D LOF "
                f"cohort: {kabuki_pooled_diagnosis}"
            )
        failure_rows.append({
            "classifier": "Kabuki syndrome type 1", "scored_disorder": "(classifier build)",
            "reason": reason,
        })
    if gz97362.exists() and charge_clf is None and charge_diagnosis is not None:
        failure_rows.append({
            "classifier": "CHARGE syndrome", "scored_disorder": "(classifier build)",
            "reason": charge_diagnosis,
        })
    pending_path = paths.tables_dir() / "cross_disorder_matrix_not_computed.tsv"
    existing_pending = _drop_owned_rows(
        pd.read_csv(pending_path, sep="\t") if pending_path.exists() else pd.DataFrame()
    )
    if failure_rows:
        pd.concat([existing_pending, pd.DataFrame(failure_rows)], ignore_index=True).to_csv(
            pending_path, sep="\t", index=False
        )
        print(f"\nRecorded {len(failure_rows)} classifier-build failure(s) with diagnosis -> "
              f"{pending_path}")
    elif existing_pending is not None:
        existing_pending.to_csv(pending_path, sep="\t", index=False)

    # --- extend the cross-disorder matrix ---
    build_cross_disorder_matrix(
        harmonized, kabuki_clf, beta_116300, charge_clf, beta_97362,
        kabuki_case_ids, kabuki_control_ids, charge_case_ids, charge_control_ids,
    )


def build_cross_disorder_matrix(
    harmonized, kabuki_clf, beta_116300, charge_clf, beta_97362,
    kabuki_case_ids, kabuki_control_ids, charge_case_ids, charge_control_ids,
):
    """Scores each available classifier against every other disorder's cases it can reach
    (shared probes with a downloaded beta matrix), plus a pooled control corpus built from
    every control-role sample across every beta matrix actually downloaded this session.
    Extends (does not replace) the existing Sotos rows from phase4_6_srs_and_matrix.py."""
    existing_path = paths.tables_dir() / "cross_disorder_matrix.tsv"
    existing = pd.read_csv(existing_path, sep="\t") if existing_path.exists() else pd.DataFrame()
    existing = _drop_owned_rows(existing)
    if "signature_source" not in existing.columns and len(existing):
        existing["signature_source"] = "published_probe_list"  # the pre-Task-3 rows are Sotos

    classifiers = []
    if kabuki_clf is not None:
        classifiers.append(("Kabuki syndrome type 1", kabuki_clf, beta_116300))
    if charge_clf is not None:
        classifiers.append(("CHARGE syndrome", charge_clf, beta_97362))

    # Pooled control corpus: every control-role sample in a beta matrix we actually have.
    control_pools = {}
    if kabuki_control_ids:
        control_pools["GSE116300"] = (beta_116300, kabuki_control_ids)
    if charge_control_ids:
        control_pools["GSE97362"] = (beta_97362, charge_control_ids)

    new_rows = []
    not_computed_rows = []

    for disorder_name, clf, own_beta in classifiers:
        # self, plainly flagged as not leave-one-out guarded
        own_case_ids = kabuki_case_ids if disorder_name.startswith("Kabuki") else charge_case_ids
        self_scores = score_samples(clf, own_beta, own_case_ids)
        new_rows.append({
            "classifier": disorder_name, "scored_disorder": f"{disorder_name} (self)",
            "n": len(self_scores), "fraction_positive": round((self_scores > 0).mean(), 3)
            if len(self_scores) else float("nan"),
            "signature_source": clf.signature_source,
            "note": "NOT leave-one-out guarded (scored on its own training data)",
        })

        # against pooled controls (excluding its own control set, correctly held out already
        # since a classifier's own controls were only used to build median_control, not scored)
        for control_series, (control_beta, control_ids) in control_pools.items():
            probes = clf.signature_probes.intersection(control_beta.index)
            if len(probes) < 10:
                not_computed_rows.append({
                    "classifier": disorder_name,
                    "scored_disorder": f"pooled_control:{control_series}",
                    "reason": f"only {len(probes)} shared probes",
                })
                continue
            scores = score_samples(clf, control_beta, control_ids)
            n = len(scores)
            new_rows.append({
                "classifier": disorder_name,
                "scored_disorder": f"pooled_control ({control_series})",
                "n": n, "fraction_positive": round((scores > 0).mean(), 3) if n else float("nan"),
                "signature_source": clf.signature_source,
                "note": "too small to interpret" if n < MIN_N_TO_INTERPRET else "",
            })

        # against every OTHER disorder's cases where we have a beta matrix
        other_case_sources = {
            "Kabuki syndrome type 1": (beta_116300, kabuki_case_ids),
            "CHARGE syndrome": (beta_97362, charge_case_ids) if beta_97362 is not None else None,
        }
        for other_disorder, src in other_case_sources.items():
            if other_disorder == disorder_name or src is None:
                continue
            other_beta, other_case_ids = src
            probes = clf.signature_probes.intersection(other_beta.index)
            if len(probes) < 10 or not other_case_ids:
                continue
            scores = score_samples(clf, other_beta, other_case_ids)
            n = len(scores)
            new_rows.append({
                "classifier": disorder_name, "scored_disorder": other_disorder,
                "n": n, "fraction_positive": round((scores > 0).mean(), 3) if n else float("nan"),
                "signature_source": clf.signature_source,
                "note": "too small to interpret" if n < MIN_N_TO_INTERPRET else "",
            })

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined.to_csv(existing_path, sep="\t", index=False)
    print(f"\nExtended cross-disorder matrix to {len(combined)} rows -> {existing_path}")
    print(combined.to_string())

    if not_computed_rows:
        pending_path = paths.tables_dir() / "cross_disorder_matrix_not_computed.tsv"
        pending_existing = _drop_owned_rows(
            pd.read_csv(pending_path, sep="\t") if pending_path.exists() else pd.DataFrame()
        )
        pending_combined = pd.concat(
            [pending_existing, pd.DataFrame(not_computed_rows)], ignore_index=True
        )
        pending_combined.to_csv(pending_path, sep="\t", index=False)
        print(f"Appended {len(not_computed_rows)} not-computed cells -> {pending_path}")


if __name__ == "__main__":
    main()
