"""Build the complete classifier corpus: one disorder-agnostic pass over all twelve disorders.

For every disorder in epigrade.corpus.registry this derives (or loads) its signature, records
the derivation, runs leave-one-study-out where a disorder spans two studies, scores every
classifier against every disorder's cases and the pooled control corpus, and calibrates each
classifier's scores into evidence bands.

Outputs (all under results/):
  tables/signature_derivation.tsv        every derivation attempted, with thresholds and counts
  tables/corpus_cohorts.tsv              who is a case/control for what, per study
  tables/leave_one_study_out.tsv         both directions for every multi-study disorder
  tables/cross_disorder_matrix.tsv       every classifier x every cohort + pooled controls
  tables/control_specificity_by_series.tsv
  tables/classifier_scores/<slug>.tsv    per-sample scores behind the matrix
  tables/evidence_bands.tsv              calibrated bands per classifier
  tables/disorder_verdicts.tsv           the 12-row verdict summary
  signatures/<slug>.tsv                  the signatures themselves (probe, direction, medians)

This replaces phase4_6_srs_and_matrix.py, phase4_kabuki_charge.py, phase5_calibration.py and
merge_cross_disorder_matrix.py, which each handled one or two disorders by hand.

Thresholds are whatever config/signature_thresholds.yaml says. A disorder that does not build
under them is recorded as a failure with its diagnosis - nothing here retries with a looser
threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epigrade import paths
from epigrade.audit.loso import leave_one_study_out
from epigrade.audit.loso import to_frame as loso_to_frame
from epigrade.corpus import build, loader, registry
from epigrade.corpus.build import BuildResult
from epigrade.corpus.calibration import PRIOR_GRID, calibration_frame, evidence_rows_at
from epigrade.signature.generic import (
    MIN_CASES_TO_BUILD_CLASSIFIER,
    Classifier,
    build_classifier,
    derivation_stats,
    probe_tests,
    score_matrix,
)

MIN_N_TO_INTERPRET = 10
TOO_SMALL = "too small to interpret"
SIBLING_LABEL = registry.SIBLING_LABEL

DERIVATION_COLUMNS = [
    "disorder", "derivation_scope", "status", "multiple_testing_method", "alpha",
    "effect_size_floor", "n_probes_significant", "n_probes_after_effect_filter", "n_cases",
    "n_controls", "signature_source", "train_series", "control_series", "controls_source",
    "note",
]
MATRIX_COLUMNS = [
    "classifier", "scored_disorder", "cell_type", "status", "n", "n_positive",
    "fraction_positive", "interpretation", "signature_source", "controls_source", "reason",
]
WITHIN_SERIES, CROSS_SERIES = "within_series", "cross_series"
# Shown wherever the matrix is displayed.
CONTROLS_NOTE = (
    "Classifiers trained against out-of-series controls detect series membership, not "
    "disease. Coffin-Siris and Nicolaides-Baraitser are both in that category."
)


def controls_source(result: BuildResult) -> str:
    """cross_series if any study the canonical classifier trains on had to take its controls
    from another series; within_series otherwise."""
    borrowed = any(a.controls_borrowed for a in result.resolved.train_arms())
    return CROSS_SERIES if borrowed else WITHIN_SERIES


def _write(df: pd.DataFrame, name: str) -> None:
    path = paths.tables_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    print(f"  wrote {len(df):>4} rows -> results/tables/{name}")


# --------------------------------------------------------------------------- derivations


def derivation_table(results: dict[str, BuildResult], loso_df: pd.DataFrame,
                     extra_rows: list[dict]) -> pd.DataFrame:
    rows = [row for r in results.values() for row in r.derivation_rows]
    rows += extra_rows
    for _, r in loso_df.iterrows():
        attempted = r["n_probes_significant"] != "NA"
        rows.append({
            "disorder": r["disorder"],
            "derivation_scope": f"leave_one_study_out (train {r['train_study']})",
            "status": (
                "built" if r["status"] == "computed"
                else "failed_underpowered" if attempted else "not_attempted_too_few_cases"
            ),
            "multiple_testing_method": r["multiple_testing_method"], "alpha": r["alpha"],
            "effect_size_floor": r["effect_size_floor"],
            "n_probes_significant": r["n_probes_significant"],
            "n_probes_after_effect_filter": r["n_signature_probes"] if attempted else "NA",
            "n_cases": r["n_train_cases"], "n_controls": r["n_train_controls"],
            "signature_source": r["signature_source"],
            "train_series": r["train_study"], "control_series": "",
            "controls_source": "", "note": r["reason"],
        })
    return pd.DataFrame(rows, columns=DERIVATION_COLUMNS)


def pooled_alternative(result: BuildResult) -> list[dict]:
    """Where the canonical classifier deliberately trains on a subset of a disorder's studies,
    also derive the all-studies pooled alternative and record it, with its measured false-
    positive rate on Sotos cases - the evidence the choice rests on, kept reproducible rather
    than asserted in a comment."""
    resolved = result.resolved
    if set(resolved.disorder.canonical_train_series) == set(resolved.disorder.studies):
        return []
    series = {s: a.series_id for a in resolved.arms for s in a.train_case_ids}
    series |= {s: a.control_series for a in resolved.arms for s in a.control_ids}
    beta = build.assemble_beta(series)
    cases = [s for a in resolved.arms for s in a.train_case_ids]
    controls = list(dict.fromkeys(s for a in resolved.arms for s in a.control_ids))
    tests = probe_tests(beta, cases, controls)
    stats = derivation_stats(beta, cases, controls, disorder=resolved.name, tests=tests)
    alt = build_classifier(resolved.name, beta, cases, controls, tests=tests)

    note = "pooled across every study; NOT the canonical classifier"
    if alt is not None:
        sotos = build.resolve(registry.BY_NAME["Sotos syndrome"])
        scores = score_matrix(alt, loader.load_beta("GSE74432"), sotos.case_ids)["score"]
        own = _fraction_positive(result.classifier, "GSE74432", sotos.case_ids)
        note += (
            f"; it scores {(scores > 0).sum()}/{len(scores)} Sotos cases positive, against "
            f"{own} for the canonical {len(result.classifier.signature_probes)}-probe "
            "classifier - a larger signature that is not a more specific one"
        )
    return [{
        **stats, "derivation_scope": "pooled_alternative_not_used",
        "status": "built" if alt is not None else "failed_underpowered",
        "train_series": "+".join(resolved.disorder.studies),
        "control_series": "+".join(dict.fromkeys(a.control_series for a in resolved.arms)),
        "controls_source": "within_series", "note": note,
    }]


def _fraction_positive(clf: Classifier, series_id: str, ids: list[str]) -> str:
    scores = score_matrix(clf, loader.load_beta(series_id), ids)["score"]
    return f"{(scores > 0).sum()}/{len(scores)}"


# --------------------------------------------------------------------------- signatures


def write_signatures(results: dict[str, BuildResult]) -> pd.DataFrame:
    out_dir = paths.repo_root() / "results" / "signatures"
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for result in results.values():
        clf = result.classifier
        if clf is None:
            continue
        table = pd.DataFrame({
            "median_case": clf.median_case, "median_control": clf.median_control,
        })
        if clf.probe_table is not None:
            keep = [c for c in ("direction", "delta_beta", "p_value", "corrected_p")
                    if c in clf.probe_table.columns]
            table = clf.probe_table[keep].join(table)
        table.index.name = "probe_id"
        table = table.sort_index()
        for col in ("delta_beta", "median_case", "median_control"):
            table[col] = table[col].round(4)
        slug = result.resolved.disorder.slug
        table.to_csv(out_dir / f"{slug}.tsv", sep="\t", float_format="%.6g")
        index_rows.append({
            "disorder": clf.disorder, "slug": slug, "n_probes": len(table),
            "n_loss": int((table["direction"] == "loss").sum()),
            "n_gain": int((table["direction"] == "gain").sum()),
            "signature_source": clf.signature_source,
            "n_train_cases": clf.n_case, "n_train_controls": clf.n_control,
            "train_series": "+".join(a.series_id for a in result.resolved.train_arms()),
        })
    index = pd.DataFrame(index_rows)
    index.to_csv(out_dir / "index.tsv", sep="\t", index=False)
    print(f"  wrote {len(index)} signatures -> results/signatures/")
    return index


# --------------------------------------------------------------------------- scoring


def scoring_store(groups: pd.DataFrame, results: dict[str, BuildResult]) -> dict:
    """Betas for every scored sample at the union of all signature probes - a few hundred MB
    instead of the full matrices, and all the cross-disorder scoring needs."""
    union = pd.Index([])
    for result in results.values():
        if result.classifier is not None:
            union = union.union(result.classifier.signature_probes)
    store = {}
    for series_id, block in groups.groupby("series_id"):
        beta = loader.load_beta(series_id)
        store[series_id] = beta.loc[beta.index.intersection(union), block.index.tolist()]
    print(f"  scoring store: {len(union)} probes x {len(groups)} samples")
    return store


def score_everything(clf: Classifier, groups: pd.DataFrame, store: dict) -> pd.DataFrame:
    parts = []
    for series_id, block in groups.groupby("series_id"):
        scores = score_matrix(clf, store[series_id], block.index.tolist())
        parts.append(scores.join(block))
    scores = pd.concat(parts)
    scores.index.name = "gsm_accession"
    scores["training_role"] = "not_in_training"
    scores.loc[scores.index.isin(clf.train_case_ids), "training_role"] = "train_case"
    scores.loc[scores.index.isin(clf.train_control_ids), "training_role"] = "train_control"
    return scores.reset_index()


def matrix_rows(result: BuildResult, scores: pd.DataFrame | None,
                results: dict[str, BuildResult], group_sizes: dict[str, int]) -> list[dict]:
    name = result.resolved.name
    targets = [r.resolved.name for r in results.values()] + [registry.POOLED_CONTROL_LABEL]
    if scores is None:
        return [{
            "classifier": name, "scored_disorder": target,
            "cell_type": "self" if target == name else "not_built",
            "status": "not_computed", "signature_source": "NA", "controls_source": "NA",
            "reason": f"no {name} classifier: {result.failure_reason}",
        } for target in targets]

    clf = result.classifier
    rows = []
    for target in targets:
        block = scores[scores["group"] == target]
        cell_type = "other_disorder"
        reason = ""
        if target == name:
            cell_type = "self"
            trained = block[block.training_role == "train_case"]
            held = block[block.training_role != "train_case"]
            reason = (
                f"sensitivity. {int((trained.score > 0).sum())}/{len(trained)} training cases "
                "positive, each scored against median profiles rebuilt without it "
                "(leave-one-out); probe selection still used every training case, so this is "
                "optimistic"
            )
            if len(held):
                reason += (
                    f". {int((held.score > 0).sum())}/{len(held)} cases not used in training "
                    "positive (held out)"
                )
        elif target == registry.POOLED_CONTROL_LABEL:
            cell_type = "pooled_control"
            n_own = int((block.training_role == "train_control").sum())
            block = block[block.training_role != "train_control"]
            reason = (
                f"false-positive rate over matched controls from {block.series_id.nunique()} "
                f"series; this classifier's own {n_own} training controls excluded"
            )
        base = {
            "classifier": name, "scored_disorder": target, "cell_type": cell_type,
            "signature_source": clf.signature_source,
            "controls_source": controls_source(result),
        }
        if block.empty:
            rows.append({
                **base, "status": "not_computed",
                "reason": f"fewer than 10 of the {len(clf.signature_probes)} signature probes "
                          f"are present for any of the {group_sizes.get(target, 0)} {target} "
                          "samples",
            })
            continue
        n, n_pos = len(block), int((block.score > 0).sum())
        rows.append({
            **base, "status": "computed", "n": n, "n_positive": n_pos,
            "fraction_positive": round(n_pos / n, 3),
            "interpretation": TOO_SMALL if n < MIN_N_TO_INTERPRET else "",
            "reason": reason,
        })
    return rows


def control_rows(name: str, clf: Classifier, scores: pd.DataFrame) -> list[dict]:
    rows = []
    controls = scores[scores["group"].isin([registry.POOLED_CONTROL_LABEL, SIBLING_LABEL])]
    for (series_id, group, role), block in controls.groupby(
        ["series_id", "group", "training_role"]
    ):
        n, n_pos = len(block), int((block.score > 0).sum())
        rows.append({
            "classifier": name, "control_series": series_id, "control_group": group,
            "relation_to_classifier": (
                "training_controls_leave_one_out" if role == "train_control"
                else "held_out_controls"
            ),
            "n": n, "n_positive": n_pos, "fraction_positive": round(n_pos / n, 3),
            "interpretation": TOO_SMALL if n < MIN_N_TO_INTERPRET else "",
            "signature_source": clf.signature_source,
        })
    return rows


def controls_source_summary(matrix_df: pd.DataFrame) -> pd.DataFrame:
    """The batch-detector finding as numbers: how often a classifier scores OTHER disorders'
    cases positive, by where its training controls came from. Off-diagonal cells only, and
    only cells with n >= 10. The mean is taken over cells (each classifier-cohort pair counts
    once); the pooled figure weights by samples."""
    computed = matrix_df[matrix_df.status == "computed"]
    rows = []
    for source in (WITHIN_SERIES, CROSS_SERIES):
        block = computed[computed.controls_source == source]
        off = block[(block.cell_type == "other_disorder") & block.interpretation.isna()
                    | (block.cell_type == "other_disorder") & (block.interpretation == "")]
        controls = block[block.cell_type == "pooled_control"]
        rows.append({
            "controls_source": source,
            "n_classifiers": block.classifier.nunique(),
            "classifiers": ", ".join(dict.fromkeys(block.classifier)),
            "n_off_diagonal_cells": len(off),
            "mean_off_diagonal_fraction_positive": round(off.fraction_positive.mean(), 4),
            "median_off_diagonal_fraction_positive": round(off.fraction_positive.median(), 4),
            "pooled_off_diagonal_fraction_positive": round(
                off.n_positive.sum() / off.n.sum(), 4),
            "n_off_diagonal_samples_scored": int(off.n.sum()),
            "mean_pooled_control_fraction_positive": round(
                controls.fraction_positive.mean(), 4),
            "note": CONTROLS_NOTE if source == CROSS_SERIES else (
                "includes Down syndrome, whose comparators are unaffected siblings from its "
                "own series"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- calibration


def evidence_rows(result: BuildResult, scores: pd.DataFrame) -> list[dict]:
    """Bands for this classifier on its own disorder: its cases against the controls it is
    defined against, each tagged with the study it came from. Study-level resampling applies
    only when both sides span >= 2 studies (epigrade.calibrate decides); a single-study side
    yields a within_study_only band, never a between-study bound."""
    resolved = result.resolved
    by_gsm = scores.set_index("gsm_accession")["score"]
    case_series = {s: resolved.case_series[s] for s in resolved.case_ids}
    df = calibration_frame(by_gsm, case_series, resolved.control_series_of())

    queries = [("typical_case_score", float(df.loc[df.label == 1, "score"].median()))]
    sotos_path = paths.tables_dir() / "sotos_scores.tsv"
    if resolved.name == "Sotos syndrome" and sotos_path.exists():
        sotos = pd.read_csv(sotos_path, sep="\t")
        missense = sotos.loc[sotos.cohort == "missense_variant", "score"]
        if len(missense):
            queries.append(("typical_missense_vous_score", float(missense.median())))
    queries.append(("borderline_score_zero", 0.0))

    rows = []
    for query_name, query_score in queries:
        rows += evidence_rows_at(resolved.name, df, query_name, query_score, PRIOR_GRID)
    return rows


# --------------------------------------------------------------------------- summaries


def cohort_table(results: dict[str, BuildResult]) -> pd.DataFrame:
    rows = []
    for result in results.values():
        train = set(result.resolved.disorder.canonical_train_series)
        for arm in result.resolved.arms:
            rows.append({
                "disorder": result.resolved.name, "series_id": arm.series_id,
                "n_cases": len(arm.case_ids), "n_cases_for_derivation": len(arm.train_case_ids),
                "n_controls": len(arm.control_ids), "control_series": arm.control_series,
                "controls_source": "cross_series" if arm.controls_borrowed else "within_series",
                "in_canonical_training": arm.series_id in train,
                "note": arm.control_note,
            })
    return pd.DataFrame(rows)


def verdict_table(results: dict[str, BuildResult], loso_df: pd.DataFrame,
                  evidence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for result in results.values():
        resolved = result.resolved
        n_cases, studies = resolved.n_cases, resolved.disorder.studies
        downloaded = all(loader.is_downloaded(s) for s in studies)
        if not downloaded:
            verdict = "not-downloaded"
        elif n_cases < MIN_CASES_TO_BUILD_CLASSIFIER:
            verdict = "too-few-cases"
        elif len(studies) >= 2:
            verdict = "graded"
        else:
            verdict = "single-study"

        loso = loso_df[loso_df.disorder == resolved.name]
        band = evidence[
            (evidence.disorder == resolved.name) & (evidence.query_point == "typical_case_score")
            & (evidence.prior == 0.10)
        ]
        clf = result.classifier
        rows.append({
            "disorder": resolved.name, "n_cases": n_cases, "n_studies": len(studies),
            "studies": ", ".join(studies), "verdict": verdict,
            "signature_built": clf is not None,
            "n_signature_probes": len(clf.signature_probes) if clf is not None else 0,
            "signature_source": clf.signature_source if clf is not None else "NA",
            "controls_source": (
                "cross_series" if any(a.controls_borrowed for a in resolved.arms)
                else "within_series"
            ),
            "leave_one_study_out": (
                f"{int((loso.status == 'computed').sum())}/{len(loso)} directions computed"
                if len(loso) else "not possible (single study)"
            ),
            "interpretation_scope": band.iloc[0]["interpretation_scope"] if len(band) else "NA",
            "band_at_typical_case_score": band.iloc[0]["band"] if len(band) else "NA",
            "reason": result.failure_reason,
        })
    order = {"graded": 0, "single-study": 1, "too-few-cases": 2, "not-downloaded": 3}
    df = pd.DataFrame(rows)
    df["_order"] = df["verdict"].map(order)
    return df.sort_values(["_order", "n_cases"], ascending=[True, False]).drop(columns="_order")


# --------------------------------------------------------------------------- main


def main() -> None:
    print("Deriving signatures...")
    results: dict[str, BuildResult] = {}
    for disorder in registry.DISORDERS:
        results[disorder.name] = build.build(disorder)
        clf = results[disorder.name].classifier
        print(f"  {disorder.name}: "
              + (f"{len(clf.signature_probes)} probes ({clf.signature_source})" if clf
                 else f"no classifier - {results[disorder.name].failure_reason[:90]}"))

    print("Leave-one-study-out...")
    loso_rows, extra_derivations = [], []
    for result in results.values():
        if len(result.resolved.arms) >= 2:
            loso_rows += leave_one_study_out(result.resolved.name,
                                             build.loso_arms(result.resolved))
        if result.classifier is not None:
            extra_derivations += pooled_alternative(result)
    loso_df = loso_to_frame(loso_rows)
    loso_df.to_csv(paths.tables_dir() / "leave_one_study_out.tsv", sep="\t", index=False,
                   na_rep="NA")
    print(f"  wrote {len(loso_df):>4} rows -> results/tables/leave_one_study_out.tsv")
    _write(derivation_table(results, loso_df, extra_derivations), "signature_derivation.tsv")
    _write(cohort_table(results), "corpus_cohorts.tsv")
    write_signatures(results)

    print("Scoring every classifier against every cohort...")
    groups = build.corpus_groups([r.resolved for r in results.values()])
    group_sizes = groups["group"].value_counts().to_dict()
    store = scoring_store(groups, results)
    matrix, controls, evidence = [], [], []
    for result in results.values():
        clf = result.classifier
        if clf is None:
            matrix += matrix_rows(result, None, results, group_sizes)
            continue
        scores = score_everything(clf, groups, store)
        out = scores[["gsm_accession", "series_id", "group", "training_role", "score",
                      "n_probes_used", "loo_guarded"]].copy()
        out["signature_source"] = clf.signature_source
        _write(out.sort_values(["group", "series_id", "gsm_accession"]),
               f"classifier_scores/{result.resolved.disorder.slug}.tsv")
        matrix += matrix_rows(result, scores, results, group_sizes)
        controls += control_rows(result.resolved.name, clf, scores)
        evidence += evidence_rows(result, scores)

    matrix_df = pd.DataFrame(matrix, columns=MATRIX_COLUMNS)
    for col in ("n", "n_positive"):
        matrix_df[col] = matrix_df[col].astype("Int64")
    _write(matrix_df, "cross_disorder_matrix.tsv")
    _write(controls_source_summary(matrix_df), "controls_source_summary.tsv")
    _write(pd.DataFrame(controls), "control_specificity_by_series.tsv")
    evidence_df = pd.DataFrame(evidence)
    _write(evidence_df, "evidence_bands.tsv")
    _write(verdict_table(results, loso_df, evidence_df), "disorder_verdicts.tsv")

    computed = matrix_df[matrix_df.status == "computed"]
    off = computed[computed.cell_type == "other_disorder"]
    worst = off.sort_values("fraction_positive", ascending=False).head(5)
    print(f"\n{len(computed)} matrix cells computed, "
          f"{(matrix_df.status == 'not_computed').sum()} not computed (each with a reason).")
    print("Strongest cross-reactivity:")
    print(worst[["classifier", "scored_disorder", "n", "fraction_positive"]].to_string(
        index=False))
    assert not np.any(matrix_df["reason"].isna() & (matrix_df.status == "not_computed"))


if __name__ == "__main__":
    main()
