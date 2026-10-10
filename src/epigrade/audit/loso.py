"""Disorder-agnostic leave-one-study-out: derive a Path B signature from every study but one,
score the held-out study with it, and report sensitivity/specificity there.

This replaces the Silver-Russell-only version that used to live inline in
scripts/phase4_6_srs_and_matrix.py (and wrote results/tables/srs_leave_one_study_out.tsv) - three
disorders in this corpus span two studies, so the routine cannot be disorder-specific.

A direction that cannot be built is a result, not an error: it is returned as a
status="not_computed" row carrying the reason (too few training cases, no training controls, or
an underpowered derivation), never dropped and never forced by loosening a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from epigrade.signature.generic import (
    MIN_CASES_TO_BUILD_CLASSIFIER,
    MIN_SIGNATURE_SIZE,
    build_classifier,
    diagnose_underpowered,
    get_thresholds,
    score_samples,
)

LOSO_COLUMNS = [
    "disorder", "train_study", "test_study", "status",
    "n_train_cases", "n_train_controls", "n_test_cases", "n_test_controls",
    "n_signature_probes", "test_case_sensitivity", "test_control_specificity",
    "sensitivity", "specificity", "multiple_testing_method", "alpha", "effect_size_floor",
    "signature_source", "reason",
]


@dataclass
class StudyArm:
    """One study's contribution to a disorder. `train_case_ids` may be a stricter subset of
    `test_case_ids` (e.g. molecularly confirmed cases only) - a signature is derived from the
    former, but a held-out study is always scored on its full case set."""

    study_id: str
    beta: pd.DataFrame
    train_case_ids: list[str]
    control_ids: list[str]
    test_case_ids: list[str] = field(default_factory=list)
    # Where the controls came from when a study deposited none of its own, e.g. "GSE125367" -
    # carried into the reason column so a borrowed-control direction is never read as an
    # ordinary within-study one.
    controls_borrowed_from: str | None = None

    def __post_init__(self) -> None:
        if not self.test_case_ids:
            self.test_case_ids = list(self.train_case_ids)


def _fraction(n_hit: int, n_total: int) -> tuple[str, float]:
    if n_total == 0:
        return "NA", float("nan")
    return f"{n_hit}/{n_total}", n_hit / n_total


def leave_one_study_out(disorder: str, arms: list[StudyArm]) -> list[dict]:
    """One row per held-out study. With two studies this is both directions (A->B and B->A)."""
    if len(arms) < 2:
        raise ValueError(
            f"{disorder}: leave-one-study-out needs >= 2 studies, got {len(arms)}"
        )
    thresholds = get_thresholds(disorder)
    rows = []
    for held_out in arms:
        train_arms = [a for a in arms if a.study_id != held_out.study_id]
        probes = held_out.beta.index
        for a in train_arms:
            probes = probes.intersection(a.beta.index)

        train_cases = [s for a in train_arms for s in a.train_case_ids]
        train_controls = [s for a in train_arms for s in a.control_ids]
        train_beta = pd.concat(
            [a.beta.loc[probes, a.train_case_ids + a.control_ids] for a in train_arms], axis=1,
        )
        train_beta = train_beta.loc[:, ~train_beta.columns.duplicated()]

        row = {
            "disorder": disorder,
            "train_study": "+".join(a.study_id for a in train_arms),
            "test_study": held_out.study_id,
            "n_train_cases": len(train_cases),
            "n_train_controls": len(train_controls),
            "n_test_cases": len(held_out.test_case_ids),
            "n_test_controls": len(set(held_out.control_ids) - set(train_controls)),
            "multiple_testing_method": thresholds["multiple_testing_method"],
            "alpha": thresholds["alpha"],
            "effect_size_floor": thresholds["effect_size_floor"],
        }
        borrowed = [
            f"{a.study_id} has no controls of its own; {a.controls_borrowed_from} controls used"
            for a in arms if a.controls_borrowed_from
        ]

        def _not_computed(reason: str, row=row, borrowed=borrowed) -> dict:
            return {
                **row, "status": "not_computed", "n_signature_probes": 0,
                "test_case_sensitivity": "NA", "test_control_specificity": "NA",
                "sensitivity": float("nan"), "specificity": float("nan"),
                "signature_source": "NA", "reason": "; ".join([reason, *borrowed]),
            }

        if len(train_cases) < MIN_CASES_TO_BUILD_CLASSIFIER:
            rows.append(_not_computed(
                f"only {len(train_cases)} training case(s) in {row['train_study']} "
                f"(< {MIN_CASES_TO_BUILD_CLASSIFIER} minimum to attempt a classifier)"
            ))
            continue
        if not train_controls:
            rows.append(_not_computed(f"no training controls in {row['train_study']}"))
            continue

        clf = build_classifier(disorder, train_beta, train_cases, train_controls)
        if clf is None:
            diagnosis = diagnose_underpowered(
                train_beta, train_cases, train_controls, disorder=disorder,
            )
            rows.append(_not_computed(
                f"{diagnosis} (minimum usable signature size is {MIN_SIGNATURE_SIZE} probes)"
            ))
            continue

        test_beta = held_out.beta.loc[probes]
        case_scores = score_samples(clf, test_beta, held_out.test_case_ids)
        # Borrowed controls can sit on both sides of the split (in either direction). Any
        # control the signature was derived against is not held out, so it never counts
        # toward specificity.
        seen_in_training = set(train_controls)
        held_out_controls = [s for s in held_out.control_ids if s not in seen_in_training]
        control_scores = score_samples(clf, test_beta, held_out_controls)
        sens_text, sens = _fraction(int((case_scores > 0).sum()), len(case_scores))
        spec_text, spec = _fraction(int((control_scores < 0).sum()), len(control_scores))
        reasons = list(borrowed)
        if held_out.control_ids and not held_out_controls:
            reasons.append(
                "specificity not computed: the held-out study's only controls are the "
                "training controls"
            )
        elif not len(control_scores):
            reasons.append("specificity not computed: held-out study has no controls")
        rows.append({
            **row, "status": "computed",
            "n_signature_probes": len(clf.signature_probes),
            "test_case_sensitivity": sens_text, "test_control_specificity": spec_text,
            "sensitivity": sens, "specificity": spec,
            "signature_source": clf.signature_source, "reason": "; ".join(reasons),
        })
    return rows


def to_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=LOSO_COLUMNS)
