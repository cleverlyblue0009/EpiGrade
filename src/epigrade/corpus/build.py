"""Resolve the registry against real data and build every classifier the corpus supports.

Nothing here writes a file - scripts/build_corpus.py owns the outputs. Functions return plain
frames/rows so they can be tested on synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from epigrade.audit.loso import StudyArm
from epigrade.corpus import loader, registry
from epigrade.corpus.registry import Disorder
from epigrade.signature import choufani
from epigrade.signature.generic import (
    MIN_CASES_TO_BUILD_CLASSIFIER,
    MIN_SIGNATURE_SIZE,
    REDERIVED_NOT_PUBLISHED,
    Classifier,
    build_classifier,
    classifier_from_probes,
    derivation_stats,
    diagnose_underpowered,
    get_thresholds,
    probe_tests,
)


@dataclass
class ResolvedArm:
    series_id: str
    case_ids: list[str]
    train_case_ids: list[str]
    control_ids: list[str]
    control_series: str
    control_note: str

    @property
    def controls_borrowed(self) -> bool:
        return self.control_series != self.series_id


@dataclass
class ResolvedDisorder:
    disorder: Disorder
    arms: list[ResolvedArm]

    @property
    def name(self) -> str:
        return self.disorder.name

    @property
    def case_ids(self) -> list[str]:
        return [s for a in self.arms for s in a.case_ids]

    @property
    def n_cases(self) -> int:
        return len(self.case_ids)

    @property
    def case_series(self) -> dict[str, str]:
        return {s: a.series_id for a in self.arms for s in a.case_ids}

    def train_arms(self) -> list[ResolvedArm]:
        keep = self.disorder.canonical_train_series
        return [a for a in self.arms if a.series_id in keep]

    def train_ids(self) -> tuple[list[str], list[str]]:
        cases = [s for a in self.train_arms() for s in a.train_case_ids]
        controls = list(dict.fromkeys(s for a in self.train_arms() for s in a.control_ids))
        return cases, controls

    def control_series_of(self) -> dict[str, str]:
        return {s: a.control_series for a in self.arms for s in a.control_ids}


def resolve(disorder: Disorder) -> ResolvedDisorder:
    arms = []
    for arm in disorder.arms:
        meta = loader.series_meta(arm.series_id)
        control_meta = loader.series_meta(arm.control_series)
        cases = arm.cases(meta)
        train = arm.train_cases(meta) if arm.train_cases else list(cases)
        arms.append(ResolvedArm(
            series_id=arm.series_id, case_ids=cases, train_case_ids=train,
            control_ids=arm.controls(control_meta), control_series=arm.control_series,
            control_note=arm.control_note,
        ))
    return ResolvedDisorder(disorder, arms)


def assemble_beta(sample_series: dict[str, str]) -> pd.DataFrame:
    """Probe x sample betas for samples drawn from one or more series, restricted to the
    probes every involved series has (e.g. the 450K/EPIC overlap for a mixed-platform set)."""
    by_series: dict[str, list[str]] = {}
    for sample_id, series_id in sample_series.items():
        by_series.setdefault(series_id, []).append(sample_id)
    parts = []
    for series_id, ids in by_series.items():
        beta = loader.load_beta(series_id)
        missing = [s for s in ids if s not in beta.columns]
        if missing:
            raise KeyError(
                f"{series_id}: {len(missing)} sample(s) absent from the beta matrix "
                f"(first: {missing[:3]})"
            )
        parts.append(beta[ids])
    if len(parts) == 1:
        return parts[0]
    return pd.concat(parts, axis=1, join="inner")


def training_frame(resolved: ResolvedDisorder) -> tuple[pd.DataFrame, list[str], list[str]]:
    cases, controls = resolved.train_ids()
    series = {s: a.series_id for a in resolved.train_arms() for s in a.train_case_ids}
    series |= {s: a.control_series for a in resolved.train_arms() for s in a.control_ids}
    return assemble_beta(series), cases, controls


@dataclass
class BuildResult:
    resolved: ResolvedDisorder
    classifier: Classifier | None
    derivation_rows: list[dict] = field(default_factory=list)
    failure_reason: str = ""


def _derivation_row(resolved: ResolvedDisorder, stats: dict, scope: str, note: str) -> dict:
    return {
        **stats,
        "derivation_scope": scope,
        "train_series": "+".join(a.series_id for a in resolved.train_arms()),
        "control_series": "+".join(dict.fromkeys(
            a.control_series for a in resolved.train_arms()
        )),
        "controls_source": (
            "cross_series" if any(a.controls_borrowed for a in resolved.train_arms())
            else "within_series"
        ),
        "status": "built",
        "note": note,
    }


def build(disorder: Disorder) -> BuildResult:
    """Build the canonical classifier for one disorder, or explain why there is none."""
    resolved = resolve(disorder)
    control_notes = "; ".join(a.control_note for a in resolved.train_arms() if a.control_note)
    note = "; ".join(n for n in (disorder.note, control_notes) if n)

    if disorder.signature == registry.NO_SIGNATURE or (
        resolved.n_cases < MIN_CASES_TO_BUILD_CLASSIFIER
    ):
        reason = (
            f"only {resolved.n_cases} confirmed cases (< {MIN_CASES_TO_BUILD_CLASSIFIER} "
            "minimum): no signature can be derived; this cohort is scored by the other "
            "classifiers only"
        )
        thresholds = get_thresholds(disorder.name)
        row = _derivation_row(resolved, {
            "disorder": disorder.name, **thresholds, "n_probes_significant": "NA",
            "n_probes_after_effect_filter": "NA", "n_cases": resolved.n_cases,
            "n_controls": len(resolved.train_ids()[1]), "signature_source": "NA",
        }, "canonical", reason)
        row["status"] = "not_attempted_too_few_cases"
        return BuildResult(resolved, None, [row], reason)

    beta, case_ids, control_ids = training_frame(resolved)

    if disorder.signature == registry.PUBLISHED:
        published = choufani.load_signature()
        thresholds = get_thresholds(disorder.name)
        clf = classifier_from_probes(
            disorder.name, published.index, beta, case_ids, control_ids,
            signature_source=choufani.SIGNATURE_SOURCE,
            probe_table=published[["delta_beta"]].assign(
                direction=lambda t: t["delta_beta"].lt(0).map({True: "loss", False: "gain"}),
            ),
        )
        row = _derivation_row(resolved, {
            "disorder": disorder.name, **thresholds,
            # The published list is used as-is; these are the paper's own counts, not a
            # re-derivation. n_probes_after_effect_filter is the size of the published list.
            "n_probes_significant": "NA",
            "n_probes_after_effect_filter": len(published),
            "n_cases": len(case_ids), "n_controls": len(control_ids),
            "signature_source": choufani.SIGNATURE_SOURCE,
        }, "canonical", f"{note}; {len(clf.signature_probes)} of {len(published)} published "
                        "probes present in the beta matrix; thresholds are the paper's own")
        return BuildResult(resolved, clf, [row])

    tests = probe_tests(beta, case_ids, control_ids)
    stats = derivation_stats(beta, case_ids, control_ids, disorder=disorder.name, tests=tests)
    clf = build_classifier(disorder.name, beta, case_ids, control_ids, tests=tests)
    row = _derivation_row(resolved, stats, "canonical", note)
    if clf is None:
        reason = diagnose_underpowered(
            beta, case_ids, control_ids, disorder=disorder.name, tests=tests,
        )
        row["status"] = "failed_underpowered"
        row["note"] = "; ".join(n for n in (reason, note) if n)
        row["signature_source"] = "NA"
        return BuildResult(resolved, None, [row], reason)
    assert clf.signature_source == REDERIVED_NOT_PUBLISHED
    assert len(clf.signature_probes) >= MIN_SIGNATURE_SIZE
    return BuildResult(resolved, clf, [row])


def loso_arms(resolved: ResolvedDisorder) -> list[StudyArm]:
    """Study arms for epigrade.audit.loso, each carrying its own betas (cases plus the controls
    it is compared against, borrowed or not)."""
    arms = []
    for arm in resolved.arms:
        series = {s: arm.series_id for s in arm.case_ids}
        series |= {s: arm.control_series for s in arm.control_ids}
        arms.append(StudyArm(
            study_id=arm.series_id, beta=assemble_beta(series),
            train_case_ids=arm.train_case_ids, control_ids=arm.control_ids,
            test_case_ids=arm.case_ids,
            controls_borrowed_from=arm.control_series if arm.controls_borrowed else None,
        ))
    return arms
