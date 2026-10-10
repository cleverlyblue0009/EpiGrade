"""The twelve disorders in the corpus: which series each draws cases and controls from, and
which samples count as what. This is the single place cohort membership is defined - every
table and figure downstream resolves it from here.

Every selector takes the harmonised metadata for one series (indexed by GSM) and returns GSM
ids. Membership follows the phase 1 triage roles; the few places that go beyond a plain role
filter are each explained where they are defined.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from epigrade.signature.choufani import classify_cohort

Selector = Callable[[pd.DataFrame], list[str]]

PUBLISHED = "published"   # scored with a published probe list (Sotos only)
PATH_B = "path_b"         # signature re-derived here
NO_SIGNATURE = "none"     # fewer than 10 cases: scored by other classifiers, never built


def _role_cases(disorder: str) -> Selector:
    return lambda meta: meta[(meta.role == "case") & (meta.disorder == disorder)].index.tolist()


def _matched_controls(meta: pd.DataFrame) -> list[str]:
    return meta[meta.role == "matched_control"].index.tolist()


def _sample_type(value: str) -> Selector:
    return lambda meta: meta[meta["sample_type"] == value].index.tolist()


def _sotos_cohort(*groups: str) -> Selector:
    return lambda meta: classify_cohort(meta).loc[lambda c: c.isin(groups)].index.tolist()


def _srs_lom_cases(meta: pd.DataFrame) -> list[str]:
    """Silver-Russell is molecularly heterogeneous: 11p15 loss of methylation is a confirmed
    epimutation, UPD(7) is a different mechanism, and 'clinical SRS' has no molecular
    confirmation. Pooling all three washed out the differential signal during development, so
    derivation uses the confirmed 11p15 LOM subtype only - the same principle as the Sotos
    paper restricting discovery to confirmed NSD1 loss-of-function. Scoring always uses every
    Silver-Russell case."""
    cases = meta[(meta.role == "case") & (meta.disorder == "Silver-Russell syndrome")]
    return cases[cases["genotype/variation"] == "11p15 LOM"].index.tolist()


def _down_siblings(meta: pd.DataFrame) -> list[str]:
    """GSE52588 has no unrelated controls: it is 29 Down syndrome cases, their 29 unaffected
    siblings and their 29 mothers. Triage excludes relatives from the retained corpus, and that
    count is unchanged. For Down syndrome alone the siblings serve as the comparison group (the
    depositing study's own design). Mothers are not used: they are a generation older, and age
    is one of the strongest drivers of blood methylation. Siblings are never added to the
    pooled control corpus."""
    return meta[
        (meta.role == "unaffected_relative") & meta["group"].str.startswith("sibling", na=False)
    ].index.tolist()


@dataclass(frozen=True)
class Arm:
    """One study's contribution to a disorder."""

    series_id: str
    cases: Selector
    controls: Selector = _matched_controls
    # Stricter subset used to DERIVE a signature; defaults to `cases`.
    train_cases: Selector | None = None
    # Series the controls are read from when this study deposited none of its own.
    controls_series: str | None = None
    control_note: str = ""

    @property
    def control_series(self) -> str:
        return self.controls_series or self.series_id

    @property
    def controls_borrowed(self) -> bool:
        return self.controls_series is not None and self.controls_series != self.series_id


@dataclass(frozen=True)
class Disorder:
    name: str
    slug: str
    arms: tuple[Arm, ...]
    signature: str = PATH_B
    # Series whose samples train the canonical classifier; default is every arm.
    train_series: tuple[str, ...] = ()
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def studies(self) -> list[str]:
        return [a.series_id for a in self.arms]

    @property
    def canonical_train_series(self) -> tuple[str, ...]:
        return self.train_series or tuple(self.studies)


DISORDERS: tuple[Disorder, ...] = (
    Disorder(
        "Silver-Russell syndrome", "silver_russell",
        arms=(
            Arm("GSE104451", _role_cases("Silver-Russell syndrome"),
                train_cases=_srs_lom_cases),
            Arm("GSE55491", _role_cases("Silver-Russell syndrome")),
        ),
        # GSE104451 alone, not pooled: pooling in GSE55491's molecularly unconfirmed cases
        # produced a larger signature (328 probes) that fired on 35 of 38 Sotos cases. More
        # probes passing the same filter is not a better signature.
        train_series=("GSE104451",),
        note="canonical classifier trained on GSE104451 11p15 LOM cases only",
    ),
    Disorder(
        "Sotos syndrome", "sotos",
        arms=(
            Arm("GSE74432", _sotos_cohort("discovery_case", "replication_case"),
                controls=_sotos_cohort("discovery_control"),
                train_cases=_sotos_cohort("discovery_case")),
        ),
        signature=PUBLISHED,
        note="published 7,085-probe list (Choufani et al. 2015); median profiles from the "
             "discovery cohort only",
    ),
    Disorder(
        "Kabuki syndrome type 1", "kabuki_1",
        arms=(
            Arm("GSE116300", _role_cases("Kabuki syndrome type 1")),
            Arm("GSE97362", _role_cases("Kabuki syndrome type 1"),
                controls=_sample_type("Control for KMT2D LOF discovery cohort")),
        ),
        note="canonical classifier pooled across both studies (GSE116300 alone finds 0 probes)",
    ),
    Disorder(
        "Down syndrome", "down",
        arms=(
            Arm("GSE52588", _role_cases("Down syndrome"), controls=_down_siblings,
                control_note="29 unaffected siblings used as within-family comparators; the "
                             "study has no unrelated controls"),
        ),
    ),
    Disorder(
        "Coffin-Siris syndrome", "coffin_siris",
        arms=(
            Arm("GSE116992", _role_cases("Coffin-Siris syndrome"), controls_series="GSE125367",
                control_note="GSE116992 deposited no controls; the 27 controls of GSE125367 "
                             "(a different laboratory, EPIC) are used, so case/control is "
                             "fully confounded with series"),
        ),
    ),
    Disorder(
        "Williams syndrome", "williams",
        arms=(Arm("GSE66552", _role_cases("Williams syndrome")),),
    ),
    Disorder(
        "CHARGE syndrome", "charge",
        arms=(
            Arm("GSE97362", _role_cases("CHARGE syndrome"),
                controls=_sample_type("Control for CHD7 LOF discovery cohort")),
        ),
    ),
    Disorder(
        "Nicolaides-Baraitser syndrome", "nicolaides_baraitser",
        arms=(
            Arm("GSE116992", _role_cases("Nicolaides-Baraitser syndrome"),
                controls_series="GSE125367",
                control_note="GSE116992 deposited no controls; GSE125367 controls used"),
            Arm("GSE125367", _role_cases("Nicolaides-Baraitser syndrome")),
        ),
        note="canonical classifier pooled across both studies (8 cases each); all controls "
             "come from GSE125367",
    ),
    Disorder(
        "ICF syndrome", "icf",
        arms=(Arm("GSE95040", _role_cases("ICF syndrome")),),
        note="ICF types 1-4 pooled as one disorder, as harmonised",
    ),
    Disorder(
        "7q11.23 duplication syndrome", "dup7q11_23",
        arms=(Arm("GSE66552", _role_cases("7q11.23 duplication syndrome")),),
    ),
    Disorder(
        "Weaver syndrome", "weaver",
        arms=(Arm("GSE74432", _sotos_cohort("weaver"),
                  controls=_sotos_cohort("discovery_control")),),
        signature=NO_SIGNATURE,
    ),
    Disorder(
        "Claes-Jensen syndrome", "claes_jensen",
        arms=(Arm("GSE108423", _role_cases("Claes-Jensen syndrome")),),
        signature=NO_SIGNATURE,
    ),
)

BY_NAME = {d.name: d for d in DISORDERS}
BY_SLUG = {d.slug: d for d in DISORDERS}

# Every series with a downloaded beta matrix. Matched controls from all of them form the
# pooled control corpus. The six population-control accessions (GSE42861, GSE87571, GSE85210,
# GSE87648, GSE99863, GSE35069) were harvested for metadata only - their beta matrices are not
# part of this build.
BETA_SERIES: tuple[str, ...] = (
    "GSE74432", "GSE104451", "GSE55491", "GSE116300", "GSE97362", "GSE66552", "GSE95040",
    "GSE108423", "GSE116992", "GSE125367", "GSE52588",
)
POOLED_CONTROL_LABEL = "pooled_control"
# Down syndrome's sibling comparators: scored, but never part of the pooled controls.
SIBLING_LABEL = "comparator_sibling"


def pooled_control_ids(meta: pd.DataFrame) -> list[str]:
    """Matched controls of one series, for the pooled control corpus."""
    return _matched_controls(meta)
