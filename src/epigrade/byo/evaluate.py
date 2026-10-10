"""Bring your own signature: evaluate a submitted probe list against every cohort in the corpus.

Two input forms:
  Form 1 - a probe list (CSV/TSV): probe_id (required), direction (optional: hypo/hyper),
           weight (optional, default 1.0).                    -> parse_signature()
  Form 2 - labelled samples plus a beta matrix; a signature is derived with the same Path B
           routine the corpus uses, then evaluated as Form 1. -> signature_from_labelled_samples()

What evaluate() reports: platform and corpus coverage, an identification verdict across all 12
cohorts, the signature's own specificity row, an evidence ceiling (cohort size and signature size),
overlap with the 10 corpus signatures, and an attribution panel.

HOW A MATCH IS DECIDED. Ranking cohorts by AUC alone does not work on this corpus: 500 random
probes separate Sotos cases from their controls at AUC 1.00 (Williams 0.99, ICF 0.99), because
those disorders shift methylation genome-wide inside a single study. So AUC is reported for
every cohort, but a cohort counts as a match only if BOTH hold:
  1. AUC of cases vs that cohort's controls is at least 0.70, and
  2. the submitted probes show a larger mean absolute case-control beta difference in that
     cohort than every one of N_NULL random probe sets of the same size (empirical p below
     1/N_NULL, which stays below 0.05 after correcting for the 12 cohorts searched).
Cohorts are ranked by how far the submitted probes exceed that random-probe null (z). If no
cohort passes both, the result is no_match - a best-of-set guess is never forced.

This module never raises on bad user input: parse errors become SignatureInputError with a
message naming the expected columns, and refusals are returned as results with a code.
"""

from __future__ import annotations

import io
import re
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

from epigrade import paths
from epigrade.byo import store
from epigrade.calibrate.calibrate import (
    local_likelihood_ratio,
    lr_to_points,
    points_to_band,
)
from epigrade.corpus.registry import POOLED_CONTROL_LABEL
from epigrade.report.attribution import explain_score, format_explanation
from epigrade.signature.generic import (
    MIN_CASES_TO_BUILD_CLASSIFIER,
    MIN_PROBES_TO_SCORE,
    classifier_from_probes,
    derive_signature,
    diagnose_underpowered,
    get_thresholds,
    probe_tests,
    score_matrix,
)

EXPECTED_COLUMNS = "probe_id (required), direction (optional: hypo/hyper), weight (optional)"
MIN_PROBES_FOUND = 10
LOW_COVERAGE_FRACTION = 0.5
AUC_THRESHOLD = 0.70
OVERLAP_FLAG = 0.5
N_NULL = 1000
N_BOOT = 500
N_CEILING_BOOT = 200
MIN_N_TO_INTERPRET = 10

REFUSED_COVERAGE = "insufficient_probe_coverage"
WARN_LOW_COVERAGE = "low_coverage_warning"

# 450K / EPIC IDs (cg########, ch.<chr>.<pos><F|R>, rs#######), optionally with the EPIC v2
# replicate suffix (_TC21, _BC11, ...), which is stripped before matching.
_PROBE_RE = re.compile(r"^(cg\d{8}|ch\.[0-9XY]+\.\d+[FR]?|rs\d+)(_[TB][CO]\d{2})?$")
_DIRECTIONS = {"hypo": "hypo", "loss": "hypo", "hyper": "hyper", "gain": "hyper"}


class SignatureInputError(ValueError):
    """A submitted file that cannot be read as a signature. `code` is machine-readable; the
    message is safe to show to a user as-is."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class ParsedSignature:
    probes: pd.DataFrame  # indexed by probe_id; columns direction (hypo/hyper/None), weight
    n_rows: int
    n_invalid_format: int = 0
    n_duplicates: int = 0
    n_invalid_direction: int = 0
    n_invalid_weight: int = 0
    invalid_examples: list[str] = field(default_factory=list)
    source: str = "probe_list"

    @property
    def n_submitted(self) -> int:
        return len(self.probes)

    @property
    def has_direction(self) -> bool:
        return bool(self.probes["direction"].notna().any())


# --------------------------------------------------------------------------- parsing


def _read_table(source, name: str) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise SignatureInputError("file_not_found", f"{path} does not exist.")
        raw = path.read_bytes()
        name = name or path.name
    elif isinstance(source, bytes):
        raw = source
    else:
        raw = source.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SignatureInputError(
            "not_text", f"The file is not a text CSV/TSV. Expected columns: {EXPECTED_COLUMNS}."
        ) from exc
    if not text.strip():
        raise SignatureInputError(
            "empty_file", f"The file is empty. Expected columns: {EXPECTED_COLUMNS}."
        )
    first_line = text.lstrip().splitlines()[0]
    sep = "\t" if first_line.count("\t") >= first_line.count(",") and "\t" in first_line else ","
    try:
        return pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False,
                           skip_blank_lines=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise SignatureInputError(
            "malformed_file",
            f"The file could not be read as CSV/TSV ({exc}). Expected columns: "
            f"{EXPECTED_COLUMNS}.",
        ) from exc


def parse_signature(source, name: str = "") -> ParsedSignature:
    """Read a Form 1 probe list from a path, bytes, file-like object or DataFrame."""
    table = _read_table(source, name)
    table.columns = [str(c).strip().lower() for c in table.columns]
    if "probe_id" not in table.columns:
        header = ", ".join(map(str, table.columns[:6])) or "(none)"
        hint = ""
        if table.columns.size and _PROBE_RE.match(str(table.columns[0])):
            hint = " The first line looks like a probe ID, so the header row may be missing."
        raise SignatureInputError(
            "missing_probe_id_column",
            f"No 'probe_id' column found (columns seen: {header}).{hint} Expected columns: "
            f"{EXPECTED_COLUMNS}.",
        )
    n_rows = len(table)
    if n_rows == 0:
        raise SignatureInputError(
            "no_rows", f"The file has a header but no probe rows. Expected columns: "
                       f"{EXPECTED_COLUMNS}."
        )

    ids = table["probe_id"].astype(str).str.strip()
    matched = ids.map(lambda s: _PROBE_RE.match(s))
    valid = matched.notna()
    invalid_examples = ids[~valid].head(5).tolist()
    table = table[valid].copy()
    table["probe_id"] = [m.group(1) for m in matched[valid]]

    direction = pd.Series([None] * len(table), index=table.index, dtype=object)
    n_bad_direction = 0
    if "direction" in table.columns:
        raw = table["direction"].astype(str).str.strip().str.lower()
        direction = raw.map(_DIRECTIONS).astype(object).where(raw.isin(_DIRECTIONS), None)
        n_bad_direction = int(((raw != "") & ~raw.isin(_DIRECTIONS)).sum())
    weight = pd.Series(1.0, index=table.index)
    n_bad_weight = 0
    if "weight" in table.columns:
        raw = table["weight"].astype(str).str.strip()
        numeric = pd.to_numeric(raw, errors="coerce")
        bad = (raw != "") & ~(numeric > 0)
        n_bad_weight = int(bad.sum())
        weight = numeric.where(numeric > 0, 1.0).fillna(1.0)

    probes = pd.DataFrame({
        "probe_id": table["probe_id"].to_numpy(), "direction": direction.to_numpy(),
        "weight": weight.to_numpy(dtype=float),
    })
    n_before = len(probes)
    probes = probes.drop_duplicates("probe_id", keep="first").set_index("probe_id")
    return ParsedSignature(
        probes=probes, n_rows=n_rows, n_invalid_format=int((~valid).sum()),
        n_duplicates=n_before - len(probes), n_invalid_direction=n_bad_direction,
        n_invalid_weight=n_bad_weight, invalid_examples=invalid_examples,
    )


def signature_from_labelled_samples(labels, beta) -> ParsedSignature:
    """Form 2: derive a signature from labelled samples with the corpus's own Path B routine
    (project-default thresholds), ready to pass to evaluate().

    `labels`: table with sample_id and label (case/control). `beta`: probe x sample betas,
    probe IDs in the first column or the index.
    """
    labels = _read_table(labels, "") if not isinstance(labels, pd.DataFrame) else labels.copy()
    labels.columns = [str(c).strip().lower() for c in labels.columns]
    if not {"sample_id", "label"} <= set(labels.columns):
        raise SignatureInputError(
            "missing_label_columns",
            "The sample sheet needs the columns sample_id and label (case/control).",
        )
    if not isinstance(beta, pd.DataFrame):
        beta = _read_table(beta, "")
        beta = beta.set_index(beta.columns[0])
    beta = beta.apply(pd.to_numeric, errors="coerce")
    beta.index = beta.index.astype(str)

    label = labels["label"].astype(str).str.strip().str.lower()
    sample = labels["sample_id"].astype(str).str.strip()
    cases = [s for s in sample[label == "case"] if s in beta.columns]
    controls = [s for s in sample[label == "control"] if s in beta.columns]
    if len(cases) < MIN_CASES_TO_BUILD_CLASSIFIER or len(controls) < 2:
        raise SignatureInputError(
            "too_few_labelled_samples",
            f"Found {len(cases)} case and {len(controls)} control samples with beta values. "
            f"At least {MIN_CASES_TO_BUILD_CLASSIFIER} cases and 2 controls are needed to "
            "derive a signature.",
        )
    tests = probe_tests(beta, cases, controls)
    sig = derive_signature(beta, cases, controls, tests=tests)
    if len(sig) < MIN_PROBES_FOUND:
        raise SignatureInputError(
            "no_signature_derived",
            diagnose_underpowered(beta, cases, controls, tests=tests),
        )
    thresholds = get_thresholds(None)
    parsed = parse_signature(pd.DataFrame({
        "probe_id": sig.index,
        "direction": np.where(sig["delta_beta"] < 0, "hypo", "hyper"),
    }))
    parsed.n_rows = len(sig)
    parsed.source = (
        f"derived_from_labelled_samples ({len(cases)} cases, {len(controls)} controls; "
        f"{thresholds['multiple_testing_method']} alpha={thresholds['alpha']}, effect floor "
        f"{thresholds['effect_size_floor']})"
    )
    return parsed


# --------------------------------------------------------------------------- pieces


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum identity (ties get mid-ranks)."""
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _auc_ci(scores: np.ndarray, labels: np.ndarray, rng) -> tuple[float, float]:
    pos, neg = np.where(labels == 1)[0], np.where(labels == 0)[0]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan"), float("nan")
    values = []
    for _ in range(N_BOOT):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        values.append(_auc(scores[idx], labels[idx]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


@lru_cache(maxsize=1)
def _tables() -> dict[str, pd.DataFrame]:
    def read(name):
        path = paths.repo_root() / "results" / "tables" / name
        return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""])
    return {
        "verdicts": read("disorder_verdicts.tsv").set_index("disorder"),
        "gate": read("confounding_gate.tsv").set_index("disorder"),
        "ceiling": read("attainable_ceiling.tsv").set_index("disorder"),
    }


@lru_cache(maxsize=1)
def corpus_signatures() -> dict[str, frozenset]:
    """disorder -> probe set, for each of the corpus's own signatures."""
    sig_dir = paths.repo_root() / "results" / "signatures"
    index = pd.read_csv(sig_dir / "index.tsv", sep="\t")
    return {
        row.disorder: frozenset(
            pd.read_csv(sig_dir / f"{row.slug}.tsv", sep="\t", usecols=["probe_id"])["probe_id"]
        )
        for row in index.itertuples()
    }


def interpretation_scope(disorder: str) -> str:
    """within_study_only unless the disorder passes the confounding gate AND its calibration
    has a between-study bound. A failed gate always carries into a bring-your-own result."""
    tables = _tables()
    if tables["gate"].loc[disorder, "status"] != "pass-structural":
        return "within_study_only"
    scope = tables["verdicts"].loc[disorder, "interpretation_scope"]
    return scope if scope == "between_study" else "within_study_only"


def coverage(parsed: ParsedSignature, index: store.CorpusIndex, source: store.BetaSource) -> dict:
    ids = parsed.probes.index.to_numpy(dtype=str)
    pos = index.positions(ids)
    known = pos >= 0
    usable = np.zeros(len(ids), dtype=bool)
    usable[known] = (~np.isnan(index.delta[pos[known]].astype(float))).any(axis=1)
    n = len(ids)
    platform = {}
    for j, name in enumerate(index.platforms):
        if name not in index.platforms_available:
            platform[name] = {"n_present": None, "fraction": None,
                              "note": "manifest not available in this resource"}
            continue
        n_on = int(index.platform_presence[pos[known], j].sum())
        platform[name] = {"n_present": n_on, "fraction": n_on / n if n else float("nan")}
    series = {
        name: int(index.series_presence[pos[known], j].sum())
        for j, name in enumerate(index.series)
    }
    n_found = int(usable.sum())
    return {
        "n_submitted": n, "n_found": n_found, "n_missing": n - n_found,
        "fraction_missing": (n - n_found) / n if n else float("nan"),
        "n_in_any_corpus_matrix": int((index.series_presence[pos[known]].any(axis=1)).sum()),
        "platform_survival": platform, "corpus_series_presence": series,
        "beta_source": source.kind,
        "n_with_sample_level_betas": int(source.has(ids[usable]).sum()),
        "_usable_ids": ids[usable],
    }


def _cohort_ids(disorder: str) -> tuple[list[str], list[str]]:
    members = store.cohort_membership()
    block = members[members.disorder == disorder]
    return (block.loc[block.role == "case", "gsm_accession"].tolist(),
            block.loc[block.role == "control", "gsm_accession"].tolist())


def _score_cohort(disorder: str, probes: np.ndarray, betas: pd.DataFrame):
    """Leave-one-out correlation-difference scores for one cohort on the submitted probes."""
    cases, controls = _cohort_ids(disorder)
    clf = classifier_from_probes(disorder, pd.Index(probes), betas, cases, controls)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        scores = score_matrix(clf, betas, cases + controls)
    return clf, scores, cases, controls


def search_cohorts(parsed: ParsedSignature, index: store.CorpusIndex,
                   source: store.BetaSource, usable_ids: np.ndarray,
                   seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per cohort (all 12), plus the sample betas used for the AUCs."""
    rng = np.random.default_rng(seed)
    pos = index.positions(usable_ids)
    weights = parsed.probes.loc[usable_ids, "weight"].to_numpy(dtype=float)
    betas = source.get(usable_ids)
    own = corpus_signatures()
    verdicts = _tables()["verdicts"]

    rows = []
    for cohort in index.cohorts:
        delta_all = np.abs(index.cohort_delta(cohort))
        universe = delta_all[~np.isnan(delta_all)]
        mine = delta_all[pos]
        in_cohort = ~np.isnan(mine)
        k = int(in_cohort.sum())
        cases, controls = _cohort_ids(cohort)
        row = {
            "disorder": cohort, "n_cases": len(cases), "n_controls": len(controls),
            "n_probes_used": k, "has_own_signature": cohort in own,
            "interpretation_scope": interpretation_scope(cohort),
            "verdict_class": verdicts.loc[cohort, "verdict"],
            "interpretation": ("too small to interpret" if len(cases) < MIN_N_TO_INTERPRET
                               else ""),
        }
        if k < MIN_PROBES_FOUND:
            rows.append({**row, "status": "not_computed",
                         "reason": f"only {k} submitted probes are usable in this cohort"})
            continue
        w = weights[in_cohort] / weights[in_cohort].sum()
        observed = float((mine[in_cohort] * w).sum())
        null = (universe[rng.integers(0, len(universe), size=(N_NULL, k))] * w).sum(axis=1)
        boot = rng.integers(0, k, size=(N_BOOT, k))
        effect_boot = (mine[in_cohort][boot] * w[boot]).sum(axis=1) / w[boot].sum(axis=1)
        row |= {
            "status": "computed", "mean_abs_delta_beta": observed,
            "effect_ci_low": float(np.percentile(effect_boot, 2.5)),
            "effect_ci_high": float(np.percentile(effect_boot, 97.5)),
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "z_vs_random_probes": float((observed - null.mean()) / null.std(ddof=1)),
            "null_percentile": float((null < observed).mean() * 100),
            "exceeds_all_random_sets": bool(observed > null.max()),
        }
        with_betas = betas.index.intersection(usable_ids[in_cohort])
        row["n_probes_with_sample_betas"] = len(with_betas)
        if len(with_betas) < MIN_PROBES_TO_SCORE:
            row |= {"auc": float("nan"), "reason": (
                f"AUC not computed: sample-level betas are available for only "
                f"{len(with_betas)} of these probes in the {source.kind}")}
        else:
            _, scores, cases, controls = _score_cohort(cohort, with_betas.to_numpy(), betas)
            labels = scores.index.isin(cases).astype(int)
            values = scores["score"].to_numpy()
            low, high = _auc_ci(values, labels, rng)
            row |= {"auc": _auc(values, labels), "auc_ci_low": low, "auc_ci_high": high,
                    "reason": ""}
        auc = row["auc"]
        row["passes_auc"] = bool(auc >= AUC_THRESHOLD) if not np.isnan(auc) else False
        row["is_candidate"] = row["passes_auc"] and row["exceeds_all_random_sets"]
        rows.append(row)
    table = pd.DataFrame(rows)
    table = table.sort_values("z_vs_random_probes", ascending=False, na_position="last")
    return table.reset_index(drop=True), betas


def identify(cohorts: pd.DataFrame) -> dict:
    computed = cohorts[cohorts.status == "computed"]
    candidates = computed[computed.is_candidate]
    base = {
        "n_disorders_searched": int(len(cohorts)),
        "n_with_own_signature": int(cohorts.has_own_signature.sum()),
        "rule": (f"match requires AUC >= {AUC_THRESHOLD:.2f} and a mean absolute case-control "
                 f"difference above all {N_NULL} random probe sets of the same size"),
    }
    naive = computed.dropna(subset=["auc"]).sort_values("auc", ascending=False)
    if len(naive):
        base["highest_auc_disorder"] = naive.iloc[0].disorder
        base["highest_auc"] = float(naive.iloc[0].auc)
        base["n_cohorts_auc_at_or_above_threshold"] = int((naive.auc >= AUC_THRESHOLD).sum())
    if candidates.empty:
        return {**base, "verdict": "no_match", "top_match": None}
    top = candidates.iloc[0]
    others = computed[computed.disorder != top.disorder]
    runner = others.iloc[0] if len(others) else None
    return {
        **base, "verdict": "match", "top_match": top.disorder,
        "top_match_auc": float(top.auc),
        "top_match_auc_ci": [float(top.auc_ci_low), float(top.auc_ci_high)],
        "top_match_z": float(top.z_vs_random_probes),
        "top_match_mean_abs_delta_beta": float(top.mean_abs_delta_beta),
        "runner_up": runner.disorder if runner is not None else None,
        "runner_up_z": float(runner.z_vs_random_probes) if runner is not None else None,
        "margin_z": (float(top.z_vs_random_probes - runner.z_vs_random_probes)
                     if runner is not None else None),
        "other_candidates": candidates.disorder.iloc[1:].tolist(),
        "interpretation_scope": top.interpretation_scope,
        "top_match_too_small_to_interpret": bool(top.interpretation),
    }


def specificity_row(disorder: str, usable_ids: np.ndarray, betas: pd.DataFrame) -> pd.DataFrame:
    """The submitted signature's own row of the cross-disorder matrix, anchored on the matched
    cohort: median profiles from that cohort's cases and controls, then every other disorder's
    cases and the pooled controls scored against them."""
    clf, _, cases, controls = _score_cohort(disorder, betas.index.to_numpy(), betas)
    samples = store.sample_table()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        scores = score_matrix(clf, betas, samples.index.tolist()).join(samples)
    rows = []
    groups = [g for g in dict.fromkeys(store.load_index().cohorts)] + [POOLED_CONTROL_LABEL]
    for group in groups:
        block = scores[scores["group"] == group]
        note = ""
        if group == POOLED_CONTROL_LABEL:
            block = block[~block.index.isin(controls)]
            note = "matched cohort's own controls excluded"
        elif group == disorder:
            note = "matched cohort (sensitivity, leave-one-out profiles)"
        if block.empty:
            rows.append({"scored_group": group, "status": "not_computed", "n": 0,
                         "reason": "too few submitted probes present for these samples"})
            continue
        n, n_pos = len(block), int((block.score > 0).sum())
        rows.append({
            "scored_group": group, "status": "computed", "n": n, "n_positive": n_pos,
            "fraction_positive": n_pos / n,
            "interpretation": "too small to interpret" if n < MIN_N_TO_INTERPRET else "",
            "reason": note,
        })
    return pd.DataFrame(rows)


def overlap_with_corpus(parsed: ParsedSignature) -> pd.DataFrame:
    mine = set(parsed.probes.index)
    rows = []
    for disorder, theirs in corpus_signatures().items():
        shared = len(mine & theirs)
        rows.append({
            "corpus_signature": disorder, "n_corpus_probes": len(theirs),
            "n_shared": shared, "jaccard": shared / len(mine | theirs) if mine else 0.0,
            "fraction_of_submitted_in_corpus_signature": shared / len(mine) if mine else 0.0,
        })
    return pd.DataFrame(rows).sort_values("n_shared", ascending=False).reset_index(drop=True)


def attribution(parsed: ParsedSignature, disorder: str, index: store.CorpusIndex,
                betas: pd.DataFrame) -> dict:
    """Why this result: direction consistency against the submitted direction column, and the
    epigrade.report.attribution panel for the median-scoring case of the matched cohort."""
    ids = betas.index.to_numpy()
    delta = pd.Series(index.cohort_delta(disorder)[index.positions(ids)], index=ids)
    submitted = parsed.probes.loc[ids, "direction"].dropna()
    observed = np.where(delta.loc[submitted.index] < 0, "hypo", "hyper")
    consistent = int((submitted.to_numpy() == observed).sum())
    result = {
        "matched_disorder": disorder,
        "direction_column_supplied": bool(len(submitted)),
        "n_probes_with_direction": int(len(submitted)),
        "n_direction_consistent": consistent,
        "n_direction_inconsistent": int(len(submitted) - consistent),
        "fraction_direction_consistent": consistent / len(submitted) if len(submitted) else None,
        "all_directions_inconsistent": bool(len(submitted)) and consistent == 0,
        "observed_direction_counts": {
            "hypo": int((delta < 0).sum()), "hyper": int((delta >= 0).sum()),
        },
    }
    clf, scores, cases, controls = _score_cohort(disorder, ids, betas)
    case_scores = scores.loc[scores.index.isin(cases), "score"]
    control_scores = scores.loc[scores.index.isin(controls), "score"]
    exemplar = (case_scores - case_scores.median()).abs().idxmin()
    explained = explain_score(
        gsm_accession=exemplar, disorder=disorder, cohort_label="median-scoring case",
        sample_beta=betas[exemplar], median_case=clf.median_case,
        median_control=clf.median_control, signature_probes=clf.signature_probes,
        case_scores=case_scores, control_scores=control_scores,
    )
    result |= {
        "exemplar_sample": exemplar, "exemplar_score": explained.score,
        "percentile_within_cases": explained.percentile_within_cases,
        "percentile_within_controls": explained.percentile_within_controls,
        "sd_from_control_mean": explained.sd_from_control_mean,
        "contribution_method": explained.contribution_method,
        "top_probes": [
            {**p, "submitted_direction": parsed.probes.loc[p["probe_id"], "direction"],
             "observed_direction": "hypo" if delta.loc[p["probe_id"]] < 0 else "hyper"}
            for p in explained.top_probes
        ],
        "explanation": format_explanation(explained),
    }
    return result


BAND_ORDER = ["NA", "No evidence", "Supporting", "Moderate", "Strong", "Very Strong"]


def signature_size_bound(disorder: str, betas: pd.DataFrame, seed: int = 0) -> dict:
    """The band this signature's SIZE can support in the matched cohort.

    A cohort ceiling says what a perfect classifier could earn at that cohort's size. It says
    nothing about how many probes the signature has: 20 probes and 500 probes matched to the
    same cohort would get the same ceiling. This adds the missing part. Each replicate
    resamples the signature's probes AND the cohort's samples, rescores the cohort with the
    resampled probes (leave-one-out profiles), and takes the local likelihood ratio at the
    typical case score. The band is read from the 2.5th percentile across replicates. Fewer
    probes means more probe-sampling uncertainty, so a lower bound.

    A replicate needs at least MIN_PROBES_TO_SCORE distinct probes. If fewer than half the
    replicates are usable the signature is too small to bound and the result is No evidence.
    """
    cfg = _band_config()
    rng = np.random.default_rng(seed)
    cases, _ = _cohort_ids(disorder)
    probes = betas.index.to_numpy()
    ratios = []
    for _ in range(N_CEILING_BOOT):
        sub = np.unique(rng.choice(probes, len(probes)))
        if len(sub) < MIN_PROBES_TO_SCORE:
            continue
        _, scores, _, _ = _score_cohort(disorder, sub, betas.loc[sub])
        labels = scores.index.isin(cases).astype(int)
        values = scores["score"].to_numpy()
        pos, neg = np.where(labels == 1)[0], np.where(labels == 0)[0]
        if len(pos) < 2 or len(neg) < 2:
            continue
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        ratio = local_likelihood_ratio(
            values[idx], labels[idx], float(np.median(values[idx][labels[idx] == 1])),
        )
        if np.isfinite(ratio):
            ratios.append(ratio)
    base = {"n_probes": len(probes), "n_replicates": N_CEILING_BOOT,
            "n_replicates_usable": len(ratios)}
    if len(ratios) < N_CEILING_BOOT / 2:
        return {**base, "band": "No evidence", "conservative_lr": float("nan"),
                "reason": (f"too few probes to bound: only {len(ratios)} of {N_CEILING_BOOT} "
                           f"resamples kept {MIN_PROBES_TO_SCORE} or more distinct probes")}
    low = float(np.percentile(ratios, 2.5))
    band = "No evidence" if low <= 1 else points_to_band(
        lr_to_points(low, cfg["odds_path_base"]), cfg["bands"])
    return {**base, "band": band, "conservative_lr": low, "reason": ""}


@lru_cache(maxsize=1)
def _band_config() -> dict:
    with open(paths.repo_root() / "config" / "evidence_bands.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evidence_ceiling_for(disorder: str, betas: pd.DataFrame, seed: int = 0) -> dict:
    """The strongest band this signature could support in the matched cohort: the lower of
    the cohort ceiling (cohort size and study structure, from attainable_ceiling.tsv) and the
    signature-size bound (see signature_size_bound)."""
    row = _tables()["ceiling"].loc[disorder]
    size = signature_size_bound(disorder, betas, seed=seed)
    cohort_band = row.attainable_band
    limiting = min(cohort_band, size["band"], key=BAND_ORDER.index)
    return {
        "disorder": disorder, "n_case": int(row.n_case), "n_control": int(row.n_control),
        "attainable_band": limiting,
        "limited_by": ("signature size" if BAND_ORDER.index(size["band"])
                       < BAND_ORDER.index(cohort_band) else "cohort size"),
        "cohort_ceiling_band": cohort_band,
        "signature_size_band": size["band"],
        "signature_size_conservative_lr": size["conservative_lr"],
        "n_probes_scored": size["n_probes"],
        "n_replicates_usable": size["n_replicates_usable"],
        "signature_size_reason": size["reason"],
        "between_study_bound_possible": bool(row.between_study_bound_possible),
        "interpretation_scope": row.interpretation_scope,
        "note": ("the lower of two bounds: what a perfect classifier could earn at this "
                 "cohort's size, and what this signature's number of probes can support when "
                 "probes and samples are both resampled"),
    }


# --------------------------------------------------------------------------- top level


@dataclass
class Evaluation:
    status: str                      # "ok" or "refused"
    refusal_code: str | None
    messages: list[str]
    warnings: list[dict]
    input_summary: dict
    coverage: dict
    cohorts: pd.DataFrame | None = None
    identification: dict | None = None
    specificity: pd.DataFrame | None = None
    evidence_ceiling: dict | None = None
    overlap: pd.DataFrame | None = None
    attribution: dict | None = None

    def to_long_table(self) -> pd.DataFrame:
        """The whole evaluation as one section/item/field/value table, for TSV download."""
        rows = [("result", "", "status", self.status),
                ("result", "", "refusal_code", self.refusal_code or "")]
        rows += [("message", str(i), "text", m) for i, m in enumerate(self.messages)]
        for w in self.warnings:
            rows += [("warning", w["code"], k, v) for k, v in w.items() if k != "code"]
        rows += [("input", "", k, v) for k, v in self.input_summary.items()]
        for k, v in self.coverage.items():
            if isinstance(v, dict):
                for name, value in v.items():
                    if isinstance(value, dict):
                        rows += [(f"coverage.{k}", name, kk, vv) for kk, vv in value.items()]
                    else:
                        rows.append((f"coverage.{k}", name, "n_present", value))
            else:
                rows.append(("coverage", "", k, v))
        for name, value in (self.identification or {}).items():
            rows.append(("identification", "", name, value))
        for section, table, key in (("cohort_search", self.cohorts, "disorder"),
                                    ("specificity", self.specificity, "scored_group"),
                                    ("overlap", self.overlap, "corpus_signature")):
            if table is not None:
                for _, r in table.iterrows():
                    rows += [(section, r[key], c, r[c]) for c in table.columns if c != key]
        rows += [("evidence_ceiling", "", k, v) for k, v in (self.evidence_ceiling or {}).items()]
        for k, v in (self.attribution or {}).items():
            if k == "top_probes":
                for p in v:
                    rows += [("attribution.top_probes", p["probe_id"], kk, vv)
                             for kk, vv in p.items() if kk != "probe_id"]
            elif k == "explanation":
                rows += [("attribution.explanation", str(i), "text", line)
                         for i, line in enumerate(v)]
            else:
                rows.append(("attribution", "", k, v))
        return pd.DataFrame(rows, columns=["section", "item", "field", "value"])


def evaluate(parsed: ParsedSignature, index: store.CorpusIndex | None = None,
             source: store.BetaSource | None = None, seed: int = 0) -> Evaluation:
    index = index or store.load_index()
    source = source or store.load_beta_source()

    messages, warnings_ = [], []
    if parsed.n_invalid_format:
        examples = ", ".join(parsed.invalid_examples)
        messages.append(
            f"{parsed.n_invalid_format} row(s) rejected: probe ID not in cg/ch/rs format "
            f"(e.g. {examples}). The remaining rows were evaluated."
        )
    if parsed.n_duplicates:
        messages.append(f"{parsed.n_duplicates} duplicate probe ID(s) removed.")
    if parsed.n_invalid_direction:
        messages.append(
            f"{parsed.n_invalid_direction} direction value(s) were not hypo/hyper and were "
            "treated as missing."
        )
    if parsed.n_invalid_weight:
        messages.append(
            f"{parsed.n_invalid_weight} weight value(s) were not positive numbers and were "
            "set to 1.0."
        )
    input_summary = {
        "source": parsed.source, "n_rows_in_file": parsed.n_rows,
        "n_rejected_bad_format": parsed.n_invalid_format,
        "n_duplicates_removed": parsed.n_duplicates, "n_submitted": parsed.n_submitted,
        "direction_column_supplied": parsed.has_direction,
    }

    cov = coverage(parsed, index, source)
    usable_ids = cov.pop("_usable_ids")
    base = {
        "n_disorders_searched": len(index.cohorts),
        "n_with_own_signature": len(corpus_signatures()),
    }
    if cov["n_found"] < MIN_PROBES_FOUND:
        messages.append(
            f"Refused: only {cov['n_found']} of {cov['n_submitted']} submitted probes are "
            f"present in any corpus cohort (minimum {MIN_PROBES_FOUND}). Nothing was scored."
        )
        return Evaluation(
            status="refused", refusal_code=REFUSED_COVERAGE, messages=messages,
            warnings=warnings_, input_summary=input_summary, coverage=cov,
            identification={**base, "verdict": "refused", "top_match": None},
            overlap=overlap_with_corpus(parsed),
        )
    if cov["fraction_missing"] > LOW_COVERAGE_FRACTION:
        warnings_.append({
            "code": WARN_LOW_COVERAGE,
            "percent_missing": round(cov["fraction_missing"] * 100, 1),
            "text": (f"{cov['fraction_missing']:.1%} of submitted probes are missing from "
                     "the corpus; results describe the remaining probes only."),
        })
    if cov["n_with_sample_level_betas"] < cov["n_found"]:
        messages.append(
            f"Sample-level betas are available for {cov['n_with_sample_level_betas']} of the "
            f"{cov['n_found']} usable probes ({source.kind}). The identification statistic "
            "uses all usable probes; AUC, the specificity row and attribution use only those "
            "with sample-level betas. Run scripts/build_byo_resource.py against the downloaded "
            "corpus for full coverage."
        )

    cohorts, betas = search_cohorts(parsed, index, source, usable_ids, seed=seed)
    identification = identify(cohorts)
    evaluation = Evaluation(
        status="ok", refusal_code=None, messages=messages, warnings=warnings_,
        input_summary=input_summary, coverage=cov, cohorts=cohorts,
        identification=identification, overlap=overlap_with_corpus(parsed),
    )
    flagged = evaluation.overlap[
        (evaluation.overlap.jaccard > OVERLAP_FLAG)
        | (evaluation.overlap.fraction_of_submitted_in_corpus_signature > OVERLAP_FLAG)
    ]
    for r in flagged.itertuples():
        messages.append(
            f"Overlap with the corpus {r.corpus_signature} signature: {r.n_shared} shared "
            f"probes (Jaccard {r.jaccard:.2f}; "
            f"{r.fraction_of_submitted_in_corpus_signature:.0%} of submitted probes). This "
            "may be a re-derivation or subset of a known signature rather than a new one."
        )

    match = identification["top_match"]
    if match is None:
        messages.append(
            f"No match: none of the {identification['n_disorders_searched']} cohorts passed "
            "both criteria. No best-of-set guess is made."
        )
        return evaluation
    cohort_probes = betas.index[~np.isnan(index.cohort_delta(match)[index.positions(
        betas.index.to_numpy())])]
    matched_betas = betas.loc[cohort_probes]
    evaluation.specificity = specificity_row(match, usable_ids, matched_betas)
    evaluation.evidence_ceiling = evidence_ceiling_for(match, matched_betas, seed=seed)
    evaluation.attribution = attribution(parsed, match, index, matched_betas)
    if identification["interpretation_scope"] == "within_study_only":
        messages.append(
            f"{match} does not support a between-study bound in this corpus "
            "(interpretation_scope = within_study_only): this match shows the signature "
            "separates that one study's cases from its controls, not that it would transfer."
        )
    if evaluation.attribution["all_directions_inconsistent"]:
        messages.append(
            "Every submitted direction disagrees with the direction observed in the matched "
            "cohort. The probes are the right ones but the direction column is inverted."
        )
    return evaluation


def evaluate_file(file, name: str = "", **kwargs) -> Evaluation:
    """Parse and evaluate in one call. Raises SignatureInputError for an unreadable file."""
    return evaluate(parse_signature(file, name), **kwargs)
