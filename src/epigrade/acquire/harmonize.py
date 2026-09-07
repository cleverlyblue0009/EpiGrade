"""Label harmonization: turn heterogeneous GEO sample metadata into (disorder, gene, role).

Two-stage design per the project spec:
  1. Rule-based matcher: canonical vocabulary (config/vocabulary.yaml) matched against a
     per-sample metadata blob, with a confidence score based on which field the match came from.
  2. Confidence router: rows at or above CONFIDENCE_THRESHOLD are auto-accepted; rows below it
     are escalated to an LLM (Gemini) that reads the sample's full metadata and returns a
     structured judgment. If no Gemini key/response is available, the row is left unresolved
     (role="exclude_other", resolution_status="needs_review") rather than guessed - see
     epigrade.acquire.llm_escalate.

Role enum: case | matched_control | population_control | unaffected_relative | under_test |
           cell_line | wrong_tissue | exclude_other
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
import yaml

from epigrade import paths

CONFIDENCE_THRESHOLD = 0.7

# Fields checked, in priority order, for a "clean" diagnosis signal (high-confidence match).
_STRUCTURED_FIELDS = [
    "disease_state", "group", "genotype/variation", "case_status", "phenotype",
    "simplified_diagnosis", "sample_type", "variant_classification",
]
# Freer-text fields: a match here only is lower confidence.
_FREETEXT_FIELDS = ["title", "source_name_ch1", "characteristics_raw"]

_VALID_ROLES = {
    "case", "matched_control", "population_control", "unaffected_relative",
    "under_test", "cell_line", "wrong_tissue", "exclude_other",
}


@dataclass
class Vocabulary:
    disorders: list[dict]
    role_keywords: dict[str, list[str]]
    blood_tissue_tokens: list[str]


def load_vocabulary() -> Vocabulary:
    with open(paths.repo_root() / "config" / "vocabulary.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Vocabulary(
        disorders=raw["disorders"],
        role_keywords=raw["role_keywords"],
        blood_tissue_tokens=[t.lower() for t in raw["blood_tissue_tokens"]],
    )


def _matches(text: str, alias: str) -> bool:
    """An alias is a plain substring unless it looks like a regex (contains ^, $, or \\b)."""
    if not text:
        return False
    if any(c in alias for c in "^$"):
        return re.search(alias, text) is not None
    return alias in text


def _find_disorder(row: pd.Series, vocab: Vocabulary) -> tuple[dict | None, str, float]:
    """Return (disorder_entry, matched_field, confidence) or (None, "", 0.0)."""
    for field_name in _STRUCTURED_FIELDS:
        val = str(row.get(field_name, "") or "").lower()
        if not val:
            continue
        for d in vocab.disorders:
            if any(_matches(val, a) for a in d["aliases"]):
                return d, field_name, 0.95
    for field_name in _FREETEXT_FIELDS:
        val = str(row.get(field_name, "") or "").lower()
        if not val:
            continue
        for d in vocab.disorders:
            if any(_matches(val, a) for a in d["aliases"]):
                return d, field_name, 0.80
    return None, "", 0.0


def _find_role_keyword(row: pd.Series, vocab: Vocabulary) -> tuple[str | None, str, float]:
    blob_structured = " | ".join(
        str(row.get(f, "") or "").lower() for f in _STRUCTURED_FIELDS
    )
    blob_free = " | ".join(str(row.get(f, "") or "").lower() for f in _FREETEXT_FIELDS)

    # matched_control checked before under_test: a phrase like "Control for validation cohort"
    # must resolve to matched_control, not to under_test on the "validation cohort" substring.
    for role in ("unaffected_relative", "cell_line", "matched_control", "under_test"):
        keywords = vocab.role_keywords.get(role, [])
        if any(_matches(blob_structured, kw) for kw in keywords):
            return role, "structured", 0.9
        if any(_matches(blob_free, kw) for kw in keywords):
            return role, "freetext", 0.75
    return None, "", 0.0


def _tissue_token(row: pd.Series, vocab: Vocabulary) -> str | None:
    for f in ("tissue", "source_name_ch1", "tissue/cell_type"):
        val = str(row.get(f, "") or "").lower()
        for tok in vocab.blood_tissue_tokens:
            if tok in val:
                return tok
        if val and val not in ("nan", ""):
            return val  # non-blood token, e.g. "fibroblast genomic dna"
    return None


def harmonize_row(
    row: pd.Series, vocab: Vocabulary, series_majority_is_blood: bool,
    series_has_in_scope_disorder: bool,
) -> dict:
    disorder, disorder_field, disorder_conf = _find_disorder(row, vocab)
    role_kw, role_field, role_conf = _find_role_keyword(row, vocab)

    disorder_name = disorder["canonical"] if disorder else None
    gene = disorder["gene"] if disorder else None
    in_scope = bool(disorder and disorder.get("in_scope", True))

    reason_parts = []
    # "matched_control" only means something when this row's own series actually has an
    # in-scope disorder to be matched against; otherwise "normal"/"healthy"/"control" text is
    # unselected population data (e.g. GSE87571's aging cohort, GSE42861's non-RA subjects).
    if role_kw == "matched_control" and not series_has_in_scope_disorder:
        role = "population_control"
        confidence = role_conf
        reason_parts.append(
            "'matched_control' keyword matched via " + role_field +
            ", but series has no in-scope disorder to match against -> population_control"
        )
    elif role_kw:
        role = role_kw
        confidence = role_conf
        reason_parts.append(f"role keyword '{role_kw}' matched via {role_field}")
    elif disorder is not None:
        if in_scope:
            role = "case"
        else:
            role = "exclude_other"  # matched disorder, but out of episignature scope (RA, IBD)
        confidence = disorder_conf
        reason_parts.append(f"disorder '{disorder_name}' matched via {disorder_field}")
    else:
        # No disorder, no role keyword: series with no in-scope disorder at all is treated as
        # a population reference cohort; otherwise it's genuinely unresolved.
        role = "population_control"
        confidence = 0.5
        reason_parts.append("no disorder/role keyword matched; defaulted as population data")

    # Tissue-mismatch override: applies regardless of the role decided above.
    tissue_tok = _tissue_token(row, vocab)
    is_blood = tissue_tok in vocab.blood_tissue_tokens if tissue_tok else None
    if series_majority_is_blood and is_blood is False and role not in ("cell_line",):
        reason_parts.append(f"tissue '{tissue_tok}' mismatches series-majority blood tissue")
        role = "wrong_tissue"
        confidence = min(confidence, 0.9)

    needs_review = confidence < CONFIDENCE_THRESHOLD

    return {
        "disorder": disorder_name,
        "gene": gene,
        "role": role,
        "confidence": confidence,
        "resolution_status": "auto_accepted" if not needs_review else "needs_review",
        "rule_rationale": "; ".join(reason_parts),
    }


def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    vocab = load_vocabulary()
    df = df.copy()

    # Per-series majority tissue (blood vs not), used for the wrong_tissue override.
    series_is_blood = {}
    # Per-series: does ANY sample carry an in-scope disorder match? Used to distinguish
    # matched_control (paired within a disorder study) from population_control (no disorder
    # in this series at all).
    series_has_disorder = {}
    for series_id, g in df.groupby("series_id"):
        toks = [_tissue_token(r, vocab) for _, r in g.iterrows()]
        blood_count = sum(1 for t in toks if t in vocab.blood_tissue_tokens)
        series_is_blood[series_id] = blood_count >= len(g) / 2

        has_disorder = False
        for _, r in g.iterrows():
            d, _, _ = _find_disorder(r, vocab)
            if d is not None and d.get("in_scope", True):
                has_disorder = True
                break
        series_has_disorder[series_id] = has_disorder

    results = [
        harmonize_row(
            row, vocab, series_is_blood[row["series_id"]],
            series_has_disorder[row["series_id"]],
        )
        for _, row in df.iterrows()
    ]
    result_df = pd.DataFrame(results)
    for col in result_df.columns:
        df[col] = result_df[col].values

    assert set(df["role"].unique()) <= _VALID_ROLES, f"invalid role produced: {df['role'].unique()}"
    return df


def main() -> None:
    from epigrade.acquire.llm_escalate import escalate

    samples_path = paths.interim_dir() / "samples.parquet"
    df = pd.read_parquet(samples_path)
    harmonized = harmonize(df)
    harmonized = escalate(harmonized)

    out_path = paths.interim_dir() / "samples_harmonized.parquet"
    harmonized.to_parquet(out_path, index=False)

    n_review = (harmonized["resolution_status"] == "needs_review").sum()
    print(f"Harmonized {len(harmonized)} samples -> {out_path}")
    print(f"  auto-accepted: {len(harmonized) - n_review}")
    print(f"  needs_review (escalation candidates): {n_review}")
    print("\nRole counts:")
    print(harmonized["role"].value_counts().to_string())
    print("\nDisorder counts (in-scope cases only):")
    print(
        harmonized[harmonized["role"] == "case"]["disorder"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()
