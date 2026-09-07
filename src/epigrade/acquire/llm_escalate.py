"""LLM escalation for ambiguous GEO sample labels (phase 1 confidence router, low branch).

Rows the rule-based harmonizer left at resolution_status="needs_review" are grouped by a
canonicalized metadata signature (same series + same characteristics pattern with per-sample
identifiers stripped) so that, e.g., 586 near-identical "Proband###" rows in one series cost one
LLM call, not 586. Each group's representative blob is sent to Gemini with a JSON schema asking
for exactly the harmonizer's own output shape, so a resolved group is written back the same way
a high-confidence rule match would have been.

If GEMINI_API_KEY is not set, or a call fails, the affected rows are left as
resolution_status="needs_review" with an explicit reason - never guessed. This module must not
be the only place a role gets decided silently.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

import pandas as pd

logger = logging.getLogger(__name__)

_VALID_ROLES = [
    "case", "matched_control", "population_control", "unaffected_relative",
    "under_test", "cell_line", "wrong_tissue", "exclude_other",
]

MODEL_NAME = os.environ.get("EPIGRADE_GEMINI_MODEL", "gemini-2.5-flash-lite")

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "disorder": {"type": "string", "nullable": True},
        "gene": {"type": "string", "nullable": True},
        "role": {"type": "string", "enum": _VALID_ROLES},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["role", "confidence", "rationale"],
}

_PROMPT_TEMPLATE = """You are triaging a DNA methylation array sample from NCBI GEO for a \
reproducibility benchmark of episignature classifiers. Given the sample's raw metadata below, \
decide:

- disorder: the specific genetic disorder/syndrome this sample's methylation profile is \
  associated with, if the metadata clearly states or strongly implies one. Use a specific \
  syndrome name, not a gene name alone. Use null if the metadata gives no diagnosis at all - \
  do NOT invent a diagnosis from a bare identifier like "Proband123".
- gene: the causal gene/locus if stated or well known for the named disorder, else null.
- role: exactly one of {roles}.
  - "case": a confirmed/diagnosed affected individual.
  - "matched_control": a control specifically recruited for this disorder's study.
  - "population_control": general population data with no disorder relevance (e.g. a smoking,
    aging, or cell-type-reference study).
  - "unaffected_relative": a parent/sibling/relative of a case, not themselves diagnosed.
  - "under_test": a variant of uncertain significance or a sample explicitly being classified/
    validated, not a confirmed case.
  - "cell_line": an immortalized cell line (lymphoblastoid, etc.), not primary tissue.
  - "wrong_tissue": tissue clearly mismatched from the rest of its study (e.g. fibroblast among
    a whole-blood cohort).
  - "exclude_other": anything else that shouldn't be pooled as a case or a clean control
    (including an unrelated disease with no bearing on any episignature disorder).
- confidence: your confidence in this specific role assignment, 0 to 1.
- rationale: one sentence, citing the specific metadata field(s) that justified your answer.

If the metadata is too sparse to support a real judgment (e.g. only a bare sample ID), say so
plainly in the rationale and set role to "exclude_other" with confidence below 0.5 rather than
guessing a specific disorder.

Series accession: {series_id}
Platform: {platform_id}
Sample metadata (one representative of {n_samples} near-identical samples in this series):
{blob}
"""


def _canonical_signature(row: pd.Series) -> str:
    """Strip numeric/individual identifiers so near-duplicate rows share one signature."""
    parts = [
        str(row.get("series_id", "")),
        re.sub(r"\d+", "#", str(row.get("characteristics_raw", "") or "")),
        re.sub(r"\d+", "#", str(row.get("title", "") or "")),
        str(row.get("source_name_ch1", "") or ""),
    ]
    return " || ".join(parts)


def _build_blob(row: pd.Series) -> str:
    fields = [
        "title", "source_name_ch1", "tissue", "characteristics_raw",
        "disease_state", "group", "genotype/variation", "case_status", "phenotype",
        "simplified_diagnosis", "sample_type", "variant_classification",
    ]
    lines = [f"  {f}: {row[f]}" for f in fields if f in row.index and pd.notna(row[f])]
    return "\n".join(lines)


def _call_gemini(client, series_id, platform_id, blob, n_samples) -> dict | None:
    prompt = _PROMPT_TEMPLATE.format(
        roles=", ".join(_VALID_ROLES), series_id=series_id, platform_id=platform_id,
        blob=blob, n_samples=n_samples,
    )
    from google.genai import types

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
            temperature=0.0,
        ),
    )
    return json.loads(response.text)


def escalate(df: pd.DataFrame, max_calls: int | None = None) -> pd.DataFrame:
    """Attempt to resolve needs_review rows via Gemini. Returns the updated dataframe."""
    df = df.copy()
    review_mask = df["resolution_status"] == "needs_review"
    n_review = review_mask.sum()
    if n_review == 0:
        logger.info("No rows need escalation.")
        return df

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning(
            "GEMINI_API_KEY not set - %d rows across the collapsed groups below stay "
            "needs_review (unresolved). No labels were guessed.", n_review,
        )
        df.loc[review_mask, "rule_rationale"] = df.loc[review_mask, "rule_rationale"] + \
            " | escalation skipped: GEMINI_API_KEY not set"
        return df

    from google import genai

    client = genai.Client(api_key=api_key)

    review_df = df[review_mask].copy()
    review_df["_sig"] = review_df.apply(_canonical_signature, axis=1)
    groups = review_df.groupby("_sig")
    logger.info(
        "Escalating %d needs_review rows collapsed into %d distinct metadata signatures.",
        n_review, groups.ngroups,
    )

    resolved_calls = 0
    for i, (_sig, group) in enumerate(groups):
        if max_calls is not None and i >= max_calls:
            logger.info("Hit max_calls=%d, leaving remaining groups unresolved this run.",
                        max_calls)
            break
        rep = group.iloc[0]
        try:
            result = _call_gemini(
                client, rep["series_id"], rep.get("platform_id", ""),
                _build_blob(rep), len(group),
            )
        except Exception as exc:  # noqa: BLE001 - one failed call must not sink the whole run
            logger.warning("Gemini call failed for signature in %s: %s", rep["series_id"], exc)
            continue

        if result["role"] not in _VALID_ROLES:
            logger.warning("Gemini returned invalid role %r, skipping group.", result["role"])
            continue

        idx = group.index
        df.loc[idx, "disorder"] = result.get("disorder")
        df.loc[idx, "gene"] = result.get("gene")
        df.loc[idx, "role"] = result["role"]
        df.loc[idx, "confidence"] = float(result["confidence"])
        df.loc[idx, "resolution_status"] = (
            "llm_resolved" if result["confidence"] >= 0.5 else "needs_review"
        )
        df.loc[idx, "rule_rationale"] = f"gemini({MODEL_NAME}): {result['rationale']}"
        resolved_calls += 1
        time.sleep(0.2)  # stay well under free-tier RPM

    logger.info("Escalation done: %d/%d groups resolved via Gemini.", resolved_calls,
                groups.ngroups)
    return df
