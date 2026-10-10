"""Evaluate the explanation layer: generate a fresh explanation for each of the five BYO
fixtures and all 12 disorders, and record how often the model's output passed the validator
first time, needed its one retry, or was thrown away in favour of the template - and why.

Writes:
  results/tables/explanation_eval.tsv          one row per subject
  results/tables/explanation_eval_summary.tsv  counts per outcome and per rejection reason
and refreshes results/cache/explanations/ with the accepted outputs, so the app can show them
without a network call.

Run with GROQ_API_KEY set. Without it every subject falls back to the template and the summary
says so (model_available = False) - those counts describe the fallback, not the model.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter

import pandas as pd

from epigrade import paths
from epigrade.byo import store
from epigrade.byo.evaluate import evaluate_file
from epigrade.corpus import registry
from epigrade.explain import groq_client
from epigrade.explain.explain import SOURCE_MODEL, explain
from epigrade.explain.facts import facts_for_byo, facts_for_disorder

FIXTURES = ["sig_sotos_true.csv", "sig_random.csv", "sig_sotos_flipped.csv", "sig_tiny.csv",
            "sig_low_coverage.csv"]


def subjects() -> list[tuple[str, str, dict]]:
    source = store.load_beta_source(prefer_full=False)
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in FIXTURES:
            evaluation = evaluate_file(paths.repo_root() / "tests" / "fixtures" / name,
                                       source=source)
            out.append(("byo", name, facts_for_byo(evaluation)))
    for disorder in registry.DISORDERS:
        out.append(("disorder", disorder.name, facts_for_disorder(disorder.name)))
    return out


def outcome(result) -> str:
    if result.source == SOURCE_MODEL:
        return "passed_first_time" if result.attempts == 1 else "passed_after_retry"
    if result.rejections:
        return "fell_back_after_two_rejections"
    return f"fell_back_{result.fallback_reason}"


def reason_category(reason: str) -> str:
    return re.split(r"[:']", reason)[0].strip()


def main() -> None:
    rows, reasons = [], Counter()
    model = groq_client.config()["model"]
    available = groq_client.has_key()
    for kind, subject, facts in subjects():
        result = explain(facts, use_cache=False)
        for rejection in result.rejections:
            for part in rejection.split("; "):
                reasons[reason_category(part)] += 1
        rows.append({
            "kind": kind, "subject": subject, "fact_hash": result.fact_hash,
            "outcome": outcome(result), "source": result.source,
            "model": result.model or "", "model_calls": result.attempts,
            "n_rejections": len(result.rejections),
            "rejection_reasons": " | ".join(result.rejections),
            "text": result.text,
        })
        print(f"  {subject}: {rows[-1]['outcome']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(paths.tables_dir() / "explanation_eval.tsv", sep="\t", index=False)
    counts = df.outcome.value_counts()
    summary = [{"measure": "model_available", "value": available},
               {"measure": "model", "value": model},
               {"measure": "n_subjects", "value": len(df)}]
    for key in ("passed_first_time", "passed_after_retry", "fell_back_after_two_rejections"):
        summary.append({"measure": key, "value": int(counts.get(key, 0))})
    for key, n in counts.items():
        if key.startswith("fell_back_") and key != "fell_back_after_two_rejections":
            summary.append({"measure": key, "value": int(n)})
    summary.append({"measure": "model_outputs_rejected_total",
                    "value": int(df.n_rejections.sum())})
    for reason, n in reasons.most_common():
        summary.append({"measure": f"rejection_reason: {reason}", "value": n})
    pd.DataFrame(summary).to_csv(paths.tables_dir() / "explanation_eval_summary.tsv", sep="\t",
                                 index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
