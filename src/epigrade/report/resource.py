"""Builds the single versioned resource (results/resource/epigrade_v1.json + a TSV mirror) that
the Streamlit app reads from. The app never recomputes anything - everything it can show must
already be in this file, produced by a script, never hand-edited.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone

import pandas as pd

from epigrade import paths

RESOURCE_VERSION = "epigrade_v1"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=paths.repo_root(), text=True,
        ).strip()
    except Exception:  # noqa: BLE001 - resource export must not hard-fail if git is unavailable
        return "unknown"


def _config_hash() -> str:
    h = hashlib.sha256()
    for name in ["paths.yaml", "vocabulary.yaml", "evidence_bands.yaml"]:
        p = paths.repo_root() / "config" / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _read_tsv(name: str) -> list[dict]:
    p = paths.tables_dir() / name
    if not p.exists():
        return []
    # keep_default_na=False + na_values=[""] : pandas' default NA-string list includes the
    # literal text "NA", which silently turns our own band="NA" refusal (Task 1) into a null
    # value indistinguishable from a formatting glitch - caught by the Task 2 clean-clone check,
    # where the resource JSON came out with "band": NaN (not even valid JSON) instead of the
    # string "NA". Only genuinely empty cells (how a NaN float is actually written by to_csv)
    # should become null; the text "NA" must survive as text.
    df = pd.read_csv(p, sep="\t", keep_default_na=False, na_values=[""])
    return df.to_dict(orient="records")


def build_resource() -> dict:
    confounding = _read_tsv("confounding_gate.tsv")
    confounding_by_disorder = {row["disorder"]: row for row in confounding}

    evidence = _read_tsv("evidence_bands.tsv")
    ceiling = _read_tsv("attainable_ceiling.tsv")
    cross_matrix = _read_tsv("cross_disorder_matrix.tsv")
    cross_matrix_pending = _read_tsv("cross_disorder_matrix_not_computed.tsv")
    sotos_summary = _read_tsv("sotos_reproduction_summary.tsv")
    sotos_scores = _read_tsv("sotos_scores.tsv")
    probe_filtering = _read_tsv("probe_filtering.tsv")
    loso = _read_tsv("srs_leave_one_study_out.tsv")
    triage = _read_tsv("sample_triage.tsv")
    agreement = _read_tsv("harmonisation_agreement.tsv")

    scoreable_agreement = [r for r in agreement if r.get("agree") is not None]
    agreement_rate = (
        sum(1 for r in scoreable_agreement if r["agree"]) / len(scoreable_agreement)
        if scoreable_agreement else None
    )

    resource = {
        "resource_version": RESOURCE_VERSION,
        "generated_from": {
            "git_sha": _git_sha(),
            "config_hash": _config_hash(),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "disclaimer": (
            "Research/benchmarking tool for laboratories and researchers. Not a diagnostic "
            "device. Never accepts patient data, raw arrays, or issues a diagnosis."
        ),
        "confounding_gate": confounding,
        "demo": {
            "disorder": "Sotos syndrome",
            "reproduction_summary": sotos_summary,
            "scores": sotos_scores,
            "probe_filtering": probe_filtering[0] if probe_filtering else None,
            "confounding": confounding_by_disorder.get("Sotos syndrome"),
        },
        "evidence_bands": evidence,
        "attainable_ceiling": ceiling,
        "cross_disorder_matrix": {
            "computed": cross_matrix,
            "not_computed": cross_matrix_pending,
        },
        "leave_one_study_out": {
            "silver_russell_syndrome": loso,
        },
        "harmonisation": {
            "n_samples_total": len({r["gsm_accession"] for r in triage}) if triage else None,
            "n_flagged_or_excluded": len(triage),
            "hand_curation_agreement_rate": agreement_rate,
            "hand_curation_n_scoreable": len(scoreable_agreement),
            "hand_curation_note": (
                "AI-performed cross-check against a second independent implementation of the "
                "same judgment, not independent human clinical curation - see docs/METHODS.md."
            ),
        },
    }
    return resource


def _sanitize_nans(obj):
    """Recursively replace float NaN with None (JSON null). Plain json.dump happily emits the
    non-standard `NaN` token for a float NaN, which is not valid JSON per the spec and would
    fail to parse in a strict JSON consumer (e.g. most non-Python languages) - a real
    reproducibility concern for a file meant to be portable. `allow_nan=False` below then
    asserts this sanitizer actually caught everything, rather than silently emitting bad JSON
    again if some other NaN sneaks in later.
    """
    if isinstance(obj, float) and obj != obj:  # NaN != NaN is the standard float NaN check
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nans(v) for v in obj]
    return obj


def main() -> None:
    resource = _sanitize_nans(build_resource())
    out_dir = paths.resource_dir()

    json_path = out_dir / f"{RESOURCE_VERSION}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resource, f, indent=2, default=str, allow_nan=False)
    print(f"Wrote {json_path}")

    # Flat TSV mirror of the top-level confounding gate + demo summary, for anyone who'd rather
    # grep a table than parse JSON.
    flat_rows = []
    for row in resource["confounding_gate"]:
        flat_rows.append({"section": "confounding_gate", **row})
    for row in resource["demo"]["reproduction_summary"]:
        flat_rows.append({"section": "sotos_demo", **row})
    tsv_path = out_dir / f"{RESOURCE_VERSION}.tsv"
    pd.DataFrame(flat_rows).to_csv(tsv_path, sep="\t", index=False)
    print(f"Wrote {tsv_path}")


if __name__ == "__main__":
    main()
