"""Task 5: generates data/external/human_curated.csv, a template for a REAL human curator to
fill in by hand. This script only builds the harness - it never fills in a verdict itself.

Sampling is deliberately weighted toward the hard cases (ambiguous roles: needs_review,
unaffected_relative, under_test, wrong_tissue, exclude_other), not the easy, obvious controls -
an agreement rate computed only on easy cases would be meaningless. A handful of easy cases are
included too, as a sanity baseline.
"""

from __future__ import annotations

import pandas as pd

from epigrade import paths
from epigrade.acquire.harmonize import harmonize

RANDOM_SEED = 2024
OUT_NAME = "human_curated.csv"

# (role or resolution_status filter, n to sample) - hard cases first, weighted heavily.
STRATA = [
    ("needs_review", 10),           # the pipeline itself wasn't confident here
    ("unaffected_relative", 5),
    ("under_test", 4),
    ("wrong_tissue", 3),             # only 7 exist total in the whole corpus
    ("exclude_other", 4),
    ("case_easy", 2),                # baseline: pipeline was confident and it's plausible
    ("population_control_easy", 2),
]


def _context_blob(row: pd.Series) -> str:
    fields = ["title", "source_name_ch1", "characteristics_raw"]
    parts = [str(row[f]) for f in fields if pd.notna(row.get(f))]
    return " | ".join(parts)[:300]


def main() -> None:
    samples = pd.read_parquet(paths.interim_dir() / "samples.parquet")
    harmonized = harmonize(samples)

    picked = []
    for stratum, n in STRATA:
        if stratum == "needs_review":
            pool = harmonized[harmonized.resolution_status == "needs_review"]
        elif stratum == "case_easy":
            pool = harmonized[
                (harmonized.role == "case") & (harmonized.resolution_status == "auto_accepted")
            ]
        elif stratum == "population_control_easy":
            pool = harmonized[
                (harmonized.role == "population_control")
                & (harmonized.resolution_status == "auto_accepted")
            ]
        else:
            pool = harmonized[harmonized.role == stratum]

        n = min(n, len(pool))
        sample = pool.sample(n=n, random_state=RANDOM_SEED)
        sample = sample.copy()
        sample["stratum"] = stratum
        picked.append(sample)

    combined = pd.concat(picked, ignore_index=True)
    combined = combined.drop_duplicates(subset="gsm_accession").head(30)

    rows = []
    for _, row in combined.iterrows():
        rows.append({
            "gsm_accession": row["gsm_accession"],
            "series_id": row["series_id"],
            "stratum": row["stratum"],
            "context": _context_blob(row),
            "pipeline_role": row["role"],
            "pipeline_disorder": row["disorder"],
            "pipeline_confidence": row["confidence"],
            # --- the human curator fills in everything from here down ---
            "human_role": "",
            "human_disorder": "",
            "curator": "",
            "notes": "",
        })

    out_path = paths.external_dir() / OUT_NAME
    if out_path.exists():
        print(f"{out_path} already exists - not overwriting (it may already have real "
              "curator input). Delete it manually first if you want a fresh template.")
        return

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)}-row human curation template -> {out_path}")
    print("Columns human_role, human_disorder, curator, notes are blank - fill by hand, then "
          "run scripts/score_human_curation.py")
    print("\nStratum counts in this template:")
    print(pd.DataFrame(rows)["stratum"].value_counts().to_string())


if __name__ == "__main__":
    main()
