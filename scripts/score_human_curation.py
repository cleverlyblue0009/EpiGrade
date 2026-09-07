"""Task 5: reads data/external/human_curated.csv once a real human has filled in verdicts, and
reports the human-vs-pipeline agreement rate SEPARATELY from the AI-vs-pipeline agreement rate
in results/tables/harmonisation_agreement.tsv (scripts/phase1_report.py). These two numbers must
never be merged into one figure - one is an independent human judgment, the other is an AI cross-
check against a second implementation of its own logic (see docs/METHODS.md for why that
distinction matters).

If the file is unfilled (or missing), this reports "human curation pending" rather than falling
back to the AI-vs-pipeline number - the two are not interchangeable.
"""

from __future__ import annotations

import pandas as pd

from epigrade import paths

IN_NAME = "human_curated.csv"


def main() -> None:
    in_path = paths.external_dir() / IN_NAME
    out_path = paths.tables_dir() / "human_curation_agreement.tsv"

    if not in_path.exists():
        print(f"{in_path} does not exist yet. Run "
              "scripts/generate_human_curation_template.py first, then have a human fill it in.")
        pd.DataFrame([{
            "status": "human curation pending",
            "reason": "template not yet generated",
        }]).to_csv(out_path, sep="\t", index=False)
        return

    df = pd.read_csv(in_path, dtype=str).fillna("")
    filled = df[(df["human_role"].str.strip() != "") | (df["human_disorder"].str.strip() != "")]

    if len(filled) == 0:
        print(f"{in_path} exists but no rows have a human_role/human_disorder filled in yet.")
        pd.DataFrame([{
            "status": "human curation pending",
            "reason": f"template generated ({len(df)} rows) but not yet filled in by a curator",
        }]).to_csv(out_path, sep="\t", index=False)
        return

    def _agree(row) -> bool | None:
        human_role = row["human_role"].strip()
        if not human_role:
            return None
        role_match = human_role == row["pipeline_role"].strip()
        human_disorder = row["human_disorder"].strip()
        disorder_match = (
            human_disorder == "" or human_disorder.lower() == "n/a"
            or human_disorder == str(row["pipeline_disorder"]).strip()
        )
        return role_match and disorder_match

    filled = filled.copy()
    filled["agree"] = filled.apply(_agree, axis=1)
    scoreable = filled[filled["agree"].notna()]

    overall_rate = scoreable["agree"].mean() if len(scoreable) else float("nan")
    per_stratum = (
        scoreable.groupby("stratum")["agree"].agg(["mean", "count"])
        .rename(columns={"mean": "agreement_rate", "count": "n"})
        if "stratum" in scoreable.columns else None
    )

    scoreable.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(scoreable)} human-vs-pipeline comparisons -> {out_path}")
    print(f"Filled: {len(filled)}/{len(df)} rows; scoreable: {len(scoreable)}")
    print(f"HUMAN-vs-pipeline agreement rate: {overall_rate:.1%}"
          if not pd.isna(overall_rate) else "HUMAN-vs-pipeline agreement rate: NA")
    if per_stratum is not None:
        print("\nPer-stratum:")
        print(per_stratum.to_string())
    print("\nNote: this is DELIBERATELY reported separately from the AI-vs-pipeline agreement "
          "rate in results/tables/harmonisation_agreement.tsv - see docs/METHODS.md. Do not "
          "average or otherwise combine the two numbers.")


if __name__ == "__main__":
    main()
