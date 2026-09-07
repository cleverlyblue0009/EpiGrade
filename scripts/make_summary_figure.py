"""Task 6: one summary figure - confirmed case counts per disorder, with the attainable
evidence ceiling (from scripts/phase5_calibration.py's simulation) annotated on each bar, and
single-study cohorts (confounded by design - see the confounding gate) hatched so the reader
sees at a glance which bars are trustworthy at all before looking at their height.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from epigrade import paths  # noqa: E402

RC = {
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 13,
}


def _nearest_ceiling(ceiling_df: pd.DataFrame, n_case: int) -> pd.Series:
    """CAVEAT: attainable_ceiling.tsv is a generic illustrative curve (n_studies_case assumed
    >=2 only for n>=30 - see phase5_calibration.py), not a per-disorder simulation using each
    disorder's ACTUAL study count. Nicolaides-Baraitser syndrome (n=16) is structurally
    multi-study per the real confounding gate but still shows ceiling=NA here, because the
    generic curve assumes single-study below n=30. Good enough for an illustrative summary
    figure, not precise enough to read a specific disorder's ceiling off directly - use
    attainable_ceiling.tsv's own n_studies_case_assumed column if you need to know why."""
    idx = (ceiling_df["n_case"] - n_case).abs().idxmin()
    return ceiling_df.loc[idx]


def main() -> None:
    gate = pd.read_csv(paths.tables_dir() / "confounding_gate.tsv", sep="\t")
    ceiling = pd.read_csv(paths.tables_dir() / "attainable_ceiling.tsv", sep="\t")

    gate = gate.sort_values("n_cases", ascending=False).reset_index(drop=True)
    ceilings = [_nearest_ceiling(ceiling, n) for n in gate["n_cases"]]
    gate["ceiling_band"] = [c["attainable_band"] for c in ceilings]
    gate["ceiling_n_match"] = [c["n_case"] for c in ceilings]

    targets = [(paths.figures_dir(), 150, {}), (paths.figures_dir() / "slides", 200, RC)]
    for out_dir, dpi, rc in targets:
        out_dir.mkdir(parents=True, exist_ok=True)
        with plt.rc_context(rc):
            fig, ax = plt.subplots(figsize=(11, 6))
            colors = {"fail": "#d9534f", "not-testable": "#f0ad4e", "pass-structural": "#5cb85c"}
            bar_colors = [colors[s] for s in gate["status"]]
            bars = ax.bar(
                gate["disorder"], gate["n_cases"], color=bar_colors, edgecolor="black",
            )
            for bar, confounded in zip(bars, gate["confounded_by_design"]):
                if confounded:
                    bar.set_hatch("///")

            ceiling_pairs = zip(bars, gate["ceiling_band"], gate["ceiling_n_match"])
            for bar, ceiling_band, n_match in ceiling_pairs:
                label = ceiling_band if pd.notna(ceiling_band) else "NA"
                ax.annotate(
                    f"ceiling:\n{label}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=rc.get("font.size", 9) - 2 if rc else 8,
                )

            ax.set_ylabel("Confirmed cases (role=case)")
            ax.set_title(
                "Confirmed case counts by disorder, with attainable evidence ceiling\n"
                "(hatched = single-study, confounded by design)"
            )
            ax.set_ylim(0, gate["n_cases"].max() * 1.15)  # headroom for the top annotation
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

            from matplotlib.patches import Patch
            legend_handles = [
                Patch(facecolor=colors["fail"], edgecolor="black",
                      label="fails confounding gate"),
                Patch(facecolor=colors["not-testable"], edgecolor="black",
                      label="not-testable (n<10)"),
                Patch(facecolor=colors["pass-structural"], edgecolor="black",
                      label="passes gate (multi-study)"),
                Patch(facecolor="white", edgecolor="black", hatch="///",
                      label="confounded by design (single study)"),
            ]
            ax.legend(handles=legend_handles, loc="upper right", fontsize=9 if not rc else 11)

            fig.tight_layout()
            fig.savefig(out_dir / "disorder_case_counts_summary.png", dpi=dpi)
            plt.close(fig)
        print(f"Wrote disorder_case_counts_summary.png -> {out_dir} (dpi={dpi})")


if __name__ == "__main__":
    main()
