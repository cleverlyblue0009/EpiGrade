"""Task 3: merges cross_disorder_matrix.tsv and cross_disorder_matrix_not_computed.tsv into a
single table with a `status` column ("computed" / "not_computed") and a reason for every
non-computed cell - replacing the two-file convention used while this matrix was being built up
incrementally. Also renders results/figures/cross_disorder_matrix.png: computed cells coloured
by fraction_positive, non-computed cells greyed and hatched. Diagonal (self) reads as
sensitivity; off-diagonal reads as cross-reactivity/specificity.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from epigrade import paths  # noqa: E402

MIN_N_TO_INTERPRET = 10


def main() -> None:
    computed_path = paths.tables_dir() / "cross_disorder_matrix.tsv"
    not_computed_path = paths.tables_dir() / "cross_disorder_matrix_not_computed.tsv"

    computed = pd.read_csv(computed_path, sep="\t") if computed_path.exists() else pd.DataFrame()
    not_computed = (
        pd.read_csv(not_computed_path, sep="\t") if not_computed_path.exists() else pd.DataFrame()
    )

    computed = computed.copy()
    # Idempotency bug fix: re-running this script previously took its OWN merged output (which
    # already has a status column, including genuine "not_computed" rows) as the "computed"
    # input and blindly overwrote status="computed" on all of it - silently relabeling honest
    # NA rows as computed. Only default missing status to "computed" now; never overwrite a
    # status a prior run already set.
    if "status" not in computed.columns:
        computed["status"] = "computed"
    else:
        computed["status"] = computed["status"].fillna("computed")
    if "reason" not in computed.columns:
        computed["reason"] = computed.get("note", "")

    not_computed = not_computed.copy()
    if len(not_computed):
        not_computed["status"] = "not_computed"
        if "reason" not in not_computed.columns:
            not_computed["reason"] = not_computed.get("note", "")
        if "fraction_positive" not in not_computed.columns:
            not_computed["fraction_positive"] = None
        if "n" not in not_computed.columns:
            not_computed["n"] = None
        if "signature_source" not in not_computed.columns:
            not_computed["signature_source"] = None

    all_cols = ["classifier", "scored_disorder", "status", "n", "fraction_positive",
                "signature_source", "reason"]
    for df in (computed, not_computed):
        for col in all_cols:
            if col not in df.columns:
                df[col] = None

    merged = pd.concat([computed[all_cols], not_computed[all_cols]], ignore_index=True)
    merged = merged.drop_duplicates(subset=["classifier", "scored_disorder"], keep="first")
    merged.to_csv(computed_path, sep="\t", index=False)
    print(f"Wrote merged matrix ({len(merged)} rows: "
          f"{(merged.status == 'computed').sum()} computed, "
          f"{(merged.status == 'not_computed').sum()} not_computed) -> {computed_path}")

    if not_computed_path.exists():
        not_computed_path.unlink()
        print(f"Removed {not_computed_path} - one unified table now, per Task 3.")

    render_heatmap(merged)


def render_heatmap(merged: pd.DataFrame) -> None:
    # Build an N x N grid over every disorder that appears as either a classifier or a scored
    # target (excluding "(self)"/"pooled_control(...)" labels, which get their own annotation
    # rather than a grid cell).
    def _clean_target(name: str) -> str | None:
        if name is None:
            return None
        if "(self)" in name or name.startswith("pooled_control") or name == "NOT COMPUTED":
            return None
        return name

    merged = merged.copy()
    merged["target_disorder"] = merged["scored_disorder"].map(_clean_target)
    grid_rows = sorted(merged["classifier"].dropna().unique())
    grid_cols = sorted(
        set(grid_rows) | set(merged["target_disorder"].dropna().unique())
    )

    n_rows, n_cols = len(grid_rows), len(grid_cols)
    values = np.full((n_rows, n_cols), np.nan)
    computed_mask = np.zeros((n_rows, n_cols), dtype=bool)

    for i, clf in enumerate(grid_rows):
        for j, target in enumerate(grid_cols):
            if clf == target:
                self_label = f"{clf} (self)"
                row = merged[
                    (merged.classifier == clf) & (merged.scored_disorder == self_label)
                ]
            else:
                row = merged[(merged.classifier == clf) & (merged.target_disorder == target)]
            is_computed = len(row) and row.iloc[0]["status"] == "computed"
            has_value = is_computed and pd.notna(row.iloc[0]["fraction_positive"])
            if has_value:
                n_val = row.iloc[0]["n"]
                if pd.notna(n_val) and n_val < MIN_N_TO_INTERPRET:
                    continue  # too small to interpret - leave as not-computed/grey
                values[i, j] = row.iloc[0]["fraction_positive"]
                computed_mask[i, j] = True

    fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.3), max(5, n_rows * 1.1)))
    cmap = plt.get_cmap("RdYlGn_r").copy()
    im = ax.imshow(np.where(computed_mask, values, np.nan), cmap=cmap, vmin=0, vmax=1)

    # Grey + hatch every non-computed cell.
    for i in range(n_rows):
        for j in range(n_cols):
            if not computed_mask[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, facecolor="#dddddd", hatch="//",
                    edgecolor="white",
                ))
            else:
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=9)
                if grid_rows[i] == grid_cols[j]:
                    # Diagonal (self/sensitivity) cells use the SAME red=high colour scale as
                    # off-diagonal (cross-reactivity) cells, where high is bad - a naive read
                    # would misinterpret a dark, "alarming"-looking 1.00 self-sensitivity cell
                    # as a bad result. Outlined in black and separately labeled below rather
                    # than given a different colour scale, to keep one consistent legend.
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.5,
                    ))

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(grid_cols, rotation=40, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(grid_rows)
    ax.set_xlabel("scored disorder")
    ax.set_ylabel("classifier")
    ax.set_title(
        "Cross-disorder specificity matrix\n"
        "black-outlined = diagonal/self (high = good, sensitivity) | "
        "plain = off-diagonal (high = bad, cross-reactivity) | grey/hatched = not computed"
    )
    fig.colorbar(im, ax=ax, label="fraction scoring positive")
    fig.tight_layout()

    for out_dir, dpi in [(paths.figures_dir(), 150), (paths.figures_dir() / "slides", 200)]:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "cross_disorder_matrix.png", dpi=dpi)
    plt.close(fig)
    print(f"Wrote cross_disorder_matrix.png -> {paths.figures_dir()} (+ slides/)")


if __name__ == "__main__":
    main()
