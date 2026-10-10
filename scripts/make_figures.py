"""Render every Phase 1 figure from the tables scripts/build_corpus.py and
scripts/compute_ceilings.py wrote. Reads results/tables/ only - no beta matrix, no network - so
it runs from a clean clone.

Each figure is written twice: results/figures/ (150 dpi) and results/figures/slides/ (200 dpi,
axis labels >= 14pt and titles >= 16pt, legible from the back of a room).

  <slug>_scores.png            score strip by group with the zero line, one per classifier
  cross_disorder_matrix.png    classifier x cohort heatmap; diagonal = sensitivity
  leave_one_study_out.png      cross-study sensitivity/specificity, both directions
  signature_sizes.png          probes per signature, annotated with case count
  attainable_ceiling.png       ceiling vs cohort size, each disorder as a point
  disorder_verdict_summary.png 12 disorders, four verdicts
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from epigrade import paths  # noqa: E402

# Colours by the job they do. Categorical slots are used in fixed order and were checked for
# colour-vision-deficiency separation as a set; grey is reserved for "not computed".
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#8a8984"
NOT_COMPUTED = "#dedcd6"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "epigrade_blue", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
)

BASE_RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "hatch.color": "#b9b7b0", "hatch.linewidth": 0.6,
}
SLIDE_RC = {
    **BASE_RC, "font.size": 14, "axes.titlesize": 17, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
}
VARIANTS = [("", 150, BASE_RC), ("slides", 200, SLIDE_RC)]

SHORT = {
    "Silver-Russell syndrome": "Silver-Russell", "Sotos syndrome": "Sotos",
    "Kabuki syndrome type 1": "Kabuki 1", "Down syndrome": "Down",
    "Coffin-Siris syndrome": "Coffin-Siris", "Williams syndrome": "Williams",
    "CHARGE syndrome": "CHARGE", "Nicolaides-Baraitser syndrome": "Nicolaides-Baraitser",
    "ICF syndrome": "ICF", "7q11.23 duplication syndrome": "7q11.23 dup",
    "Weaver syndrome": "Weaver", "Claes-Jensen syndrome": "Claes-Jensen",
    "pooled_control": "Pooled controls",
}


def _table(name: str) -> pd.DataFrame:
    return pd.read_csv(paths.tables_dir() / name, sep="\t", keep_default_na=False,
                       na_values=["", "NA"])


def render(name: str, draw, figsize: tuple[float, float]) -> None:
    """Draw one figure in both variants. `draw(fig, ax, scale)` gets a font scale so absolute
    sizes (markers, annotations) follow the variant."""
    for subdir, dpi, rc in VARIANTS:
        out_dir = paths.figures_dir() / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        with plt.rc_context(rc):
            scale = rc["font.size"] / BASE_RC["font.size"]
            fig, ax = plt.subplots(figsize=(figsize[0] * scale ** 0.5,
                                            figsize[1] * scale ** 0.5))
            draw(fig, ax, scale)
            fig.savefig(out_dir / name, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
    print(f"  {name}")


# --------------------------------------------------------------------------- score strips


def score_figures() -> None:
    index = pd.read_csv(paths.repo_root() / "results" / "signatures" / "index.tsv", sep="\t")
    cohorts = _table("corpus_cohorts.tsv")
    order = list(dict.fromkeys(cohorts.disorder))
    for _, sig in index.iterrows():
        scores = _table(f"classifier_scores/{sig.slug}.tsv")
        own = sig.disorder
        own_controls = scores[scores.training_role == "train_control"]
        groups = [("Training\ncontrols", own_controls, NEUTRAL),
                  (f"{SHORT[own]}\ncases", scores[scores.group == own], BLUE)]
        for other in order:
            if other != own:
                groups.append((SHORT[other], scores[scores.group == other], ORANGE))
        held_out = scores[
            scores.group.isin(["pooled_control", "comparator_sibling"])
            & (scores.training_role != "train_control")
        ]
        groups.append(("Other\ncontrols", held_out, NEUTRAL))

        def draw(fig, ax, scale, groups=groups, own=own, sig=sig):
            rng = np.random.default_rng(0)
            for pos, (_, block, colour) in enumerate(groups):
                vals = block["score"].to_numpy()
                ax.scatter(pos + rng.uniform(-0.22, 0.22, len(vals)), vals, s=16 * scale,
                           color=colour, alpha=0.75, linewidths=0.4, edgecolors=SURFACE)
                if len(vals):
                    ax.hlines(np.median(vals), pos - 0.3, pos + 0.3, color=INK, linewidth=1.6)
                    ax.annotate(f"{(vals > 0).mean():.0%}", (pos, 1.0),
                                xycoords=("data", "axes fraction"), ha="center", va="bottom",
                                fontsize=8.5 * scale, color=INK_SECONDARY)
            ax.axhline(0, color=INK, linestyle=(0, (4, 3)), linewidth=1)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([g[0] for g in groups], rotation=45, ha="right")
            ax.grid(axis="x", visible=False)
            ax.set_ylabel("Score (r to case profile - r to control profile)")
            ax.set_title(
                f"{SHORT[own]} classifier: scores by group ({sig.n_probes:,} probes)",
                pad=26 * scale,
            )
            ax.legend(handles=[
                Patch(color=BLUE, label="own disorder (training cases scored leave-one-out)"),
                Patch(color=ORANGE, label="other disorders"),
                Patch(color=NEUTRAL, label="controls"),
            ], loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3,
                title="Number above each group = percent scoring positive; bar = median; "
                      "dashed line = decision threshold")

        render(f"{sig.slug}_scores.png", draw, (11, 5.2))


# --------------------------------------------------------------------------- heatmap


def matrix_figure() -> None:
    m = _table("cross_disorder_matrix.tsv")
    rows = list(dict.fromkeys(m.classifier))
    cols = list(dict.fromkeys(m.scored_disorder))
    cell = {(r.classifier, r.scored_disorder): r for r in m.itertuples()}

    def draw(fig, ax, scale):
        ax.grid(False)
        for i, clf in enumerate(rows):
            for j, target in enumerate(cols):
                r = cell[(clf, target)]
                box = (j - 0.5, i - 0.5)
                if r.status != "computed":
                    ax.add_patch(Rectangle(box, 1, 1, facecolor=NOT_COMPUTED, hatch="////",
                                           edgecolor=SURFACE, linewidth=1.5))
                    continue
                value = r.fraction_positive
                ax.add_patch(Rectangle(box, 1, 1, facecolor=SEQUENTIAL(value),
                                       edgecolor=SURFACE, linewidth=1.5))
                too_small = isinstance(r.interpretation, str) and r.interpretation
                if too_small:
                    ax.add_patch(Rectangle(box, 1, 1, fill=False, hatch="..",
                                           edgecolor=SURFACE, linewidth=0))
                ax.text(j, i, f"{value:.2f}" + ("*" if too_small else ""), ha="center",
                        va="center", fontsize=8.5 * scale,
                        color="white" if value > 0.55 else INK)
                if clf == target:
                    ax.add_patch(Rectangle(box, 1, 1, fill=False, edgecolor=INK,
                                           linewidth=2.2))
        ax.set_xlim(-0.5, len(cols) - 0.5)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([SHORT[c] for c in cols], rotation=45, ha="right")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([SHORT[r] for r in rows])
        ax.set_xlabel("Cohort scored (cases of each disorder; pooled controls)")
        ax.set_ylabel("Classifier")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title("Cross-disorder matrix: fraction of each cohort scoring positive",
                     pad=12 * scale)
        sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL, norm=plt.Normalize(0, 1))
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("fraction positive")
        cbar.outline.set_visible(False)
        ax.legend(handles=[
            Patch(facecolor="none", edgecolor=INK, linewidth=2,
                  label="diagonal = sensitivity (high is good)"),
            Patch(facecolor=SEQUENTIAL(0.5), edgecolor=SURFACE,
                  label="off-diagonal = cross-reactivity (high is bad)"),
            Patch(facecolor=NOT_COMPUTED, hatch="////", edgecolor=SURFACE,
                  label="not computed (no classifier: < 10 cases)"),
            Patch(facecolor="none", edgecolor="none", label="* n < 10: too small to interpret"),
        ], loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2)

    render("cross_disorder_matrix.png", draw, (11.5, 8))


# --------------------------------------------------------------------------- LOSO


def loso_figure() -> None:
    loso = _table("leave_one_study_out.tsv")

    def draw(fig, ax, scale):
        width = 0.36
        labels = []
        for i, r in enumerate(loso.itertuples()):
            name = SHORT[r.disorder].replace("Nicolaides-", "Nicolaides-\n")
            labels.append(f"{name}\ntrain {r.train_study}\ntest {r.test_study}")
            if r.status != "computed":
                ax.add_patch(Rectangle((i - width, 0), 2 * width, 1, facecolor=NOT_COMPUTED,
                                       hatch="////", edgecolor=SURFACE))
                why = ("no signature:\n0 significant\nprobes"
                       if "0 significant probes" in r.reason
                       else f"not attempted:\nonly {r.n_train_cases}\ntraining cases")
                ax.text(i, 0.5, why, ha="center", va="center", fontsize=8 * scale,
                        color=INK, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 3})
                continue
            for offset, value, text, colour in (
                (-width / 2, r.sensitivity, r.test_case_sensitivity, BLUE),
                (width / 2, r.specificity, r.test_control_specificity, ORANGE),
            ):
                ax.bar(i + offset, value, width - 0.03, color=colour)
                ax.text(i + offset, value + 0.02, f"{value:.0%}\n({text})", ha="center",
                        va="bottom", fontsize=8.5 * scale, color=INK)
        ax.set_xticks(range(len(loso)))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.22)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("Performance in the held-out study")
        n_ok = int((loso.status == "computed").sum())
        ax.set_title(
            f"Leave-one-study-out: {n_ok} of {len(loso)} directions could be tested",
            pad=12 * scale,
        )
        ax.legend(handles=[
            Patch(color=BLUE, label="sensitivity (held-out cases positive)"),
            Patch(color=ORANGE, label="specificity (held-out controls negative)"),
            Patch(facecolor=NOT_COMPUTED, hatch="////", edgecolor=SURFACE,
                  label="direction not computed"),
        ], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)

    render("leave_one_study_out.png", draw, (11.5, 5.2))


# --------------------------------------------------------------------------- signature sizes


def signature_sizes_figure() -> None:
    verdicts = _table("disorder_verdicts.tsv").sort_values("n_signature_probes")

    def draw(fig, ax, scale):
        for i, r in enumerate(verdicts.itertuples()):
            if not r.signature_built:
                ax.text(12, i, f"no signature: {r.n_cases} cases (< 10)", va="center",
                        fontsize=9 * scale, color=INK_SECONDARY)
                continue
            published = r.signature_source == "published_probe_list"
            ax.barh(i, r.n_signature_probes, height=0.62, color=AQUA if published else BLUE,
                    hatch="////" if r.controls_source == "cross_series" else None,
                    edgecolor=SURFACE)
            ax.text(r.n_signature_probes * 1.12, i,
                    f"{r.n_signature_probes:,} probes  |  {r.n_cases} cases", va="center",
                    fontsize=9 * scale, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(10, 300_000)
        ax.set_ylim(-0.6, len(verdicts) - 0.4)
        ax.set_yticks(range(len(verdicts)))
        ax.set_yticklabels([SHORT[d] for d in verdicts.disorder])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Probes in signature (log scale)")
        ax.set_title("Signature size per disorder, with the case count behind it",
                     pad=12 * scale)
        ax.legend(handles=[
            Patch(color=BLUE, label="re-derived here (not published)"),
            Patch(color=AQUA, label="published probe list (Sotos only)"),
            Patch(facecolor=BLUE, hatch="////", edgecolor=SURFACE,
                  label="controls borrowed from another series"),
        ], loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=3)

    render("signature_sizes.png", draw, (11, 5.6))


# --------------------------------------------------------------------------- ceiling


# Label placement only (offset points): several cohorts sit within a few cases of each other,
# so their labels are fanned out by hand and tied back with a leader line.
CEILING_LABEL_OFFSETS = {
    "Claes-Jensen syndrome": (0, 14), "Weaver syndrome": (0, 14),
    "7q11.23 duplication syndrome": (-6, 14), "ICF syndrome": (0, -18),
    "Nicolaides-Baraitser syndrome": (-34, 46), "CHARGE syndrome": (0, 62),
    "Williams syndrome": (10, -34), "Coffin-Siris syndrome": (40, -52),
    "Down syndrome": (4, 30), "Kabuki syndrome type 1": (0, -20),
    "Sotos syndrome": (0, 14), "Silver-Russell syndrome": (0, -20),
}


def ceiling_figure() -> None:
    curve = _table("attainable_ceiling_curve.tsv")
    points = _table("attainable_ceiling.tsv")
    with open(paths.repo_root() / "config" / "evidence_bands.yaml", encoding="utf-8") as f:
        bands = [b for b in yaml.safe_load(f)["bands"] if b["min_points"] > 0]

    def draw(fig, ax, scale):
        for band in bands:
            ax.axhline(band["min_points"], color=GRID, linewidth=1.2)
            ax.text(215, band["min_points"], f" {band['name']}", va="center",
                    fontsize=9 * scale, color=INK_SECONDARY)
        styles = {"one_study": (ORANGE, "one study per side (within-study bound only)"),
                  "two_studies": (BLUE, "two studies per side (between-study bound)")}
        for structure, (colour, label) in styles.items():
            block = curve[curve.study_structure == structure].sort_values("n_case")
            ax.plot(block.n_case, block.points_conservative, color=colour, linewidth=2,
                    label=label)
        for r in points.itertuples():
            between = r.interpretation_scope == "between_study"
            ax.scatter(r.n_case, r.points_conservative, s=52 * scale,
                       color=BLUE if between else ORANGE, edgecolors=SURFACE, linewidths=1.4,
                       zorder=3)
        for r in points.itertuples():
            dx, dy = CEILING_LABEL_OFFSETS.get(r.disorder, (0, 14))
            ax.annotate(SHORT[r.disorder], (r.n_case, r.points_conservative),
                        textcoords="offset points", xytext=(dx * scale, dy * scale),
                        ha="center", fontsize=8 * scale, color=INK,
                        arrowprops={"arrowstyle": "-", "color": NEUTRAL, "linewidth": 0.6}
                        if abs(dy) > 20 else None)
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 40, 80, 200])
        ax.set_xticklabels(["5", "10", "20", "40", "80", "200"])
        ax.set_xlim(4.5, 330)
        ax.set_ylim(0, max(10, curve.points_conservative.max() + 0.8))
        ax.set_xlabel("Cases in cohort (log scale)")
        ax.set_ylabel("Evidence points attainable\nby a perfect classifier")
        ax.set_title("Attainable evidence ceiling vs cohort size", pad=12 * scale)
        ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2,
                  title="Line = simulated ceiling with equal case and control counts; "
                        "point = a real disorder at its own case and control counts")

    render("attainable_ceiling.png", draw, (10.5, 5.6))


# --------------------------------------------------------------------------- verdicts


def verdict_figure() -> None:
    verdicts = _table("disorder_verdicts.tsv").iloc[::-1]
    colours = {"graded": BLUE, "single-study": ORANGE, "too-few-cases": AQUA,
               "not-downloaded": NEUTRAL}
    meaning = {"graded": "graded: cases span two studies",
               "single-study": "single-study: confounded with batch",
               "too-few-cases": "too-few-cases: under 10, no signature",
               "not-downloaded": "not-downloaded"}

    def draw(fig, ax, scale):
        for i, r in enumerate(verdicts.itertuples()):
            ax.barh(i, r.n_cases, height=0.62, color=colours[r.verdict], edgecolor=SURFACE)
            studies = "study" if r.n_studies == 1 else "studies"
            ax.text(max(r.n_cases, 10) + 1, i, f"{r.n_cases} cases, {r.n_studies} {studies}",
                    va="center", fontsize=9 * scale, color=INK)
        ax.axvline(10, color=INK, linestyle=(0, (4, 3)), linewidth=1)
        ax.text(10, len(verdicts) - 0.35, " 10-case minimum", fontsize=8.5 * scale,
                color=INK_SECONDARY, va="bottom")
        ax.set_yticks(range(len(verdicts)))
        ax.set_yticklabels([SHORT[d] for d in verdicts.disorder])
        ax.set_xlim(0, 82)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Confirmed cases in the corpus")
        counts = verdicts.verdict.value_counts()
        ax.set_title("Twelve disorders, four verdicts", pad=12 * scale)
        ax.legend(handles=[
            Patch(color=colours[v], label=f"{meaning[v]} ({int(counts.get(v, 0))})")
            for v in colours
        ], loc="upper center", bbox_to_anchor=(0.42, -0.15), ncol=2)

    render("disorder_verdict_summary.png", draw, (10.5, 6))


def main() -> None:
    print("Rendering figures...")
    score_figures()
    matrix_figure()
    loso_figure()
    signature_sizes_figure()
    verdict_figure()
    if (paths.tables_dir() / "attainable_ceiling_curve.tsv").exists():
        ceiling_figure()
    else:
        print("  attainable_ceiling.png skipped: run scripts/compute_ceilings.py first")


if __name__ == "__main__":
    main()
