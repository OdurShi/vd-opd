#!/usr/bin/env python
"""Draw figures from the supplementary experiment tables (``results/supplementary``).

    python scripts/plot_supplementary.py                 # all figures -> assets/supplementary/
    python scripts/plot_supplementary.py --only e07 e17  # a subset
    python scripts/plot_supplementary.py --list

Every figure is drawn from the CSV files through ``vdopd.supplementary``; nothing is
hard-coded here, so the figures follow the tables when the numbers are updated. Requires
matplotlib (``pip install -e ".[plots]"``).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from vdopd.supplementary import BENCHMARKS, Index, load_index, macro_average

REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPOSITORY / "assets" / "supplementary"

# Chart surface and ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Series colours: two categorical hues, a lighter step of the first, a de-emphasis grey.
BLUE = "#2a78d6"
BLUE_LIGHT = "#86b6ef"
ORANGE = "#eb6834"
GREY = MUTED

STUDENTS = (("E02", "Qwen3-1.7B"), ("E03", "Qwen3-4B"), ("E04", "Llama-3.2-3B"))


def _style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def _frame(ax, value_axis: str = "x") -> None:
    """Recessive chrome: hairline grid on the value axis, no box, no tick marks."""
    for side in ("top", "right", "left" if value_axis == "x" else "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis=value_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _dot(ax, x, y, color, size: float = 7.0, **kwargs) -> None:
    """A filled marker with a ring in the surface colour."""
    ax.plot(x, y, "o", ms=size, mfc=color, mec=SURFACE, mew=1.2, zorder=3, clip_on=False, **kwargs)


def _titles(fig, title: str, subtitle: str) -> None:
    """Title and subtitle in a header band of fixed height, then lay out the panels below."""
    height = fig.get_figheight()
    fig.suptitle(
        title, x=0.01, y=1 - 0.07 / height, ha="left", va="top", fontsize=10.5, fontweight="bold"
    )
    fig.text(
        0.01, 1 - 0.33 / height, subtitle, ha="left", va="top", fontsize=8.5, color=INK_SECONDARY
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.58 / height))


def plot_e01(index: Index, plt):
    """E01: gain of VD-OPD over EOPD(full) per student and benchmark, with 95% CIs."""
    table = index["E01"]
    students = list(dict.fromkeys(row["student"] for row in table.rows))
    fig, axes = plt.subplots(1, len(students), figsize=(9.2, 3.1), sharex=True, sharey=True)
    for ax, student in zip(axes, students, strict=True):
        rows = [row for row in table.rows if row["student"] == student]
        for position, row in enumerate(rows):
            gain = table.number(row, "vd_gain")
            low, high = table.number(row, "vd_gain_ci_low"), table.number(row, "vd_gain_ci_high")
            color = BLUE if low > 0 else GREY
            ax.hlines(position, low, high, color=color, linewidth=2, capstyle="round", zorder=2)
            _dot(ax, gain, position, color)
        ax.axvline(0, color=AXIS, linewidth=1, zorder=1)
        ax.set_yticks(range(len(rows)), [row["benchmark"] for row in rows])
        ax.set_title(student)
        ax.set_xlabel("Gain in Avg@8 (percentage points)")
        _frame(ax)
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([], [], color=BLUE, marker="o", mec=SURFACE, lw=2, label="95% CI excludes zero"),
        plt.Line2D([], [], color=GREY, marker="o", mec=SURFACE, lw=2, label="95% CI spans zero"),
    ]
    fig.legend(handles=handles, loc="upper right", ncols=2, bbox_to_anchor=(0.995, 1.0))
    _titles(
        fig,
        "Gain of VD-OPD over EOPD(full), three students (E01)",
        "Mean over three seeds; paired question bootstrap intervals",
    )
    return fig


def plot_e02_e04(index: Index, plt):
    """E02-E04: macro Avg@8 over the six benchmarks for every method and student."""
    fig, axes = plt.subplots(1, len(STUDENTS), figsize=(9.2, 3.6), sharex=True, sharey=True)
    for ax, (experiment_id, student) in zip(axes, STUDENTS, strict=True):
        table = index[experiment_id]
        macro = {
            row["method"]: macro_average(
                table.number(row, f"{benchmark.lower()}_avg8") for benchmark in BENCHMARKS
            )
            for row in table.rows
        }
        methods = [method for method in macro if method != "Frozen teacher"]
        for position, method in enumerate(methods):
            emphasised = method == "VD-OPD"
            _dot(ax, macro[method], position, BLUE if emphasised else GREY)
            if emphasised or position == 0:
                ax.annotate(
                    f"{macro[method]:.1f}",
                    (macro[method], position),
                    xytext=(-7, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=8,
                    color=INK,
                )
        teacher = macro.get("Frozen teacher")
        if teacher is not None:
            ax.axvline(teacher, color=INK_SECONDARY, linewidth=1, zorder=1)
            ax.annotate(
                f"teacher {teacher:.1f}",
                (teacher, -0.9),
                xytext=(-4, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=8,
                color=INK_SECONDARY,
            )
        ax.set_yticks(range(len(methods)), methods)
        ax.set_ylim(len(methods) - 0.5, -1.4)
        ax.set_title(student)
        ax.set_xlabel("Macro Avg@8 (%)")
        _frame(ax)
    _titles(
        fig,
        "Macro Avg@8 by method and student (E02-E04)",
        "Mean of the six benchmarks in each table; the vertical line is the frozen teacher",
    )
    return fig


def plot_e07(index: Index, plt):
    """E07: target x weight factorial, macro Avg@8 with the SD across seeds."""
    table = index["E07"]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    labels = []
    for position, row in enumerate(table.rows):
        mean, sd = table.number(row, "macro_avg8_mean"), table.number(row, "macro_avg8_sd")
        emphasised = table.number(row, "vd_gain") is None
        color = BLUE if emphasised else GREY
        ax.hlines(position, mean - sd, mean + sd, color=color, linewidth=2, capstyle="round")
        _dot(ax, mean, position, color)
        if emphasised or position == 0:
            ax.annotate(
                row["macro_avg8_mean"],
                (mean + sd, position),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
        target = row["target"].split(" (")[0].lower()
        suffix = "  (VD-OPD)" if emphasised else ""
        labels.append(f"{target} target, weight {row['weight']}{suffix}")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Macro Avg@8 (%)")
    _frame(ax)
    _titles(
        fig,
        "Target and weight ablation (E07)",
        "Mean ± SD over three seeds",
    )
    return fig


def plot_e10(index: Index, plt):
    """E10: accuracy of EOPD(2V) and VD-OPD against the mean agreement of the view pool."""
    table = index["E10"]
    rows = sorted(table.rows, key=lambda row: table.number(row, "mean_z"))
    agreement = [table.number(row, "mean_z") for row in rows]
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    series = (("VD-OPD", "vd_opd", BLUE), ("EOPD(2V)", "eopd_2v", ORANGE))
    for label, prefix, color in series:
        means = [table.number(row, f"{prefix}_mean") for row in rows]
        sds = [table.number(row, f"{prefix}_sd") for row in rows]
        ax.plot(agreement, means, color=color, linewidth=2, solid_capstyle="round", zorder=2)
        ax.vlines(
            agreement,
            [m - s for m, s in zip(means, sds, strict=True)],
            [m + s for m, s in zip(means, sds, strict=True)],
            color=color,
            linewidth=2,
            capstyle="round",
            zorder=2,
        )
        _dot(ax, agreement, means, color, label=label)
        ax.annotate(
            label,
            (agreement[-1], means[-1]),
            xytext=(9, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    reference = table.number(rows[0], "eopd_full")
    ax.axhline(reference, color=GREY, linewidth=1.2, zorder=1, label="EOPD(full)")
    ax.annotate(
        "EOPD(full)",
        (agreement[-1], reference),
        xytext=(9, -4),
        textcoords="offset points",
        va="top",
        fontsize=8,
    )
    ax.set_xticks(agreement, [f"{row['view_pool']}\nmean Z = {row['mean_z']}" for row in rows])
    span = agreement[-1] - agreement[0]
    ax.set_xlim(agreement[0] - 0.12 * span, agreement[-1] + 0.3 * span)
    ax.set_ylabel("Macro Avg@8 (%)", labelpad=8)
    ax.legend(loc="upper left")
    _frame(ax, value_axis="y")
    _titles(
        fig,
        "Sensitivity to the training view pool (E10)",
        "Mean ± SD over three seeds; EOPD(full) keeps the original prompt; pools ordered by mean Z",
    )
    return fig


def plot_e14(index: Index, plt):
    """E14: MeanView and WorstView on three families of held-out prompt views."""
    table = index["E14"]
    families = (
        ("prefix", "Unseen prefix wrappers"),
        ("structural", "Structural changes"),
        ("paraphrase", "Verified paraphrases"),
    )
    methods = [row["method"] for row in table.rows]
    fig, axes = plt.subplots(1, len(families), figsize=(9.6, 2.9), sharex=True, sharey=True)
    for ax, (prefix, title) in zip(axes, families, strict=True):
        for position, row in enumerate(table.rows):
            worst = table.number(row, f"{prefix}_worst_view")
            mean = table.number(row, f"{prefix}_mean_view")
            ax.hlines(position, worst, mean, color=AXIS, linewidth=2, zorder=1)
            ax.plot(
                table.number(row, "canonical"), position, "|", ms=11, mew=1.6, color=INK_SECONDARY
            )
            _dot(ax, worst, position, BLUE_LIGHT)
            _dot(ax, mean, position, BLUE)
        ax.set_yticks(range(len(methods)), methods)
        ax.set_title(title)
        ax.set_xlabel("Macro accuracy (%)")
        _frame(ax)
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([], [], color=BLUE_LIGHT, marker="o", mec=SURFACE, ls="", label="WorstView"),
        plt.Line2D([], [], color=BLUE, marker="o", mec=SURFACE, ls="", label="MeanView"),
        plt.Line2D(
            [], [], color=INK_SECONDARY, marker="|", ms=10, mew=1.6, ls="", label="Canonical prompt"
        ),
    ]
    fig.legend(handles=handles, loc="upper right", ncols=3, bbox_to_anchor=(0.995, 1.0))
    _titles(
        fig,
        "Accuracy on held-out prompt views (E14)",
        "Four views per family, fixed before evaluation",
    )
    return fig


def plot_e17(index: Index, plt):
    """E17: effect of one matched update in the low-Z and the high-Z quartile, with 95% CIs."""
    table = index["E17"]
    metrics = (
        ("oracle_kl_decrease", "Oracle KL decrease (nats)"),
        ("shared_wrong_prob_change", "Shared-wrong probability change"),
        ("correctness_change_pp", "Correctness change (pp)"),
    )
    updates = list(dict.fromkeys(row["update"] for row in table.rows))
    strata = (
        ("Low", "Low Z (bottom quartile)", BLUE, -0.17),
        ("High", "High Z (top quartile)", ORANGE, 0.17),
    )
    fig, axes = plt.subplots(1, len(metrics), figsize=(9.6, 3.2), sharey=True)
    for ax, (name, title) in zip(axes, metrics, strict=True):
        for stratum, _, color, offset in strata:
            for position, update in enumerate(updates):
                row = table.row(stratum=stratum, update=update)
                y = position + offset
                ax.hlines(
                    y,
                    table.number(row, f"{name}_ci_low"),
                    table.number(row, f"{name}_ci_high"),
                    color=color,
                    linewidth=2,
                    capstyle="round",
                    zorder=2,
                )
                _dot(ax, table.number(row, name), y, color, size=6.5)
        ax.axvline(0, color=AXIS, linewidth=1, zorder=1)
        ax.set_yticks(range(len(updates)), updates)
        ax.set_title(title)
        _frame(ax)
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([], [], color=color, marker="o", mec=SURFACE, lw=2, label=label)
        for _, label, color, _ in strata
    ]
    fig.legend(handles=handles, loc="upper right", ncols=2, bbox_to_anchor=(0.995, 1.0))
    _titles(
        fig,
        "Matched local updates by agreement stratum (E17)",
        "Point estimates with 95% confidence intervals, one middle checkpoint",
    )
    return fig


FIGURES: dict[str, tuple[str, Callable]] = {
    "e01": ("e01_gain_by_benchmark", plot_e01),
    "e02": ("e02_e04_macro_by_method", plot_e02_e04),
    "e07": ("e07_target_weight", plot_e07),
    "e10": ("e10_view_pool", plot_e10),
    "e14": ("e14_heldout_views", plot_e14),
    "e17": ("e17_local_updates", plot_e17),
}


def draw(
    output_dir: Path = OUTPUT_DIR,
    only: list[str] | None = None,
    root: Path | None = None,
    dpi: int = 200,
    suffix: str = "png",
) -> list[Path]:
    """Draw the selected figures and return the written paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    index = load_index(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for key in only or list(FIGURES):
        name, function = FIGURES[key.lower()]
        figure = function(index, plt)
        path = output_dir / f"{name}.{suffix}"
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="directory with index.yaml")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES), help="figures to draw")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--format", default="png", choices=("png", "pdf", "svg"))
    parser.add_argument("--list", action="store_true", help="list the figures and exit")
    args = parser.parse_args(argv)
    if args.list:
        for key, (name, function) in FIGURES.items():
            print(f"{key}  {name}  {function.__doc__.splitlines()[0]}")
        return 0
    for path in draw(args.output_dir, args.only, args.root, args.dpi, args.format):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
