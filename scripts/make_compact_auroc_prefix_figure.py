#!/usr/bin/env python
"""Build the compact AUROC-vs-prefix-utility reproduction figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
OUT = ROOT / "outputs"
BUDGET_OUT = OUT / "budget_cpcc"
FIXED_SCORE_OUT = OUT / "fixed_score_60_20_20"
ALPHA = 0.05


SCORES = ["random", "token_format", "step_combined", "qwen_prm", "step_qwen"]
SCORE_LABELS = {
    "random": "Random",
    "token_format": "Token/format",
    "step_combined": "Trace features",
    "qwen_prm": "Direct PRM",
    "step_qwen": "Trace features + PRM",
}
SCORE_COLORS = {
    "random": "#a7b0ba",
    "token_format": "#4b5563",
    "step_combined": "#d95f02",
    "qwen_prm": "#0072B2",
    "step_qwen": "#009E73",
}
DATASET_COLORS = {
    "Arithmetic": "#1f77b4",
    "Boolean": "#7f7f7f",
    "GSM8K": "#2ca02c",
    "ProcessBench": "#9467bd",
    "Math-Shepherd": "#8c564b",
    "PRMBench": "#e377c2",
    "PRM800K": "#17becf",
}
DOMAIN_TITLES = {
    "arithmetic": "Arithmetic",
    "boolean": "Boolean",
    "gsm8k": "GSM8K",
}
PANELS = ["Arithmetic", "GSM8K", "ProcessBench", "Math-Shepherd", "PRMBench", "PRM800K"]
EXTERNAL_DATASETS = [
    ("ProcessBench", "processbench"),
    ("Math-Shepherd", "math_shepherd"),
    ("PRMBench", "prmbench"),
    ("PRM800K", "prm800k"),
]
EXTERNAL_SCORE_MAP = {
    "random": "random",
    "token_format": "artifact_token_formatting_logistic_l2",
    "step_combined": "combined_logistic_l2",
    "qwen_prm": "qwen_prm_error",
    "step_qwen": "step_qwen_combined_logistic_l2",
}


def load_points() -> pd.DataFrame:
    target_auroc = pd.read_csv(BUDGET_OUT / "table_target_domain_auroc_summary.csv")
    target_prefix = pd.read_csv(BUDGET_OUT / "table_target_domain_mondrian_raw.csv")
    target_prefix = (
        target_prefix[
            target_prefix["domain"].isin(DOMAIN_TITLES)
            & target_prefix["score"].isin(SCORES)
        ]
        .groupby(["domain", "score"], as_index=False)["prefix_kept"]
        .mean()
    )
    target = target_auroc[
        target_auroc["domain"].isin(DOMAIN_TITLES)
        & target_auroc["score"].isin(SCORES)
    ][["domain", "score", "step_auroc_mean"]].merge(target_prefix, on=["domain", "score"], how="inner")
    target = target.assign(
        dataset=target["domain"].map(DOMAIN_TITLES),
        auroc=target["step_auroc_mean"],
        prefix_kept_pct=100.0 * target["prefix_kept"],
    )[["dataset", "score", "auroc", "prefix_kept_pct"]]

    external_rows = []
    for dataset, dataset_dir in EXTERNAL_DATASETS:
        summary = pd.read_csv(FIXED_SCORE_OUT / dataset_dir / "table_prefix_aware_summary.csv")
        summary = summary[np.isclose(summary["alpha"].astype(float), ALPHA)]
        for score, external_score in EXTERNAL_SCORE_MAP.items():
            subset = summary[summary["score"] == external_score]
            if subset.empty:
                raise ValueError(f"Missing summary row for {dataset} / {external_score}")
            row = subset.iloc[0]
            external_rows.append(
                {
                    "dataset": dataset,
                    "score": score,
                    "auroc": float(row["auroc_mean"]),
                    "prefix_kept_pct": 100.0 * float(row["prefix_retained_fraction_mean"]),
                }
            )
    external = pd.DataFrame(external_rows)
    return pd.concat([target, external], ignore_index=True)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_points()
    build_overlay(df)
    build_column_small_multiples(df)
    build_grid_small_multiples(
        df,
        rows=1,
        cols=6,
        figsize=(7.0, 1.72),
        output_stem="fig_auroc_vs_prefix_utility_lines_grid_1x6",
        title_fontsize=7.4,
        tick_fontsize=5.8,
        label_fontsize=7.3,
        legend_fontsize=5.7,
        marker_size=21,
        line_width=1.10,
        red_line_width=1.30,
        left=0.052,
        right=0.997,
        bottom=0.165,
        top=0.860,
        wspace=0.070,
        hspace=0.055,
        legend_ncol=5,
        legend_y=0.955,
        xlabel_y=0.095,
        ylabel_x=0.012,
    )
    build_grid_small_multiples(
        df,
        rows=1,
        cols=6,
        figsize=(7.25, 1.72),
        output_stem="fig_auroc_vs_prefix_utility_lines_grid_1x6_free_y",
        title_fontsize=7.4,
        tick_fontsize=5.4,
        label_fontsize=7.3,
        legend_fontsize=5.7,
        marker_size=21,
        line_width=1.10,
        red_line_width=1.30,
        left=0.052,
        right=0.997,
        bottom=0.165,
        top=0.860,
        wspace=0.175,
        hspace=0.055,
        legend_ncol=5,
        legend_y=0.955,
        xlabel_y=0.095,
        ylabel_x=0.012,
        free_y=True,
        title_above=True,
    )
    build_grid_small_multiples(
        df,
        rows=2,
        cols=3,
        figsize=(4.60, 3.22),
        output_stem="fig_auroc_vs_prefix_utility_lines_grid_2x3",
        title_fontsize=7.6,
        tick_fontsize=6.1,
        label_fontsize=8.0,
        legend_fontsize=6.0,
        marker_size=21,
        line_width=1.10,
        red_line_width=1.30,
        left=0.090,
        right=0.997,
        bottom=0.100,
        top=0.895,
        wspace=0.065,
        hspace=0.080,
        legend_ncol=5,
        legend_y=0.985,
        xlabel_y=0.030,
        ylabel_x=0.018,
    )


def build_overlay(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.2))

    for dataset in PANELS:
        sub = df[df["dataset"] == dataset].sort_values("auroc")
        if sub.empty:
            continue
        color = DATASET_COLORS[dataset]
        ax.plot(
            sub["auroc"],
            sub["prefix_kept_pct"],
            color=color,
            linewidth=1.7,
            alpha=0.88,
            marker=None,
            label=dataset,
            zorder=1,
        )
        for row in sub.itertuples():
            ax.scatter(
                row.auroc,
                row.prefix_kept_pct,
                s=34,
                color=SCORE_COLORS[row.score],
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )

    ax.set_xlim(0.43, 1.01)
    ax.set_ylim(-2.0, 102.0)
    ax.set_xticks([0.5, 0.7, 0.9])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Step AUROC", fontsize=10)
    ax.set_ylabel("Certified prefix kept (%)", fontsize=10)
    ax.grid(True, alpha=0.30, linewidth=0.65)
    ax.tick_params(axis="both", labelsize=9)

    dataset_handles = [
        plt.Line2D([0], [0], color=DATASET_COLORS[name], linewidth=1.8, label=name)
        for name in PANELS
    ]
    score_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=SCORE_COLORS[score],
            markeredgecolor="white",
            label=SCORE_LABELS[score],
            markersize=5.8,
        )
        for score in SCORES
    ]
    leg1 = ax.legend(
        handles=dataset_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.66),
        frameon=False,
        fontsize=7.4,
        title="Dataset",
        title_fontsize=8,
        borderaxespad=0.0,
        handlelength=1.6,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=score_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.13),
        frameon=False,
        fontsize=7.4,
        title="Score source",
        title_fontsize=8,
        borderaxespad=0.0,
        handlelength=1.0,
    )

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_auroc_vs_prefix_utility_lines_compact.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_auroc_vs_prefix_utility_lines_compact.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_column_small_multiples(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        len(PANELS),
        1,
        figsize=(3.35, 4.95),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.08},
    )

    for ax, dataset in zip(axes, PANELS):
        sub = df[df["dataset"] == dataset].sort_values("auroc")
        for left, right in zip(sub.iloc[:-1].itertuples(), sub.iloc[1:].itertuples()):
            decreases = right.prefix_kept_pct < left.prefix_kept_pct
            ax.plot(
                [left.auroc, right.auroc],
                [left.prefix_kept_pct, right.prefix_kept_pct],
                color="#c9362b" if decreases else "#b7bfc3",
                linewidth=1.25 if decreases else 1.0,
                alpha=0.95 if decreases else 0.8,
                zorder=1,
            )
        for row in sub.itertuples():
            ax.scatter(
                row.auroc,
                row.prefix_kept_pct,
                s=18,
                color=SCORE_COLORS[row.score],
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
        ax.text(
            0.02,
            0.78,
            dataset,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.2,
            color="#111827",
        )
        ax.set_xlim(0.43, 1.01)
        ax.set_ylim(-3.0, 103.0)
        ax.set_yticks([0, 50, 100])
        ax.grid(True, alpha=0.30, linewidth=0.45)
        ax.tick_params(axis="both", labelsize=5.8, length=2.3, pad=1.3)

    axes[-1].set_xticks([0.5, 0.7, 0.9])
    axes[-1].set_xlabel("Step AUROC", fontsize=7.2, labelpad=1.5)
    fig.supylabel("Certified prefix kept (%)", fontsize=7.2, x=0.024)

    score_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=SCORE_COLORS[score],
            markeredgecolor="white",
            label=SCORE_LABELS[score],
            markersize=4.5,
        )
        for score in SCORES
    ]
    fig.legend(
        handles=score_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 1.006),
        frameon=False,
        fontsize=5.4,
        ncol=3,
        columnspacing=0.7,
        handletextpad=0.3,
    )

    fig.tight_layout(rect=(0.08, 0.035, 1.0, 0.955))
    fig.savefig(FIGURES / "fig_auroc_vs_prefix_utility_lines_column.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_auroc_vs_prefix_utility_lines_column.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_grid_small_multiples(
    df: pd.DataFrame,
    *,
    rows: int,
    cols: int,
    figsize: tuple[float, float],
    output_stem: str,
    title_fontsize: float,
    tick_fontsize: float,
    label_fontsize: float,
    legend_fontsize: float,
    marker_size: float,
    line_width: float,
    red_line_width: float,
    left: float,
    right: float,
    bottom: float,
    top: float,
    wspace: float,
    hspace: float,
    legend_ncol: int,
    legend_y: float,
    xlabel_y: float,
    ylabel_x: float,
    free_y: bool = False,
    title_above: bool = False,
) -> None:
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        sharex=True,
        sharey=not free_y,
    )
    flat_axes = axes.ravel()
    plot_axes = flat_axes[: len(PANELS)]
    extra_axes = flat_axes[len(PANELS) :]

    for ax, dataset in zip(plot_axes, PANELS):
        ax.set_box_aspect(1)
        sub = df[df["dataset"] == dataset].sort_values("auroc")
        for left_point, right_point in zip(sub.iloc[:-1].itertuples(), sub.iloc[1:].itertuples()):
            decreases = right_point.prefix_kept_pct < left_point.prefix_kept_pct
            ax.plot(
                [left_point.auroc, right_point.auroc],
                [left_point.prefix_kept_pct, right_point.prefix_kept_pct],
                color="#c9362b" if decreases else "#b7bfc3",
                linewidth=red_line_width if decreases else line_width,
                alpha=0.95 if decreases else 0.8,
                zorder=1,
            )
        for row in sub.itertuples():
            ax.scatter(
                row.auroc,
                row.prefix_kept_pct,
                s=marker_size,
                color=SCORE_COLORS[row.score],
                edgecolor="white",
                linewidth=0.25,
                zorder=3,
            )
        if title_above:
            ax.set_title(dataset, fontsize=title_fontsize, color="#111827", pad=1.6)
        else:
            ax.text(
                0.04,
                0.80,
                dataset,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=title_fontsize,
                color="#111827",
            )
        ax.set_xlim(0.43, 1.01)
        if free_y:
            ax.set_ylim(*panel_y_limits(sub["prefix_kept_pct"]))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
        else:
            ax.set_ylim(-3.0, 103.0)
            ax.set_yticks([0, 50, 100])
        ax.set_xticks([0.5, 0.7, 0.9])
        ax.grid(True, alpha=0.28, linewidth=0.35)
        ax.tick_params(axis="both", labelsize=tick_fontsize, length=1.9, pad=0.8)

    for ax in extra_axes:
        ax.set_box_aspect(1)
        ax.axis("off")

    score_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=SCORE_COLORS[score],
            markeredgecolor="white",
            label=SCORE_LABELS[score],
            markersize=max(3.2, marker_size**0.5),
        )
        for score in SCORES
    ]

    fig.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=wspace,
        hspace=hspace,
    )
    if extra_axes.size:
        extra_axes[0].legend(
            handles=score_handles,
            loc="center",
            frameon=False,
            fontsize=legend_fontsize,
            handlelength=1.05,
            handletextpad=0.28,
            labelspacing=0.36,
            borderaxespad=0.0,
        )
    else:
        fig.legend(
            handles=score_handles,
            loc="upper center",
            bbox_to_anchor=(0.53, legend_y),
            frameon=False,
            fontsize=legend_fontsize,
            ncol=legend_ncol,
            columnspacing=0.9,
            handlelength=1.05,
            handletextpad=0.28,
        )

    fig.text(0.53, xlabel_y, "Step AUROC", ha="center", va="center", fontsize=label_fontsize)
    fig.text(
        ylabel_x,
        0.54,
        "Certified prefix kept (%)",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=label_fontsize,
    )
    fig.savefig(FIGURES / f"{output_stem}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(FIGURES / f"{output_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def panel_y_limits(values: pd.Series) -> tuple[float, float]:
    y_min = float(values.min())
    y_max = float(values.max())
    span = max(y_max - y_min, 1.0)
    pad = max(3.0, 0.12 * span)
    lower = max(0.0, y_min - pad)
    upper = min(100.0, y_max + pad)
    if upper - lower < 18.0:
        center = 0.5 * (lower + upper)
        lower = max(0.0, center - 9.0)
        upper = min(100.0, center + 9.0)
        if upper - lower < 18.0:
            if lower == 0.0:
                upper = min(100.0, lower + 18.0)
            else:
                lower = max(0.0, upper - 18.0)
    return lower, upper


if __name__ == "__main__":
    main()
