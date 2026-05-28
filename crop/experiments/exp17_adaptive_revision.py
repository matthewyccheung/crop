"""Revision outputs for adaptive CPCC experiments.

This module builds the next-round report tables and figures from the adaptive
adapter outputs, and optionally runs the targeted cost-aware cascade and
expanded threshold-crossing diagnostics that require per-step scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crop.data import TraceRecord, load_many_npz
from crop.experiments.exp16_adaptive_adapters import (
    ADAPTER_SPECS,
    AdapterBundle,
    AdaptiveSplit,
    _adaptive_split_like,
    _build_views,
    _calibrate_and_eval,
    _fit_adapter,
    _read_qwen_scores,
    _score_map_coverage,
    _selection_frequencies,
    _slug,
    _summarize,
    _tex,
    split_traces_four_way,
)
from crop.metrics import full_trace_accept_rate, prefix_contamination_rate
from crop.risk_control import prefix_lengths, prefix_losses_by_lambda, select_lambda_crc
from crop.utils import ensure_dir, write_json


DATASET_ORDER = ["Target", "ProcessBench", "Math-Shepherd", "PRMBench", "PRM800K"]
LABELS = {spec.score: spec.label for spec in ADAPTER_SPECS}
ALPHA_MAIN = 0.05
COST_LAMBDAS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
PREFIX_TAUS = [0.25, 0.40, 0.50, 0.60, 0.75]
SCORE_QUANTILES = [0.50, 0.60, 0.70, 0.80, 0.90]
AMBIGUITY_BANDS = [0.02, 0.05, 0.10, 0.15]
COST_CHEAP = ["token_format", "step_combined", "prefix_combined", "hazard_combined"]
COST_STRONG = ["qwen_prm", "step_qwen", "prefix_qwen", "hazard_qwen"]
FOCUS_CROSSING = ["qwen_prm", "step_qwen", "prefix_qwen", "hazard_qwen"]


def _dataset_configs(args) -> list[tuple[str, str, str, str, list[int]]]:
    configs = [
        (
            "Target",
            "data/cheap_baselines/crop_target_text_steps.npz",
            "data/strengthened/crop_target_combined_steps.npz",
            "outputs/strengthened/final/process_repeated_qwen_prm/qwen_prm_scores.csv",
            args.target_seeds,
        ),
        (
            "ProcessBench",
            "outputs/strengthened/final/external_process/processbench/processbench_text_steps.npz",
            "outputs/strengthened/final/external_process/processbench/processbench_combined_steps.npz",
            "outputs/strengthened/final/external_process/processbench_qwen_prm/qwen_prm_scores.csv",
            args.external_seeds,
        ),
        (
            "Math-Shepherd",
            "outputs/strengthened/final/external_process/math_shepherd/math_shepherd_text_steps.npz",
            "outputs/strengthened/final/external_process/math_shepherd/math_shepherd_combined_steps.npz",
            "outputs/strengthened/final/external_process/math_shepherd_qwen_prm/qwen_prm_scores.csv",
            args.external_seeds,
        ),
        (
            "PRMBench",
            "outputs/strengthened/final/external_process/prmbench/prmbench_text_steps.npz",
            "outputs/strengthened/final/external_process/prmbench/prmbench_combined_steps.npz",
            "outputs/strengthened/final/external_process/prmbench_full_qwen_prm/qwen_prm_scores.csv",
            args.external_seeds,
        ),
        (
            "PRM800K",
            "outputs/strengthened/final/external_process/prm800k/prm800k_text_steps.npz",
            "outputs/strengthened/final/external_process/prm800k/prm800k_combined_steps.npz",
            "outputs/strengthened/final/external_process/prm800k_qwen_prm/qwen_prm_scores.csv",
            args.external_seeds,
        ),
    ]
    if args.datasets:
        wanted = {_slug(name) for name in args.datasets}
        configs = [cfg for cfg in configs if _slug(cfg[0]) in wanted]
    if not configs:
        raise ValueError(f"No datasets matched {args.datasets!r}")
    return configs


def _fmt_pct(value: float) -> str:
    return "--" if not np.isfinite(value) else f"{100.0 * value:.1f}"


def _write_simple_tex_table(path: Path, caption: str, label: str, colspec: str, header: str, rows: list[str]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\footnotesize",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        header,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines))


def build_headline_outputs(base_dir: Path, outdir: Path) -> pd.DataFrame:
    tables_dir = ensure_dir(outdir / "tables")
    root_tables = ensure_dir("tables")
    figures = ensure_dir("figures")
    raw = pd.read_csv(base_dir / "table_adaptive_all.csv")
    deltas = pd.read_csv(base_dir / "table_paired_deltas.csv")
    freqs = pd.read_csv(base_dir / "table_selection_frequencies.csv")
    rows = []
    for dataset in DATASET_ORDER:
        group = raw[(raw["dataset"] == dataset) & (raw["alpha"].round(4) == ALPHA_MAIN)]
        adaptive = group[group["score"] == "adaptive_max_feasible"]
        best = group[group["score"] == "best_fixed_adapter"]
        qwen = group[group["score"] == "qwen_prm"]
        if adaptive.empty or best.empty or qwen.empty:
            continue
        freq = freqs[(freqs["dataset"] == dataset) & (freqs["alpha"].round(4) == ALPHA_MAIN)]
        mode = str(freq.iloc[0]["mode_adapter"]) if not freq.empty else ""
        best_mode = best["selected_adapter"].astype(str).mode().iloc[0] if "selected_adapter" in best else ""
        d_qwen = deltas[
            (deltas["dataset"] == dataset)
            & (deltas["alpha"].round(4) == ALPHA_MAIN)
            & (deltas["comparison"] == "Adaptive - Qwen")
        ]
        d_best = deltas[
            (deltas["dataset"] == dataset)
            & (deltas["alpha"].round(4) == ALPHA_MAIN)
            & (deltas["comparison"] == "Adaptive - best fixed adapter on test")
        ]
        adaptive_kept = float(adaptive["prefix_retained_fraction"].mean())
        best_kept = float(best["prefix_retained_fraction"].mean())
        qwen_kept = float(qwen["prefix_retained_fraction"].mean())
        row = {
            "dataset": dataset,
            "adaptive_kept_pct": 100.0 * adaptive_kept,
            "adaptive_risk_pct": 100.0 * float(adaptive["prefix_contamination"].mean()),
            "best_fixed_adapter": LABELS.get(best_mode, best_mode),
            "best_fixed_kept_pct": 100.0 * best_kept,
            "gap_to_best_fixed_pp": 100.0 * (best_kept - adaptive_kept),
            "qwen_prm_kept_pct": 100.0 * qwen_kept,
            "adaptive_minus_qwen_pp": 100.0 * (adaptive_kept - qwen_kept),
            "modal_selected_adapter": LABELS.get(mode, mode),
            "modal_selection_frequency_pct": 100.0 * float(freq.iloc[0]["mode_fraction"]) if not freq.empty else np.nan,
        }
        if not d_qwen.empty:
            row["adaptive_minus_qwen_ci_low_pp"] = 100.0 * float(d_qwen.iloc[0]["delta_kept_ci_low"])
            row["adaptive_minus_qwen_ci_high_pp"] = 100.0 * float(d_qwen.iloc[0]["delta_kept_ci_high"])
        if not d_best.empty:
            row["gap_to_best_fixed_ci_low_pp"] = -100.0 * float(d_best.iloc[0]["delta_kept_ci_high"])
            row["gap_to_best_fixed_ci_high_pp"] = -100.0 * float(d_best.iloc[0]["delta_kept_ci_low"])
        rows.append(row)
    headline = pd.DataFrame(rows)
    headline["dataset"] = pd.Categorical(headline["dataset"], DATASET_ORDER, ordered=True)
    headline = headline.sort_values("dataset").reset_index(drop=True)
    headline.to_csv(tables_dir / "table_main_headline.csv", index=False)
    headline.to_markdown(tables_dir / "table_main_headline.md", index=False)
    tex_rows = []
    for row in headline.itertuples(index=False):
        tex_rows.append(
            f"{_tex(row.dataset)} & {row.adaptive_kept_pct:.1f} & {row.adaptive_risk_pct:.1f} & "
            f"{_tex(row.best_fixed_adapter)} & {row.best_fixed_kept_pct:.1f} & "
            f"{row.gap_to_best_fixed_pp:.1f} & {row.qwen_prm_kept_pct:.1f} & "
            f"{row.adaptive_minus_qwen_pp:+.1f} & {_tex(row.modal_selected_adapter)} ({row.modal_selection_frequency_pct:.0f}\\%) \\\\"
        )
    caption = (
        "Adaptive CPCC scoring recovers near-best fixed-adapter utility across annotation protocols. "
        "Candidate adapters are fit on training traces, selected on a disjoint selection split, and "
        "finally recalibrated by CPCC on a fresh calibration split. ``Best fixed'' is the hindsight "
        "best single adapter from the same candidate family, used only as a diagnostic reference."
    )
    (tables_dir / "table_main_headline.tex").write_text(
        "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering",
                rf"\caption{{{caption}}}",
                r"\label{tab:main_headline_adaptive}",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{3pt}",
                r"\resizebox{\textwidth}{!}{%",
                r"\begin{tabular}{lrrlrrrrl}",
                r"\toprule",
                r"Dataset & Adapt kept & Risk & Best fixed & Best kept & Gap & Qwen kept & $\Delta$Qwen & Modal selected \\",
                r"\midrule",
                *tex_rows,
                r"\bottomrule",
                r"\end{tabular}%",
                r"}",
                r"\end{table}",
                "",
            ]
        )
    )
    shutil.copyfile(tables_dir / "table_main_headline.tex", root_tables / "table_main_headline.tex")

    gap = headline[["dataset", "gap_to_best_fixed_pp"]].copy()
    gap.to_csv(outdir / "fig_near_oracle_gap.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    x = np.arange(len(gap))
    vals = gap["gap_to_best_fixed_pp"].to_numpy(float)
    ax.bar(x, vals, color="#5B2A86", width=0.65)
    for idx, value in enumerate(vals):
        ax.text(idx, value + 0.04, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(gap["dataset"].astype(str), rotation=20, ha="right")
    ax.set_ylabel("Gap to best fixed adapter (pp)")
    ax.set_ylim(0.0, max(1.4, float(np.nanmax(vals)) + 0.35))
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(figures / "fig_near_oracle_gap.pdf")
    fig.savefig(figures / "fig_near_oracle_gap.png", dpi=180)
    plt.close(fig)
    return headline


def build_neartie_outputs(base_dir: Path, outdir: Path) -> pd.DataFrame:
    rows = []
    for dataset in DATASET_ORDER:
        path = base_dir / _slug(dataset) / "table_selection_detail.csv"
        if not path.exists():
            continue
        detail = pd.read_csv(path)
        detail = detail[(detail["alpha"].round(4) == ALPHA_MAIN) & (detail["selection_rule"] == "max_feasible")]
        for (seed, alpha), group in detail.groupby(["seed", "alpha"], dropna=False):
            feasible = group[group["selection_feasible"].astype(bool)].copy()
            pool = feasible if not feasible.empty else group.copy()
            pool = pool.sort_values(["selection_prefix_kept", "selection_corrected_risk"], ascending=[False, True])
            selected = pool.iloc[0]
            runner = pool.iloc[1] if len(pool) > 1 else selected
            selected_utility = float(selected["selection_prefix_kept"])
            margin = selected_utility - float(runner["selection_prefix_kept"])
            rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "alpha": alpha,
                    "selected_adapter": selected["score"],
                    "selected_label": selected["label"],
                    "runner_up_adapter": runner["score"],
                    "runner_up_label": runner["label"],
                    "selected_utility": selected_utility,
                    "runner_up_utility": float(runner["selection_prefix_kept"]),
                    "margin": margin,
                    "margin_pp": 100.0 * margin,
                    "n_feasible": len(feasible),
                    "n_within_0p5pp": int(np.sum(selected_utility - pool["selection_prefix_kept"].to_numpy(float) <= 0.005 + 1e-12)),
                    "n_within_1pp": int(np.sum(selected_utility - pool["selection_prefix_kept"].to_numpy(float) <= 0.010 + 1e-12)),
                }
            )
    nearties = pd.DataFrame(rows)
    nearties["dataset"] = pd.Categorical(nearties["dataset"], DATASET_ORDER, ordered=True)
    nearties = nearties.sort_values(["dataset", "seed"]).reset_index(drop=True)
    nearties.to_csv(outdir / "selection_nearties.csv", index=False)
    summary = _summarize(nearties, ["dataset"])
    summary.to_csv(outdir / "selection_nearties_summary.csv", index=False)

    root_tables = ensure_dir("tables")
    tex_rows = []
    for row in summary.itertuples(index=False):
        tex_rows.append(
            f"{_tex(row.dataset)} & {row.margin_pp_mean:.2f} & {row.margin_pp_ci95:.2f} & "
            f"{row.n_within_0p5pp_mean:.1f} & {row.n_within_1pp_mean:.1f} & {row.n_feasible_mean:.1f} \\\\"
        )
    _write_simple_tex_table(
        root_tables / "table_selection_nearties.tex",
        "Selection-split near-ties for the primary adaptive rule at $\\alpha=0.05$. Margins use selection-split utility only.",
        "tab:selection_nearties",
        "lrrrrr",
        r"Dataset & Mean margin & 95\% CI & Within 0.5pp & Within 1.0pp & Feasible \\",
        tex_rows,
    )
    figures = ensure_dir("figures")
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    summary["dataset"] = pd.Categorical(summary["dataset"], DATASET_ORDER, ordered=True)
    summary = summary.sort_values("dataset")
    x = np.arange(len(summary))
    ax.bar(x, summary["margin_pp_mean"].to_numpy(float), color="#117733", width=0.65)
    ax.errorbar(x, summary["margin_pp_mean"], yerr=summary["margin_pp_ci95"], fmt="none", ecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["dataset"].astype(str), rotation=20, ha="right")
    ax.set_ylabel("Selection margin (pp)")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(figures / "fig_selection_margin_by_dataset.pdf")
    plt.close(fig)
    return nearties


def build_ablation_outputs(base_dir: Path, outdir: Path) -> pd.DataFrame:
    ab_dir = ensure_dir(outdir / "ablations")
    raw = pd.read_csv(base_dir / "table_adaptive_all.csv")
    subset_map = {
        "all": ["random", "token_format", "qwen_prm", "step_combined", "prefix_combined", "hazard_combined", "step_qwen", "prefix_qwen", "hazard_qwen"],
        "cheap_only": ["random", "token_format", "step_combined", "prefix_combined", "hazard_combined"],
        "qwen_only": ["qwen_prm", "step_qwen", "prefix_qwen", "hazard_qwen"],
        "no_hazard": ["random", "token_format", "qwen_prm", "step_combined", "prefix_combined", "step_qwen", "prefix_qwen"],
        "no_prefix": ["random", "token_format", "qwen_prm", "step_combined", "hazard_combined", "step_qwen", "hazard_qwen"],
        "no_step": ["random", "token_format", "qwen_prm", "prefix_combined", "hazard_combined", "prefix_qwen", "hazard_qwen"],
    }
    rows = []
    for dataset in DATASET_ORDER:
        detail_path = base_dir / _slug(dataset) / "table_selection_detail.csv"
        if not detail_path.exists():
            continue
        detail = pd.read_csv(detail_path)
        detail = detail[(detail["alpha"].round(4) == ALPHA_MAIN) & (detail["selection_rule"] == "max_feasible")]
        fixed = raw[
            (raw["dataset"] == dataset)
            & (raw["alpha"].round(4) == ALPHA_MAIN)
            & (raw["row_type"] == "fixed")
        ].copy()
        by_key = {(row.seed, row.score): row for row in fixed.itertuples(index=False)}
        for subset, candidates in subset_map.items():
            for seed, group in detail.groupby("seed", dropna=False):
                pool = group[group["score"].isin(candidates)].copy()
                feasible = pool[pool["selection_feasible"].astype(bool)]
                select_pool = feasible if not feasible.empty else pool
                if select_pool.empty:
                    continue
                selected = select_pool.sort_values(["selection_prefix_kept", "selection_corrected_risk"], ascending=[False, True]).iloc[0]
                fixed_row = by_key.get((seed, selected["score"]))
                if fixed_row is None:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "alpha": ALPHA_MAIN,
                        "candidate_family": subset,
                        "selected_adapter": selected["score"],
                        "selected_label": selected["label"],
                        "selection_prefix_kept": selected["selection_prefix_kept"],
                        "selection_corrected_risk": selected["selection_corrected_risk"],
                        "prefix_contamination": fixed_row.prefix_contamination,
                        "prefix_retained_fraction": fixed_row.prefix_retained_fraction,
                        "prefix_full_trace_rate": fixed_row.prefix_full_trace_rate,
                    }
                )
        for score in ["step_qwen", "prefix_qwen", "hazard_qwen"]:
            sub = fixed[fixed["score"] == score]
            for row in sub.itertuples(index=False):
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(row.seed),
                        "alpha": ALPHA_MAIN,
                        "candidate_family": f"fixed_{score}",
                        "selected_adapter": score,
                        "selected_label": LABELS.get(score, score),
                        "selection_prefix_kept": np.nan,
                        "selection_corrected_risk": np.nan,
                        "prefix_contamination": row.prefix_contamination,
                        "prefix_retained_fraction": row.prefix_retained_fraction,
                        "prefix_full_trace_rate": row.prefix_full_trace_rate,
                    }
                )
    ab = pd.DataFrame(rows)
    ab.to_csv(ab_dir / "table_candidate_family_ablation.csv", index=False)
    summary = _summarize(ab, ["dataset", "candidate_family"])
    summary.to_csv(ab_dir / "table_candidate_family_ablation_summary.csv", index=False)
    root_tables = ensure_dir("tables")
    focus = summary[summary["candidate_family"].isin(["all", "cheap_only", "qwen_only", "no_hazard", "no_step"])].copy()
    focus["dataset"] = pd.Categorical(focus["dataset"], DATASET_ORDER, ordered=True)
    focus = focus.sort_values(["dataset", "candidate_family"])
    tex_rows = [
        f"{_tex(row.dataset)} & {_tex(row.candidate_family)} & "
        f"{_fmt_pct(row.prefix_contamination_mean)} & {_fmt_pct(row.prefix_retained_fraction_mean)} & "
        f"{_fmt_pct(row.prefix_full_trace_rate_mean)} \\\\"
        for row in focus.itertuples(index=False)
    ]
    _write_simple_tex_table(
        ab_dir / "table_candidate_family_ablation.tex",
        "Candidate-family ablations for adaptive CPCC selection at $\\alpha=0.05$.",
        "tab:candidate_family_ablation",
        "llrrr",
        r"Dataset & Family & Risk & Kept & Full accept \\",
        tex_rows,
    )
    shutil.copyfile(ab_dir / "table_candidate_family_ablation.tex", root_tables / "table_candidate_family_ablation.tex")
    figures = ensure_dir("figures")
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    pivot = focus.pivot(index="dataset", columns="candidate_family", values="prefix_retained_fraction_mean")
    pivot = pivot.reindex(DATASET_ORDER)
    x = np.arange(len(pivot))
    families = [c for c in ["all", "cheap_only", "qwen_only", "no_hazard", "no_step"] if c in pivot.columns]
    width = 0.13
    offsets = (np.arange(len(families)) - (len(families) - 1) / 2) * width
    for off, family in zip(offsets, families):
        ax.bar(x + off, 100.0 * pivot[family].to_numpy(float), width=width, label=family)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.astype(str), rotation=20, ha="right")
    ax.set_ylabel("Prefix kept (%)")
    ax.legend(ncol=3, fontsize=7, frameon=False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(figures / "fig_candidate_family_ablation.pdf")
    plt.close(fig)
    return ab


def build_alpha_sweep_outputs(full_alpha_dir: Path, outdir: Path) -> pd.DataFrame:
    sweep_dir = ensure_dir(outdir / "alpha_sweep")
    raw = pd.read_csv(full_alpha_dir / "table_adaptive_all.csv")
    summary = _summarize(
        raw[raw["score"].isin(["adaptive_max_feasible", "qwen_prm", "step_qwen", "hazard_qwen", "token_format", "best_fixed_adapter"])],
        ["dataset", "alpha", "score", "label", "row_type"],
    )
    rows = []
    for (dataset, alpha), group in raw.groupby(["dataset", "alpha"], dropna=False):
        adaptive = group[group["score"] == "adaptive_max_feasible"]
        qwen = group[group["score"] == "qwen_prm"]
        best = group[group["score"] == "best_fixed_adapter"]
        if adaptive.empty or qwen.empty or best.empty:
            continue
        rows.append(
            {
                "dataset": dataset,
                "alpha": alpha,
                "adaptive_risk": adaptive["prefix_contamination"].mean(),
                "adaptive_kept": adaptive["prefix_retained_fraction"].mean(),
                "qwen_kept": qwen["prefix_retained_fraction"].mean(),
                "best_fixed_kept": best["prefix_retained_fraction"].mean(),
                "gap_to_best_fixed": best["prefix_retained_fraction"].mean() - adaptive["prefix_retained_fraction"].mean(),
                "adaptive_minus_qwen": adaptive["prefix_retained_fraction"].mean() - qwen["prefix_retained_fraction"].mean(),
            }
        )
    sweep = pd.DataFrame(rows)
    sweep.to_csv(sweep_dir / "table_alpha_sweep.csv", index=False)
    summary.to_csv(sweep_dir / "table_alpha_sweep_methods.csv", index=False)
    root_tables = ensure_dir("tables")
    focus = sweep[sweep["alpha"].isin([0.025, 0.05, 0.10])].copy()
    focus["dataset"] = pd.Categorical(focus["dataset"], DATASET_ORDER, ordered=True)
    focus = focus.sort_values(["dataset", "alpha"])
    tex_rows = [
        f"{_tex(row.dataset)} & {row.alpha:.3f} & {_fmt_pct(row.adaptive_risk)} & "
        f"{_fmt_pct(row.adaptive_kept)} & {100.0 * row.gap_to_best_fixed:.1f} & {100.0 * row.adaptive_minus_qwen:+.1f} \\\\"
        for row in focus.itertuples(index=False)
    ]
    _write_simple_tex_table(
        sweep_dir / "table_alpha_sweep.tex",
        "Alpha-sweep summary for adaptive CPCC. Alpha 0.05 remains the main-text setting.",
        "tab:alpha_sweep",
        "lrrrrr",
        r"Dataset & $\alpha$ & Risk & Kept & Gap & $\Delta$Qwen \\",
        tex_rows,
    )
    shutil.copyfile(sweep_dir / "table_alpha_sweep.tex", root_tables / "table_alpha_sweep.tex")
    figures = ensure_dir("figures")
    for name, datasets, path in [
        ("target", ["Target"], figures / "fig_alpha_sweep_target.pdf"),
        ("external", [d for d in DATASET_ORDER if d != "Target"], figures / "fig_alpha_sweep_external.pdf"),
    ]:
        fig, ax = plt.subplots(figsize=(6.0 if name == "target" else 7.4, 3.4))
        for dataset in datasets:
            sub = sweep[sweep["dataset"] == dataset].sort_values("alpha")
            ax.plot(sub["alpha"], 100.0 * sub["adaptive_kept"], marker="o", label=dataset)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Adaptive prefix kept (%)")
        ax.grid(True, alpha=0.25, linewidth=0.7)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return sweep


def _length_losses(traces: list[TraceRecord], lengths: np.ndarray) -> np.ndarray:
    losses = []
    for trace, length in zip(traces, np.asarray(lengths, dtype=int)):
        losses.append(bool(length > 0 and np.any(trace.y_errors[:length] > 0)))
    return np.asarray(losses, dtype=int)


def _lengths_by_lambda(scores_by_trace: list[np.ndarray], lambdas: np.ndarray) -> np.ndarray:
    lengths = np.zeros((len(lambdas), len(scores_by_trace)), dtype=int)
    for col, scores in enumerate(scores_by_trace):
        scores = np.asarray(scores, dtype=float)
        if len(scores) == 0:
            continue
        prefix_max = np.maximum.accumulate(scores)
        lengths[:, col] = np.searchsorted(prefix_max, lambdas, side="right")
    return lengths


def _losses_from_lengths(traces: list[TraceRecord], lengths: np.ndarray) -> np.ndarray:
    first_errors = np.asarray(
        [
            len(trace.steps) + 1 if trace.first_error is None else int(trace.first_error)
            for trace in traces
        ],
        dtype=int,
    )
    return (lengths > first_errors[None, :]).astype(int)


def _cascade_losses_from_lengths(
    traces: list[TraceRecord],
    cheap_lengths: np.ndarray,
    strong_lengths: np.ndarray,
    route: np.ndarray,
) -> np.ndarray:
    lengths = np.where(route[None, :], strong_lengths, cheap_lengths)
    return _losses_from_lengths(traces, lengths)


def _cascade_lengths(
    cheap_scores: list[np.ndarray],
    strong_scores: list[np.ndarray],
    route: np.ndarray,
    lambda_: float,
) -> np.ndarray:
    cheap_lengths = prefix_lengths(cheap_scores, lambda_)
    strong_lengths = prefix_lengths(strong_scores, lambda_)
    return np.where(route, strong_lengths, cheap_lengths)


def _cascade_losses_by_lambda(
    traces: list[TraceRecord],
    cheap_scores: list[np.ndarray],
    strong_scores: list[np.ndarray],
    route: np.ndarray,
    lambdas: np.ndarray,
) -> np.ndarray:
    return np.vstack([
        _length_losses(traces, _cascade_lengths(cheap_scores, strong_scores, route, float(lambda_)))
        for lambda_ in lambdas
    ])


def _route_mask(scores_by_trace: list[np.ndarray], route_rule: str, value: float, cheap_lambda: float) -> np.ndarray:
    totals = np.asarray([len(scores) for scores in scores_by_trace], dtype=float)
    cheap_lengths = prefix_lengths(scores_by_trace, cheap_lambda)
    cheap_fraction = cheap_lengths / np.maximum(totals, 1.0)
    max_scores = np.asarray([float(np.max(scores)) if len(scores) else 0.0 for scores in scores_by_trace])
    if route_rule == "route_if_short_prefix":
        return cheap_fraction < value
    if route_rule == "route_if_empty_or_short":
        return (cheap_lengths == 0) | (cheap_fraction < value)
    if route_rule == "route_if_high_risk":
        return max_scores > value
    if route_rule == "route_if_ambiguous":
        near = []
        for scores in scores_by_trace:
            scores = np.asarray(scores, dtype=float)
            near.append(bool(len(scores) and np.min(np.abs(scores - cheap_lambda)) <= value))
        return np.asarray(near, dtype=bool)
    raise ValueError(f"Unknown route_rule={route_rule!r}")


def _route_specs(cheap: AdapterBundle, split: AdaptiveSplit, alpha: float, lambdas: np.ndarray) -> list[dict]:
    losses = prefix_losses_by_lambda(split.select, cheap.select_scores_by_trace, lambdas)
    cheap_lambda, _ = select_lambda_crc(losses, lambdas, alpha=alpha, direction="increasing")
    flat = np.concatenate(cheap.select_scores_by_trace) if cheap.select_scores_by_trace else np.asarray([0.0])
    specs = []
    for tau in PREFIX_TAUS:
        specs.append({"route_rule": "route_if_short_prefix", "route_value": tau, "cheap_route_lambda": cheap_lambda})
        specs.append({"route_rule": "route_if_empty_or_short", "route_value": tau, "cheap_route_lambda": cheap_lambda})
    for quantile in SCORE_QUANTILES:
        specs.append(
            {
                "route_rule": "route_if_high_risk",
                "route_value": float(np.quantile(flat, quantile)),
                "route_quantile": quantile,
                "cheap_route_lambda": cheap_lambda,
            }
        )
    for band in AMBIGUITY_BANDS:
        specs.append({"route_rule": "route_if_ambiguous", "route_value": band, "cheap_route_lambda": cheap_lambda})
    return specs


def _cascade_eval_lengths(
    traces: list[TraceRecord],
    cheap_scores: list[np.ndarray],
    strong_scores: list[np.ndarray],
    route: np.ndarray,
    lambda_: float,
) -> dict[str, float]:
    lengths = _cascade_lengths(cheap_scores, strong_scores, route, lambda_)
    totals = np.asarray([len(trace.steps) for trace in traces], dtype=float)
    losses = _length_losses(traces, lengths)
    suffix = np.maximum(totals - lengths, 0.0)
    return {
        "prefix_contamination": prefix_contamination_rate(losses),
        "prefix_retained_fraction": float(np.mean(lengths / np.maximum(totals, 1.0))) if len(lengths) else float("nan"),
        "prefix_retained_steps": float(np.mean(lengths)) if len(lengths) else float("nan"),
        "prefix_full_trace_rate": full_trace_accept_rate(lengths, totals),
        "qwen_call_rate": float(np.mean(route)) if len(route) else float("nan"),
        "review_steps_routed": float(np.mean(suffix)) if len(suffix) else float("nan"),
    }


def _lambda_index(lambdas: np.ndarray, lambda_: float) -> int:
    return int(np.argmin(np.abs(lambdas - float(lambda_))))


def _cascade_eval_from_lengths(
    traces: list[TraceRecord],
    cheap_lengths: np.ndarray,
    strong_lengths: np.ndarray,
    route: np.ndarray,
    lambda_idx: int,
) -> dict[str, float]:
    lengths = np.where(route, strong_lengths[lambda_idx], cheap_lengths[lambda_idx])
    totals = np.asarray([len(trace.steps) for trace in traces], dtype=float)
    losses = _losses_from_lengths(traces, lengths[None, :])[0]
    suffix = np.maximum(totals - lengths, 0.0)
    return {
        "prefix_contamination": prefix_contamination_rate(losses),
        "prefix_retained_fraction": float(np.mean(lengths / np.maximum(totals, 1.0))) if len(lengths) else float("nan"),
        "prefix_retained_steps": float(np.mean(lengths)) if len(lengths) else float("nan"),
        "prefix_full_trace_rate": full_trace_accept_rate(lengths, totals),
        "qwen_call_rate": float(np.mean(route)) if len(route) else float("nan"),
        "review_steps_routed": float(np.mean(suffix)) if len(suffix) else float("nan"),
    }


def _cost_aware_dataset(args, dataset_name: str, text_features: str, combined_features: str, qwen_csv: str, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined = load_many_npz([combined_features], ["mixed"], allow_nan=True)
    text = load_many_npz([text_features], ["mixed"], allow_nan=True)
    scores_by_trace_id = _read_qwen_scores(qwen_csv, args.qwen_score_col)
    views = _build_views(combined, text, scores_by_trace_id)
    lambdas = np.linspace(0.0, 1.0, args.lambda_grid_size)
    policy_rows = []
    selected_rows = []
    crossing_rows = []
    spec_by_score = {spec.score: spec for spec in ADAPTER_SPECS}
    for seed in seeds:
        reference = split_traces_four_way(
            combined,
            train_frac=args.train_frac,
            select_frac=args.select_frac,
            cal_frac=args.cal_frac,
            test_frac=args.test_frac,
            seed=seed,
        )
        split_by_view = {name: _adaptive_split_like(reference, traces) for name, traces in views.items()}
        adapters: dict[str, AdapterBundle] = {}
        for spec in ADAPTER_SPECS:
            split = reference if spec.source == "qwen" else split_by_view[spec.view]
            adapters[spec.score] = _fit_adapter(spec, split, seed, args.class_weight, scores_by_trace_id)
        length_mats = {
            score: {
                "select": _lengths_by_lambda(adapter.select_scores_by_trace, lambdas),
                "cal": _lengths_by_lambda(adapter.cal_scores_by_trace, lambdas),
                "test": _lengths_by_lambda(adapter.test_scores_by_trace, lambdas),
            }
            for score, adapter in adapters.items()
        }
        fixed_metrics = {
            score: _calibrate_and_eval(
                reference.cal,
                adapter.cal_scores_by_trace,
                reference.test,
                adapter.test_scores_by_trace,
                alpha=ALPHA_MAIN,
                lambdas=lambdas,
            )
            for score, adapter in adapters.items()
        }
        policies = []
        for cheap_score in COST_CHEAP:
            cheap = adapters[cheap_score]
            for strong_score in COST_STRONG:
                strong = adapters[strong_score]
                for route_spec in _route_specs(cheap, reference, ALPHA_MAIN, lambdas):
                    route_select = _route_mask(
                        cheap.select_scores_by_trace,
                        route_spec["route_rule"],
                        route_spec["route_value"],
                        route_spec["cheap_route_lambda"],
                    )
                    losses = _cascade_losses_from_lengths(
                        reference.select,
                        length_mats[cheap_score]["select"],
                        length_mats[strong_score]["select"],
                        route_select,
                    )
                    lambda_hat, corrected = select_lambda_crc(losses, lambdas, alpha=ALPHA_MAIN, direction="increasing")
                    metrics = _cascade_eval_from_lengths(
                        reference.select,
                        length_mats[cheap_score]["select"],
                        length_mats[strong_score]["select"],
                        route_select,
                        _lambda_index(lambdas, lambda_hat),
                    )
                    row = {
                        "dataset": dataset_name,
                        "seed": seed,
                        "alpha": ALPHA_MAIN,
                        "cheap_score": cheap_score,
                        "cheap_label": cheap.spec.label,
                        "strong_score": strong_score,
                        "strong_label": strong.spec.label,
                        **route_spec,
                        "selection_lambda": lambda_hat,
                        "selection_corrected_risk": corrected,
                        "selection_feasible": corrected <= ALPHA_MAIN,
                        "selection_prefix_kept": metrics["prefix_retained_fraction"],
                        "selection_risk": metrics["prefix_contamination"],
                        "selection_qwen_call_rate": metrics["qwen_call_rate"],
                    }
                    policies.append(row)
                    policy_rows.append(row)
        policy_df = pd.DataFrame(policies)
        for cost_lambda in COST_LAMBDAS:
            candidates = policy_df.copy()
            candidates["cost_lambda"] = cost_lambda
            candidates["selection_objective"] = candidates["selection_prefix_kept"] - cost_lambda * candidates["selection_qwen_call_rate"]
            feasible = candidates[candidates["selection_feasible"].astype(bool)]
            fallback = feasible.empty
            pool = feasible if not fallback else candidates.sort_values("selection_corrected_risk")
            selected = pool.sort_values(
                ["selection_objective", "selection_corrected_risk", "selection_qwen_call_rate"],
                ascending=[False, True, True],
            ).iloc[0]
            cheap = adapters[str(selected["cheap_score"])]
            strong = adapters[str(selected["strong_score"])]
            route_cal = _route_mask(
                cheap.cal_scores_by_trace,
                str(selected["route_rule"]),
                float(selected["route_value"]),
                float(selected["cheap_route_lambda"]),
            )
            cal_losses = _cascade_losses_from_lengths(reference.cal, length_mats[str(selected["cheap_score"])]["cal"], length_mats[str(selected["strong_score"])]["cal"], route_cal)
            lambda_cal, cal_risk = select_lambda_crc(cal_losses, lambdas, alpha=ALPHA_MAIN, direction="increasing")
            route_test = _route_mask(
                cheap.test_scores_by_trace,
                str(selected["route_rule"]),
                float(selected["route_value"]),
                float(selected["cheap_route_lambda"]),
            )
            test_metrics = _cascade_eval_from_lengths(
                reference.test,
                length_mats[str(selected["cheap_score"])]["test"],
                length_mats[str(selected["strong_score"])]["test"],
                route_test,
                _lambda_index(lambdas, lambda_cal),
            )
            cheap_kept = fixed_metrics[str(selected["cheap_score"])]["prefix_retained_fraction"]
            strong_kept = fixed_metrics[str(selected["strong_score"])]["prefix_retained_fraction"]
            denom = strong_kept - cheap_kept
            unstable = denom <= 0.02
            gain = float("nan") if unstable else (test_metrics["prefix_retained_fraction"] - cheap_kept) / denom
            selected_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "alpha": ALPHA_MAIN,
                    "cost_lambda": cost_lambda,
                    "fallback_used": bool(fallback),
                    "cheap_score": selected["cheap_score"],
                    "cheap_label": selected["cheap_label"],
                    "strong_score": selected["strong_score"],
                    "strong_label": selected["strong_label"],
                    "route_rule": selected["route_rule"],
                    "route_value": selected["route_value"],
                    "route_quantile": selected.get("route_quantile", np.nan),
                    "cheap_route_lambda": selected["cheap_route_lambda"],
                    "selection_objective": selected["selection_objective"],
                    "selection_prefix_kept": selected["selection_prefix_kept"],
                    "selection_corrected_risk": selected["selection_corrected_risk"],
                    "selection_qwen_call_rate": selected["selection_qwen_call_rate"],
                    "prefix_lambda": lambda_cal,
                    "prefix_cal_corrected_risk": cal_risk,
                    "cheap_only_kept": cheap_kept,
                    "strong_only_kept": strong_kept,
                    "qwen_prm_kept": fixed_metrics["qwen_prm"]["prefix_retained_fraction"],
                    "step_qwen_kept": fixed_metrics["step_qwen"]["prefix_retained_fraction"],
                    "absolute_gain_over_cheap": test_metrics["prefix_retained_fraction"] - cheap_kept,
                    "absolute_gap_to_strong": strong_kept - test_metrics["prefix_retained_fraction"],
                    "absolute_improvement_over_qwen": test_metrics["prefix_retained_fraction"] - fixed_metrics["qwen_prm"]["prefix_retained_fraction"],
                    "absolute_improvement_over_step_qwen": test_metrics["prefix_retained_fraction"] - fixed_metrics["step_qwen"]["prefix_retained_fraction"],
                    "gain_recovery_denominator": denom,
                    "gain_recovery_unstable": bool(unstable),
                    "gain_recovered": gain,
                    **test_metrics,
                }
            )
        crossing_rows.extend(_expanded_crossings(dataset_name, seed, reference, adapters, lambdas))
    return pd.DataFrame(policy_rows), pd.DataFrame(selected_rows), pd.DataFrame(crossing_rows)


def _expanded_crossings(
    dataset: str,
    seed: int,
    split: AdaptiveSplit,
    adapters: dict[str, AdapterBundle],
    lambdas: np.ndarray,
) -> list[dict]:
    rows = []
    for score in FOCUS_CROSSING:
        adapter = adapters[score]
        losses = prefix_losses_by_lambda(split.cal, adapter.cal_scores_by_trace, lambdas)
        lambda_hat, cal_risk = select_lambda_crc(losses, lambdas, alpha=ALPHA_MAIN, direction="increasing")
        for trace, scores in zip(split.test, adapter.test_scores_by_trace):
            first_error = trace.first_error
            if first_error is None or first_error >= len(scores):
                continue
            scores = np.asarray(scores, dtype=float)
            crossing = np.flatnonzero(scores > lambda_hat)
            if len(crossing):
                first_crossing = int(crossing[0])
                offset = first_crossing - int(first_error)
                no_crossing = False
            else:
                first_crossing = -1
                offset = np.nan
                no_crossing = True
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "alpha": ALPHA_MAIN,
                    "trace_id": trace.trace_id,
                    "score": score,
                    "label": adapter.spec.label,
                    "threshold": lambda_hat,
                    "cal_corrected_risk": cal_risk,
                    "first_error_index": int(first_error),
                    "first_crossing_index": first_crossing,
                    "offset": offset,
                    "abs_offset": abs(offset) if np.isfinite(offset) else np.nan,
                    "crossing_before_first_error": bool(np.isfinite(offset) and offset < 0),
                    "crossing_at_or_near_1": bool(np.isfinite(offset) and abs(offset) <= 1),
                    "crossing_at_or_near_2": bool(np.isfinite(offset) and abs(offset) <= 2),
                    "no_threshold_crossing": no_crossing,
                    "score_before_first_error": float(scores[first_error - 1]) if first_error > 0 else np.nan,
                    "score_at_first_error": float(scores[first_error]),
                }
            )
    return rows


def run_cost_aware_and_hazard(args, outdir: Path) -> None:
    cost_dir = ensure_dir(outdir / "cost_aware_cascades")
    hazard_dir = ensure_dir(outdir / "hazard_diagnostics")
    policy_frames = []
    selected_frames = []
    crossing_frames = []
    for dataset_name, text_features, combined_features, qwen_csv, seeds in _dataset_configs(args):
        print(f"Running cost-aware cascades for {dataset_name}", flush=True)
        policies, selected, crossings = _cost_aware_dataset(args, dataset_name, text_features, combined_features, qwen_csv, seeds)
        policy_frames.append(policies)
        selected_frames.append(selected)
        crossing_frames.append(crossings)
        dataset_dir = ensure_dir(cost_dir / _slug(dataset_name))
        policies.to_csv(dataset_dir / "raw_policy_selection.csv", index=False)
        selected.to_csv(dataset_dir / "selected_cost_frontier.csv", index=False)
        crossings.to_csv(ensure_dir(hazard_dir / _slug(dataset_name)) / "threshold_crossings.csv", index=False)
    policy = pd.concat(policy_frames, ignore_index=True)
    selected = pd.concat(selected_frames, ignore_index=True)
    crossings = pd.concat(crossing_frames, ignore_index=True)
    policy.to_csv(cost_dir / "raw_policy_selection.csv", index=False)
    selected.to_csv(cost_dir / "selected_cost_frontier.csv", index=False)
    crossings.to_csv(hazard_dir / "threshold_crossings.csv", index=False)
    _summarize_cost_aware(selected, cost_dir)
    _summarize_hazard(crossings, hazard_dir)
    write_json(cost_dir / "run_config.json", vars(args))


def merge_cost_aware_parts(part_dirs: list[str], outdir: Path) -> None:
    cost_dir = ensure_dir(outdir / "cost_aware_cascades")
    hazard_dir = ensure_dir(outdir / "hazard_diagnostics")
    policy_frames = []
    selected_frames = []
    crossing_frames = []

    for raw_part in part_dirs:
        part = Path(raw_part)
        part_cost = part / "cost_aware_cascades"
        part_hazard = part / "hazard_diagnostics"
        policy_path = part_cost / "raw_policy_selection.csv"
        selected_path = part_cost / "selected_cost_frontier.csv"
        crossing_path = part_hazard / "threshold_crossings.csv"
        if not policy_path.exists() or not selected_path.exists() or not crossing_path.exists():
            raise FileNotFoundError(f"Missing cost-aware or hazard output in {part}")

        policy_frames.append(pd.read_csv(policy_path))
        selected_frames.append(pd.read_csv(selected_path))
        crossing_frames.append(pd.read_csv(crossing_path))

        for child in part_cost.iterdir():
            if child.is_dir():
                shutil.copytree(child, cost_dir / child.name, dirs_exist_ok=True)
        for child in part_hazard.iterdir():
            if child.is_dir():
                shutil.copytree(child, hazard_dir / child.name, dirs_exist_ok=True)

    policy = pd.concat(policy_frames, ignore_index=True)
    selected = pd.concat(selected_frames, ignore_index=True)
    crossings = pd.concat(crossing_frames, ignore_index=True)

    sort_cols = [col for col in ["dataset", "seed", "alpha", "cost_lambda", "selection_objective"] if col in policy.columns]
    if sort_cols:
        policy = policy.sort_values(sort_cols)
    sort_cols = [col for col in ["dataset", "seed", "alpha", "cost_lambda"] if col in selected.columns]
    if sort_cols:
        selected = selected.sort_values(sort_cols)
    sort_cols = [col for col in ["dataset", "score", "trace_id"] if col in crossings.columns]
    if sort_cols:
        crossings = crossings.sort_values(sort_cols)

    policy.to_csv(cost_dir / "raw_policy_selection.csv", index=False)
    selected.to_csv(cost_dir / "selected_cost_frontier.csv", index=False)
    crossings.to_csv(hazard_dir / "threshold_crossings.csv", index=False)
    _summarize_cost_aware(selected, cost_dir)
    _summarize_hazard(crossings, hazard_dir)
    write_json(cost_dir / "merged_parts.json", {"part_dirs": [str(Path(p)) for p in part_dirs]})


def _summarize_cost_aware(selected: pd.DataFrame, cost_dir: Path) -> None:
    selected = selected.copy()
    if {"gain_recovery_denominator", "prefix_retained_fraction", "cheap_only_kept"}.issubset(selected.columns):
        invalid = selected["gain_recovery_denominator"] <= 0.02
        selected["gain_recovery_unstable"] = invalid
        selected["gain_recovered"] = np.where(
            invalid,
            np.nan,
            (selected["prefix_retained_fraction"] - selected["cheap_only_kept"]) / selected["gain_recovery_denominator"],
        )
    summary = _summarize(selected, ["dataset", "cost_lambda"])
    summary.to_csv(cost_dir / "table_cost_frontier.csv", index=False)
    summary.to_csv(cost_dir / "table_cost_frontier_summary.csv", index=False)
    root_tables = ensure_dir("tables")
    focus = summary[summary["cost_lambda"].isin(COST_LAMBDAS)].copy()
    focus["dataset"] = pd.Categorical(focus["dataset"], DATASET_ORDER, ordered=True)
    focus = focus.sort_values(["dataset", "cost_lambda"])
    def _cost_rows(frame: pd.DataFrame) -> list[str]:
        rows = []
        for row in frame.itertuples(index=False):
            gain = "--" if getattr(row, "gain_recovered_n", 0) == 0 else _fmt_pct(row.gain_recovered_mean)
            rows.append(
                f"{_tex(row.dataset)} & {row.cost_lambda:.2f} & {_fmt_pct(row.prefix_contamination_mean)} & "
                f"{_fmt_pct(row.prefix_retained_fraction_mean)} & {_fmt_pct(row.qwen_call_rate_mean)} & "
                f"{100.0 * row.absolute_gain_over_cheap_mean:+.1f} & {100.0 * row.absolute_gap_to_strong_mean:+.1f} & {gain} \\\\"
            )
        return rows

    tex_rows = _cost_rows(focus)
    report_focus = focus[(focus["dataset"] == "Target") | ((focus["dataset"] != "Target") & (focus["cost_lambda"].isin([0.10, 0.50])))].copy()
    report_rows = _cost_rows(report_focus)
    _write_simple_tex_table(
        cost_dir / "table_cost_frontier.tex",
        "Cost-aware directly calibrated cascade frontier. Policy selection uses selection-split utility minus a Qwen-call penalty; final returned-prefix risk is recalibrated on calibration traces.",
        "tab:cost_frontier",
        "lrrrrrrr",
        r"Dataset & Cost & Risk & Kept & Qwen calls & Gain cheap & Gap strong & Gain rec. \\",
        tex_rows,
    )
    _write_simple_tex_table(
        root_tables / "table_cost_frontier.tex",
        "Representative cost-aware directly calibrated cascade frontier. The output directory contains the full cost grid; external rows show two operating points per dataset.",
        "tab:cost_frontier",
        "lrrrrrrr",
        r"Dataset & Cost & Risk & Kept & Qwen calls & Gain cheap & Gap strong & Gain rec. \\",
        report_rows,
    )
    figures = ensure_dir("figures")
    for target_only, filename, title in [
        (True, "fig_cost_frontier_target.pdf", "Target"),
        (False, "fig_cost_frontier_external.pdf", "External datasets"),
    ]:
        fig, ax = plt.subplots(figsize=(6.2 if target_only else 7.4, 3.6))
        datasets = ["Target"] if target_only else [d for d in DATASET_ORDER if d != "Target"]
        for dataset in datasets:
            sub = summary[summary["dataset"] == dataset].sort_values("cost_lambda")
            ax.plot(100.0 * sub["qwen_call_rate_mean"], 100.0 * sub["prefix_retained_fraction_mean"], marker="o", label=dataset)
            if target_only:
                for row in sub.itertuples(index=False):
                    ax.text(100.0 * row.qwen_call_rate_mean, 100.0 * row.prefix_retained_fraction_mean, f"{row.cost_lambda:.2g}", fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("Qwen call rate (%)")
        ax.set_ylabel("Prefix kept (%)")
        ax.grid(True, alpha=0.25, linewidth=0.7)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(figures / filename)
        fig.savefig(cost_dir / filename)
        plt.close(fig)
    lines = ["# Cost-Aware Cascades", ""]
    target = summary[summary["dataset"] == "Target"].sort_values("cost_lambda")
    if not target.empty:
        best = target[(target["qwen_call_rate_mean"] <= 0.40) & (target["prefix_retained_fraction_mean"] >= 0.90)]
        lines.append("## Target Frontier")
        lines.append("")
        cols = ["cost_lambda", "prefix_contamination_mean", "prefix_retained_fraction_mean", "qwen_call_rate_mean", "absolute_gain_over_cheap_mean", "absolute_gap_to_strong_mean", "gain_recovered_mean"]
        table = target[cols].copy()
        for col in cols[1:]:
            table[col] = (100.0 * table[col]).round(2)
        lines.append(table.to_markdown(index=False))
        lines.append("")
        if not best.empty:
            lines.append("A cost-aware target cascade satisfies the working target-distribution criterion: at least 90% prefix retained with no more than 40% Qwen calls.")
        else:
            lines.append("No target cascade point satisfied both 90% retained-prefix utility and 40% Qwen-call rate.")
    lines.append("")
    lines.append("Gain recovered is omitted when the strong-minus-cheap denominator is not greater than 2 percentage points.")
    (cost_dir / "ANALYSIS.md").write_text("\n".join(lines))


def _summarize_hazard(crossings: pd.DataFrame, hazard_dir: Path) -> None:
    rows = []
    for keys, group in crossings.groupby(["dataset", "score", "label"], dropna=False):
        dataset, score, label = keys
        offsets = group["offset"].to_numpy(float)
        rows.append(
            {
                "dataset": dataset,
                "score": score,
                "label": label,
                "n_error_traces": len(group),
                "median_offset": float(np.nanmedian(offsets)),
                "mean_abs_offset": float(np.nanmean(group["abs_offset"])),
                "fraction_crossing_before_first_error": float(group["crossing_before_first_error"].mean()),
                "fraction_crossing_at_or_near_1": float(group["crossing_at_or_near_1"].mean()),
                "fraction_crossing_at_or_near_2": float(group["crossing_at_or_near_2"].mean()),
                "fraction_no_threshold_crossing": float(group["no_threshold_crossing"].mean()),
                "median_score_before_first_error": float(group["score_before_first_error"].median()),
                "median_score_at_first_error": float(group["score_at_first_error"].median()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["dataset"] = pd.Categorical(summary["dataset"], DATASET_ORDER, ordered=True)
    summary = summary.sort_values(["dataset", "score"])
    summary.to_csv(hazard_dir / "table_threshold_crossings.csv", index=False)
    root_tables = ensure_dir("tables")
    tex_rows = [
        f"{_tex(row.dataset)} & {_tex(row.label)} & {row.median_offset:.1f} & {row.mean_abs_offset:.1f} & "
        f"{_fmt_pct(row.fraction_crossing_before_first_error)} & {_fmt_pct(row.fraction_crossing_at_or_near_1)} & "
        f"{_fmt_pct(row.fraction_crossing_at_or_near_2)} \\\\"
        for row in summary.itertuples(index=False)
    ]
    _write_simple_tex_table(
        hazard_dir / "table_threshold_crossings.tex",
        "Threshold-crossing diagnostics relative to the first annotated error. Negative offsets cross before the first error.",
        "tab:threshold_crossings",
        "llrrrrr",
        r"Dataset & Score & Med. offset & Mean abs. & Before FE & Near 1 & Near 2 \\",
        tex_rows,
    )
    shutil.copyfile(hazard_dir / "table_threshold_crossings.tex", root_tables / "table_threshold_crossings.tex")
    figures = ensure_dir("figures")

    def _draw_boxplot(ax, data: list[np.ndarray], labels: list[str]) -> None:
        try:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
        except TypeError:
            ax.boxplot(data, labels=labels, showfliers=False)

    for dataset in DATASET_ORDER:
        sub = crossings[(crossings["dataset"] == dataset) & (~crossings["offset"].isna())].copy()
        if sub.empty:
            continue
        labels = [LABELS.get(score, score) for score in FOCUS_CROSSING]
        data = [sub[sub["score"] == score]["offset"].clip(-10, 10).to_numpy(float) for score in FOCUS_CROSSING]
        fig, ax = plt.subplots(figsize=(6.3, 3.4))
        _draw_boxplot(ax, data, labels)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.9)
        ax.set_ylabel("First crossing - first error")
        ax.set_title(dataset)
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
        fig.tight_layout()
        filename = f"fig_crossing_offsets_{_slug(dataset)}.pdf"
        fig.savefig(figures / filename)
        fig.savefig(hazard_dir / filename)
        plt.close(fig)

    fig, axes = plt.subplots(len(DATASET_ORDER), 1, figsize=(7.0, 10.5), sharex=False)
    labels = [LABELS.get(score, score) for score in FOCUS_CROSSING]
    for ax, dataset in zip(axes, DATASET_ORDER):
        sub = crossings[(crossings["dataset"] == dataset) & (~crossings["offset"].isna())].copy()
        if sub.empty:
            ax.axis("off")
            continue
        data = [sub[sub["score"] == score]["offset"].clip(-10, 10).to_numpy(float) for score in FOCUS_CROSSING]
        _draw_boxplot(ax, data, labels)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(dataset, fontsize=10)
        ax.set_ylabel("Offset")
        ax.tick_params(axis="x", labelrotation=18, labelsize=8)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(figures / "fig_crossing_offsets_all.pdf")
    fig.savefig(hazard_dir / "fig_crossing_offsets_all.pdf")
    plt.close(fig)
    lines = ["# Hazard Threshold-Crossing Diagnostics", ""]
    for dataset in ["PRMBench", "ProcessBench"]:
        sub = summary[summary["dataset"] == dataset]
        if not sub.empty:
            lines.append(f"## {dataset}")
            table = sub[["label", "median_offset", "mean_abs_offset", "fraction_crossing_before_first_error", "fraction_crossing_at_or_near_1", "fraction_crossing_at_or_near_2"]].copy()
            for col in ["fraction_crossing_before_first_error", "fraction_crossing_at_or_near_1", "fraction_crossing_at_or_near_2"]:
                table[col] = (100.0 * table[col]).round(2)
            lines.append(table.to_markdown(index=False))
            lines.append("")
    (hazard_dir / "ANALYSIS.md").write_text("\n".join(lines))


def build_static_outputs(args, outdir: Path) -> None:
    base_dir = Path(args.base_dir)
    full_alpha_dir = Path(args.full_alpha_dir)
    build_headline_outputs(base_dir, outdir)
    build_neartie_outputs(base_dir, outdir)
    build_ablation_outputs(base_dir, outdir)
    build_alpha_sweep_outputs(full_alpha_dir, outdir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/adaptive_adapters_extensions")
    parser.add_argument("--base_dir", default="outputs/adaptive_adapters_extensions")
    parser.add_argument("--full_alpha_dir", default="outputs/adaptive_adapters")
    parser.add_argument("--qwen_score_col", default="qwen_prm_error")
    parser.add_argument("--target_seeds", nargs="*", type=int, default=list(range(2806, 2826)))
    parser.add_argument("--external_seeds", nargs="*", type=int, default=list(range(2806, 2816)))
    parser.add_argument("--lambda_grid_size", type=int, default=101)
    parser.add_argument("--class_weight", default="balanced")
    parser.add_argument("--train_frac", type=float, default=0.5)
    parser.add_argument("--select_frac", type=float, default=0.15)
    parser.add_argument("--cal_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--run_cost_aware", action="store_true")
    parser.add_argument("--merge_part_dirs", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.target_seeds = args.target_seeds[:1]
        args.external_seeds = args.external_seeds[:1]
        args.lambda_grid_size = min(args.lambda_grid_size, 51)
    outdir = ensure_dir(args.output_dir)
    build_static_outputs(args, outdir)
    if args.run_cost_aware:
        run_cost_aware_and_hazard(args, outdir)
    if args.merge_part_dirs:
        merge_cost_aware_parts(args.merge_part_dirs, outdir)
    write_json(outdir / "revision_run_config.json", vars(args))
    print(f"Wrote revision outputs to {outdir}", flush=True)


if __name__ == "__main__":
    main()
