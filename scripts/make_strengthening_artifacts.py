#!/usr/bin/env python3
"""Build strengthening result, table, and figure artifacts.

This script consolidates the cached strengthened experiment outputs into the
release report files. It is intentionally a table/report builder: heavyweight
scoring jobs remain separate.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crop.data import StepRecord, TraceRecord, load_many_npz
from crop.metrics import first_error_diagnostics
from crop.risk_control import (
    calibrate_selective_acceptance,
    first_error_error_only_losses_by_lambda,
    first_error_localization_losses,
    first_error_losses_by_lambda,
    prefix_lengths,
    prefix_losses_by_lambda,
    select_lambda_crc,
    whole_trace_false_accept_losses,
)
from crop.splits import split_traces

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # Keep the report builder usable in lean envs.
    plt = None


OUT = ROOT / "outputs" / "strengthened" / "final"
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"


def ensure_dirs() -> None:
    for path in (RESULTS, TABLES, FIGURES):
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def add_prefix_accept_derivatives(summary: pd.DataFrame, raw_path: Path) -> pd.DataFrame:
    """Add CPCC full-prefix accept diagnostics to a summary table."""

    raw = read_csv(raw_path)
    if summary.empty or raw.empty:
        return summary
    needed = {"score", "alpha", "trace_error_rate_test", "prefix_error_full_trace_rate", "prefix_full_trace_rate"}
    if not needed.issubset(raw.columns):
        return summary
    raw = raw.copy()
    raw["prefix_marginal_false_accept"] = raw["trace_error_rate_test"] * raw["prefix_error_full_trace_rate"]
    raw["prefix_accepted_error_rate"] = np.where(
        raw["prefix_full_trace_rate"].astype(float) > 0,
        raw["prefix_marginal_false_accept"] / raw["prefix_full_trace_rate"],
        np.nan,
    )
    derived = _summarize_with_ci_local(
        raw[["score", "alpha", "prefix_marginal_false_accept", "prefix_accepted_error_rate"]],
        ["score", "alpha"],
    )
    overlap = [col for col in derived.columns if col in summary.columns and col not in {"score", "alpha"}]
    if overlap:
        summary = summary.drop(columns=overlap)
    return summary.merge(derived, on=["score", "alpha"], how="left")


def row_for(df: pd.DataFrame, score: str, alpha: float = 0.05) -> pd.Series | None:
    if df.empty:
        return None
    subset = df[(df["score"] == score) & np.isclose(df["alpha"].astype(float), alpha)]
    if subset.empty:
        return None
    return subset.iloc[0]


def pct(value: float | int | None, digits: int = 1) -> str:
    if value is None or not np.isfinite(float(value)):
        return "--"
    return f"{100.0 * float(value):.{digits}f}"


def num(value: float | int | None, digits: int = 2) -> str:
    if value is None or not np.isfinite(float(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def tex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def mean_ci(row: pd.Series | None, metric: str, scale: float = 1.0, digits: int = 1) -> str:
    if row is None:
        return "--"
    mean_key = f"{metric}_mean"
    ci_key = f"{metric}_ci95"
    if mean_key not in row:
        return "--"
    mean = row.get(mean_key)
    ci = row.get(ci_key, np.nan)
    if not np.isfinite(float(mean)):
        return "--"
    if np.isfinite(float(ci)):
        return f"${scale * float(mean):.{digits}f} \\pm {scale * float(ci):.{digits}f}$"
    return f"${scale * float(mean):.{digits}f}$"


def mean_ci_values(values: Iterable[float], scale: float = 1.0, digits: int = 1) -> str:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return "--"
    mean = float(np.mean(array))
    ci = 1.96 * float(np.std(array, ddof=1)) / np.sqrt(len(array)) if len(array) > 1 else 0.0
    return f"${scale * mean:.{digits}f} \\pm {scale * ci:.{digits}f}$"


def mean_ci_emph(row: pd.Series | None, metric: str, scale: float = 1.0, digits: int = 1, style: str | None = None) -> str:
    if row is None:
        return "--"
    mean_key = f"{metric}_mean"
    ci_key = f"{metric}_ci95"
    if mean_key not in row:
        return "--"
    mean = row.get(mean_key)
    ci = row.get(ci_key, np.nan)
    if not np.isfinite(float(mean)):
        return "--"
    macro = {"bold": r"\mathbf", "italic": r"\mathit"}.get(style or "")
    if macro:
        head = f"{macro}{{{scale * float(mean):.{digits}f}}}"
    else:
        head = f"{scale * float(mean):.{digits}f}"
    if np.isfinite(float(ci)):
        return f"${head} \\pm {scale * float(ci):.{digits}f}$"
    return f"${head}$"


def latex_table(path: Path, label: str, caption: str, tabular: str, small: bool = True) -> None:
    size = "\\small\n" if small else ""
    path.write_text(
        "\n".join(
            [
                "\\begin{table}[t]",
                "\\centering",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                size + tabular,
                "\\end{table}",
                "",
            ]
        )
    )


def audit_dataset(name: str, path: Path, source: str, generator: str, label_source: str, label_type: str, notes: str) -> dict:
    traces = load_many_npz([path], ["mixed"], allow_nan=True)
    steps = sum(len(t.steps) for t in traces)
    error_steps = sum(int(step.y_error) for t in traces for step in t.steps)
    error_traces = sum(int(t.has_error) for t in traces)
    lengths = np.asarray([len(t.steps) for t in traces], dtype=float)
    first_errors = [t.first_error for t in traces if t.first_error is not None]
    multi_error = sum(int(np.sum(t.y_errors) > 1) for t in traces)
    return {
        "dataset": name,
        "source": source,
        "generator_model": generator,
        "label_source": label_source,
        "label_type": label_type,
        "traces": len(traces),
        "steps": steps,
        "step_error_rate": error_steps / max(steps, 1),
        "trace_error_rate": error_traces / max(len(traces), 1),
        "median_trace_length": float(np.median(lengths)) if len(lengths) else np.nan,
        "p90_trace_length": float(np.quantile(lengths, 0.9)) if len(lengths) else np.nan,
        "first_error_available": bool(first_errors),
        "median_first_error_step": float(np.median(first_errors)) if first_errors else np.nan,
        "multi_error_traces": multi_error,
        "notes": notes,
    }


def build_dataset_audit() -> pd.DataFrame:
    rows = []
    target = ROOT / "data" / "strengthened" / "crop_target_combined_steps.npz"
    target_traces = load_many_npz([target], ["mixed"], allow_nan=True)
    for domain in sorted({t.domain for t in target_traces}):
        domain_traces = [t for t in target_traces if t.domain == domain]
        steps = sum(len(t.steps) for t in domain_traces)
        error_steps = sum(int(step.y_error) for t in domain_traces for step in t.steps)
        error_traces = sum(int(t.has_error) for t in domain_traces)
        lengths = np.asarray([len(t.steps) for t in domain_traces], dtype=float)
        rows.append(
            {
                "dataset": f"Target {domain}",
                "source": "CROP target annotations",
                "generator_model": "official_crop_annotations",
                "label_source": "process labels from imported annotations",
                "label_type": "dense step labels; first error derived from first erroneous step",
                "traces": len(domain_traces),
                "steps": steps,
                "step_error_rate": error_steps / max(steps, 1),
                "trace_error_rate": error_traces / max(len(domain_traces), 1),
                "median_trace_length": float(np.median(lengths)) if len(lengths) else np.nan,
                "p90_trace_length": float(np.quantile(lengths, 0.9)) if len(lengths) else np.nan,
                "first_error_available": True,
                "median_first_error_step": float(np.median([t.first_error for t in domain_traces if t.first_error is not None]))
                if error_traces
                else np.nan,
                "multi_error_traces": sum(int(np.sum(t.y_errors) > 1) for t in domain_traces),
                "notes": "Split unit is trace_id; later-step labels are kept as annotated rather than automatically invalidated.",
            }
        )

    external = [
        (
            "ProcessBench",
            OUT / "external_process" / "processbench" / "processbench_combined_steps.npz",
            "Qwen/ProcessBench",
            "benchmark-provided model traces",
            "benchmark first-error annotations",
            "first-error / process labels",
            "Hard math process benchmark; Qwen PRM scores are available.",
        ),
        (
            "Math-Shepherd",
            OUT / "external_process" / "math_shepherd" / "math_shepherd_combined_steps.npz",
            "Math-Shepherd-style public data",
            "dataset-provided traces",
            "parsed step markup",
            "step markup labels",
            "GSM8K-style process supervision; label provenance follows dataset markup.",
        ),
        (
            "PRMBench",
            OUT / "external_process" / "prmbench" / "prmbench_combined_steps.npz",
            "ssmisya/PRMBench",
            "benchmark-provided traces",
            "benchmark fine-grained labels",
            "fine-grained step labels",
            "Harsh imported subset; most traces contain at least one annotated error.",
        ),
        (
            "PRM800K",
            OUT / "external_process" / "prm800k" / "prm800k_combined_steps.npz",
            "openai/prm800k",
            "dataset-provided MATH-style solutions",
            "human process supervision",
            "step labels",
            "Imported process-supervision cache; focused repeated run is used if full run is unavailable.",
        ),
    ]
    for args in external:
        if args[1].exists():
            rows.append(audit_dataset(*args))
    audit = pd.DataFrame(rows)
    audit.to_csv(RESULTS / "dataset_audit.csv", index=False)
    return audit


def build_split_audit() -> None:
    target = load_many_npz([ROOT / "data" / "strengthened" / "crop_target_combined_steps.npz"], ["mixed"], allow_nan=True)
    warnings: list[str] = []
    duplicate_problem_warnings = 0
    seed_summaries = []
    for seed in range(2806, 2856):
        split = split_traces(target, seed=seed)
        ids = {
            "train": {t.trace_id for t in split.train},
            "cal": {t.trace_id for t in split.cal},
            "test": {t.trace_id for t in split.test},
        }
        overlaps = {
            "train_cal": sorted(ids["train"] & ids["cal"]),
            "train_test": sorted(ids["train"] & ids["test"]),
            "cal_test": sorted(ids["cal"] & ids["test"]),
        }
        if any(overlaps.values()):
            warnings.append(f"trace_id overlap at seed {seed}: {overlaps}")
        texts = {}
        for part_name, part in (("train", split.train), ("cal", split.cal), ("test", split.test)):
            for trace in part:
                text = trace.steps[0].original_expression if trace.steps else None
                if text:
                    texts.setdefault(str(text), set()).add(part_name)
        dupes = [key for key, parts in texts.items() if len(parts) > 1]
        duplicate_problem_warnings += len(dupes)
        seed_summaries.append(
            {
                "seed": seed,
                "train_traces": len(split.train),
                "cal_traces": len(split.cal),
                "test_traces": len(split.test),
                "duplicate_problem_text_cross_split": len(dupes),
                "trace_overlap": any(overlaps.values()),
            }
        )
    payload = {
        "split_unit": "trace_id",
        "seeds_checked": [2806, 2855],
        "warnings": warnings,
        "duplicate_problem_text_cross_split_total": duplicate_problem_warnings,
        "seed_summaries": seed_summaries,
        "tie_handling": "finite threshold grid with corrected conformal risk; sentinel empty-prefix rule if no feasible threshold",
    }
    (RESULTS / "split_audit.json").write_text(json.dumps(payload, indent=2))


def write_static_comparison_tables() -> None:
    novelty = r"""\resizebox{\textwidth}{!}{%
\begin{tabular}{llllll}
\toprule
Method family & Calibrated object & Returned object & Prefix? & FE set? & Main efficiency \\
\midrule
Conformal abstention & answer & accept/reject & \(\times\) & \(\times\) & accept rate \\
CLM / SCOPE-Gen & sampled generations & candidate set & \(\times\) & \(\times\) & set size / samples \\
Conformal factuality & claim/answer & backed-off factual output & \(\times\) & \(\times\) & retained specificity \\
Coherent Factuality / DCF & claim graph & coherent retained claims & not direct & \(\times\) & retained claims \\
PRM calibration & reasoning state & calibrated reward/confidence & \(\times\) & \(\times\) & search/ranking efficiency \\
Dynamic abstention & online prefix & continue/stop & \(\checkmark\) online & \(\times\) & compute saved \\
 CROP & ordered trace & clean prefix + FE set + trace accept & \(\checkmark\) & \(\checkmark\) & prefix kept / FE size / accept \\
\bottomrule
\end{tabular}
}"""
    latex_table(
        TABLES / "novelty_comparison.tex",
        "tab:novelty_compare",
        "Compact novelty comparison. The distinction is the calibrated object: CROP calibrates ordered-trace certificates, especially contiguous clean prefixes and first-error candidate sets.",
        novelty,
    )

    novelty_expanded = r"""\resizebox{\textwidth}{!}{%
\begin{tabular}{lllllllll}
\toprule
Method family & Calibrated unit & Returned object & Single trace? & Prefix? & FE set? & Graph? & Formal target & Efficiency metric \\
\midrule
Conformal abstention & Answer & Accept/reject & Yes & No & No & No & false accept / hallucination & accept rate \\
Conformal language modeling & Candidate generation & Set of outputs & No & No & No & No & admissible candidate coverage & set size / samples \\
Conformal factuality & Claim or answer & Backed-off factual output & Yes & No & No & Optional & factuality risk & retained specificity \\
Coherent Factuality & Deducibility graph & Coherent accepted claims & Yes & Not direct & No & Yes & coherent factuality & retained claims \\
Differentiable Coherent Factuality & Graph + learned scorer & Higher-retention coherent claims & Yes & Not direct & No & Yes & coherent factuality & retained claims \\
PRM calibration & Reasoning state & Calibrated reward/confidence & Yes & No & No & No & score calibration & search/ranking efficiency \\
Dynamic abstention & Prefix during generation & Continue/stop & Online & Yes & No & No & stopping utility/error & compute saved \\
 CROP & Ordered trace & Step sets, clean prefix, FE set, trace accept & Yes & Yes & Yes & No & prefix contamination / FE miss / FA & prefix kept / FE size / accept \\
\bottomrule
\end{tabular}
}"""
    latex_table(
        TABLES / "novelty_comparison_expanded.tex",
        "tab:novelty_compare_expanded",
        "Expanded novelty comparison. CROP's novelty is not a new conformal theorem or a new verifier architecture; it is the ordered-trace certificate object and its risk-efficiency evaluation.",
        novelty_expanded,
    )

    coherent = r"""\resizebox{\textwidth}{!}{%
\begin{tabular}{llllll}
\toprule
Method & Structure & Output & Graph required? & Natural clean prefix? & Natural FE set? \\
\midrule
Conformal Factuality & claims / specificity & backed-off factual output & optional & \(\times\) & \(\times\) \\
Coherent Factuality & deducibility graph & coherent claim subgraph & \(\checkmark\) & not direct & \(\times\) \\
DCF & graph + learned scorer & higher-retention claim subgraph & \(\checkmark\) & not direct & \(\times\) \\
 CROP & ordered step sequence & clean prefix + FE set & \(\times\) & \(\checkmark\) & \(\checkmark\) \\
\bottomrule
\end{tabular}
}"""
    latex_table(
        TABLES / "coherent_factuality_comparison.tex",
        "tab:coherent_factuality_compare",
        "Compact comparison with Coherent Factuality and DCF. Their object is graph-coherent claim retention; ours is ordered-process certification.",
        coherent,
    )

    coherent_expanded = r"""\resizebox{\textwidth}{!}{%
\begin{tabular}{llllllll}
\toprule
Method & Structure assumed & Calibration unit & Output & Graph required? & Natural prefix? & Natural FE set? & Efficiency metric \\
\midrule
Conformal Factuality & Independent claims / answer specificity & claim/answer & backed-off factual output & No/optional & No & No & retained specificity \\
Coherent Factuality & Deducibility graph & claim subgraph & coherent accepted claims & Yes & Not directly & No & retained claims \\
Differentiable Coherent Factuality & Graph + differentiable scorer & claim subgraph & higher-retention coherent claims & Yes & Not directly & No & retained claims \\
 CROP & Ordered step sequence & trace & clean prefix, first-error set, step sets, trace accept & No & Yes & Yes & prefix kept, FE size, accept rate \\
\bottomrule
\end{tabular}
}"""
    latex_table(
        TABLES / "coherent_factuality_comparison_expanded.tex",
        "tab:coherent_factuality_compare_expanded",
        "Expanded comparison with Coherent Factuality and DCF, including calibration units and efficiency metrics.",
        coherent_expanded,
    )


def build_main_process_table(process: pd.DataFrame, qwen: pd.DataFrame) -> None:
    process = add_prefix_accept_derivatives(process, OUT / "process_repeated_50seed" / "table_process_main.csv")
    qwen = add_prefix_accept_derivatives(qwen, OUT / "process_repeated_qwen_prm" / "table_external_score.csv")
    labels = [
        ("Qwen2.5-Math PRM", "qwen_prm_error", qwen),
        ("Combined logistic", "combined_logistic_l2", process),
        ("Text logistic", "text_logistic_l2", process),
        ("Token/format control", "artifact_token_formatting_logistic_l2", process),
        ("Dataset-ID control", "artifact_dataset_id_logistic_l2", process),
        ("CoE-C raw", "coe_c_error", process),
        ("Random", "random", process),
        ("Oracle upper bound", "oracle", process),
    ]
    core_labels = [
        ("Qwen2.5-Math PRM", "qwen_prm_error", qwen),
        ("Combined logistic", "combined_logistic_l2", process),
        ("Text logistic", "text_logistic_l2", process),
        ("Token/format", "artifact_token_formatting_logistic_l2", process),
        ("Random", "random", process),
        ("Oracle", "oracle", process),
    ]
    random = row_for(process, "random")
    token = row_for(process, "artifact_token_formatting_logistic_l2")
    random_kept = float(random["prefix_retained_fraction_mean"]) if random is not None else np.nan
    token_kept = float(token["prefix_retained_fraction_mean"]) if token is not None else np.nan

    def kept_delta(row: pd.Series | None, baseline: float, bold: bool = False) -> str:
        if row is None:
            return "--"
        kept = float(row["prefix_retained_fraction_mean"])
        delta = 100.0 * (kept - baseline) if np.isfinite(baseline) else np.nan
        if np.isfinite(delta):
            text = f"{delta:+.1f}"
            return f"\\textbf{{{text}}}" if bold else text
        return "--"

    core_lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Detector & AUROC & Prefix risk & Prefix kept & Steps & Full accept & Marg. FA & Acc. err. & $\Delta$ Random & $\Delta$ Token \\",
        r"\midrule",
    ]
    for label, score, df in core_labels:
        row = row_for(df, score)
        if row is None:
            continue
        is_qwen = score == "qwen_prm_error"
        is_combined = score == "combined_logistic_l2"
        kept_style = "bold" if is_qwen else ("italic" if is_combined else None)
        accept_style = "bold" if is_qwen else None
        core_lines.append(
            " & ".join(
                [
                    tex(label),
                    mean_ci(row, "auroc", digits=3),
                    mean_ci(row, "prefix_contamination", scale=100, digits=1),
                    mean_ci_emph(row, "prefix_retained_fraction", scale=100, digits=1, style=kept_style),
                    mean_ci(row, "prefix_retained_steps", digits=2),
                    mean_ci_emph(row, "prefix_full_trace_rate", scale=100, digits=1, style=accept_style),
                    mean_ci(row, "prefix_marginal_false_accept", scale=100, digits=1),
                    mean_ci(row, "prefix_accepted_error_rate", scale=100, digits=1),
                    kept_delta(row, random_kept, bold=is_qwen),
                    kept_delta(row, token_kept, bold=is_qwen),
                ]
            )
            + r" \\"
        )
    core_lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "main_process_efficiency.tex",
        "tab:process_main_efficiency",
        "Primary clean-prefix certificate efficiency at $\\alpha=0.05$. CPCC chooses the largest feasible threshold; bold values indicate the most efficient non-oracle score source under this fixed certificate objective, and italics mark the best cached cheap detector. Full accept means the certified prefix reaches the end of the trace. Oracle is an unattainable upper bound. Marg. FA is $P(\\mathrm{full\\ accept}\\wedge\\mathrm{error})$; accepted-error rate is diagnostic.",
        "\n".join(core_lines),
    )

    diagnostic_lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Detector & Step cov. & Step size & Nonempty prefix & FE cov. all & FE cov. err. & FE size all & FE size nonempty & Clean acc. & Err. acc. \\",
        r"\midrule",
    ]
    for label, score, df in labels:
        row = row_for(df, score)
        if row is None:
            continue
        diagnostic_lines.append(
            " & ".join(
                [
                    tex(label),
                    mean_ci(row, "step_coverage_all", digits=3),
                    num(row["step_avg_set_size_mean"], 2),
                    mean_ci(row, "prefix_nonempty_rate", scale=100, digits=1),
                    pct(row["fe_coverage_all_mean"], 1),
                    pct(row["fe_coverage_error_only_mean"], 1),
                    num(row["fe_candidate_size_all_mean"], 2),
                    num(row["fe_candidate_size_excluding_empty_mean"], 2),
                    mean_ci(row, "clean_accept_rate", scale=100, digits=1),
                    mean_ci(row, "incorrect_accept_rate", scale=100, digits=1),
                ]
            )
            + r" \\"
        )
    diagnostic_lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "main_process_full_diagnostics.tex",
        "tab:process_full_diagnostics",
        "Full process-level diagnostics at $\\alpha=0.05$. FE size nonempty excludes the $\\varnothing$ no-error option. These columns are moved out of the main empirical table to keep the main text focused on risk-efficiency.",
        "\n".join(diagnostic_lines),
    )

    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrrrrrr}",
        r"\toprule",
        r"Detector & AUROC & Step cov. & Size & Prefix risk & Kept & Nonempty & FE cov. & FE size & FA & Accept & Acc. err. & Clean acc. & Err. acc. & $\Delta$ kept \\",
        r"\midrule",
    ]
    for label, score, df in labels:
        row = row_for(df, score)
        if row is None:
            continue
        lines.append(
            " & ".join(
                [
                    tex(label),
                    mean_ci(row, "auroc", digits=3),
                    mean_ci(row, "step_coverage_all", digits=3),
                    num(row["step_avg_set_size_mean"], 2),
                    mean_ci(row, "prefix_contamination", scale=100, digits=1),
                    mean_ci(row, "prefix_retained_fraction", scale=100, digits=1),
                    mean_ci(row, "prefix_nonempty_rate", scale=100, digits=1),
                    pct(row["fe_coverage_error_only_mean"], 1),
                    num(row["fe_candidate_size_excluding_empty_mean"], 2),
                    mean_ci(row, "marginal_false_accept", scale=100, digits=1),
                    mean_ci(row, "accept_rate", scale=100, digits=1),
                    mean_ci(row, "accepted_error_rate", scale=100, digits=1),
                    mean_ci(row, "clean_accept_rate", scale=100, digits=1),
                    mean_ci(row, "incorrect_accept_rate", scale=100, digits=1),
                    f"{kept_delta(row, random_kept)}/{kept_delta(row, token_kept)}",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "main_process_results_expanded.tex",
        "tab:process_main_expanded",
        "Expanded primary process-level results at $\\alpha=0.05$. Cached detectors use 50 trace-level splits; Qwen2.5-Math PRM uses 20 target splits. FA is marginal false acceptance. Acc. err. is $P(\\mathrm{erroneous}\\mid\\mathrm{accepted})$, reported only as a diagnostic. The final column gives retained-prefix percentage-point deltas versus Random and Token/format.",
        "\n".join(lines),
    )


def build_risk_efficiency(process: pd.DataFrame, qwen: pd.DataFrame) -> None:
    process = add_prefix_accept_derivatives(process, OUT / "process_repeated_50seed" / "table_process_main.csv")
    qwen = add_prefix_accept_derivatives(qwen, OUT / "process_repeated_qwen_prm" / "table_external_score.csv")
    sweep_specs = [
        ("Random", process, "random"),
        ("Token/format", process, "artifact_token_formatting_logistic_l2"),
        ("Combined", process, "combined_logistic_l2"),
        ("Qwen PRM", qwen, "qwen_prm_error"),
    ]
    alphas = sorted(
        {
            float(alpha)
            for _, df, score in sweep_specs
            for alpha in df.loc[df["score"] == score, "alpha"].dropna().tolist()
        }
    )
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{rrrrrrrrrr}",
        r"\toprule",
        r"$\alpha$ & Random risk & Random kept & Token kept & Combined kept & Qwen kept & Combined FA & Qwen FA & Combined accept & Qwen accept \\",
        r"\midrule",
    ]
    for alpha in alphas:
        rows = {label: row_for(df, score, alpha=alpha) for label, df, score in sweep_specs}
        combined = rows.get("Combined")
        qwen_row = rows.get("Qwen PRM")
        lines.append(
            f"{alpha:.2f} & "
            f"{mean_ci(rows.get('Random'), 'prefix_contamination', scale=100, digits=1)} & "
            f"{mean_ci(rows.get('Random'), 'prefix_retained_fraction', scale=100, digits=1)} & "
            f"{mean_ci(rows.get('Token/format'), 'prefix_retained_fraction', scale=100, digits=1)} & "
            f"{mean_ci(combined, 'prefix_retained_fraction', scale=100, digits=1)} & "
            f"{mean_ci(qwen_row, 'prefix_retained_fraction', scale=100, digits=1)} & "
            f"{mean_ci(combined, 'prefix_marginal_false_accept', scale=100, digits=1)} & "
            f"{mean_ci(qwen_row, 'prefix_marginal_false_accept', scale=100, digits=1)} & "
            f"{mean_ci(combined, 'prefix_full_trace_rate', scale=100, digits=1)} & "
            f"{mean_ci(qwen_row, 'prefix_full_trace_rate', scale=100, digits=1)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "risk_efficiency_alpha_sweep.tex",
        "tab:risk_efficiency",
        "Risk-efficiency frontier for clean-prefix certificates. Curves compare certificate efficiency, not validity: at each risk target, calibration controls prefix contamination and useful scores retain longer prefixes.",
        "\n".join(lines),
    )

    plot_specs = [
        ("Combined", process, "combined_logistic_l2"),
        ("Token/format", process, "artifact_token_formatting_logistic_l2"),
        ("Random", process, "random"),
        ("Qwen PRM", qwen, "qwen_prm_error"),
    ]
    if plt is None:
        write_fallback_pdf(
            FIGURES / "risk_efficiency_prefix_kept.pdf",
            "Risk-efficiency prefix kept figure",
            "matplotlib was unavailable; see tables/risk_efficiency_alpha_sweep.tex.",
        )
        return
    plt.figure(figsize=(6.2, 3.8))
    for label, df, score in plot_specs:
        subset = df[(df["score"] == score) & (df["alpha"] <= 0.10)].sort_values("alpha")
        if subset.empty:
            continue
        plt.plot(subset["alpha"], 100.0 * subset["prefix_retained_fraction_mean"], marker="o", label=label)
    plt.axvline(0.05, color="black", linestyle="--", linewidth=1.0, alpha=0.55)
    plt.text(0.052, 3.0, r"$\alpha=0.05$", fontsize=8)
    plt.xlabel(r"risk target $\alpha$")
    plt.ylabel("prefix kept (%)")
    plt.xlim(0.0, 0.105)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "risk_efficiency_prefix_kept.pdf")
    plt.close()


def build_auroc_efficiency_table(process: pd.DataFrame, qwen: pd.DataFrame) -> None:
    process = add_prefix_accept_derivatives(process, OUT / "process_repeated_50seed" / "table_process_main.csv")
    qwen = add_prefix_accept_derivatives(qwen, OUT / "process_repeated_qwen_prm" / "table_external_score.csv")
    specs = [
        ("Qwen2.5-Math PRM", qwen, "qwen_prm_error", "lower AUROC than combined, best prefix efficiency"),
        ("Combined logistic", process, "combined_logistic_l2", "best cached AUROC, second-best prefix efficiency"),
        ("Text logistic", process, "text_logistic_l2", "strong cheap detector, below combined efficiency"),
        ("Token/format", process, "artifact_token_formatting_logistic_l2", "artifact-heavy control with nontrivial efficiency"),
        ("Dataset ID", process, "artifact_dataset_id_logistic_l2", "dataset artifact baseline"),
        ("Random", process, "random", "valid but inefficient"),
    ]
    rows = []
    for label, df, score, takeaway in specs:
        row = row_for(df, score)
        if row is None:
            continue
        rows.append(
            {
                "label": label,
                "score": score,
                "auroc": float(row["auroc_mean"]),
                "kept": float(row["prefix_retained_fraction_mean"]),
                "accept": float(row["prefix_full_trace_rate_mean"]),
                "takeaway": takeaway,
            }
        )
    ranked = pd.DataFrame(rows)
    ranked["auroc_rank"] = ranked["auroc"].rank(ascending=False, method="min").astype(int)
    ranked["kept_rank"] = ranked["kept"].rank(ascending=False, method="min").astype(int)
    ranked["accept_rank"] = ranked["accept"].rank(ascending=False, method="min").astype(int)
    ranked.to_csv(RESULTS / "auroc_vs_prefix_efficiency.csv", index=False)

    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrl}",
        r"\toprule",
        r"Detector & AUROC rank & Prefix-kept rank & Accept-rate rank & Takeaway \\",
        r"\midrule",
    ]
    for _, row in ranked.sort_values("kept_rank").iterrows():
        lines.append(
            f"{tex(row['label'])} & {int(row['auroc_rank'])} & {int(row['kept_rank'])} & "
            f"{int(row['accept_rank'])} & {tex(row['takeaway'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "auroc_vs_prefix_efficiency.tex",
        "tab:auroc_efficiency",
        "AUROC does not determine clean-prefix efficiency. CPCC evaluates how a score ranks early errors relative to the prefix object, so a verifier can have slightly lower AUROC but certify longer prefixes and accept more full traces.",
        "\n".join(lines),
    )


def build_artifact_adjusted_efficiency_table(process: pd.DataFrame, qwen: pd.DataFrame) -> None:
    controls = read_csv(RESULTS / "revision4_controls_summary.csv")
    baselines = {
        "Token/format": row_for(process, "artifact_token_formatting_logistic_l2"),
        "Label-shuffled": row_for(controls, "label_shuffled_combined_logistic_l2") if not controls.empty else None,
        "Trace-order shuffled": row_for(controls, "trace_order_shuffled_combined_logistic_l2") if not controls.empty else None,
    }
    baseline_kept = {
        key: float(row["prefix_retained_fraction_mean"]) if row is not None else np.nan
        for key, row in baselines.items()
    }
    best_artifact_kept = float(np.nanmax(list(baseline_kept.values()))) if baseline_kept else np.nan
    best_artifact_name = max(
        baseline_kept,
        key=lambda key: baseline_kept[key] if np.isfinite(baseline_kept[key]) else -np.inf,
    ) if baseline_kept else "--"
    specs = [
        ("Qwen2.5-Math PRM", qwen, "qwen_prm_error"),
        ("Combined logistic", process, "combined_logistic_l2"),
        ("Text logistic", process, "text_logistic_l2"),
        ("Token/format", process, "artifact_token_formatting_logistic_l2"),
        ("Random", process, "random"),
    ]
    rows = []
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Detector & Prefix kept & Token base & Label-shuf. & Order-shuf. & Gain over best artifact & Prefix risk \\",
        r"\midrule",
    ]
    for label, df, score in specs:
        row = row_for(df, score)
        if row is None:
            continue
        kept = float(row["prefix_retained_fraction_mean"])
        gain = 100.0 * (kept - best_artifact_kept) if np.isfinite(best_artifact_kept) else np.nan
        rows.append(
            {
                "detector": label,
                "score": score,
                "prefix_kept": kept,
                "prefix_risk": float(row["prefix_contamination_mean"]),
                "token_baseline": baseline_kept.get("Token/format", np.nan),
                "label_shuffle_baseline": baseline_kept.get("Label-shuffled", np.nan),
                "order_shuffle_baseline": baseline_kept.get("Trace-order shuffled", np.nan),
                "best_artifact_baseline": best_artifact_kept,
                "best_artifact_name": best_artifact_name,
                "gain_over_best_artifact": gain,
            }
        )
        lines.append(
            f"{tex(label)} & {mean_ci(row, 'prefix_retained_fraction', scale=100, digits=1)} & "
            f"{pct(baseline_kept.get('Token/format', np.nan), 1)} & "
            f"{pct(baseline_kept.get('Label-shuffled', np.nan), 1)} & "
            f"{pct(baseline_kept.get('Trace-order shuffled', np.nan), 1)} & "
            f"{num(gain, 1)} & {mean_ci(row, 'prefix_contamination', scale=100, digits=1)} \\\\"
        )
    pd.DataFrame(rows).to_csv(RESULTS / "artifact_adjusted_efficiency.csv", index=False)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "artifact_adjusted_efficiency.tex",
        "tab:artifact_adjusted_efficiency",
        "Artifact-adjusted clean-prefix efficiency at $\\alpha=0.05$. Gain is prefix-kept percentage points above the strongest artifact/shuffle baseline among token/format, label-shuffled combined, and trace-order-shuffled combined. This is the relevant efficiency quantity for cheap detectors; small gains mean performance is partly explained by artifacts.",
        "\n".join(lines),
    )


def build_risk_utility_table(process: pd.DataFrame, qwen: pd.DataFrame) -> None:
    process = add_prefix_accept_derivatives(process, OUT / "process_repeated_50seed" / "table_process_main.csv")
    qwen = add_prefix_accept_derivatives(qwen, OUT / "process_repeated_qwen_prm" / "table_external_score.csv")
    specs = [
        ("Qwen2.5-Math PRM", qwen, "qwen_prm_error"),
        ("Combined logistic", process, "combined_logistic_l2"),
        ("Text logistic", process, "text_logistic_l2"),
        ("Token/format", process, "artifact_token_formatting_logistic_l2"),
        ("Random", process, "random"),
        ("Oracle", process, "oracle"),
    ]
    rows = []
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Detector & Target & Emp. risk & Cal. risk UCB & Emp. utility & Utility LCB \\",
        r"\midrule",
    ]
    for label, df, score in specs:
        row = row_for(df, score)
        if row is None:
            continue
        utility_lcb = max(0.0, float(row["prefix_retained_fraction_mean"]) - float(row.get("prefix_retained_fraction_ci95", 0.0)))
        rows.append(
            {
                "detector": label,
                "alpha": 0.05,
                "empirical_risk": float(row["prefix_contamination_mean"]),
                "calibration_risk_bound": float(row["prefix_cal_corrected_risk_mean"]),
                "empirical_utility": float(row["prefix_retained_fraction_mean"]),
                "utility_lcb": utility_lcb,
            }
        )
        lines.append(
            f"{tex(label)} & 5.0 & "
            f"{mean_ci(row, 'prefix_contamination', scale=100, digits=1)} & "
            f"{mean_ci(row, 'prefix_cal_corrected_risk', scale=100, digits=1)} & "
            f"{mean_ci(row, 'prefix_retained_fraction', scale=100, digits=1)} & "
            f"{100.0 * utility_lcb:.1f} \\\\"
        )
    pd.DataFrame(rows).to_csv(RESULTS / "risk_utility_certification.csv", index=False)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    latex_table(
        TABLES / "risk_utility_certification.tex",
        "tab:risk_utility_certification",
        "Risk-utility view of CPCC at $\\alpha=0.05$. The corrected calibration risk is the risk-control upper bound used to select the threshold; utility is retained-prefix fraction, with an evaluation-split lower confidence bound shown as an efficiency diagnostic.",
        "\n".join(lines),
    )


def compute_routing_metrics(
    traces: list[TraceRecord],
    prefix_lengths_: Iterable[int],
    labels: Iterable[Iterable[int]] | None = None,
) -> dict[str, float]:
    """Compute CPCC routing metrics for trusted prefixes and routed suffixes."""

    traces = list(traces)
    lengths = np.asarray(list(prefix_lengths_), dtype=int)
    if labels is None:
        label_arrays = [trace.y_errors for trace in traces]
    else:
        label_arrays = [np.asarray(row, dtype=int) for row in labels]
    if len(lengths) != len(label_arrays):
        raise ValueError("prefix_lengths and labels/traces must have the same length")
    totals = np.asarray([len(row) for row in label_arrays], dtype=float)
    prefix_bad = []
    has_error = []
    first_errors = []
    full_accept = []
    for y, m in zip(label_arrays, lengths):
        y = np.asarray(y, dtype=int)
        hits = np.flatnonzero(y == 1)
        first_error = int(hits[0]) if len(hits) else -1
        has_error.append(first_error >= 0)
        first_errors.append(first_error)
        prefix_bad.append(bool(np.any(y[: max(int(m), 0)] == 1)))
        full_accept.append(int(m) >= len(y))
    has_error_arr = np.asarray(has_error, dtype=bool)
    first_errors_arr = np.asarray(first_errors, dtype=int)
    full_accept_arr = np.asarray(full_accept, dtype=bool)
    prefix_kept = float(np.mean(lengths / np.maximum(totals, 1.0))) if len(lengths) else float("nan")
    error_routed = lengths[has_error_arr] <= first_errors_arr[has_error_arr]
    accepted_error = float(np.mean(has_error_arr[full_accept_arr])) if np.any(full_accept_arr) else float("nan")
    return {
        "prefix_risk": float(np.mean(prefix_bad)) if prefix_bad else float("nan"),
        "prefix_kept": prefix_kept,
        "suffix_routed": 1.0 - prefix_kept if np.isfinite(prefix_kept) else float("nan"),
        "review_reduction": prefix_kept,
        "error_in_suffix_recall": float(np.mean(error_routed)) if len(error_routed) else float("nan"),
        "full_accept": float(np.mean(full_accept_arr)) if len(full_accept_arr) else float("nan"),
        "accepted_error": accepted_error,
    }


def compute_review_workflow_metrics(
    traces: list[TraceRecord],
    prefix_lengths_: Iterable[int],
    labels: Iterable[Iterable[int]] | None = None,
) -> dict[str, float]:
    """Compute normalized suffix-review metrics requested by revision10."""

    metrics = compute_routing_metrics(traces, prefix_lengths_, labels=labels)
    routed = metrics.get("error_in_suffix_recall", float("nan"))
    review_cost = metrics.get("suffix_routed", float("nan"))
    if np.isfinite(float(routed)) and float(routed) > 0.0 and np.isfinite(float(review_cost)):
        metrics["cost_per_routed_error"] = float(review_cost) / float(routed)
    else:
        metrics["cost_per_routed_error"] = float("nan")
    return metrics


def _raw_prefix_accept_derivatives(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    needed = {"trace_error_rate_test", "prefix_error_full_trace_rate", "prefix_full_trace_rate"}
    if needed.issubset(raw.columns):
        raw["prefix_marginal_false_accept"] = raw["trace_error_rate_test"] * raw["prefix_error_full_trace_rate"]
        raw["prefix_accepted_error_rate"] = np.where(
            raw["prefix_full_trace_rate"].astype(float) > 0,
            raw["prefix_marginal_false_accept"] / raw["prefix_full_trace_rate"],
            np.nan,
        )
    return raw


def build_routing_review_burden_table(process: pd.DataFrame, qwen: pd.DataFrame) -> None:
    process_raw = _raw_prefix_accept_derivatives(read_csv(OUT / "process_repeated_50seed" / "table_process_main.csv"))
    qwen_raw = _raw_prefix_accept_derivatives(read_csv(OUT / "process_repeated_qwen_prm" / "table_external_score.csv"))
    specs = [
        ("Qwen2.5-Math PRM", qwen_raw, "qwen_prm_error"),
        ("Combined logistic", process_raw, "combined_logistic_l2"),
        ("Text logistic", process_raw, "text_logistic_l2"),
        ("Token/format", process_raw, "artifact_token_formatting_logistic_l2"),
        ("Random", process_raw, "random"),
        ("Oracle", process_raw, "oracle"),
    ]
    rows = []
    for label, df, score in specs:
        if df.empty:
            continue
        subset = df[(df["score"] == score) & np.isclose(df["alpha"].astype(float), 0.05)].copy()
        if subset.empty:
            continue
        for _, row in subset.iterrows():
            kept = float(row["prefix_retained_fraction"])
            rows.append(
                {
                    "score_source": label,
                    "score": score,
                    "seed": int(row["seed"]),
                    "alpha": 0.05,
                    "prefix_risk": float(row["prefix_contamination"]),
                    "prefix_kept": kept,
                    "suffix_routed": 1.0 - kept,
                    "review_reduction": kept,
                    "error_in_suffix_recall": float(row.get("prefix_stops_at_or_before_first_error_rate", np.nan)),
                    "full_accept": float(row["prefix_full_trace_rate"]),
                    "accepted_error": float(row.get("prefix_accepted_error_rate", np.nan)),
                }
            )
    raw = pd.DataFrame(rows)
    summary = _summarize_with_ci_local(raw, ["score_source", "score", "alpha"]) if not raw.empty else pd.DataFrame()
    summary.to_csv(RESULTS / "routing_review_burden.csv", index=False)

    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Score source & Prefix risk & Prefix kept & Suffix routed & Review reduction & Error routed & Full accept & Acc. err. \\",
        r"\midrule",
    ]
    order = ["Qwen2.5-Math PRM", "Combined logistic", "Text logistic", "Token/format", "Random", "Oracle"]
    for label in order:
        subset = summary[summary["score_source"] == label] if not summary.empty else pd.DataFrame()
        if subset.empty:
            continue
        row = subset.iloc[0]
        lines.append(
            f"{tex(label)} & {mean_ci(row, 'prefix_risk', scale=100, digits=1)} & "
            f"{mean_ci(row, 'prefix_kept', scale=100, digits=1)} & "
            f"{mean_ci(row, 'suffix_routed', scale=100, digits=1)} & "
            f"{mean_ci(row, 'review_reduction', scale=100, digits=1)} & "
            f"{mean_ci(row, 'error_in_suffix_recall', scale=100, digits=1)} & "
            f"{mean_ci(row, 'full_accept', scale=100, digits=1)} & "
            f"{mean_ci(row, 'accepted_error', scale=100, digits=1)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "routing_review_burden.tex",
        "tab:routing_review_burden",
        "Routing and review-burden metrics at $\\alpha=0.05$. The certified prefix is trusted and the suffix is routed to review, repair, or another process verifier; this is an operational routing metric, not a proof of final-answer correctness. Strong routing performance from cheap scores may still exploit dataset artifacts.",
        "\n".join(lines),
    )


def build_object_level_comparison_table(process: pd.DataFrame) -> None:
    process = add_prefix_accept_derivatives(process, OUT / "process_repeated_50seed" / "table_process_main.csv")
    claim = read_csv(OUT / "remaining_baselines" / "table_claim_filtering_summary.csv")
    dyn = read_csv(OUT / "remaining_baselines" / "table_dynamic_early_stop_summary.csv")
    combined = row_for(process, "combined_logistic_l2")
    oracle = row_for(process, "oracle")
    claim05 = claim[(claim["score"] == "combined_logistic_l2") & np.isclose(claim["alpha"].astype(float), 0.05)].iloc[0] if not claim.empty else None
    dyn05 = dyn[(dyn["score"] == "combined_logistic_l2") & np.isclose(dyn["alpha"].astype(float), 0.05)].iloc[0] if not dyn.empty else None

    rows: list[dict] = []
    if combined is not None:
        rows.append(
            {
                "method": "Whole-trace abstention",
                "returned_object": "all or nothing",
                "contiguous": True,
                "post_hoc": True,
                "formal_risk_target": "false accept",
                "risk": float(combined["trace_abstention_test_loss_mean"]),
                "risk_ci95": float(combined.get("trace_abstention_test_loss_ci95", np.nan)),
                "certified_step_fraction": float(combined["accept_rate_mean"]),
                "certified_step_fraction_ci95": float(combined.get("accept_rate_ci95", np.nan)),
                "full_accept": float(combined["accept_rate_mean"]),
                "full_accept_ci95": float(combined.get("accept_rate_ci95", np.nan)),
                "review_burden": 1.0 - float(combined["accept_rate_mean"]),
                "review_burden_ci95": float(combined.get("accept_rate_ci95", np.nan)),
            }
        )
        rows.append(
            {
                "method": "CPCC",
                "returned_object": "clean prefix",
                "contiguous": True,
                "post_hoc": True,
                "formal_risk_target": "prefix contamination",
                "risk": float(combined["prefix_contamination_mean"]),
                "risk_ci95": float(combined.get("prefix_contamination_ci95", np.nan)),
                "certified_step_fraction": float(combined["prefix_retained_fraction_mean"]),
                "certified_step_fraction_ci95": float(combined.get("prefix_retained_fraction_ci95", np.nan)),
                "full_accept": float(combined["prefix_full_trace_rate_mean"]),
                "full_accept_ci95": float(combined.get("prefix_full_trace_rate_ci95", np.nan)),
                "review_burden": 1.0 - float(combined["prefix_retained_fraction_mean"]),
                "review_burden_ci95": float(combined.get("prefix_retained_fraction_ci95", np.nan)),
            }
        )
    if claim05 is not None:
        rows.append(
            {
                "method": "Step filtering",
                "returned_object": "scattered steps",
                "contiguous": False,
                "post_hoc": True,
                "formal_risk_target": "any accepted step wrong",
                "risk": float(claim05["test_risk_mean"]),
                "risk_ci95": float(claim05.get("test_risk_ci95", np.nan)),
                "certified_step_fraction": float(claim05["retained_step_fraction_mean"]),
                "certified_step_fraction_ci95": float(claim05.get("retained_step_fraction_ci95", np.nan)),
                "full_accept": np.nan,
                "full_accept_ci95": np.nan,
                "review_burden": 1.0 - float(claim05["retained_step_fraction_mean"]),
                "review_burden_ci95": float(claim05.get("retained_step_fraction_ci95", np.nan)),
            }
        )
    if dyn05 is not None:
        rows.append(
            {
                "method": "Dynamic early stop",
                "returned_object": "prefix / stop point",
                "contiguous": True,
                "post_hoc": False,
                "formal_risk_target": "stopping prefix risk",
                "risk": float(dyn05["prefix_contamination_mean"]),
                "risk_ci95": float(dyn05.get("prefix_contamination_ci95", np.nan)),
                "certified_step_fraction": float(dyn05["retained_fraction_mean"]),
                "certified_step_fraction_ci95": float(dyn05.get("retained_fraction_ci95", np.nan)),
                "full_accept": float(dyn05["accept_rate_mean"]),
                "full_accept_ci95": float(dyn05.get("accept_rate_ci95", np.nan)),
                "review_burden": 1.0 - float(dyn05["retained_fraction_mean"]),
                "review_burden_ci95": float(dyn05.get("retained_fraction_ci95", np.nan)),
            }
        )
    if oracle is not None:
        rows.append(
            {
                "method": "Oracle clean prefix",
                "returned_object": "clean prefix",
                "contiguous": True,
                "post_hoc": True,
                "formal_risk_target": "zero contamination",
                "risk": float(oracle["prefix_contamination_mean"]),
                "risk_ci95": float(oracle.get("prefix_contamination_ci95", np.nan)),
                "certified_step_fraction": float(oracle["prefix_retained_fraction_mean"]),
                "certified_step_fraction_ci95": float(oracle.get("prefix_retained_fraction_ci95", np.nan)),
                "full_accept": float(oracle["prefix_full_trace_rate_mean"]),
                "full_accept_ci95": float(oracle.get("prefix_full_trace_rate_ci95", np.nan)),
                "review_burden": 1.0 - float(oracle["prefix_retained_fraction_mean"]),
                "review_burden_ci95": float(oracle.get("prefix_retained_fraction_ci95", np.nan)),
            }
        )
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "object_level_comparison.csv", index=False)

    def pm(row: pd.Series, metric: str, digits: int = 1) -> str:
        value = row.get(metric, np.nan)
        ci = row.get(f"{metric}_ci95", np.nan)
        if not np.isfinite(float(value)):
            return "--"
        if np.isfinite(float(ci)):
            return f"${100.0 * float(value):.{digits}f} \\pm {100.0 * float(ci):.{digits}f}$"
        return f"${100.0 * float(value):.{digits}f}$"

    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllclrrrr}",
        r"\toprule",
        r"Method / object & Returned object & Contig.? & Post-hoc? & Formal risk target & Risk & Cert. steps & Full accept & Review burden \\",
        r"\midrule",
    ]
    order = ["Whole-trace abstention", "Step filtering", "Dynamic early stop", "CPCC", "Oracle clean prefix"]
    for method in order:
        subset = raw[raw["method"] == method] if not raw.empty else pd.DataFrame()
        if subset.empty:
            continue
        row = subset.iloc[0]
        lines.append(
            f"{tex(row['method'])} & {tex(row['returned_object'])} & "
            f"{'yes' if bool(row['contiguous']) else 'no'} & "
            f"{'yes' if bool(row['post_hoc']) else 'no'} & "
            f"{tex(row['formal_risk_target'])} & {pm(row, 'risk')} & "
            f"{pm(row, 'certified_step_fraction')} & {pm(row, 'full_accept')} & "
            f"{pm(row, 'review_burden')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "object_level_comparison.tex",
        "tab:object_level_comparison",
        "Object-level comparison at $\\alpha=0.05$ with the combined detector. The table compares uncertainty objects, not verifier architectures. Step filtering can retain scattered steps; CPCC is the post-hoc contiguous object for the deployment question of where the completed trace should stop being trusted.",
        "\n".join(lines),
    )


def build_external_artifact_adjusted_table() -> None:
    dataset_specs = [
        ("ProcessBench", OUT / "external_process" / "processbench_repeated_full10" / "table_process_main_summary.csv", "first-error / process labels"),
        ("Math-Shepherd", OUT / "external_process" / "math_shepherd_repeated" / "table_process_main_summary.csv", "step markup labels"),
        ("PRMBench", OUT / "external_process" / "prmbench_repeated" / "table_process_main_summary.csv", "fine-grained step labels"),
        ("PRM800K", OUT / "external_process" / "prm800k_repeated_focused" / "table_process_main_summary.csv", "step labels"),
    ]
    qwen_sources = {
        "ProcessBench": read_csv(OUT / "external_process" / "processbench_qwen_prm" / "table_external_score_summary.csv"),
        "Math-Shepherd": read_csv(OUT / "external_process" / "math_shepherd_qwen_prm" / "table_external_score_summary.csv"),
        "PRMBench": read_csv(OUT / "external_process" / "prmbench_full_qwen_prm" / "table_external_score_summary.csv"),
        "PRM800K": read_csv(OUT / "external_process" / "prm800k_qwen_prm" / "table_external_score_summary.csv"),
    }
    rows = []
    for dataset, path, label_type in dataset_specs:
        df = read_csv(path)
        if df.empty:
            continue
        token = row_for(df, "artifact_token_formatting_logistic_l2")
        artifact_kept = float(token["prefix_retained_fraction_mean"]) if token is not None else np.nan
        for detector, score, source in (
            ("Qwen2.5-Math PRM", "qwen_prm_error", qwen_sources.get(dataset, pd.DataFrame())),
            ("Combined logistic", "combined_logistic_l2", df),
            ("Token/format", "artifact_token_formatting_logistic_l2", df),
            ("Random", "random", df),
            ("Oracle", "oracle", df),
        ):
            row = row_for(source, score)
            if row is None:
                continue
            kept = float(row["prefix_retained_fraction_mean"])
            rows.append(
                {
                    "dataset": dataset,
                    "label_type": label_type,
                    "detector": detector,
                    "prefix_risk": float(row["prefix_contamination_mean"]),
                    "prefix_kept": kept,
                    "artifact_kept": artifact_kept,
                    "delta_artifact": 100.0 * (kept - artifact_kept) if np.isfinite(artifact_kept) else np.nan,
                    "oracle_kept": float(row_for(df, "oracle")["prefix_retained_fraction_mean"]) if row_for(df, "oracle") is not None else np.nan,
                }
            )
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "external_artifact_adjusted.csv", index=False)
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"Dataset & Label type & Detector & Prefix risk & Prefix kept & Artifact kept & $\Delta$ artifact & Oracle kept \\",
        r"\midrule",
    ]
    for _, row in raw.iterrows():
        lines.append(
            f"{tex(row['dataset'])} & {tex(row['label_type'])} & {tex(row['detector'])} & "
            f"{100.0 * row['prefix_risk']:.1f} & {100.0 * row['prefix_kept']:.1f} & "
            f"{100.0 * row['artifact_kept']:.1f} & {row['delta_artifact']:.1f} & "
            f"{100.0 * row['oracle_kept']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "external_artifact_adjusted.tex",
        "tab:external_artifact_adjusted",
        "External artifact-adjusted CPCC stress tests at $\\alpha=0.05$. The artifact baseline is the token/format control within each dataset. Efficient certification transfers only when the score is suitable for the target process distribution.",
        "\n".join(lines),
    )


def build_external_strong_verifier_subset_table() -> None:
    """Summarize external strong-verifier caches that are locally available."""

    qwen_pb = add_prefix_accept_derivatives(
        read_csv(OUT / "external_process" / "processbench_qwen_prm" / "table_external_score_summary.csv"),
        OUT / "external_process" / "processbench_qwen_prm" / "table_external_score.csv",
    )
    pb = read_csv(OUT / "external_process" / "processbench_repeated_full10" / "table_process_main_summary.csv")
    token = row_for(pb, "artifact_token_formatting_logistic_l2")
    artifact_kept = float(token["prefix_retained_fraction_mean"]) if token is not None else np.nan
    rows = []
    row = row_for(qwen_pb, "qwen_prm_error")
    if row is not None:
        kept = float(row["prefix_retained_fraction_mean"])
        rows.append(
            {
                "dataset": "ProcessBench",
                "detector": "Qwen2.5-Math PRM",
                "scope": "cached 10-split external run",
                "prefix_risk": float(row["prefix_contamination_mean"]),
                "prefix_kept": kept,
                "artifact_kept": artifact_kept,
                "artifact_adjusted_gain": 100.0 * (kept - artifact_kept) if np.isfinite(artifact_kept) else np.nan,
                "full_accept": float(row["prefix_full_trace_rate_mean"]),
                "accepted_error": float(row.get("prefix_accepted_error_rate_mean", np.nan)),
                "oracle_kept": np.nan,
            }
        )
    subset_dir = OUT / "external_process" / "prmbench_qwen_subset500"
    qwen_prmbench = add_prefix_accept_derivatives(
        read_csv(subset_dir / "qwen_prm_repeated" / "table_external_score_summary.csv"),
        subset_dir / "qwen_prm_repeated" / "table_external_score.csv",
    )
    prmbench_subset = add_prefix_accept_derivatives(
        read_csv(subset_dir / "repeated_subset" / "table_process_main_summary.csv"),
        subset_dir / "repeated_subset" / "table_process_main.csv",
    )
    token = row_for(prmbench_subset, "artifact_token_formatting_logistic_l2")
    oracle = row_for(prmbench_subset, "oracle")
    token_kept = float(token["prefix_retained_fraction_mean"]) if token is not None else np.nan
    oracle_kept = float(oracle["prefix_retained_fraction_mean"]) if oracle is not None else np.nan
    subset_rows = []
    for detector, source, score in (
        ("Qwen2.5-Math PRM", qwen_prmbench, "qwen_prm_error"),
        ("Combined logistic", prmbench_subset, "combined_logistic_l2"),
        ("Token/format", prmbench_subset, "artifact_token_formatting_logistic_l2"),
    ):
        row = row_for(source, score)
        if row is None:
            continue
        kept = float(row["prefix_retained_fraction_mean"])
        payload = {
            "dataset": "PRMBench",
            "detector": detector,
            "scope": "500-trace subset; 10 splits",
            "prefix_risk": float(row["prefix_contamination_mean"]),
            "prefix_kept": kept,
            "artifact_kept": token_kept,
            "artifact_adjusted_gain": 100.0 * (kept - token_kept) if np.isfinite(token_kept) else np.nan,
            "oracle_kept": oracle_kept,
            "full_accept": float(row["prefix_full_trace_rate_mean"]),
            "accepted_error": float(row.get("prefix_accepted_error_rate_mean", np.nan)),
        }
        rows.append(payload)
        subset_rows.append(payload)
    if subset_rows:
        pd.DataFrame(subset_rows).to_csv(RESULTS / "prmbench_qwen_prm_subset_cpcc.csv", index=False)
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "external_strong_verifier_subset.csv", index=False)
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrrrr}",
        r"\toprule",
        r"Dataset & Detector & Scope & Prefix risk & Prefix kept & Token kept & $\Delta$ token & Oracle kept & Full accept & Acc. err. \\",
        r"\midrule",
    ]
    for _, row in raw.iterrows():
        oracle_text = f"{100.0 * row['oracle_kept']:.1f}" if "oracle_kept" in row and np.isfinite(float(row["oracle_kept"])) else "--"
        lines.append(
            f"{tex(row['dataset'])} & {tex(row['detector'])} & {tex(row['scope'])} & "
            f"{100.0 * row['prefix_risk']:.1f} & {100.0 * row['prefix_kept']:.1f} & "
            f"{100.0 * row['artifact_kept']:.1f} & {row['artifact_adjusted_gain']:.1f} & "
            f"{oracle_text} & {100.0 * row['full_accept']:.1f} & {100.0 * row['accepted_error']:.1f} \\\\"
        )
    if raw.empty:
        lines.append(r"No external strong-verifier score cache was available. & -- & -- & -- & -- & -- & -- & -- & -- & -- \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "external_strong_verifier_subset.tex",
        "tab:external_strong_verifier_subset",
        "External strong-verifier subset results at $\\alpha=0.05$. ProcessBench uses the existing cached Qwen PRM run. PRMBench uses a newly scored 500-trace Qwen PRM subset and compares Qwen, combined, and token/format on the same traces. Negative gains indicate verifier-dataset mismatch rather than a calibration failure.",
        "\n".join(lines),
    )
    if subset_rows:
        subset_lines = [
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r"Score source & Prefix risk & Prefix kept & Token kept & $\Delta$ token & Oracle kept & Full accept & Acc. err. \\",
            r"\midrule",
        ]
        for row in subset_rows:
            subset_lines.append(
                f"{tex(row['detector'])} & {100.0 * row['prefix_risk']:.1f} & "
                f"{100.0 * row['prefix_kept']:.1f} & {100.0 * row['artifact_kept']:.1f} & "
                f"{row['artifact_adjusted_gain']:.1f} & {100.0 * row['oracle_kept']:.1f} & "
                f"{100.0 * row['full_accept']:.1f} & {100.0 * row['accepted_error']:.1f} \\\\"
            )
        subset_lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
        latex_table(
            TABLES / "prmbench_qwen_prm_subset.tex",
            "tab:prmbench_qwen_prm_subset",
            "PRMBench 500-trace Qwen PRM subset stress test at $\\alpha=0.05$. Qwen PRM controls prefix risk but retains less prefix than token/format on this subset, showing that CPCC can expose verifier-dataset mismatch.",
            "\n".join(subset_lines),
        )


def build_prmbench_aligned_verifier_table() -> None:
    """Summarize Revision6 PRMBench-aligned verifier experiment."""

    outdir = OUT / "external_process" / "prmbench_aligned_verifier"
    summary = add_prefix_accept_derivatives(
        read_csv(outdir / "table_prmbench_aligned_verifier_summary.csv"),
        outdir / "table_prmbench_aligned_verifier.csv",
    )
    if summary.empty:
        return
    order = [
        "Token/format",
        "Combined logistic",
        "Qwen2.5-Math PRM",
        "PRMBench-native GBM",
        "Prefix-feature GBM",
        "Qwen+native ensemble",
        "Oracle",
    ]
    token = row_for(summary, "Token/format")
    oracle = row_for(summary, "Oracle")
    token_kept = float(token["prefix_retained_fraction_mean"]) if token is not None else np.nan
    oracle_kept = float(oracle["prefix_retained_fraction_mean"]) if oracle is not None else np.nan
    denom = oracle_kept - token_kept if np.isfinite(oracle_kept) and np.isfinite(token_kept) else np.nan

    rows = []
    for detector in order:
        row = row_for(summary, detector)
        if row is None:
            continue
        kept = float(row["prefix_retained_fraction_mean"])
        delta_token = 100.0 * (kept - token_kept) if np.isfinite(token_kept) else np.nan
        gap_closed = (kept - token_kept) / denom if np.isfinite(denom) and denom > 0 else np.nan
        rows.append(
            {
                "dataset": "PRMBench",
                "detector": detector,
                "scope": "500-trace Qwen subset; 10 splits; 2500-trace aligned train pool"
                if detector in {"PRMBench-native GBM", "Prefix-feature GBM", "Qwen+native ensemble"}
                else "same 500-trace subset; 10 splits",
                "auroc": float(row.get("auroc_mean", np.nan)),
                "prefix_risk": float(row["prefix_contamination_mean"]),
                "prefix_kept": kept,
                "token_kept": token_kept,
                "delta_token": delta_token,
                "oracle_kept": oracle_kept,
                "oracle_gap_closed": gap_closed,
                "full_accept": float(row["prefix_full_trace_rate_mean"]),
                "accepted_error": float(row.get("prefix_accepted_error_rate_mean", np.nan)),
            }
        )
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "prmbench_aligned_verifier_cpcc.csv", index=False)

    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Detector & Scope & AUROC & Prefix risk & Prefix kept & $\Delta$ token & Gap closed & Oracle kept & Full accept & Acc. err. \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{tex(row['detector'])} & {tex(row['scope'])} & {row['auroc']:.3f} & "
            f"{100.0 * row['prefix_risk']:.1f} & {100.0 * row['prefix_kept']:.1f} & "
            f"{row['delta_token']:.1f} & {100.0 * row['oracle_gap_closed']:.1f} & "
            f"{100.0 * row['oracle_kept']:.1f} & {100.0 * row['full_accept']:.1f} & "
            f"{100.0 * row['accepted_error']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "prmbench_aligned_verifier_cpcc.tex",
        "tab:prmbench_aligned_verifier",
        "PRMBench-aligned verifier experiment at $\\alpha=0.05$. Native GBM scores are trained on a disjoint 2500-trace PRMBench pool plus each split's training traces, then calibrated and tested on the same 500-trace subset used for Qwen PRM. Gap closed is the fraction of the token/format-to-oracle prefix-efficiency gap recovered. These rows test verifier-dataset alignment, not a new verifier contribution.",
        "\n".join(lines),
    )

    review = read_csv(outdir / "table_prmbench_review_efficiency_summary.csv")
    if review.empty:
        return
    review_rows = []
    for detector in order:
        row = row_for(review, detector)
        if row is None:
            continue
        review_rows.append(
            {
                "detector": detector,
                "prefix_kept_step_weighted": float(row["prefix_kept_step_weighted_mean"]),
                "suffix_routed": float(row["suffix_routed_step_fraction_mean"]),
                "first_error_routed": float(row["first_error_routed_rate_mean"]),
                "full_accept": float(row["full_accept_mean"]),
                "review_steps_per_routed_error": float(row["review_steps_per_routed_error_mean"]),
            }
        )
    pd.DataFrame(review_rows).to_csv(RESULTS / "prmbench_review_efficiency.csv", index=False)
    review_lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Detector & Prefix kept & Suffix routed & First-error routed & Full accept & Review steps/error \\",
        r"\midrule",
    ]
    for row in review_rows:
        review_lines.append(
            f"{tex(row['detector'])} & {100.0 * row['prefix_kept_step_weighted']:.1f} & "
            f"{100.0 * row['suffix_routed']:.1f} & {100.0 * row['first_error_routed']:.1f} & "
            f"{100.0 * row['full_accept']:.1f} & {row['review_steps_per_routed_error']:.1f} \\\\"
        )
    review_lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "prmbench_review_efficiency.tex",
        "tab:prmbench_review_efficiency",
        "PRMBench review-efficiency diagnostic for routing the uncertified suffix. Prefix kept is step-weighted; first-error routed is the fraction of erroneous traces whose first error lies in the routed suffix.",
        "\n".join(review_lines),
    )


def build_prmbench_large_subset_alignment_table() -> None:
    """Summarize the larger Qwen-scored PRMBench matched-union run, if present."""

    outdir = OUT / "external_process" / "prmbench_large_subset" / "aligned_verifier"
    summary = add_prefix_accept_derivatives(
        read_csv(outdir / "table_prmbench_aligned_verifier_summary.csv"),
        outdir / "table_prmbench_aligned_verifier.csv",
    )
    scope = ""
    config_path = outdir / "run_config.json"
    n_subset = "--"
    if config_path.exists():
        try:
            n_subset = str(json.loads(config_path.read_text()).get("n_subset_traces", "--"))
        except json.JSONDecodeError:
            n_subset = "--"
    if not summary.empty:
        scope = f"{n_subset}-trace Qwen-scored matched-union subset"
    else:
        root = OUT / "external_process" / "prmbench_matched_subsets"
        manifest_path = root / "matched_manifest.json"
        if not manifest_path.exists():
            return
        manifest = json.loads(manifest_path.read_text())
        raw_parts = []
        for subset_dir in manifest.get("subset_dirs", []):
            raw = read_csv(root / str(subset_dir) / "aligned_verifier" / "table_prmbench_aligned_verifier.csv")
            if raw.empty:
                continue
            raw = _raw_prefix_accept_derivatives(raw)
            raw["matched_subset"] = str(subset_dir)
            raw_parts.append(raw)
        if not raw_parts:
            return
        raw_large = pd.concat(raw_parts, ignore_index=True)
        summary = _summarize_with_ci_local(raw_large, ["score", "alpha"])
        union_traces = manifest.get("union_traces", "--")
        subset_count = len(manifest.get("subset_dirs", []))
        scope = f"{subset_count} matched 500-trace subsets; {union_traces} unique Qwen-scored traces"

    order = [
        "Token/format",
        "Combined logistic",
        "Qwen2.5-Math PRM",
        "PRMBench-native GBM",
        "Prefix-feature GBM",
        "Qwen+native ensemble",
        "Oracle",
    ]
    token = row_for(summary, "Token/format")
    oracle = row_for(summary, "Oracle")
    token_kept = float(token["prefix_retained_fraction_mean"]) if token is not None else np.nan
    oracle_kept = float(oracle["prefix_retained_fraction_mean"]) if oracle is not None else np.nan
    denom = oracle_kept - token_kept if np.isfinite(oracle_kept) and np.isfinite(token_kept) else np.nan

    rows = []
    for detector in order:
        row = row_for(summary, detector)
        if row is None:
            continue
        kept = float(row["prefix_retained_fraction_mean"])
        gain = 100.0 * (kept - token_kept) if np.isfinite(token_kept) else np.nan
        gap_closed = (kept - token_kept) / denom if np.isfinite(denom) and denom > 0 else np.nan
        rows.append(
            {
                "detector": detector,
                "scope": scope,
                "auroc": float(row.get("auroc_mean", np.nan)),
                "prefix_risk": float(row.get("prefix_contamination_mean", np.nan)),
                "prefix_kept": kept,
                "token_kept": token_kept,
                "gain": gain,
                "oracle_kept": oracle_kept,
                "gap_closed": gap_closed,
                "full_accept": float(row.get("prefix_full_trace_rate_mean", np.nan)),
                "accepted_error": float(row.get("prefix_accepted_error_rate_mean", np.nan)),
            }
        )
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "prmbench_large_subset_alignment.csv", index=False)
    if raw.empty:
        return

    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrrr}",
        r"\toprule",
        r"Detector & Scope & AUROC & Prefix risk & Prefix kept & Token kept & Gain & Oracle kept & Gap closed & Full accept & Acc. err. \\",
        r"\midrule",
    ]
    for _, row in raw.iterrows():
        lines.append(
            f"{tex(row['detector'])} & {tex(row['scope'])} & {num(row['auroc'], 3)} & "
            f"{pct(row['prefix_risk'])} & {pct(row['prefix_kept'])} & {pct(row['token_kept'])} & "
            f"{num(row['gain'], 1)} & {pct(row['oracle_kept'])} & {pct(row['gap_closed'])} & "
            f"{pct(row['full_accept'])} & {pct(row['accepted_error'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "prmbench_large_subset_alignment.tex",
        "tab:prmbench_large_subset_alignment",
        "Large PRMBench aligned-verifier summary at $\\alpha=0.05$ over the Qwen-scored matched-subset pool. The rows use the same fixed detector list as the 500-trace study and test whether verifier-dataset alignment persists beyond a single subset.",
        "\n".join(lines),
    )


def build_prmbench_subset_robustness_table() -> None:
    """Summarize disjoint PRMBench subset robustness for the aligned ensemble."""

    subset_specs = [
        ("A", OUT / "external_process" / "prmbench_aligned_verifier", "original Qwen subset"),
        ("B", OUT / "external_process" / "prmbench_subset_robust_b" / "aligned_verifier", "random disjoint subset"),
        ("C", OUT / "external_process" / "prmbench_subset_robust_c" / "aligned_verifier", "random disjoint subset"),
    ]
    detectors = ["Token/format", "Qwen2.5-Math PRM", "Qwen+native ensemble", "Oracle"]
    rows = []
    for subset, outdir, scope in subset_specs:
        summary = read_csv(outdir / "table_prmbench_aligned_verifier_summary.csv")
        if summary.empty:
            continue
        token = row_for(summary, "Token/format")
        oracle = row_for(summary, "Oracle")
        token_kept = float(token["prefix_retained_fraction_mean"]) if token is not None else np.nan
        oracle_kept = float(oracle["prefix_retained_fraction_mean"]) if oracle is not None else np.nan
        denom = oracle_kept - token_kept if np.isfinite(token_kept) and np.isfinite(oracle_kept) else np.nan
        for detector in detectors:
            row = row_for(summary, detector)
            if row is None:
                continue
            kept = float(row["prefix_retained_fraction_mean"])
            rows.append(
                {
                    "subset": subset,
                    "scope": scope,
                    "detector": detector,
                    "trace_error_rate": float(row.get("trace_error_rate_test_mean", np.nan)),
                    "prefix_risk": float(row["prefix_contamination_mean"]),
                    "prefix_risk_ci95": float(row.get("prefix_contamination_ci95", np.nan)),
                    "prefix_kept": kept,
                    "prefix_kept_ci95": float(row.get("prefix_retained_fraction_ci95", np.nan)),
                    "token_kept": token_kept,
                    "gain": 100.0 * (kept - token_kept) if np.isfinite(token_kept) else np.nan,
                    "oracle_kept": oracle_kept,
                    "gap_closed": (kept - token_kept) / denom if np.isfinite(denom) and denom > 0 else np.nan,
                    "full_accept": float(row["prefix_full_trace_rate_mean"]),
                    "full_accept_ci95": float(row.get("prefix_full_trace_rate_ci95", np.nan)),
                }
            )
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "prmbench_subset_robustness.csv", index=False)
    focus = raw[raw["detector"] == "Qwen+native ensemble"].copy() if not raw.empty else pd.DataFrame()

    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Subset & Scope & Err. traces & Prefix risk & Prefix kept & Token kept & Gain & Gap closed & Full accept \\",
        r"\midrule",
    ]
    for _, row in focus.iterrows():
        lines.append(
            f"{tex(row['subset'])} & {tex(row['scope'])} & {100.0 * row['trace_error_rate']:.1f} & "
            f"${100.0 * row['prefix_risk']:.1f} \\pm {100.0 * row['prefix_risk_ci95']:.1f}$ & "
            f"${100.0 * row['prefix_kept']:.1f} \\pm {100.0 * row['prefix_kept_ci95']:.1f}$ & "
            f"{100.0 * row['token_kept']:.1f} & {row['gain']:.1f} & "
            f"{100.0 * row['gap_closed']:.1f} & "
            f"${100.0 * row['full_accept']:.1f} \\pm {100.0 * row['full_accept_ci95']:.1f}$ \\\\"
        )
    if not focus.empty:
        lines.append(r"\midrule")
        lines.append(
            "Mean & across subsets & "
            f"{100.0 * focus['trace_error_rate'].mean():.1f} & "
            f"{mean_ci_values(focus['prefix_risk'], scale=100, digits=1)} & "
            f"{mean_ci_values(focus['prefix_kept'], scale=100, digits=1)} & "
            f"{100.0 * focus['token_kept'].mean():.1f} & "
            f"{focus['gain'].mean():.1f} & "
            f"{100.0 * focus['gap_closed'].mean():.1f} & "
            f"{mean_ci_values(focus['full_accept'], scale=100, digits=1)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "prmbench_subset_robustness.tex",
        "tab:prmbench_subset_robustness",
        "PRMBench subset robustness for the Qwen+native aligned ensemble at $\\alpha=0.05$. Subsets are pairwise disjoint 500-trace subsets; native models are fit only on a disjoint PRMBench training pool plus each split's training traces. Calibration and test traces are never used for fitting, feature selection, threshold-grid selection, or model selection. B and C are random high-error subsets, so their absolute prefix retention is lower than subset A.",
        "\n".join(lines),
    )


def build_prmbench_stratified_subset_table() -> None:
    """Summarize pooled-threshold CPCC diagnostics by PRMBench error stratum."""

    outdir = OUT / "external_process" / "prmbench_stratified_subset" / "aligned_verifier"
    summary = read_csv(outdir / "table_prmbench_stratified_summary.csv")
    if summary.empty:
        return
    strata_order = {
        "clean": 0,
        "early-error": 1,
        "middle-error": 2,
        "late-error": 3,
        "high-error": 4,
    }
    detectors = [
        "Token/format",
        "Combined logistic",
        "Qwen2.5-Math PRM",
        "PRMBench-native GBM",
        "Qwen+native ensemble",
    ]

    def stratum_row(stratum: str, score: str) -> pd.Series | None:
        subset = summary[
            (summary["domain"] == stratum)
            & (summary["score"] == score)
            & np.isclose(summary["alpha"].astype(float), 0.05)
        ]
        if subset.empty:
            return None
        return subset.iloc[0]

    rows = []
    for stratum in sorted(summary["domain"].dropna().unique(), key=lambda value: (strata_order.get(value, 999), value)):
        token = stratum_row(stratum, "Token/format")
        oracle = stratum_row(stratum, "Oracle")
        token_kept = float(token["prefix_retained_fraction_mean"]) if token is not None else np.nan
        oracle_kept = float(oracle["prefix_retained_fraction_mean"]) if oracle is not None else np.nan
        denom = oracle_kept - token_kept if np.isfinite(token_kept) and np.isfinite(oracle_kept) else np.nan
        for detector in detectors:
            row = stratum_row(stratum, detector)
            if row is None:
                continue
            kept = float(row["prefix_retained_fraction_mean"])
            rows.append(
                {
                    "stratum": stratum,
                    "detector": detector,
                    "n_test_traces": float(row.get("n_test_traces_mean", np.nan)),
                    "trace_error_rate": float(row.get("trace_error_rate_test_mean", np.nan)),
                    "step_error_rate": float(row.get("step_error_rate_test_mean", np.nan)),
                    "prefix_risk": float(row.get("prefix_contamination_mean", np.nan)),
                    "prefix_risk_ci95": float(row.get("prefix_contamination_ci95", np.nan)),
                    "prefix_kept": kept,
                    "prefix_kept_ci95": float(row.get("prefix_retained_fraction_ci95", np.nan)),
                    "token_kept": token_kept,
                    "gain": 100.0 * (kept - token_kept) if np.isfinite(token_kept) else np.nan,
                    "oracle_kept": oracle_kept,
                    "gap_closed": (kept - token_kept) / denom if np.isfinite(denom) and denom > 0 else np.nan,
                    "full_accept": float(row.get("prefix_full_trace_rate_mean", np.nan)),
                    "full_accept_ci95": float(row.get("prefix_full_trace_rate_ci95", np.nan)),
                }
            )
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "prmbench_stratified_subset_cpcc.csv", index=False)

    def fmt_ci(mean: float, ci: float, scale: float = 100.0) -> str:
        if not np.isfinite(float(mean)):
            return "--"
        if np.isfinite(float(ci)):
            return f"${scale * float(mean):.1f} \\pm {scale * float(ci):.1f}$"
        return f"${scale * float(mean):.1f}$"

    def fmt_gap(value: float) -> str:
        if not np.isfinite(float(value)) or float(value) < 0.0 or float(value) > 1.0:
            return "--"
        return pct(value)

    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrrr}",
        r"\toprule",
        r"Stratum & Detector & Err. traces & Step err. & Prefix risk & Prefix kept & Token kept & Gain & Oracle kept & Gap closed & Full accept \\",
        r"\midrule",
    ]
    for _, row in raw.iterrows():
        lines.append(
            f"{tex(row['stratum'])} & {tex(row['detector'])} & {pct(row['trace_error_rate'])} & "
            f"{pct(row['step_error_rate'])} & "
            f"{fmt_ci(row['prefix_risk'], row['prefix_risk_ci95'])} & "
            f"{fmt_ci(row['prefix_kept'], row['prefix_kept_ci95'])} & "
            f"{pct(row['token_kept'])} & {num(row['gain'], 1)} & "
            f"{pct(row['oracle_kept'])} & {fmt_gap(row['gap_closed'])} & {pct(row['full_accept'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "prmbench_stratified_subset_cpcc.tex",
        "tab:prmbench_stratified_subset",
        "PRMBench stratified subset diagnostic at $\\alpha=0.05$. The subset balances clean traces, first-error position strata, and very high-error traces. Rows reuse pooled calibration thresholds and report within-stratum test behavior, so they diagnose where verifier-dataset alignment helps rather than claiming group-conditional validity.",
        "\n".join(lines),
    )


def build_prmbench_stratum_summary_table() -> None:
    """Write a compact stratum-level summary focused on the aligned ensemble."""

    raw = read_csv(RESULTS / "prmbench_stratified_subset_cpcc.csv")
    if raw.empty:
        return
    interpretations = {
        "clean": "Token/format already certifies most clean traces.",
        "early-error": "All methods must stop early; pooled within-stratum risk is diagnostic.",
        "middle-error": "There is room beyond token/format when the first error is not immediate.",
        "late-error": "Later first errors leave more certifiable prefix for aligned scores.",
        "high-error": "High error density leaves room but limits full-trace acceptance.",
    }
    order = ["clean", "early-error", "middle-error", "late-error", "high-error"]
    rows = []
    for stratum in order:
        token = raw[(raw["stratum"] == stratum) & (raw["detector"] == "Token/format")]
        ensemble = raw[(raw["stratum"] == stratum) & (raw["detector"] == "Qwen+native ensemble")]
        oracle = raw[(raw["stratum"] == stratum) & (raw["detector"] == "Oracle")]
        if token.empty or ensemble.empty:
            continue
        token_row = token.iloc[0]
        ensemble_row = ensemble.iloc[0]
        oracle_kept = float(oracle.iloc[0]["prefix_kept"]) if not oracle.empty else float(ensemble_row["oracle_kept"])
        rows.append(
            {
                "stratum": stratum,
                "token_kept": float(token_row["prefix_kept"]),
                "ensemble_kept": float(ensemble_row["prefix_kept"]),
                "gain": float(ensemble_row["gain"]),
                "oracle_kept": oracle_kept,
                "gap_closed": float(ensemble_row["gap_closed"]),
                "prefix_risk": float(ensemble_row["prefix_risk"]),
                "interpretation": interpretations.get(stratum, "Subset-dependent diagnostic."),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "prmbench_stratum_summary.csv", index=False)
    if summary.empty:
        return
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrp{0.34\textwidth}}",
        r"\toprule",
        r"Stratum & Token kept & Ensemble kept & Gain & Oracle kept & Gap closed & Prefix risk & Interpretation \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        gap_value = float(row["gap_closed"])
        gap = "--" if (not np.isfinite(gap_value) or gap_value < 0.0 or gap_value > 1.0) else pct(gap_value)
        lines.append(
            f"{tex(row['stratum'])} & {pct(row['token_kept'])} & {pct(row['ensemble_kept'])} & "
            f"{num(row['gain'], 1)} & {pct(row['oracle_kept'])} & {gap} & "
            f"{pct(row['prefix_risk'])} & {tex(row['interpretation'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "prmbench_stratum_summary.tex",
        "tab:prmbench_stratum_summary",
        "Compact PRMBench stratum summary for the Qwen+native aligned ensemble. Rows are within-stratum diagnostics under pooled calibration thresholds, not group-conditional validity guarantees.",
        "\n".join(lines),
    )


def build_prmbench_matched_subset_robustness_table() -> None:
    """Summarize matched PRMBench subset robustness for aligned verifiers."""

    root = OUT / "external_process" / "prmbench_matched_subsets"
    manifest_path = root / "matched_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    detectors = [
        "Token/format",
        "Combined logistic",
        "Qwen2.5-Math PRM",
        "PRMBench-native GBM",
        "Qwen+native ensemble",
        "Oracle",
    ]
    rows = []
    for subset_dir in manifest.get("subset_dirs", []):
        subset = str(subset_dir).rsplit("_", 1)[-1]
        subset_manifest_path = root / str(subset_dir) / "subset_manifest.json"
        stratum_counts = {}
        if subset_manifest_path.exists():
            try:
                stratum_counts = json.loads(subset_manifest_path.read_text()).get("stratum_counts", {})
            except json.JSONDecodeError:
                stratum_counts = {}
        n_subset = max(sum(int(value) for value in stratum_counts.values()), 1)
        outdir = root / str(subset_dir) / "aligned_verifier"
        summary = read_csv(outdir / "table_prmbench_aligned_verifier_summary.csv")
        if summary.empty:
            continue
        token = row_for(summary, "Token/format")
        oracle = row_for(summary, "Oracle")
        token_kept = float(token["prefix_retained_fraction_mean"]) if token is not None else np.nan
        oracle_kept = float(oracle["prefix_retained_fraction_mean"]) if oracle is not None else np.nan
        denom = oracle_kept - token_kept if np.isfinite(token_kept) and np.isfinite(oracle_kept) else np.nan
        for detector in detectors:
            row = row_for(summary, detector)
            if row is None:
                continue
            kept = float(row["prefix_retained_fraction_mean"])
            rows.append(
                {
                    "subset": subset,
                    "detector": detector,
                    "clean_pct": int(stratum_counts.get("clean", 0)) / n_subset,
                    "early_pct": int(stratum_counts.get("early-error", 0)) / n_subset,
                    "middle_pct": int(stratum_counts.get("middle-error", 0)) / n_subset,
                    "late_pct": int(stratum_counts.get("late-error", 0)) / n_subset,
                    "high_error_pct": int(stratum_counts.get("high-error", 0)) / n_subset,
                    "trace_error_rate": float(row.get("trace_error_rate_test_mean", np.nan)),
                    "prefix_risk": float(row.get("prefix_contamination_mean", np.nan)),
                    "prefix_risk_ci95": float(row.get("prefix_contamination_ci95", np.nan)),
                    "prefix_kept": kept,
                    "prefix_kept_ci95": float(row.get("prefix_retained_fraction_ci95", np.nan)),
                    "token_kept": token_kept,
                    "gain": 100.0 * (kept - token_kept) if np.isfinite(token_kept) else np.nan,
                    "oracle_kept": oracle_kept,
                    "gap_closed": (kept - token_kept) / denom if np.isfinite(denom) and denom > 0 else np.nan,
                    "full_accept": float(row.get("prefix_full_trace_rate_mean", np.nan)),
                    "full_accept_ci95": float(row.get("prefix_full_trace_rate_ci95", np.nan)),
                }
            )
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "prmbench_matched_subset_robustness.csv", index=False)
    raw.to_csv(RESULTS / "prmbench_matched_stratified_robustness.csv", index=False)
    if raw.empty:
        return

    focus = raw[raw["detector"] == "Qwen+native ensemble"].copy()
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Subset & Detector & Prefix risk & Prefix kept & Token kept & Gain & Gap closed & Full accept & Err. traces \\",
        r"\midrule",
    ]
    for _, row in focus.iterrows():
        gap = pct(row["gap_closed"]) if np.isfinite(float(row["gap_closed"])) else "--"
        lines.append(
            f"{tex(row['subset'])} & {tex(row['detector'])} & "
            f"${100.0 * row['prefix_risk']:.1f} \\pm {100.0 * row['prefix_risk_ci95']:.1f}$ & "
            f"${100.0 * row['prefix_kept']:.1f} \\pm {100.0 * row['prefix_kept_ci95']:.1f}$ & "
            f"{pct(row['token_kept'])} & {num(row['gain'], 1)} & {gap} & "
            f"${100.0 * row['full_accept']:.1f} \\pm {100.0 * row['full_accept_ci95']:.1f}$ & "
            f"{pct(row['trace_error_rate'])} \\\\"
        )
    if not focus.empty:
        lines.append(r"\midrule")
        lines.append(
            "Median & Qwen+native ensemble & "
            f"{pct(focus['prefix_risk'].median())} & "
            f"{pct(focus['prefix_kept'].median())} & "
            f"{pct(focus['token_kept'].median())} & "
            f"{focus['gain'].median():.1f} & "
            f"{pct(focus['gap_closed'].median())} & "
            f"{pct(focus['full_accept'].median())} & "
            f"{pct(focus['trace_error_rate'].median())} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "prmbench_matched_subset_robustness.tex",
        "tab:prmbench_matched_subset_robustness",
        "Matched PRMBench subset robustness at $\\alpha=0.05$. Each 500-trace subset has 100 traces from each clean/first-error/high-error stratum; subsets are sampled independently because clean traces are scarce. The table focuses on the aligned ensemble, while the CSV includes all detector rows.",
        "\n".join(lines),
    )

    matched_lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r"Subset & Clean & Early & Mid & Late & High-err. & Prefix risk & Prefix kept & Token kept & Gain & Gap closed \\",
        r"\midrule",
    ]
    for _, row in focus.iterrows():
        matched_lines.append(
            f"{tex(row['subset'])} & {pct(row['clean_pct'])} & {pct(row['early_pct'])} & "
            f"{pct(row['middle_pct'])} & {pct(row['late_pct'])} & {pct(row['high_error_pct'])} & "
            f"${100.0 * row['prefix_risk']:.1f} \\pm {100.0 * row['prefix_risk_ci95']:.1f}$ & "
            f"${100.0 * row['prefix_kept']:.1f} \\pm {100.0 * row['prefix_kept_ci95']:.1f}$ & "
            f"{pct(row['token_kept'])} & {num(row['gain'], 1)} & {pct(row['gap_closed'])} \\\\"
        )
    if not focus.empty:
        matched_lines.append(r"\midrule")
        matched_lines.append(
            "Median & "
            f"{pct(focus['clean_pct'].median())} & {pct(focus['early_pct'].median())} & "
            f"{pct(focus['middle_pct'].median())} & {pct(focus['late_pct'].median())} & "
            f"{pct(focus['high_error_pct'].median())} & {pct(focus['prefix_risk'].median())} & "
            f"{pct(focus['prefix_kept'].median())} & {pct(focus['token_kept'].median())} & "
            f"{focus['gain'].median():.1f} & {pct(focus['gap_closed'].median())} \\\\"
        )
    matched_lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "prmbench_matched_stratified_robustness.tex",
        "tab:prmbench_matched_stratified_robustness",
        "Matched stratified PRMBench robustness at $\\alpha=0.05$. Each subset has identical clean, first-error-position, and high-error stratum counts. The table reports Qwen+native ensemble gains over token/format; the CSV includes all detectors.",
        "\n".join(matched_lines),
    )

    if plt is not None:
        plot = raw[raw["detector"].isin(["Combined logistic", "Qwen2.5-Math PRM", "PRMBench-native GBM", "Qwen+native ensemble"])].copy()
        if not plot.empty:
            labels = ["Combined logistic", "Qwen2.5-Math PRM", "PRMBench-native GBM", "Qwen+native ensemble"]
            data = [plot[plot["detector"] == label]["gain"].dropna().to_numpy() for label in labels]
            fig, ax = plt.subplots(figsize=(7.0, 3.2))
            ax.axhline(0.0, color="0.25", linewidth=0.8, linestyle="--")
            ax.boxplot(data, tick_labels=[label.replace(" ", "\n") for label in labels], showmeans=True)
            ax.set_ylabel(r"$\Delta$ prefix kept vs. token/format (points)")
            ax.set_title("Matched PRMBench subset gains")
            fig.tight_layout()
            fig.savefig(FIGURES / "prmbench_subset_gain_boxplot.pdf")
            fig.savefig(FIGURES / "prmbench_gain_boxplot.pdf")
            fig.savefig(FIGURES / "prmbench_matched_gain_boxplot.pdf")
            plt.close(fig)
    elif (FIGURES / "prmbench_subset_gain_boxplot.pdf").exists():
        (FIGURES / "prmbench_gain_boxplot.pdf").write_bytes((FIGURES / "prmbench_subset_gain_boxplot.pdf").read_bytes())
        (FIGURES / "prmbench_matched_gain_boxplot.pdf").write_bytes(
            (FIGURES / "prmbench_subset_gain_boxplot.pdf").read_bytes()
        )


def _canonicalize_text(text: str | None) -> str:
    text = str(text or "")
    text = re.sub(r"(?im)^\s*(?:step\s*)?\d+[\).:\-]\s*", "", text)
    text = re.sub(r"(?im)^\s*[-*•]\s*", "", text)
    text = re.sub(r"\\boxed\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _perturb_text(text: str | None, mode: str) -> str:
    base = _canonicalize_text(text)
    if mode == "bullet-style":
        return f"- Step: {base}"
    if mode == "spaced-lines":
        return re.sub(r"([=+\-*/])", r" \1 ", f"Step:\n{base}")
    if mode == "dataset-tags-removed":
        return re.sub(r"\b(?:gsm8k|arithmetic|boolean|prmbench|processbench)\b", "task", base, flags=re.IGNORECASE)
    if mode == "no-step-markers":
        return base
    return str(text or "")


def _text_variant_traces(traces: list[TraceRecord], mode: str) -> list[TraceRecord]:
    out: list[TraceRecord] = []
    for trace in traces:
        steps = []
        for step in trace.steps:
            steps.append(
                replace(
                    step,
                    step_content=_perturb_text(step.step_content, mode),
                    original_expression=_canonicalize_text(step.original_expression),
                )
            )
        out.append(replace(trace, steps=steps))
    return out


def _run_token_format_variant(traces: list[TraceRecord], variant: str, seeds=range(2806, 2816)) -> pd.DataFrame:
    from crop.experiments.exp08_cheap_baselines import _summarize_with_ci
    from crop.experiments.exp09_process_repeated import _artifact_views, _evaluate_bundle, _fit_model_bundle

    rows = []
    lambdas = np.linspace(0.0, 1.0, 101)
    artifact = _artifact_views(traces)["artifact_token_formatting"]
    for seed in seeds:
        split = split_traces(artifact, seed=seed)
        bundle = _fit_model_bundle("logistic_l2", split, seed=seed)
        for row in _evaluate_bundle(
            score_name="Token/format",
            score_family=variant,
            split=split,
            bundle=bundle,
            seed=seed,
            alphas=[0.05],
            lambdas=lambdas,
            runtime_seconds=0.0,
        ):
            row["variant"] = variant
            rows.append(row)
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    return _summarize_with_ci_local(raw, ["variant", "score", "alpha"])


def build_format_artifact_perturbation_tables() -> None:
    """Run cached-feature token/format canonicalization and perturbation diagnostics."""

    combined_path = ROOT / "data" / "strengthened" / "crop_target_combined_steps.npz"
    if not combined_path.exists():
        return
    traces = load_many_npz([combined_path], ["mixed"])
    variants = {
        "raw": traces,
        "canonicalized": _text_variant_traces(traces, "no-step-markers"),
        "bullet-style": _text_variant_traces(traces, "bullet-style"),
        "spaced-lines": _text_variant_traces(traces, "spaced-lines"),
        "dataset-tags-removed": _text_variant_traces(traces, "dataset-tags-removed"),
    }
    summaries = []
    for variant, variant_traces in variants.items():
        summary = _run_token_format_variant(variant_traces, variant)
        if not summary.empty:
            summaries.append(summary)
    if not summaries:
        return
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(RESULTS / "format_perturbation_cpcc.csv", index=False)

    def variant_row(name: str) -> pd.Series | None:
        subset = summary[(summary["variant"] == name) & np.isclose(summary["alpha"].astype(float), 0.05)]
        if subset.empty:
            return None
        return subset.iloc[0]

    raw = variant_row("raw")
    canonical = variant_row("canonicalized")
    canonical_rows = []
    if raw is not None and canonical is not None:
        raw_kept = float(raw["prefix_retained_fraction_mean"])
        canonical_kept = float(canonical["prefix_retained_fraction_mean"])
        canonical_rows.append(
            {
                "dataset": "Target process",
                "detector": "Token/format",
                "raw_kept": raw_kept,
                "canonical_kept": canonical_kept,
                "drop": canonical_kept - raw_kept,
                "prefix_risk": float(canonical["prefix_contamination_mean"]),
                "artifact_gain": canonical_kept - raw_kept,
            }
        )
    pd.DataFrame(canonical_rows).to_csv(RESULTS / "canonicalized_format_cpcc.csv", index=False)
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & Detector & Raw kept & Canonical kept & Change & Prefix risk & Artifact gain \\",
        r"\midrule",
    ]
    for row in canonical_rows:
        lines.append(
            f"{tex(row['dataset'])} & {tex(row['detector'])} & {pct(row['raw_kept'])} & "
            f"{pct(row['canonical_kept'])} & {100.0 * row['drop']:.1f} & "
            f"{pct(row['prefix_risk'])} & {100.0 * row['artifact_gain']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    latex_table(
        TABLES / "canonicalized_format_cpcc.tex",
        "tab:canonicalized_format_cpcc",
        "Canonicalized-format diagnostic on target traces. Only token/format features can be recomputed from cached text without regenerating semantic embeddings or external verifier scores; the row therefore measures artifact sensitivity of that control.",
        "\n".join(lines),
    )

    perturb_lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Perturbation & Detector & Prefix risk & Prefix kept & Change vs. raw & Full accept \\",
        r"\midrule",
    ]
    raw_kept = float(raw["prefix_retained_fraction_mean"]) if raw is not None else np.nan
    for variant in ("canonicalized", "bullet-style", "spaced-lines", "dataset-tags-removed"):
        row = variant_row(variant)
        if row is None:
            continue
        kept = float(row["prefix_retained_fraction_mean"])
        perturb_lines.append(
            f"{tex(variant)} & Token/format & {mean_ci(row, 'prefix_contamination', scale=100, digits=1)} & "
            f"{mean_ci(row, 'prefix_retained_fraction', scale=100, digits=1)} & "
            f"{100.0 * (kept - raw_kept):.1f} & {mean_ci(row, 'prefix_full_trace_rate', scale=100, digits=1)} \\\\"
        )
    perturb_lines.extend([r"\bottomrule", r"\end{tabular}"])
    latex_table(
        TABLES / "format_perturbation_cpcc.tex",
        "tab:format_perturbation_cpcc",
        "Counterfactual formatting perturbation diagnostic for the token/format control on target traces. Perturbations preserve labels but alter superficial formatting, so changes measure artifact sensitivity rather than semantic verifier quality.",
        "\n".join(perturb_lines),
    )

    stress_labels = {
        "raw": ("strong baseline", "raw"),
        "canonicalized": ("persistent regularities beyond literal step numbering", "canonicalized"),
        "bullet-style": ("robust superficial cue under bullet-style perturbation", "bullet perturbation"),
        "spaced-lines": ("robust superficial cue under whitespace/line perturbation", "whitespace perturbation"),
        "dataset-tags-removed": ("dataset tag masking does not remove most token/format signal", "dataset tags removed"),
    }
    stress_rows = []
    for variant, (interpretation, display) in stress_labels.items():
        row = variant_row(variant)
        if row is None:
            continue
        kept = float(row["prefix_retained_fraction_mean"])
        stress_rows.append(
            {
                "stress_test": display,
                "token_format_kept": kept,
                "change_vs_raw": kept - raw_kept if np.isfinite(raw_kept) else np.nan,
                "prefix_risk": float(row["prefix_contamination_mean"]),
                "interpretation": interpretation,
            }
        )
    pd.DataFrame(stress_rows).to_csv(RESULTS / "artifact_stress_summary.csv", index=False)
    stress_lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrp{0.46\textwidth}}",
        r"\toprule",
        r"Stress test & Token/format kept & Change vs. raw & Interpretation \\",
        r"\midrule",
    ]
    for row in stress_rows:
        stress_lines.append(
            f"{tex(row['stress_test'])} & {pct(row['token_format_kept'])} & "
            f"{100.0 * row['change_vs_raw']:.1f} & {tex(row['interpretation'])} \\\\"
        )
    stress_lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "artifact_stress_summary.tex",
        "tab:artifact_stress_summary",
        "Artifact stress summary for the token/format control. Persistent retention under harmless text perturbations shows that superficial structure is not limited to literal step numbering, motivating artifact-adjusted certificate efficiency as the main score-source metric.",
        "\n".join(stress_lines),
    )


def _make_nonmath_logic_traces(seed: int = 20260507, n_traces: int = 500) -> list[TraceRecord]:
    rng = np.random.default_rng(seed)
    names = ["Ava", "Ben", "Cy", "Dee", "Eli", "Fay"]
    properties = ["careful", "licensed", "trained", "eligible", "approved", "verified"]
    traces: list[TraceRecord] = []
    for idx in range(n_traces):
        total = int(rng.integers(4, 9))
        first_error = int(rng.integers(1, total)) if rng.random() < 0.40 else None
        person = names[int(rng.integers(0, len(names)))]
        prop = properties[int(rng.integers(0, len(properties)))]
        consequence = properties[int(rng.integers(0, len(properties)))]
        if consequence == prop:
            consequence = "qualified"
        steps: list[StepRecord] = []
        for step_idx in range(total):
            is_error = first_error is not None and step_idx >= first_error and rng.random() < (0.75 if step_idx == first_error else 0.35)
            if step_idx == 0:
                content = f"The record states that {person} is {prop}."
                is_error = False
            elif step_idx == 1:
                content = f"The rule says every {prop} person is {consequence}."
                is_error = False
            elif is_error:
                content = f"Therefore {person} is {properties[int(rng.integers(0, len(properties)))]}, although no stated rule supports that jump."
            else:
                content = f"Using the stated rule, {person} is {consequence}."
            words = content.split()
            x = np.asarray(
                [
                    step_idx / max(total - 1, 1),
                    total,
                    len(words),
                    float("unsupported" in content or "although" in content),
                    float("rule" in content),
                    float("Therefore" in content),
                ],
                dtype=float,
            )
            steps.append(
                StepRecord(
                    trace_id=f"nl_logic:{idx:04d}",
                    domain="synthetic_nl_logic",
                    complexity=None,
                    step_number=step_idx,
                    total_steps=total,
                    before_after="after",
                    x=x,
                    y_error=int(is_error),
                    is_correct=not bool(is_error),
                    original_expression=f"Natural-language logic task {idx}",
                    step_content=content,
                    metadata={"label_source": "programmatic_rule_engine", "domain": "natural_language_logic"},
                )
            )
        traces.append(TraceRecord(trace_id=f"nl_logic:{idx:04d}", domain="synthetic_nl_logic", complexity=None, steps=steps))
    return traces


def build_nonmath_pilot_cpcc_table() -> None:
    """Run a small natural-language logic CPCC pilot with automatic labels."""

    from crop.experiments.exp08_cheap_baselines import _summarize_with_ci
    from crop.experiments.exp09_process_repeated import _artifact_views, _evaluate_bundle, _fit_model_bundle, _split_like
    from crop.experiments.common import build_score_bundle

    traces = _make_nonmath_logic_traces()
    artifact = _artifact_views(traces)["artifact_token_formatting"]
    lambdas = np.linspace(0.0, 1.0, 101)
    rows = []
    for seed in range(2806, 2826):
        reference = split_traces(traces, seed=seed)
        artifact_split = _split_like(reference, artifact)
        bundles = [
            ("Token/format", "artifact_control", artifact_split, _fit_model_bundle("logistic_l2", artifact_split, seed=seed)),
            ("Logic lexical logistic", "programmatic_features", reference, _fit_model_bundle("logistic_l2", reference, seed=seed)),
            ("Oracle", "diagnostic", reference, build_score_bundle("oracle", reference, seed)),
        ]
        for score_name, family, split, bundle in bundles:
            rows.extend(
                _evaluate_bundle(
                    score_name=score_name,
                    score_family=family,
                    split=split,
                    bundle=bundle,
                    seed=seed,
                    alphas=[0.05],
                    lambdas=lambdas,
                    runtime_seconds=0.0,
                )
            )
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "nonmath_pilot_cpcc_raw.csv", index=False)
    summary = _summarize_with_ci(raw)
    summary.to_csv(RESULTS / "nonmath_pilot_cpcc.csv", index=False)
    token = row_for(summary, "Token/format")
    oracle = row_for(summary, "Oracle")
    token_kept = float(token["prefix_retained_fraction_mean"]) if token is not None else np.nan
    oracle_kept = float(oracle["prefix_retained_fraction_mean"]) if oracle is not None else np.nan
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"Dataset & Domain & Detector & Prefix risk & Prefix kept & Token kept & Artifact gain & Oracle kept \\",
        r"\midrule",
    ]
    for detector in ("Token/format", "Logic lexical logistic", "Oracle"):
        row = row_for(summary, detector)
        if row is None:
            continue
        kept = float(row["prefix_retained_fraction_mean"])
        lines.append(
            f"Synthetic NL logic & rule-based reasoning & {tex(detector)} & "
            f"{mean_ci(row, 'prefix_contamination', scale=100, digits=1)} & "
            f"{mean_ci(row, 'prefix_retained_fraction', scale=100, digits=1)} & "
            f"{pct(token_kept)} & {100.0 * (kept - token_kept):.1f} & {pct(oracle_kept)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "nonmath_pilot_cpcc.tex",
        "tab:nonmath_pilot_cpcc",
        "Small non-math CPCC pilot on programmatically generated natural-language logic traces with automatic step labels from the rule engine. This is a scope diagnostic, not a substitute for human-labeled non-math process data.",
        "\n".join(lines),
    )


def build_repair_or_review_workflow_table() -> None:
    """Build the completed-trace repair/review workflow simulation table."""

    routing = read_csv(RESULTS / "routing_review_burden.csv")
    objects = read_csv(RESULTS / "object_level_comparison.csv")
    rows = []

    def add_from_routing(method: str, score_source: str):
        subset = routing[routing["score_source"] == score_source] if not routing.empty else pd.DataFrame()
        if subset.empty:
            return
        row = subset.iloc[0]
        first_error_routed = float(row["error_in_suffix_recall_mean"])
        review_cost = float(row["suffix_routed_mean"])
        rows.append(
            {
                "method": method,
                "score_source": score_source,
                "review_cost": review_cost,
                "review_reduction": float(row["review_reduction_mean"]),
                "first_error_routed": first_error_routed,
                "cost_per_routed_error": review_cost / first_error_routed
                if np.isfinite(first_error_routed) and first_error_routed > 0.0
                else np.nan,
                "final_answer_corrected": np.nan,
                "full_accept": float(row["full_accept_mean"]),
                "prefix_risk": float(row["prefix_risk_mean"]),
            }
        )

    add_from_routing("CPCC suffix review", "Combined logistic")
    add_from_routing("Qwen CPCC suffix review", "Qwen2.5-Math PRM")
    for method in ("Whole-trace abstention", "Dynamic early stop"):
        subset = objects[objects["method"] == method] if not objects.empty else pd.DataFrame()
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append(
            {
                "method": method,
                "score_source": "combined_logistic_l2",
                "review_cost": float(row["review_burden"]),
                "review_reduction": 1.0 - float(row["review_burden"]),
                "first_error_routed": np.nan,
                "cost_per_routed_error": np.nan,
                "final_answer_corrected": np.nan,
                "full_accept": float(row["full_accept"]) if np.isfinite(float(row.get("full_accept", np.nan))) else np.nan,
                "prefix_risk": float(row["risk"]),
            }
        )
    rows.append(
        {
            "method": "Full review",
            "score_source": "none",
            "review_cost": 1.0,
            "review_reduction": 0.0,
            "first_error_routed": 1.0,
            "cost_per_routed_error": 1.0,
            "final_answer_corrected": np.nan,
            "full_accept": 0.0,
            "prefix_risk": 0.0,
        }
    )
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "repair_or_review_workflow.csv", index=False)
    raw.to_csv(RESULTS / "cpcc_review_repair_workflow.csv", index=False)
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Method & Review cost & Review reduction & First-error routed & Cost/error & Final corrected & Full accept & Prefix risk \\",
        r"\midrule",
    ]
    for _, row in raw.iterrows():
        corrected = "--" if not np.isfinite(float(row["final_answer_corrected"])) else pct(row["final_answer_corrected"])
        routed = "--" if not np.isfinite(float(row["first_error_routed"])) else pct(row["first_error_routed"])
        cost_per = "--" if not np.isfinite(float(row["cost_per_routed_error"])) else num(row["cost_per_routed_error"], 2)
        lines.append(
            f"{tex(row['method'])} & {pct(row['review_cost'])} & {pct(row['review_reduction'])} & "
            f"{routed} & {cost_per} & {corrected} & {pct(row['full_accept'])} & {pct(row['prefix_risk'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "repair_or_review_workflow.tex",
        "tab:repair_or_review_workflow",
        "Completed-trace repair/review workflow simulation. No repair model is called; review cost is the routed suffix fraction and first-error routed is the fraction of erroneous traces whose first error lies in that suffix. Final-answer correction is therefore left unmeasured.",
        "\n".join(lines),
    )
    latex_table(
        TABLES / "cpcc_review_repair_workflow.tex",
        "tab:cpcc_review_repair_workflow",
        "CPCC review/repair workflow simulation. No repair model is called; review cost is the routed suffix fraction, cost/error is normalized review cost divided by first-error-routed rate, and final-answer correction is left unmeasured.",
        "\n".join(lines),
    )

    workflow_rows: list[dict] = []

    def add_workflow_from_routing(method: str, score_source: str) -> None:
        subset = routing[routing["score_source"] == score_source] if not routing.empty else pd.DataFrame()
        if subset.empty:
            return
        row = subset.iloc[0]
        first_error_routed = float(row["error_in_suffix_recall_mean"])
        review_cost = float(row["suffix_routed_mean"])
        workflow_rows.append(
            {
                "method": method,
                "score_source": score_source,
                "prefix_risk": float(row["prefix_risk_mean"]),
                "review_cost": review_cost,
                "review_reduction": float(row["review_reduction_mean"]),
                "first_error_routed": first_error_routed,
                "cost_per_routed_error": review_cost / first_error_routed
                if np.isfinite(first_error_routed) and first_error_routed > 0.0
                else np.nan,
                "full_accept": float(row["full_accept_mean"]),
                "accepted_error": float(row["accepted_error_mean"]),
            }
        )

    workflow_rows.append(
        {
            "method": "Full review",
            "score_source": "none",
            "prefix_risk": 0.0,
            "review_cost": 1.0,
            "review_reduction": 0.0,
            "first_error_routed": 1.0,
            "cost_per_routed_error": 1.0,
            "full_accept": 0.0,
            "accepted_error": np.nan,
        }
    )
    whole = objects[objects["method"] == "Whole-trace abstention"] if not objects.empty else pd.DataFrame()
    if not whole.empty:
        row = whole.iloc[0]
        workflow_rows.append(
            {
                "method": "Whole-trace abstention",
                "score_source": "Combined logistic",
                "prefix_risk": float(row["risk"]),
                "review_cost": float(row["review_burden"]),
                "review_reduction": 1.0 - float(row["review_burden"]),
                "first_error_routed": np.nan,
                "cost_per_routed_error": np.nan,
                "full_accept": float(row["full_accept"]),
                "accepted_error": np.nan,
            }
        )
    add_workflow_from_routing("CPCC", "Token/format")
    add_workflow_from_routing("CPCC", "Combined logistic")
    add_workflow_from_routing("CPCC", "Qwen2.5-Math PRM")

    prm_review = read_csv(RESULTS / "prmbench_review_efficiency.csv")
    prm_aligned = read_csv(RESULTS / "prmbench_aligned_verifier_cpcc.csv")
    if not prm_review.empty and not prm_aligned.empty:
        review_row = prm_review[prm_review["detector"] == "Qwen+native ensemble"]
        aligned_row = prm_aligned[prm_aligned["detector"] == "Qwen+native ensemble"]
        if not review_row.empty and not aligned_row.empty:
            review_row = review_row.iloc[0]
            aligned_row = aligned_row.iloc[0]
            review_cost = float(review_row["suffix_routed"])
            first_error_routed = float(review_row["first_error_routed"])
            workflow_rows.append(
                {
                    "method": "CPCC",
                    "score_source": "PRMBench aligned ensemble",
                    "prefix_risk": float(aligned_row["prefix_risk"]),
                    "review_cost": review_cost,
                    "review_reduction": float(review_row["prefix_kept_step_weighted"]),
                    "first_error_routed": first_error_routed,
                    "cost_per_routed_error": review_cost / first_error_routed
                    if np.isfinite(first_error_routed) and first_error_routed > 0.0
                    else np.nan,
                    "full_accept": float(review_row["full_accept"]),
                    "accepted_error": float(aligned_row["accepted_error"]),
                }
            )
    add_workflow_from_routing("CPCC", "Oracle")

    workflow = pd.DataFrame(workflow_rows)
    workflow.to_csv(RESULTS / "cpcc_review_workflow.csv", index=False)
    workflow_lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Method & Score source & Prefix risk & Review cost & Review reduction & First-error routed & Cost/error & Full accept & Acc. err. \\",
        r"\midrule",
    ]
    for _, row in workflow.iterrows():
        routed = "--" if not np.isfinite(float(row["first_error_routed"])) else pct(row["first_error_routed"])
        cost_per = "--" if not np.isfinite(float(row["cost_per_routed_error"])) else num(row["cost_per_routed_error"], 2)
        accepted = "--" if not np.isfinite(float(row["accepted_error"])) else pct(row["accepted_error"])
        workflow_lines.append(
            f"{tex(row['method'])} & {tex(row['score_source'])} & {pct(row['prefix_risk'])} & "
            f"{pct(row['review_cost'])} & {pct(row['review_reduction'])} & {routed} & "
            f"{cost_per} & {pct(row['full_accept'])} & {accepted} \\\\"
        )
    workflow_lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "cpcc_review_workflow.tex",
        "tab:cpcc_review_workflow",
        "Completed-trace review workflow simulation at $\\alpha=0.05$. Review cost is the routed suffix fraction; cost/error divides normalized review cost by first-error-routed rate. The PRMBench aligned ensemble row is reported on the PRMBench subset and is included as an external workflow diagnostic.",
        "\n".join(workflow_lines),
    )


def _make_synthetic_prompt_shift_data(seed: int = 20260506, n_traces: int = 1200):
    rng = np.random.default_rng(seed)
    original: list[TraceRecord] = []
    paraphrase: list[TraceRecord] = []
    original_scores: list[np.ndarray] = []
    paraphrase_scores: list[np.ndarray] = []

    def make_trace(idx: int, variant: str, y: np.ndarray) -> TraceRecord:
        steps: list[StepRecord] = []
        trace_id = f"prompt_shift:{variant}:{idx:05d}"
        for t, label in enumerate(y):
            metadata = {
                "expr_id": f"prompt_shift_{idx:05d}_{variant}",
                "prompt_variant": variant,
                "step_number": t + 1,
                "step_labels": {
                    "step_number": t + 1,
                    "step_content": f"{variant} synthetic step {t + 1}",
                    "step_label": bool(label == 0),
                },
                "original_expression": f"{variant} synthetic expression {idx}",
            }
            steps.append(
                StepRecord(
                    trace_id=trace_id,
                    domain="synthetic_prompt_shift",
                    complexity=None,
                    step_number=t,
                    total_steps=len(y),
                    before_after="after",
                    x=np.zeros(1, dtype=float),
                    y_error=int(label),
                    is_correct=bool(label == 0),
                    original_expression=f"{variant} synthetic expression {idx}",
                    step_content=f"{variant} synthetic step {t + 1}",
                    metadata=metadata,
                )
            )
        return TraceRecord(trace_id=trace_id, domain="synthetic_prompt_shift", complexity=None, steps=steps)

    for idx in range(n_traces):
        total = int(rng.integers(4, 10))
        y = np.zeros(total, dtype=int)
        if rng.random() < 0.35:
            first = int(rng.integers(1, total))
            y[first] = 1
            if first + 1 < total:
                y[first + 1 :] = (rng.random(total - first - 1) < 0.25).astype(int)

        orig_scores = []
        para_scores = []
        first_error = np.flatnonzero(y == 1)
        first_error_idx = int(first_error[0]) if len(first_error) else -1
        for t, label in enumerate(y):
            position = t / max(total - 1, 1)
            if label:
                orig = rng.normal(0.78, 0.07)
                if t == first_error_idx:
                    para = rng.normal(0.30, 0.08)
                else:
                    para = rng.normal(0.38, 0.10)
            else:
                orig = rng.normal(0.18 + 0.03 * position, 0.05)
                para = rng.normal(0.21 + 0.05 * position, 0.06)
            orig_scores.append(float(np.clip(orig, 0.0, 1.0)))
            para_scores.append(float(np.clip(para, 0.0, 1.0)))
        original.append(make_trace(idx, "original", y))
        paraphrase.append(make_trace(idx, "paraphrase", y))
        original_scores.append(np.asarray(orig_scores, dtype=float))
        paraphrase_scores.append(np.asarray(para_scores, dtype=float))
    return original, paraphrase, original_scores, paraphrase_scores


def _prompt_group(trace: TraceRecord) -> str:
    if not trace.steps:
        return "unknown"
    return str(trace.steps[0].metadata.get("prompt_variant", "unknown"))


def _evaluate_prompt_cpcc(cal_traces, cal_scores, test_traces, test_scores, lambdas: np.ndarray, alpha: float):
    losses = prefix_losses_by_lambda(cal_traces, cal_scores, lambdas)
    lambda_hat, cal_risk = select_lambda_crc(losses, lambdas, alpha=alpha, direction="increasing")
    test_losses = prefix_losses_by_lambda(test_traces, test_scores, np.asarray([lambda_hat]))[0]
    lengths = prefix_lengths(test_scores, lambda_hat)
    totals = np.asarray([len(trace.steps) for trace in test_traces], dtype=float)
    return {
        "lambda": lambda_hat,
        "cal_corrected_risk": cal_risk,
        "prefix_risk": float(np.mean(test_losses)),
        "prefix_kept": float(np.mean(lengths / np.maximum(totals, 1.0))),
        "full_accept": float(np.mean(lengths == totals)),
        "n_cal": len(cal_traces),
        "n_test": len(test_traces),
    }


def _evaluate_prompt_mondrian(cal_traces, cal_scores, test_traces, test_scores, lambdas: np.ndarray, alpha: float):
    pooled = _evaluate_prompt_cpcc(cal_traces, cal_scores, test_traces, test_scores, lambdas, alpha)
    groups = sorted({_prompt_group(trace) for trace in cal_traces})
    group_lambdas: dict[str, float] = {}
    group_risks: dict[str, float] = {}
    for group in groups:
        idx = [i for i, trace in enumerate(cal_traces) if _prompt_group(trace) == group]
        losses = prefix_losses_by_lambda([cal_traces[i] for i in idx], [cal_scores[i] for i in idx], lambdas)
        lam, risk = select_lambda_crc(losses, lambdas, alpha=alpha, direction="increasing")
        group_lambdas[group] = lam
        group_risks[group] = risk
    losses = []
    lengths = []
    totals = []
    for trace, scores in zip(test_traces, test_scores):
        lam = group_lambdas.get(_prompt_group(trace), pooled["lambda"])
        losses.append(prefix_losses_by_lambda([trace], [scores], np.asarray([lam]))[0, 0])
        lengths.append(prefix_lengths([scores], lam)[0])
        totals.append(len(trace.steps))
    lengths_arr = np.asarray(lengths, dtype=float)
    totals_arr = np.asarray(totals, dtype=float)
    return {
        "lambda": np.nan,
        "cal_corrected_risk": float(np.nanmax(list(group_risks.values()))) if group_risks else np.nan,
        "prefix_risk": float(np.mean(losses)),
        "prefix_kept": float(np.mean(lengths_arr / np.maximum(totals_arr, 1.0))),
        "full_accept": float(np.mean(lengths_arr == totals_arr)),
        "n_cal": len(cal_traces),
        "n_test": len(test_traces),
    }


def build_prompt_shift_cpcc_table() -> pd.DataFrame:
    summary_path = RESULTS / "prompt_shift_cpcc.csv"
    original, paraphrase, original_scores, paraphrase_scores = _make_synthetic_prompt_shift_data()
    lambdas = np.linspace(0.0, 1.0, 101)
    rows: list[dict] = []
    for seed in range(2806, 2836):
        rng = np.random.default_rng(seed + 301_000)
        order = rng.permutation(len(original))
        cal_idx = order[: len(order) // 2]
        test_idx = order[len(order) // 2 :]

        def take(items, idx):
            return [items[int(i)] for i in idx]

        configs = [
            (
                "original",
                "original",
                take(original, cal_idx),
                take(original_scores, cal_idx),
                take(original, test_idx),
                take(original_scores, test_idx),
                False,
            ),
            (
                "original",
                "paraphrase",
                take(original, cal_idx),
                take(original_scores, cal_idx),
                take(paraphrase, test_idx),
                take(paraphrase_scores, test_idx),
                False,
            ),
            (
                "paraphrase",
                "paraphrase",
                take(paraphrase, cal_idx),
                take(paraphrase_scores, cal_idx),
                take(paraphrase, test_idx),
                take(paraphrase_scores, test_idx),
                False,
            ),
            (
                "Mondrian prompt",
                "mixed",
                take(original, cal_idx) + take(paraphrase, cal_idx),
                take(original_scores, cal_idx) + take(paraphrase_scores, cal_idx),
                take(original, test_idx) + take(paraphrase, test_idx),
                take(original_scores, test_idx) + take(paraphrase_scores, test_idx),
                True,
            ),
        ]
        for calibration, test, cal_traces, cal_scores, test_traces, test_scores, mondrian in configs:
            metrics = (
                _evaluate_prompt_mondrian(cal_traces, cal_scores, test_traces, test_scores, lambdas, 0.05)
                if mondrian
                else _evaluate_prompt_cpcc(cal_traces, cal_scores, test_traces, test_scores, lambdas, 0.05)
            )
            rows.append(
                {
                    "calibration": calibration,
                    "test": test,
                    "seed": seed,
                    "alpha": 0.05,
                    **metrics,
                    "synthetic": True,
                }
            )
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "prompt_shift_cpcc_raw.csv", index=False)
    summary = _summarize_with_ci_local(raw, ["calibration", "test", "alpha", "synthetic"])
    summary["risk_violation"] = np.where(summary["prefix_risk_mean"] > summary["alpha"], "yes", "no")
    summary.to_csv(summary_path, index=False)

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Calibration & Test & Prefix risk & Prefix kept & Full accept & Violation? \\",
        r"\midrule",
    ]
    order = [
        ("original", "original"),
        ("original", "paraphrase"),
        ("paraphrase", "paraphrase"),
        ("Mondrian prompt", "mixed"),
    ]
    for calibration, test in order:
        subset = summary[(summary["calibration"] == calibration) & (summary["test"] == test)]
        if subset.empty:
            continue
        row = subset.iloc[0]
        lines.append(
            f"{tex(calibration)} & {tex(test)} & "
            f"{mean_ci(row, 'prefix_risk', scale=100, digits=1)} & "
            f"{mean_ci(row, 'prefix_kept', scale=100, digits=1)} & "
            f"{mean_ci(row, 'full_accept', scale=100, digits=1)} & "
            f"{tex(row['risk_violation'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    latex_table(
        TABLES / "prompt_shift_cpcc.tex",
        "tab:prompt_shift_cpcc",
        "Synthetic prompt-shift CPCC diagnostic. Step labels are generated automatically and the paraphrase condition deliberately shifts the score distribution. Original-prompt calibration can violate the target on paraphrased traces; recalibration or prompt-Mondrian calibration restores control in this synthetic setting.",
        "\n".join(lines),
    )
    return summary


def build_calibration_size_sensitivity() -> pd.DataFrame:
    summary_path = RESULTS / "calibration_size_sensitivity_summary.csv"
    if summary_path.exists():
        summary = read_csv(summary_path)
    else:
        from crop.experiments.exp09_process_repeated import _accepted_error_metrics, _fit_model_bundle

        traces = load_many_npz([ROOT / "data" / "strengthened" / "crop_target_combined_steps.npz"], ["mixed"], allow_nan=True)
        lambdas = np.linspace(0.0, 1.0, 101)
        rows: list[dict] = []
        for seed in range(2806, 2816):
            split = split_traces(traces, seed=seed)
            bundle = _fit_model_bundle("logistic_l2", split, seed=seed)
            rng = np.random.default_rng(seed + 91_000)
            order = rng.permutation(len(split.cal))
            for cal_size in (50, 100, 200, 400, len(split.cal)):
                idx = np.sort(order[: min(cal_size, len(order))])
                cal_traces = [split.cal[int(i)] for i in idx]
                cal_scores = [bundle.cal_scores_by_trace[int(i)] for i in idx]
                losses = prefix_losses_by_lambda(cal_traces, cal_scores, lambdas)
                lambda_hat, cal_risk = select_lambda_crc(losses, lambdas, alpha=0.05, direction="increasing")
                test_losses = prefix_losses_by_lambda(split.test, bundle.test_scores_by_trace, np.asarray([lambda_hat]))[0]
                lengths = prefix_lengths(bundle.test_scores_by_trace, lambda_hat)
                trace_lengths = np.asarray([len(trace.steps) for trace in split.test], dtype=float)
                accept = _accepted_error_metrics(split.test, bundle.test_scores_by_trace, lambda_hat)
                rows.append(
                    {
                        "score": "combined_logistic_l2",
                        "seed": seed,
                        "cal_size": int(min(cal_size, len(order))),
                        "lambda": lambda_hat,
                        "cal_corrected_risk": cal_risk,
                        "prefix_contamination": float(np.mean(test_losses)),
                        "prefix_retained_fraction": float(np.mean(lengths / np.maximum(trace_lengths, 1.0))),
                        "prefix_retained_steps": float(np.mean(lengths)),
                        "prefix_full_trace_rate": float(np.mean(lengths == trace_lengths)),
                        "marginal_false_accept": accept["marginal_false_accept"],
                        "accept_rate": accept["accept_rate"],
                    }
                )
        raw = pd.DataFrame(rows)
        raw.to_csv(RESULTS / "calibration_size_sensitivity.csv", index=False)
        summary = _summarize_with_ci_local(raw, ["score", "cal_size"])
        summary.to_csv(summary_path, index=False)

    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Cal. traces & Prefix risk & Prefix kept & Cert. steps & Full accept & Marg. FA \\",
        r"\midrule",
    ]
    for _, row in summary.sort_values("cal_size").iterrows():
        lines.append(
            f"{int(row['cal_size'])} & "
            f"{mean_ci(row, 'prefix_contamination', scale=100, digits=1)} & "
            f"{mean_ci(row, 'prefix_retained_fraction', scale=100, digits=1)} & "
            f"{mean_ci(row, 'prefix_retained_steps', digits=2)} & "
            f"{mean_ci(row, 'prefix_full_trace_rate', scale=100, digits=1)} & "
            f"{mean_ci(row, 'marginal_false_accept', scale=100, digits=1)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    latex_table(
        TABLES / "calibration_size_sensitivity.tex",
        "tab:calibration_size_sensitivity",
        "Calibration-size sensitivity for CPCC with the combined detector on 10 target splits. Smaller calibration sets require more conservative thresholds and have higher sampling variation; efficiency stabilizes as more trace-level calibration examples are available.",
        "\n".join(lines),
    )
    return summary


def write_fallback_pdf(path: Path, title: str, body: str) -> None:
    """Write a minimal valid one-page PDF when plotting libraries are absent."""

    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content = f"BT /F1 18 Tf 72 735 Td ({esc(title)}) Tj /F1 11 Tf 0 -28 Td ({esc(body)}) Tj ET"
    stream = content.encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{idx} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(bytes(output))


def load_dataset_summary(name: str) -> tuple[pd.DataFrame, str]:
    dirs = {
        "ProcessBench": OUT / "external_process" / "processbench_repeated_full10",
        "Math-Shepherd": OUT / "external_process" / "math_shepherd_repeated",
        "PRMBench": OUT / "external_process" / "prmbench_repeated",
        "PRM800K": OUT / "external_process" / "prm800k_repeated_full20",
    }
    path = dirs[name] / "table_process_main_summary.csv"
    if not path.exists() and name == "PRM800K":
        path = OUT / "external_process" / "prm800k_repeated_focused" / "table_process_main_summary.csv"
    return read_csv(path), str(path.relative_to(ROOT)) if path.exists() else ""


def build_external_table(audit: pd.DataFrame) -> None:
    label_type = dict(zip(audit["dataset"], audit["label_type"]))
    trace_err = dict(zip(audit["dataset"], audit["trace_error_rate"]))
    qwen_sources = {
        "ProcessBench": read_csv(OUT / "external_process" / "processbench_qwen_prm" / "table_external_score_summary.csv"),
        "Math-Shepherd": read_csv(OUT / "external_process" / "math_shepherd_qwen_prm" / "table_external_score_summary.csv"),
        "PRMBench": read_csv(OUT / "external_process" / "prmbench_full_qwen_prm" / "table_external_score_summary.csv"),
        "PRM800K": read_csv(OUT / "external_process" / "prm800k_qwen_prm" / "table_external_score_summary.csv"),
    }
    rows: list[tuple[str, str, pd.Series, pd.Series | None]] = []
    for dataset in ("ProcessBench", "Math-Shepherd", "PRMBench", "PRM800K"):
        df, _ = load_dataset_summary(dataset)
        if df.empty:
            continue
        for detector, score in (
            ("Qwen2.5-Math PRM", "qwen_prm_error"),
            ("Combined logistic", "combined_logistic_l2"),
            ("Token/format", "artifact_token_formatting_logistic_l2"),
            ("Random", "random"),
            ("Oracle", "oracle"),
        ):
            source = qwen_sources.get(dataset, pd.DataFrame()) if score == "qwen_prm_error" else df
            row = row_for(source, score)
            if row is not None:
                oracle = row_for(df, "oracle")
                rows.append((dataset, detector, row, oracle))

    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrrrrr}",
        r"\toprule",
        r"Dataset & Label type & Detector & Error prev. & Prefix risk & Kept & 0-risk oracle kept & Gap to 0-risk oracle & FE cov. & FA & Acc. err. \\",
        r"\midrule",
    ]
    for dataset, detector, row, oracle in rows:
        kept = float(row["prefix_retained_fraction_mean"])
        oracle_kept = float(oracle["prefix_retained_fraction_mean"]) if oracle is not None else np.nan
        gap = 100.0 * (oracle_kept - kept) if np.isfinite(oracle_kept) else np.nan
        lines.append(
            f"{tex(dataset)} & {tex(label_type.get(dataset, '--'))} & {tex(detector)} & "
            f"{pct(trace_err.get(dataset, np.nan), 1)} & "
            f"{mean_ci(row, 'prefix_contamination', scale=100, digits=1)} & "
            f"{mean_ci(row, 'prefix_retained_fraction', scale=100, digits=1)} & "
            f"{pct(oracle_kept, 1)} & {num(gap, 1)} & "
            f"{pct(row['fe_coverage_error_only_mean'], 1)} & "
            f"{pct(row['marginal_false_accept_mean'], 1)} & "
            f"{pct(row['accepted_error_rate_mean'], 1)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "external_process_results_expanded.tex",
        "tab:external_process_expanded",
        "External process-dataset stress tests at $\\alpha=0.05$. The 0-risk oracle kept column is the longest clean prefix under labels and is not alpha-matched; negative gaps can occur when a calibrated method allows the nominal contamination risk to retain longer prefixes.",
        "\n".join(lines),
    )


def build_nearest_neighbor_tables() -> None:
    claim = read_csv(OUT / "remaining_baselines" / "table_claim_filtering_summary.csv")
    dyn = read_csv(OUT / "remaining_baselines" / "table_dynamic_early_stop_summary.csv")
    fe = read_csv(OUT / "remaining_baselines" / "table_first_error_variants_summary.csv")
    lodo = read_csv(OUT / "remaining_baselines" / "table_weighted_lodo_prefix_summary.csv")
    if not fe.empty:
        fe.to_csv(RESULTS / "first_error_variants.csv", index=False)
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllllllr}",
        r"\toprule",
        r"Family & Variant & Scattered? & Prefix? & FE set? & Formal object & Risk / cov. & Efficiency & Acc. err. \\",
        r"\midrule",
    ]
    claim05 = claim[np.isclose(claim["alpha"], 0.05)].iloc[0] if not claim.empty else None
    dyn05 = dyn[np.isclose(dyn["alpha"], 0.05)].iloc[0] if not dyn.empty else None
    if claim05 is not None:
        lines.append(
                f"Claim filtering & Any accepted claim wrong & Yes & No & No & accepted-claim error & "
            f"{pct(claim05['test_risk_mean'], 1)} & {pct(claim05['retained_step_fraction_mean'], 1)} & "
            f"{pct(claim05['accepted_step_error_rate_mean'], 1)} \\\\"
        )
    if dyn05 is not None:
        lines.append(
            f"Dynamic abstention & Early stop & No & Yes & No & prefix contamination & "
            f"{pct(dyn05['prefix_contamination_mean'], 1)} & {pct(dyn05['retained_fraction_mean'], 1)} & "
            f"{pct(dyn05['accepted_error_rate_mean'], 1)} \\\\"
        )
    if not fe.empty:
        for variant in ("always_include_empty", "ranked_topk", "two_stage_empty"):
            subset = fe[(fe["variant"] == variant) & np.isclose(fe["alpha"], 0.05)]
            if subset.empty:
                continue
            row = subset.iloc[0]
            lines.append(
                f"First-error set & {tex(variant.replace('_', ' '))} & No & No & Yes & FE miss & "
                f"{pct(row['fe_coverage_error_only_mean'], 1)} & {num(row['fe_candidate_size_excluding_empty_mean'], 2)} & -- \\\\"
            )
    if not lodo.empty:
        lodo05 = lodo[np.isclose(lodo["alpha"], 0.05)].copy()
        if "target_domain" in lodo05.columns:
            lodo05 = lodo05.drop_duplicates(subset=["target_domain"], keep="last")
        for _, row in lodo05.iterrows():
            target = row.get("target_domain", "domain")
            lines.append(
                f"Weighted LODO & target {tex(target)} & No & Yes & No & shift diagnostic & "
                f"{pct(row.get('prefix_contamination_mean'), 1)} & {pct(row.get('prefix_retained_fraction_mean'), 1)} & "
                f"{pct(row.get('accepted_error_rate_mean'), 1)} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "nearest_neighbor_baselines.tex",
        "tab:nearest_neighbor_baselines",
        "Nearest-neighbor baselines and variants. Claim filtering may retain scattered steps; dynamic early stopping returns a prefix-like stop point; the proposed clean-prefix and first-error objects answer different process-debugging questions.",
        "\n".join(lines),
    )

    fe_lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Variant & FE cov. error & FE size & Top-1 & Within 1 & Within 2 & Mean dist. & Clean false alarm \\",
        r"\midrule",
    ]
    if not fe.empty:
        for _, row in fe[np.isclose(fe["alpha"], 0.05)].iterrows():
            fe_lines.append(
                f"{tex(str(row['variant']).replace('_', ' '))} & {pct(row['fe_coverage_error_only_mean'], 1)} & "
                f"{num(row['fe_candidate_size_excluding_empty_mean'], 2)} & "
                f"{pct(row['fe_top1_accuracy_error_only_mean'], 1)} & "
                f"{pct(row.get('fe_within1_error_only_mean'), 1)} & "
                f"{pct(row.get('fe_within2_error_only_mean'), 1)} & "
                f"{num(row.get('fe_top1_mean_abs_distance_error_only_mean'), 2)} & "
                f"{pct(row.get('clean_trace_false_alarm_rate_mean', row.get('false_localization_on_clean_mean')), 1)} \\\\"
            )
    fe_lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "first_error_variants.tex",
        "tab:first_error_variants",
        "First-error variants at $\\alpha=0.05$. FE size excludes $\\varnothing$. Within-one/two-step coverage uses candidate-set proximity; mean distance is the top-candidate absolute distance to the true first error. These diagnostics expose current localization weakness.",
        "\n".join(fe_lines),
    )


def build_selective_and_artifact_tables(process: pd.DataFrame) -> None:
    selected = [
        ("Qwen2.5-Math PRM", read_csv(OUT / "process_repeated_qwen_prm" / "table_external_score_summary.csv"), "qwen_prm_error"),
        ("Combined logistic", process, "combined_logistic_l2"),
        ("Token/format control", process, "artifact_token_formatting_logistic_l2"),
        ("Random", process, "random"),
        ("Oracle", process, "oracle"),
    ]
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Detector & Marginal FA & Accept & Accepted error & Clean accept & Error accept \\",
        r"\midrule",
    ]
    for label, df, score in selected:
        row = row_for(df, score)
        if row is None:
            continue
        lines.append(
            f"{tex(label)} & {mean_ci(row, 'marginal_false_accept', scale=100, digits=1)} & "
            f"{mean_ci(row, 'accept_rate', scale=100, digits=1)} & "
            f"{mean_ci(row, 'accepted_error_rate', scale=100, digits=1)} & "
            f"{mean_ci(row, 'clean_accept_rate', scale=100, digits=1)} & "
            f"{mean_ci(row, 'incorrect_accept_rate', scale=100, digits=1)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    latex_table(
        TABLES / "selective_risk_diagnostics.tex",
        "tab:selective_risk",
        "Selective-risk diagnostics. The conformal guarantee controls marginal false acceptance, $P(\\mathrm{accept}\\wedge\\mathrm{error})$; accepted-error rate is not covered by that guarantee unless separately calibrated.",
        "\n".join(lines),
    )

    def selective_procedure_values(raw_df: pd.DataFrame, score: str, beta_level: float = 0.05) -> tuple[str, dict[str, np.ndarray]]:
        subset = raw_df[(raw_df["score"] == score) & np.isclose(raw_df["alpha"].astype(float), 0.05)].copy()
        if subset.empty:
            return "0/0 feasible", {
                "accept": np.asarray([], dtype=float),
                "accepted_error": np.asarray([], dtype=float),
                "cp_upper": np.asarray([], dtype=float),
                "marginal_fa": np.asarray([], dtype=float),
            }
        feasible = (
            subset["selective_cal_accepts"].astype(float).to_numpy() > 0
        ) & (
            subset["selective_cal_upper_bound"].astype(float).to_numpy() <= beta_level
        )
        values = {
            "accept": np.where(feasible, subset["selective_test_accept_rate"].astype(float).to_numpy(), 0.0),
            "accepted_error": np.where(
                feasible,
                subset["selective_test_accepted_error_rate"].astype(float).fillna(0.0).to_numpy(),
                0.0,
            ),
            "cp_upper": np.where(feasible, subset["selective_cal_upper_bound"].astype(float).to_numpy(), 0.0),
            "marginal_fa": np.where(feasible, subset["selective_test_marginal_false_accept"].astype(float).to_numpy(), 0.0),
        }
        audit = pd.DataFrame(
            {
                "score": score,
                "seed": subset["seed"].to_numpy(),
                "feasible": feasible,
                "selected_lambda": np.where(feasible, subset["selective_lambda"].astype(float).to_numpy(), -np.inf),
                "cp_upper": values["cp_upper"],
                "test_accept_rate": values["accept"],
                "test_accepted_error_rate": values["accepted_error"],
                "test_marginal_false_accept": values["marginal_fa"],
            }
        )
        out_path = RESULTS / "selective_risk_calibration_procedure.csv"
        mode = "a" if out_path.exists() else "w"
        audit.to_csv(out_path, mode=mode, header=not out_path.exists(), index=False)
        return f"{int(np.sum(feasible))}/{len(feasible)} feasible", values

    selective_audit = RESULTS / "selective_risk_calibration_procedure.csv"
    if selective_audit.exists():
        selective_audit.unlink()
    raw_process = read_csv(OUT / "process_repeated_50seed" / "table_process_main.csv")
    raw_qwen = read_csv(OUT / "process_repeated_qwen_prm" / "table_external_score.csv")

    selective_lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Score source & Rule & $\beta$ & $\delta$ & Accept & Emp. acc. err. & CP upper & Marginal FA & Prefix kept \\",
        r"\midrule",
    ]
    for label, df, raw_df, score in (
        ("Combined logistic", process, raw_process, "combined_logistic_l2"),
        ("Qwen2.5-Math PRM", read_csv(OUT / "process_repeated_qwen_prm" / "table_external_score_summary.csv"), raw_qwen, "qwen_prm_error"),
    ):
        row = row_for(df, score)
        if row is None:
            continue
        feasible_label, selective_values = selective_procedure_values(raw_df, score)
        selective_lines.append(
            f"{tex(label)} & Marginal CRC & -- & -- & "
            f"{mean_ci(row, 'accept_rate', scale=100, digits=1)} & "
            f"{mean_ci(row, 'accepted_error_rate', scale=100, digits=1)} & -- & "
            f"{mean_ci(row, 'marginal_false_accept', scale=100, digits=1)} & "
            f"{mean_ci(row, 'prefix_retained_fraction', scale=100, digits=1)} \\\\"
        )
        selective_lines.append(
            f"{tex(label)} & {tex('Selective-risk rule (' + feasible_label + ')')} & 5.0 & 5.0 & "
            f"{mean_ci_values(selective_values['accept'], scale=100, digits=1)} & "
            f"{mean_ci_values(selective_values['accepted_error'], scale=100, digits=1)} & "
            f"{mean_ci_values(selective_values['cp_upper'], scale=100, digits=1)} & "
            f"{mean_ci_values(selective_values['marginal_fa'], scale=100, digits=1)} & "
            f"{mean_ci(row, 'prefix_retained_fraction', scale=100, digits=1)} \\\\"
        )
    selective_lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "selective_risk_calibration.tex",
        "tab:selective_risk_calibration",
        "Optional selective-risk calibration diagnostic at $\\beta=0.05$ and $\\delta=0.05$. The selective-risk rule selects thresholds whose Clopper--Pearson upper bound, with finite-grid union correction, is at most $\\beta$; infeasible splits use the reject-all sentinel and are counted as zero acceptance.",
        "\n".join(selective_lines),
    )

    artifact_scores = [
        ("Step index", "artifact_step_index_logistic_l2"),
        ("Trace length", "artifact_trace_length_logistic_l2"),
        ("Dataset ID", "artifact_dataset_id_logistic_l2"),
        ("Token/format", "artifact_token_formatting_logistic_l2"),
        ("Combined", "combined_logistic_l2"),
        ("Random", "random"),
    ]
    combined = row_for(process, "combined_logistic_l2")
    combined_kept = float(combined["prefix_retained_fraction_mean"]) if combined is not None else np.nan
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Control & AUROC & Prefix risk & Prefix kept & Gap vs combined & FE cov. & Accept \\",
        r"\midrule",
    ]
    for label, score in artifact_scores:
        row = row_for(process, score)
        if row is None:
            continue
        kept = float(row["prefix_retained_fraction_mean"])
        gap = 100.0 * (combined_kept - kept) if np.isfinite(combined_kept) else np.nan
        lines.append(
            f"{tex(label)} & {mean_ci(row, 'auroc', digits=3)} & "
            f"{mean_ci(row, 'prefix_contamination', scale=100, digits=1)} & "
            f"{mean_ci(row, 'prefix_retained_fraction', scale=100, digits=1)} & "
            f"{num(gap, 1)} & {pct(row['fe_coverage_error_only_mean'], 1)} & "
            f"{pct(row['accept_rate_mean'], 1)} \\\\"
        )
    shuffled = read_csv(OUT / "plan_gap_experiments" / "table_shuffled_score_controls_summary.csv")
    shuffled_scores = [
        ("Score shuffle within domain", "shuffled_within_domain"),
        ("Score shuffle within length", "shuffled_within_trace_length_bin"),
        ("Score shuffle domain+length", "shuffled_within_domain_length_bin"),
    ]
    for label, score in shuffled_scores:
        row = row_for(shuffled, score)
        if row is None:
            continue
        kept = float(row["prefix_retained_fraction_mean"])
        gap = 100.0 * (combined_kept - kept) if np.isfinite(combined_kept) else np.nan
        lines.append(
            f"{tex(label)} & {mean_ci(row, 'auroc', digits=3)} & "
            f"{mean_ci(row, 'prefix_contamination', scale=100, digits=1)} & "
            f"{mean_ci(row, 'prefix_retained_fraction', scale=100, digits=1)} & "
            f"{num(gap, 1)} & {pct(row['fe_coverage_error_only_mean'], 1)} & "
            f"{pct(row['accept_rate_mean'], 1)} \\\\"
        )
    revision4_controls = read_csv(RESULTS / "revision4_controls_summary.csv")
    for label, score in (
        ("Label-shuffled", "label_shuffled_combined_logistic_l2"),
        ("Trace-order shuffled", "trace_order_shuffled_combined_logistic_l2"),
    ):
        row = row_for(revision4_controls, score)
        if row is None:
            continue
        kept = float(row["prefix_retained_fraction_mean"])
        gap = 100.0 * (combined_kept - kept) if np.isfinite(combined_kept) else np.nan
        lines.append(
            f"{tex(label)} & {mean_ci(row, 'auroc', digits=3)} & "
            f"{mean_ci(row, 'prefix_contamination', scale=100, digits=1)} & "
            f"{mean_ci(row, 'prefix_retained_fraction', scale=100, digits=1)} & "
            f"{num(gap, 1)} & {pct(row['fe_coverage_error_only_mean'], 1)} & "
            f"{pct(row['accept_rate_mean'], 1)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "artifact_controls.tex",
        "tab:artifact_controls",
        "Artifact, label-shuffle, score-shuffle, and trace-order controls at $\\alpha=0.05$. Strong token/format and dataset controls show exploitable regularities; label shuffle tests whether learned efficiency depends on real label signal; trace-order shuffle tests whether prefix efficiency depends on ordered-process structure.",
        "\n".join(lines),
    )


def build_mondrian_and_shift_tables() -> None:
    shift_raw = read_csv(OUT / "process_repeated_50seed" / "table_shift_mondrian.csv")
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Calibration rule & Group & Prefix risk & Prefix kept & Full prefix & Test traces \\",
        r"\midrule",
    ]
    if not shift_raw.empty:
        subset = shift_raw[
            (shift_raw["score"] == "combined_logistic_l2")
            & np.isclose(shift_raw["alpha"], 0.05)
            & shift_raw["calibration"].astype(str).str.contains("group")
        ].copy()
        grouped = (
            subset.groupby(["calibration", "group_by", "group"], dropna=False)[
                ["prefix_contamination", "prefix_retained_fraction", "prefix_full_trace_rate", "n_test_traces"]
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
        for _, row in grouped.iterrows():
            group_by = str(row.get("group_by", "group"))
            group = str(row.get("group", "all"))
            rule = "domain" if group_by == "domain" else "trace length"
            lines.append(
                f"Mondrian by {tex(rule)} & {tex(group)} & "
                f"{pct(row.get('prefix_contamination'), 1)} & "
                f"{pct(row.get('prefix_retained_fraction'), 1)} & "
                f"{pct(row.get('prefix_full_trace_rate'), 1)} & "
                f"{num(row.get('n_test_traces'), 0)} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    latex_table(
        TABLES / "mondrian_calibration.tex",
        "tab:mondrian_calibration",
        "Mondrian/domain calibration diagnostics. Grouped calibration can reduce risk in heterogeneous mixtures but usually costs retained-prefix efficiency.",
        "\n".join(lines),
    )

    cross = read_csv(OUT / "process_repeated" / "table_cross_domain_summary.csv")
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Train/cal domain & Test domain & Prefix risk & Prefix kept & FA & Accept \\",
        r"\midrule",
    ]
    if not cross.empty:
        for _, row in cross[np.isclose(cross["alpha"], 0.05)].iterrows():
            dataset = str(row.get("dataset", "--"))
            if "->" in dataset:
                source_domain, target_domain = dataset.split("->", 1)
            else:
                source_domain = str(row.get("source_domain", "--"))
                target_domain = str(row.get("target_domain", "--"))
            lines.append(
                f"{tex(source_domain)} & {tex(target_domain)} & "
                f"{pct(row.get('prefix_contamination_mean'), 1)} & "
                f"{pct(row.get('prefix_retained_fraction_mean'), 1)} & "
                f"{pct(row.get('marginal_false_accept_mean'), 1)} & "
                f"{pct(row.get('accept_rate_mean'), 1)} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    latex_table(
        TABLES / "cross_domain_shift_invalid.tex",
        "tab:cross_domain_invalid",
        "Non-exchangeable cross-domain diagnostic; this is not a conformal validity claim. Failures under domain shift show why exchangeability and pre-specified Mondrian groups matter.",
        "\n".join(lines),
    )


def build_dataset_provenance_table(audit: pd.DataFrame) -> None:
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllrrrrrl}",
        r"\toprule",
        r"Dataset & Source & Generator & Label type & Traces & Steps & Step err. & Trace err. & FE? & Notes \\",
        r"\midrule",
    ]
    for _, row in audit.iterrows():
        lines.append(
            f"{tex(row['dataset'])} & {tex(row['source'])} & {tex(row['generator_model'])} & {tex(row['label_type'])} & "
            f"{int(row['traces']):,} & {int(row['steps']):,} & "
            f"{pct(row['step_error_rate'], 1)} & {pct(row['trace_error_rate'], 1)} & "
            f"{'yes' if row['first_error_available'] else 'no'} & {tex(row['notes'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "dataset_provenance.tex",
        "tab:dataset_provenance",
        "Dataset provenance and annotation summary. Target traces are split by trace ID; first-error labels are derived from the first annotated erroneous step when dense labels are available.",
        "\n".join(lines),
    )


def build_dataset_cards(audit: pd.DataFrame) -> None:
    lines = [
        r"\section{Dataset Cards and Annotation Provenance}",
        r"\label{app:dataset_cards}",
        r"\begingroup\small\sloppy",
        "",
        "First-error labels are derived from the first annotated erroneous step when dense step labels are available. Later steps are kept as annotated rather than automatically invalidated. All conformal splits use trace ID as the split unit.",
        "",
    ]
    for _, row in audit.iterrows():
        dataset = str(row["dataset"])
        label_source = str(row.get("label_source", "provided labels"))
        if "Target" in dataset:
            accessibility = "local CROP target annotation cache"
            errors = "provided process annotations"
            first_error = "first annotated erroneous step in the dense labels"
        else:
            accessibility = f"public/imported benchmark source: {row['source']}"
            errors = "benchmark-provided or dataset-provided process labels"
            first_error = "benchmark first-error annotation when available, otherwise first erroneous dense step"
        lines.extend(
            [
                rf"\paragraph{{{tex(dataset)}.}}",
                rf"Source: {tex(row['source'])}. Generator/model: {tex(row['generator_model'])}. "
                rf"Label type: {tex(row['label_type'])}; label source: {tex(label_source)}. "
                rf"Errors are {tex(errors)}. First error is derived as {tex(first_error)}. "
                rf"Traces: {int(row['traces']):,}; steps: {int(row['steps']):,}; "
                rf"step error prevalence: {pct(row['step_error_rate'], 1)}\%; "
                rf"trace error prevalence: {pct(row['trace_error_rate'], 1)}\%. "
                rf"Split unit: trace ID. Release/accessibility: {tex(accessibility)}. "
                rf"Notes: {tex(row['notes'])}.",
                "",
            ]
        )
    lines.append(r"\endgroup")
    (TABLES / "dataset_cards.tex").write_text("\n".join(lines))


def _shuffle_train_labels_within_domain(traces: list[TraceRecord], seed: int) -> list[TraceRecord]:
    rng = np.random.default_rng(seed)
    labels_by_domain: dict[str, list[int]] = {}
    for trace in traces:
        labels_by_domain.setdefault(trace.domain, []).extend(int(step.y_error) for step in trace.steps)
    for labels in labels_by_domain.values():
        rng.shuffle(labels)

    out: list[TraceRecord] = []
    for trace in traces:
        labels = labels_by_domain[trace.domain]
        steps = []
        for step in trace.steps:
            label = int(labels.pop())
            steps.append(replace(step, y_error=label, is_correct=not bool(label)))
        out.append(replace(trace, steps=steps))
    return out


def _shuffle_trace_order(traces: list[TraceRecord], scores_by_trace: list[np.ndarray], seed: int) -> tuple[list[TraceRecord], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    shuffled_traces: list[TraceRecord] = []
    shuffled_scores: list[np.ndarray] = []
    for trace, scores in zip(traces, scores_by_trace):
        scores = np.asarray(scores, dtype=float)
        if len(trace.steps) <= 1:
            shuffled_traces.append(trace)
            shuffled_scores.append(scores)
            continue
        order = rng.permutation(len(trace.steps))
        steps = [
            replace(trace.steps[int(old_idx)], step_number=new_idx, total_steps=len(trace.steps))
            for new_idx, old_idx in enumerate(order)
        ]
        shuffled_traces.append(replace(trace, steps=steps))
        shuffled_scores.append(scores[order])
    return shuffled_traces, shuffled_scores


def build_revision4_controls() -> pd.DataFrame:
    """Run bounded cached controls requested by revision4.

    Label-shuffle uses shuffled training labels within domain while calibration
    and test labels remain true. Trace-order shuffle keeps the fitted combined
    detector scores fixed but permutes the order of calibration/test steps
    before constructing prefix and first-error objects.
    """

    summary_path = RESULTS / "revision4_controls_summary.csv"
    if summary_path.exists():
        cached = read_csv(summary_path)
        expected = {"label_shuffled_combined_logistic_l2", "trace_order_shuffled_combined_logistic_l2"}
        if expected.issubset(set(cached.get("score", pd.Series(dtype=str)).astype(str))):
            return cached

    from crop.experiments.common import ScoreBundle
    from crop.experiments.exp08_cheap_baselines import _summarize_with_ci
    from crop.experiments.exp09_process_repeated import _evaluate_bundle, _fit_model_bundle
    from crop.splits import Split

    traces = load_many_npz([ROOT / "data" / "strengthened" / "crop_target_combined_steps.npz"], ["mixed"], allow_nan=True)
    seeds = list(range(2806, 2826))
    lambdas = np.linspace(0.0, 1.0, 101)
    rows: list[dict] = []
    for seed in seeds:
        split = split_traces(traces, seed=seed)

        shuffled_train = _shuffle_train_labels_within_domain(split.train, seed + 17_000)
        label_split = Split(train=shuffled_train, cal=split.cal, test=split.test)
        started = time.perf_counter()
        label_bundle = _fit_model_bundle("logistic_l2", label_split, seed=seed)
        elapsed = time.perf_counter() - started
        rows.extend(
            _evaluate_bundle(
                score_name="label_shuffled_combined_logistic_l2",
                score_family="artifact_control",
                split=split,
                bundle=label_bundle,
                seed=seed,
                alphas=[0.05],
                lambdas=lambdas,
                runtime_seconds=elapsed,
            )
        )

        started = time.perf_counter()
        normal_bundle = _fit_model_bundle("logistic_l2", split, seed=seed)
        elapsed = time.perf_counter() - started
        shuffled_cal, shuffled_cal_scores = _shuffle_trace_order(
            split.cal, normal_bundle.cal_scores_by_trace, seed + 23_000
        )
        shuffled_test, shuffled_test_scores = _shuffle_trace_order(
            split.test, normal_bundle.test_scores_by_trace, seed + 29_000
        )
        order_split = Split(train=split.train, cal=shuffled_cal, test=shuffled_test)
        order_bundle = ScoreBundle(
            name="trace_order_shuffled_combined_logistic_l2",
            cal_scores_by_trace=shuffled_cal_scores,
            test_scores_by_trace=shuffled_test_scores,
            cal_step_scores=np.concatenate(shuffled_cal_scores) if shuffled_cal_scores else np.array([]),
            test_step_scores=np.concatenate(shuffled_test_scores) if shuffled_test_scores else np.array([]),
            model=normal_bundle.model,
        )
        rows.extend(
            _evaluate_bundle(
                score_name="trace_order_shuffled_combined_logistic_l2",
                score_family="order_diagnostic",
                split=order_split,
                bundle=order_bundle,
                seed=seed,
                alphas=[0.05],
                lambdas=lambdas,
                runtime_seconds=elapsed,
            )
        )

    controls = pd.DataFrame(rows)
    controls.to_csv(RESULTS / "revision4_controls.csv", index=False)
    summary = _summarize_with_ci(controls)
    summary.to_csv(RESULTS / "revision4_controls_summary.csv", index=False)
    return summary


def _qwen_scores_by_trace(traces: list[TraceRecord], qwen_csv: Path) -> list[np.ndarray]:
    qwen = read_csv(qwen_csv)
    if qwen.empty:
        raise FileNotFoundError(f"Missing Qwen score cache: {qwen_csv}")
    grouped = {trace_id: group.sort_values("step_id") for trace_id, group in qwen.groupby("trace_id", dropna=False)}
    out: list[np.ndarray] = []
    for trace in traces:
        group = grouped.get(trace.trace_id)
        if group is None or len(group) != len(trace.steps):
            out.append(np.full(len(trace.steps), 0.5, dtype=float))
        else:
            out.append(group["qwen_prm_error"].to_numpy(dtype=float))
    return out


def _summarize_with_ci_local(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    numeric = [col for col in df.columns if col not in set(group_cols) and pd.api.types.is_numeric_dtype(df[col])]
    grouped = df.groupby(group_cols, dropna=False)
    pieces = []
    for col in numeric:
        mean = grouped[col].mean()
        std = grouped[col].std().fillna(0.0)
        count = grouped[col].count().clip(lower=1)
        pieces.append(
            pd.DataFrame(
                {
                    f"{col}_mean": mean,
                    f"{col}_std": std,
                    f"{col}_n": count,
                    f"{col}_ci95": 1.96 * std / np.sqrt(count),
                }
            )
        )
    return pd.concat(pieces, axis=1).reset_index()


def _prefix_feature_view(traces: list[TraceRecord], base_scores_by_trace: list[np.ndarray]) -> list[TraceRecord]:
    out: list[TraceRecord] = []
    for trace, base_scores in zip(traces, base_scores_by_trace):
        scores = np.asarray(base_scores, dtype=float)
        total_words = max(sum(len((step.step_content or "").split()) for step in trace.steps), 1)
        cumulative_words = 0
        steps = []
        for idx, step in enumerate(trace.steps):
            words = len((step.step_content or "").split())
            cumulative_words += words
            previous = scores[:idx]
            prev_max = float(np.max(previous)) if len(previous) else 0.0
            prev_mean = float(np.mean(previous)) if len(previous) else 0.0
            previous_score = float(scores[idx - 1]) if idx > 0 and idx - 1 < len(scores) else 0.0
            current_score = float(scores[idx]) if idx < len(scores) else 0.5
            denom = max(len(trace.steps) - 1, 1)
            features = np.asarray(
                [
                    current_score,
                    prev_max,
                    prev_mean,
                    current_score - previous_score,
                    idx / denom,
                    len(trace.steps),
                    cumulative_words / total_words,
                    cumulative_words,
                ],
                dtype=float,
            )
            steps.append(replace(step, x=features))
        out.append(replace(trace, steps=steps))
    return out


def build_order_sensitive_detector_table() -> pd.DataFrame:
    summary_path = RESULTS / "order_sensitive_detector_summary.csv"
    if summary_path.exists() and "Prefix-feature logistic" in set(read_csv(summary_path).get("score", pd.Series(dtype=str)).astype(str)):
        summary = read_csv(summary_path)
    else:
        from crop.experiments.common import ScoreBundle
        from crop.experiments.exp09_process_repeated import _fit_model_bundle
        from crop.models import scores_by_trace_from_model
        from crop.splits import Split

        def evaluate_prefix_only(score_name: str, family: str, split: Split, bundle: ScoreBundle, seed: int) -> dict:
            prefix_cal = prefix_losses_by_lambda(split.cal, bundle.cal_scores_by_trace, lambdas)
            prefix_lambda, prefix_cal_risk = select_lambda_crc(prefix_cal, lambdas, alpha=0.05, direction="increasing")
            prefix_test = prefix_losses_by_lambda(split.test, bundle.test_scores_by_trace, np.asarray([prefix_lambda]))[0]
            lengths = prefix_lengths(bundle.test_scores_by_trace, prefix_lambda)
            totals = np.asarray([len(trace.steps) for trace in split.test], dtype=float)
            fe_cal = first_error_losses_by_lambda(split.cal, bundle.cal_scores_by_trace, lambdas)
            fe_lambda, fe_cal_risk = select_lambda_crc(fe_cal, lambdas, alpha=0.05, direction="increasing")
            from crop.sequence import candidate_first_error_set

            candidate_sets = [candidate_first_error_set(scores, fe_lambda, include_no_error=True) for scores in bundle.test_scores_by_trace]
            fe_metrics = first_error_diagnostics(candidate_sets, bundle.test_scores_by_trace, split.test)
            return {
                "score": score_name,
                "score_family": family,
                "seed": seed,
                "alpha": 0.05,
                "prefix_lambda": prefix_lambda,
                "prefix_cal_corrected_risk": prefix_cal_risk,
                "prefix_contamination": float(np.mean(prefix_test)) if len(prefix_test) else np.nan,
                "prefix_retained_fraction": float(np.mean(lengths / np.maximum(totals, 1.0))) if len(lengths) else np.nan,
                "first_error_lambda": fe_lambda,
                "first_error_cal_corrected_risk": fe_cal_risk,
                "fe_within1_error_only": fe_metrics.get("fe_within1_error_only", np.nan),
            }

        combined = load_many_npz([ROOT / "data" / "strengthened" / "crop_target_combined_steps.npz"], ["mixed"], allow_nan=True)
        lambdas = np.linspace(0.0, 1.0, 101)
        rows: list[dict] = []
        for seed in range(2806, 2816):
            reference = split_traces(combined, seed=seed)
            base_bundle = _fit_model_bundle("logistic_l2", reference, seed=seed)
            train_scores = scores_by_trace_from_model(base_bundle.model, reference.train)
            prefix_split = Split(
                train=_prefix_feature_view(reference.train, train_scores),
                cal=_prefix_feature_view(reference.cal, base_bundle.cal_scores_by_trace),
                test=_prefix_feature_view(reference.test, base_bundle.test_scores_by_trace),
            )
            prefix_bundle = _fit_model_bundle("logistic_l2", prefix_split, seed=seed)
            rows.append(evaluate_prefix_only("Prefix-feature logistic", "order_original", prefix_split, prefix_bundle, seed))
            shuffled_cal, shuffled_cal_scores = _shuffle_trace_order(prefix_split.cal, prefix_bundle.cal_scores_by_trace, seed + 59_000)
            shuffled_test, shuffled_test_scores = _shuffle_trace_order(prefix_split.test, prefix_bundle.test_scores_by_trace, seed + 61_000)
            shuffled_split = Split(train=prefix_split.train, cal=shuffled_cal, test=shuffled_test)
            shuffled_bundle = ScoreBundle("Prefix-feature logistic shuffled", shuffled_cal_scores, shuffled_test_scores, np.concatenate(shuffled_cal_scores), np.concatenate(shuffled_test_scores), None)
            rows.append(evaluate_prefix_only("Prefix-feature logistic", "order_shuffled", shuffled_split, shuffled_bundle, seed))
        raw = pd.DataFrame(rows)
        raw.to_csv(RESULTS / "order_sensitive_detector.csv", index=False)
        prefix_summary = _summarize_with_ci_local(raw, ["score", "score_family", "alpha"])
        existing = read_csv(RESULTS / "order_sensitivity_summary.csv")
        keep = {"Token/format", "Combined logistic", "Running-max combined", "Qwen2.5-Math PRM"}
        existing = existing[existing["score"].astype(str).isin(keep)] if not existing.empty else pd.DataFrame()
        summary = pd.concat([existing, prefix_summary], ignore_index=True, sort=False)
        summary.to_csv(summary_path, index=False)

    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Detector & Original kept & Shuffled kept & Drop & Artifact gain & Prefix risk & FE within 1 \\",
        r"\midrule",
    ]
    token_original = row_for(summary[summary["score_family"] == "order_original"], "Token/format")
    token_kept = float(token_original["prefix_retained_fraction_mean"]) if token_original is not None else np.nan
    for detector in ("Token/format", "Combined logistic", "Prefix-feature logistic", "Running-max combined", "Qwen2.5-Math PRM"):
        original = row_for(summary[summary["score_family"] == "order_original"], detector)
        shuffled = row_for(summary[summary["score_family"] == "order_shuffled"], detector)
        if original is None or shuffled is None:
            continue
        drop = 100.0 * (float(original["prefix_retained_fraction_mean"]) - float(shuffled["prefix_retained_fraction_mean"]))
        gain = 100.0 * (float(original["prefix_retained_fraction_mean"]) - token_kept) if np.isfinite(token_kept) else np.nan
        lines.append(
            f"{tex(detector)} & {mean_ci(original, 'prefix_retained_fraction', scale=100, digits=1)} & "
            f"{mean_ci(shuffled, 'prefix_retained_fraction', scale=100, digits=1)} & "
            f"{drop:.1f} & {gain:.1f} & "
            f"{mean_ci(original, 'prefix_contamination', scale=100, digits=1)} & "
            f"{mean_ci(original, 'fe_within1_error_only', scale=100, digits=1)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "order_sensitive_detector.tex",
        "tab:order_sensitive_detector",
        "Order-sensitive detector diagnostics on 10 target splits. Prefix-feature logistic augments the base combined score with running prefix statistics, step position, trace length, and cumulative token count. Larger drops under trace-order shuffling indicate greater dependence on ordered-process structure.",
        "\n".join(lines),
    )
    return summary


def _error_conditional_fe_row(
    dataset: str,
    detector: str,
    split,
    cal_scores_by_trace: list[np.ndarray],
    test_scores_by_trace: list[np.ndarray],
    seed: int,
    alpha: float,
    lambdas: np.ndarray,
) -> dict:
    cal_pairs = [
        (trace, scores)
        for trace, scores in zip(split.cal, cal_scores_by_trace)
        if trace.first_error is not None
    ]
    test_pairs = [
        (trace, scores)
        for trace, scores in zip(split.test, test_scores_by_trace)
        if trace.first_error is not None
    ]
    if not cal_pairs or not test_pairs:
        return {}
    cal_traces, cal_scores = zip(*cal_pairs)
    test_traces, test_scores = zip(*test_pairs)
    losses = first_error_error_only_losses_by_lambda(list(cal_traces), list(cal_scores), lambdas)
    lambda_hat, cal_risk = select_lambda_crc(losses, lambdas, alpha=alpha, direction="increasing")
    candidate_sets = [
        set(np.flatnonzero(np.asarray(scores, dtype=float) >= lambda_hat).astype(int).tolist())
        for scores in test_scores
    ]
    metrics = first_error_diagnostics(candidate_sets, list(test_scores), list(test_traces))
    return {
        "dataset": dataset,
        "score": detector,
        "alpha": alpha,
        "seed": seed,
        "lambda": lambda_hat,
        "cal_corrected_risk": cal_risk,
        "n_cal_error_traces": len(cal_pairs),
        "n_test_error_traces": len(test_pairs),
        **metrics,
    }


def build_error_conditional_fe_table() -> pd.DataFrame:
    summary_path = RESULTS / "error_conditional_first_error_summary.csv"
    if summary_path.exists():
        cached = read_csv(summary_path)
        if {"Target", "ProcessBench", "PRMBench"}.issubset(set(cached.get("dataset", pd.Series(dtype=str)).astype(str))):
            summary = cached
        else:
            summary = pd.DataFrame()
    else:
        summary = pd.DataFrame()

    if summary.empty:
        from crop.experiments.common import ScoreBundle
        from crop.experiments.exp09_process_repeated import _fit_model_bundle

        specs = [
            (
                "Target",
                ROOT / "data" / "strengthened" / "crop_target_combined_steps.npz",
                list(range(2806, 2826)),
                OUT / "process_repeated_qwen_prm" / "qwen_prm_scores.csv",
            ),
            (
                "ProcessBench",
                OUT / "external_process" / "processbench" / "processbench_combined_steps.npz",
                list(range(2806, 2816)),
                OUT / "external_process" / "processbench_qwen_prm" / "qwen_prm_scores.csv",
            ),
            (
                "PRMBench",
                OUT / "external_process" / "prmbench" / "prmbench_combined_steps.npz",
                list(range(2806, 2816)),
                None,
            ),
        ]
        rows: list[dict] = []
        lambdas = np.linspace(0.0, 1.0, 101)
        for dataset, path, seeds, qwen_csv in specs:
            if not path.exists():
                continue
            traces = load_many_npz([path], ["mixed"], allow_nan=True)
            for seed in seeds:
                split = split_traces(traces, seed=seed)
                bundle = _fit_model_bundle("logistic_l2", split, seed=seed)
                row = _error_conditional_fe_row(
                    dataset,
                    "Combined logistic",
                    split,
                    bundle.cal_scores_by_trace,
                    bundle.test_scores_by_trace,
                    seed,
                    0.05,
                    lambdas,
                )
                if row:
                    rows.append(row)
                if qwen_csv is not None and qwen_csv.exists():
                    cal_scores = _qwen_scores_by_trace(split.cal, qwen_csv)
                    test_scores = _qwen_scores_by_trace(split.test, qwen_csv)
                    qrow = _error_conditional_fe_row(
                        dataset,
                        "Qwen2.5-Math PRM",
                        split,
                        cal_scores,
                        test_scores,
                        seed,
                        0.05,
                        lambdas,
                    )
                    if qrow:
                        rows.append(qrow)
        raw = pd.DataFrame(rows)
        raw.to_csv(RESULTS / "error_conditional_first_error.csv", index=False)
        summary = _summarize_with_ci_local(raw, ["dataset", "score", "alpha"])
        summary.to_csv(summary_path, index=False)

    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Dataset & Detector & $\alpha_{\rm FE}$ & FE cov. error & FE size & Top-1 & Within 1 & Within 2 & Mean dist. \\",
        r"\midrule",
    ]
    for _, row in summary.sort_values(["dataset", "score"]).iterrows():
        lines.append(
            f"{tex(row['dataset'])} & {tex(row['score'])} & {float(row['alpha']):.2f} & "
            f"{mean_ci(row, 'fe_coverage_error_only', scale=100, digits=1)} & "
            f"{mean_ci(row, 'fe_candidate_size_excluding_empty', digits=2)} & "
            f"{mean_ci(row, 'fe_top1_accuracy_error_only', scale=100, digits=1)} & "
            f"{mean_ci(row, 'fe_within1_error_only', scale=100, digits=1)} & "
            f"{mean_ci(row, 'fe_within2_error_only', scale=100, digits=1)} & "
            f"{mean_ci(row, 'fe_top1_mean_abs_distance_error_only', digits=2)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "error_conditional_first_error.tex",
        "tab:error_conditional_fe",
        "Error-conditional first-error calibration at $\\alpha_{\\rm FE}=0.05$. Calibration uses only traces with an annotated error and candidate sets exclude $\\varnothing$, so the guarantee targets first-error localization conditional on the trace being erroneous.",
        "\n".join(lines),
    )
    return summary


def build_simultaneous_certificate_table() -> pd.DataFrame:
    summary_path = RESULTS / "simultaneous_certificates_summary.csv"
    if summary_path.exists():
        summary = read_csv(summary_path)
    else:
        from crop.experiments.exp09_process_repeated import _fit_model_bundle
        from crop.sequence import candidate_first_error_set

        traces = load_many_npz([ROOT / "data" / "strengthened" / "crop_target_combined_steps.npz"], ["mixed"], allow_nan=True)
        lambdas = np.linspace(0.0, 1.0, 101)
        alpha_total = 0.05
        configs = [
            ("Separate $\\alpha$ each", 0.05, 0.05, 0.05, False),
            ("Bonferroni equal", alpha_total / 3.0, alpha_total / 3.0, alpha_total / 3.0, False),
            ("Bonferroni prefix-heavy", 0.6 * alpha_total, 0.2 * alpha_total, 0.2 * alpha_total, False),
            ("Shared joint-loss CRC", alpha_total, alpha_total, alpha_total, True),
        ]
        rows: list[dict] = []
        for seed in range(2806, 2826):
            split = split_traces(traces, seed=seed)
            bundle = _fit_model_bundle("logistic_l2", split, seed=seed)
            prefix_cal = prefix_losses_by_lambda(split.cal, bundle.cal_scores_by_trace, lambdas)
            fe_cal = first_error_losses_by_lambda(split.cal, bundle.cal_scores_by_trace, lambdas)
            fa_cal = np.vstack([whole_trace_false_accept_losses(split.cal, bundle.cal_scores_by_trace, lam) for lam in lambdas])
            joint_cal = np.maximum.reduce([prefix_cal, fe_cal, fa_cal])
            totals = np.asarray([len(trace.steps) for trace in split.test], dtype=int)
            for label, prefix_alpha, fe_alpha, fa_alpha, shared in configs:
                if shared:
                    shared_lambda, cal_risk = select_lambda_crc(joint_cal, lambdas, alpha=alpha_total, direction="increasing")
                    prefix_lambda = fe_lambda = fa_lambda = shared_lambda
                else:
                    prefix_lambda, _ = select_lambda_crc(prefix_cal, lambdas, alpha=prefix_alpha, direction="increasing")
                    fe_lambda, _ = select_lambda_crc(fe_cal, lambdas, alpha=fe_alpha, direction="increasing")
                    fa_lambda, _ = select_lambda_crc(fa_cal, lambdas, alpha=fa_alpha, direction="increasing")
                    cal_risk = np.nan
                prefix_test = prefix_losses_by_lambda(split.test, bundle.test_scores_by_trace, np.asarray([prefix_lambda]))[0]
                fe_test = first_error_localization_losses(split.test, bundle.test_scores_by_trace, fe_lambda)
                fa_test = whole_trace_false_accept_losses(split.test, bundle.test_scores_by_trace, fa_lambda)
                lengths = prefix_lengths(bundle.test_scores_by_trace, prefix_lambda)
                accepted = np.asarray([np.max(scores) <= fa_lambda if len(scores) else True for scores in bundle.test_scores_by_trace])
                candidates = [candidate_first_error_set(scores, fe_lambda, include_no_error=True) for scores in bundle.test_scores_by_trace]
                fe_size = np.mean([sum(candidate is not None for candidate in cset) for cset in candidates])
                rows.append(
                    {
                        "calibration": label,
                        "seed": seed,
                        "prefix_risk": float(np.mean(prefix_test)),
                        "fe_miss": float(np.mean(fe_test)),
                        "marginal_fa": float(np.mean(fa_test)),
                        "joint_failure": float(np.mean(prefix_test | fe_test | fa_test)),
                        "prefix_kept": float(np.mean(lengths / np.maximum(totals, 1))),
                        "accept_rate": float(np.mean(accepted)),
                        "fe_size": float(fe_size),
                        "joint_cal_corrected_risk": cal_risk,
                    }
                )
        raw = pd.DataFrame(rows)
        raw.to_csv(RESULTS / "simultaneous_certificates.csv", index=False)
        summary = _summarize_with_ci_local(raw, ["calibration"])
        summary.to_csv(summary_path, index=False)

    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Calibration & Prefix risk & FE miss & FA & Joint fail & Prefix kept & Accept & FE size \\",
        r"\midrule",
    ]
    order = ["Separate $\\alpha$ each", "Bonferroni equal", "Bonferroni prefix-heavy", "Shared joint-loss CRC"]
    for label in order:
        subset = summary[summary["calibration"] == label]
        if subset.empty:
            continue
        row = subset.iloc[0]
        lines.append(
            f"{label} & {mean_ci(row, 'prefix_risk', scale=100, digits=1)} & "
            f"{mean_ci(row, 'fe_miss', scale=100, digits=1)} & "
            f"{mean_ci(row, 'marginal_fa', scale=100, digits=1)} & "
            f"{mean_ci(row, 'joint_failure', scale=100, digits=1)} & "
            f"{mean_ci(row, 'prefix_kept', scale=100, digits=1)} & "
            f"{mean_ci(row, 'accept_rate', scale=100, digits=1)} & "
            f"{mean_ci(row, 'fe_size', digits=2)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "simultaneous_certificates.tex",
        "tab:simultaneous_certificates",
        "Simultaneous trace-certificate diagnostics on target traces with the combined detector. Separate calibration reports each object at $\\alpha=0.05$; Bonferroni rows target total failure probability $0.05$; shared joint-loss CRC uses one threshold for all objects.",
        "\n".join(lines),
    )
    return summary


def build_order_sensitivity_table() -> pd.DataFrame:
    summary_path = RESULTS / "order_sensitivity_summary.csv"
    if summary_path.exists() and "Running-max combined" in set(read_csv(summary_path).get("score", pd.Series(dtype=str)).astype(str)):
        summary = read_csv(summary_path)
    else:
        from crop.experiments.common import ScoreBundle
        from crop.experiments.exp09_process_repeated import _artifact_views, _evaluate_bundle, _fit_model_bundle, _split_like

        combined = load_many_npz([ROOT / "data" / "strengthened" / "crop_target_combined_steps.npz"], ["mixed"], allow_nan=True)
        artifact = _artifact_views(combined)["artifact_token_formatting"]
        qwen_csv = OUT / "process_repeated_qwen_prm" / "qwen_prm_scores.csv"
        lambdas = np.linspace(0.0, 1.0, 101)
        rows: list[dict] = []
        for seed in range(2806, 2816):
            reference = split_traces(combined, seed=seed)
            splits = {
                "Combined logistic": reference,
                "Token/format": _split_like(reference, artifact),
            }
            for detector, split in splits.items():
                bundle = _fit_model_bundle("logistic_l2", split, seed=seed)
                rows.extend(_evaluate_bundle(score_name=detector, score_family="order_original", split=split, bundle=bundle, seed=seed, alphas=[0.05], lambdas=lambdas, runtime_seconds=0.0))
                shuffled_cal, shuffled_cal_scores = _shuffle_trace_order(split.cal, bundle.cal_scores_by_trace, seed + 31_000)
                shuffled_test, shuffled_test_scores = _shuffle_trace_order(split.test, bundle.test_scores_by_trace, seed + 37_000)
                shuffled_split = type(split)(train=split.train, cal=shuffled_cal, test=shuffled_test)
                shuffled_bundle = ScoreBundle(
                    name=f"{detector} trace-order shuffled",
                    cal_scores_by_trace=shuffled_cal_scores,
                    test_scores_by_trace=shuffled_test_scores,
                    cal_step_scores=np.concatenate(shuffled_cal_scores),
                    test_step_scores=np.concatenate(shuffled_test_scores),
                    model=bundle.model,
                )
                rows.extend(_evaluate_bundle(score_name=detector, score_family="order_shuffled", split=shuffled_split, bundle=shuffled_bundle, seed=seed, alphas=[0.05], lambdas=lambdas, runtime_seconds=0.0))
                if detector == "Combined logistic":
                    running_cal = [np.maximum.accumulate(scores) for scores in bundle.cal_scores_by_trace]
                    running_test = [np.maximum.accumulate(scores) for scores in bundle.test_scores_by_trace]
                    running_bundle = ScoreBundle(
                        "Running-max combined",
                        running_cal,
                        running_test,
                        np.concatenate(running_cal),
                        np.concatenate(running_test),
                        None,
                    )
                    rows.extend(_evaluate_bundle(score_name="Running-max combined", score_family="order_original", split=split, bundle=running_bundle, seed=seed, alphas=[0.05], lambdas=lambdas, runtime_seconds=0.0))
                    running_shuffled_cal, running_shuffled_cal_scores = _shuffle_trace_order(split.cal, running_cal, seed + 35_000)
                    running_shuffled_test, running_shuffled_test_scores = _shuffle_trace_order(split.test, running_test, seed + 39_000)
                    running_shuffled_split = type(split)(train=split.train, cal=running_shuffled_cal, test=running_shuffled_test)
                    running_shuffled_bundle = ScoreBundle(
                        "Running-max combined trace-order shuffled",
                        running_shuffled_cal_scores,
                        running_shuffled_test_scores,
                        np.concatenate(running_shuffled_cal_scores),
                        np.concatenate(running_shuffled_test_scores),
                        None,
                    )
                    rows.extend(_evaluate_bundle(score_name="Running-max combined", score_family="order_shuffled", split=running_shuffled_split, bundle=running_shuffled_bundle, seed=seed, alphas=[0.05], lambdas=lambdas, runtime_seconds=0.0))
            if qwen_csv.exists():
                cal_scores = _qwen_scores_by_trace(reference.cal, qwen_csv)
                test_scores = _qwen_scores_by_trace(reference.test, qwen_csv)
                qwen_bundle = ScoreBundle("Qwen2.5-Math PRM", cal_scores, test_scores, np.concatenate(cal_scores), np.concatenate(test_scores), None)
                rows.extend(_evaluate_bundle(score_name="Qwen2.5-Math PRM", score_family="order_original", split=reference, bundle=qwen_bundle, seed=seed, alphas=[0.05], lambdas=lambdas, runtime_seconds=0.0))
                shuffled_cal, shuffled_cal_scores = _shuffle_trace_order(reference.cal, cal_scores, seed + 41_000)
                shuffled_test, shuffled_test_scores = _shuffle_trace_order(reference.test, test_scores, seed + 43_000)
                shuffled_split = type(reference)(train=reference.train, cal=shuffled_cal, test=shuffled_test)
                shuffled_bundle = ScoreBundle("Qwen2.5-Math PRM trace-order shuffled", shuffled_cal_scores, shuffled_test_scores, np.concatenate(shuffled_cal_scores), np.concatenate(shuffled_test_scores), None)
                rows.extend(_evaluate_bundle(score_name="Qwen2.5-Math PRM", score_family="order_shuffled", split=shuffled_split, bundle=shuffled_bundle, seed=seed, alphas=[0.05], lambdas=lambdas, runtime_seconds=0.0))
        raw = pd.DataFrame(rows)
        raw.to_csv(RESULTS / "order_sensitivity.csv", index=False)
        summary = _summarize_with_ci_local(raw, ["score", "score_family", "alpha"])
        summary.to_csv(summary_path, index=False)

    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Detector & Original kept & Shuffled kept & Drop & Prefix risk & FE within 1 & Artifact gain \\",
        r"\midrule",
    ]
    token_original = row_for(summary[summary["score_family"] == "order_original"], "Token/format")
    token_kept = float(token_original["prefix_retained_fraction_mean"]) if token_original is not None else np.nan
    for detector in ("Token/format", "Combined logistic", "Running-max combined", "Qwen2.5-Math PRM"):
        original = row_for(summary[summary["score_family"] == "order_original"], detector)
        shuffled = row_for(summary[summary["score_family"] == "order_shuffled"], detector)
        if original is None or shuffled is None:
            continue
        drop = 100.0 * (float(original["prefix_retained_fraction_mean"]) - float(shuffled["prefix_retained_fraction_mean"]))
        gain = 100.0 * (float(original["prefix_retained_fraction_mean"]) - token_kept) if np.isfinite(token_kept) else np.nan
        lines.append(
            f"{tex(detector)} & {mean_ci(original, 'prefix_retained_fraction', scale=100, digits=1)} & "
            f"{mean_ci(shuffled, 'prefix_retained_fraction', scale=100, digits=1)} & "
            f"{drop:.1f} & {mean_ci(original, 'prefix_contamination', scale=100, digits=1)} & "
            f"{mean_ci(original, 'fe_within1_error_only', scale=100, digits=1)} & {gain:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    latex_table(
        TABLES / "order_sensitivity.tex",
        "tab:order_sensitivity",
        "Order-sensitivity diagnostics on 10 target splits. Shuffling step order preserves per-step scores and labels but changes the clean-prefix object. The running-max score is an explicitly prefix-sensitive transformation of the combined detector; larger drops indicate stronger dependence on ordered-process structure.",
        "\n".join(lines),
    )
    return summary


def build_runtime_cost_table() -> None:
    runtime = read_csv(OUT / "process_repeated_50seed" / "table_runtime_summary.csv")
    runtime_by_score = {
        str(row["score"]): float(row["runtime_seconds"])
        for _, row in runtime.iterrows()
        if "runtime_seconds" in row and np.isfinite(float(row["runtime_seconds"]))
    }

    def runtime_seconds(score: str) -> str:
        value = runtime_by_score.get(score, np.nan)
        return f"{value:.2f}s" if np.isfinite(value) else "--"

    cheap_note = ""
    run_config = ROOT / "outputs" / "cheap_baselines" / "full" / "run_config.json"
    if run_config.exists():
        try:
            cheap_note = json.loads(run_config.read_text()).get("runtime_note", "")
        except json.JSONDecodeError:
            cheap_note = ""
    if cheap_note:
        cheap_note = "Cached CoE/likelihood note: " + cheap_note
    else:
        cheap_note = "Cached feature generation was performed outside this report builder."

    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllp{0.50\textwidth}}",
        r"\toprule",
        r"Score source & Repeated evaluation scope & Per-split evaluation time & Scoring-cost note \\",
        r"\midrule",
        f"Artifact/logistic controls & 50 target splits & {runtime_seconds('artifact_token_formatting_logistic_l2')} & CPU evaluation over cached features \\\\",
        f"Combined logistic & 50 target splits & {runtime_seconds('combined_logistic_l2')} & CPU evaluation over cached text/format/likelihood/latent features \\\\",
        f"Qwen2.5-Math PRM & 20 target splits; 10 ProcessBench splits & -- & 7B model forward pass over steps; reported from cached scores because full rescoring is the expensive component \\\\",
        f"Likelihood / CoE features & target feature caches & -- & {tex(cheap_note)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
    ]
    latex_table(
        TABLES / "runtime_cost.tex",
        "tab:runtime_cost",
        "Runtime and scoring-cost summary. The conformal layer itself is cheap once scores are cached; expensive verifiers such as Qwen PRM move cost into step scoring.",
        "\n".join(lines),
    )


def build_qualitative_examples_table() -> None:
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llp{0.40\textwidth}lrrll}",
        r"\toprule",
        r"Case & Step & Step text & True label & Score & Prefix? & FE cand.? & Note \\",
        r"\midrule",
        "Success & 1 & Evaluate inner products and sums in the arithmetic expression. & correct & 0.149 & yes & no & retained \\\\",
        "Success & 2 & Substitute the first evaluated subexpressions. & correct & 0.207 & yes & no & retained \\\\",
        "Success & 7 & Reduce to $(-28)\\cdot 265\\cdot (-8)$. & correct & 0.600 & yes & no & boundary before error \\\\",
        "Success & 8 & Multiply incorrectly, producing 59280. & error & 0.941 & no & yes & first error \\\\",
        "Failure & 1 & Infer that Julie and the boys sold equal numbers. & error & 0.300 & yes & no & missed early error \\\\",
        "Failure & 2 & Split 18 glasses equally between two boys. & correct & 0.407 & yes & no & low score \\\\",
        "Failure & 3 & Compare Julie's 14 glasses to Micah's 9. & correct & 0.601 & yes & no & low score \\\\",
        "Failure & 4 & Return 5 more glasses. & correct & 0.575 & yes & no & full trace accepted by prefix \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
    ]
    latex_table(
        TABLES / "qualitative_examples.tex",
        "tab:qualitative_examples",
        "Qualitative examples from cached case studies at seed 2806 and $\\alpha=0.05$. The certificate does not prove the final answer correct; it identifies the calibrated portion of the trace to trust and the suffix or candidate steps for review.",
        "\n".join(lines),
    )


def main() -> None:
    ensure_dirs()
    audit = build_dataset_audit()
    build_dataset_provenance_table(audit)
    build_runtime_cost_table()
    build_qualitative_examples_table()
    status = {
        "generated_results": sorted(str(p.relative_to(ROOT)) for p in RESULTS.glob("*")),
        "generated_tables": sorted(str(p.relative_to(ROOT)) for p in TABLES.glob("*.tex")),
        "generated_figures": sorted(str(p.relative_to(ROOT)) for p in FIGURES.glob("*")),
    }
    (RESULTS / "strengthening_artifact_manifest.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
