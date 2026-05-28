"""Build the paper boundary-deviation table under the fixed-score split."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crop.experiments.exp19_trace_gating_next_steps import _prepare_seed_context  # noqa: E402
from crop.paper_repro import (  # noqa: E402
    TARGET_COMBINED,
    TARGET_QWEN,
    TARGET_TEXT,
    boundary_metrics,
    target_args,
    trace_totals,
)
from scripts.run_repeated_split_repair import calibrate_domains  # noqa: E402


LOCKED_SEED = 2856
DOMAINS = ("arithmetic", "gsm8k")
DOMAIN_LABELS = {"arithmetic": "Arithmetic", "gsm8k": "GSM8K"}
DEFAULT_SCORE = "step_qwen"
DOMAIN_SCORES = {"gsm8k": "qwen_prm"}


def _fmt(value: float) -> str:
    return f"{100.0 * value:.1f}"


def _domain_rows(ctx) -> pd.DataFrame:
    crop_lengths, whole_accepts, _ = calibrate_domains(ctx, DEFAULT_SCORE, DOMAIN_SCORES, set(DOMAINS))
    rows = []
    for domain in DOMAINS:
        traces = [trace for trace in ctx.split.test if trace.domain == domain]
        totals = trace_totals(traces)
        methods = [
            ("Question-only reference", np.zeros(len(traces), dtype=int)),
            ("Full-trace reference", totals),
            (
                "Whole-trace abstention",
                np.asarray([totals[i] if whole_accepts[trace.trace_id] else 0 for i, trace in enumerate(traces)]),
            ),
            ("CROP prefix", np.asarray([crop_lengths[trace.trace_id] for trace in traces], dtype=int)),
        ]
        for method, retained in methods:
            metrics = boundary_metrics(traces, retained)
            rows.append(
                {
                    "Dataset": DOMAIN_LABELS[domain],
                    "Object / method": method,
                    "Over-withholding ↓": metrics.over_withholding,
                    "Unsafe overshoot ↓": metrics.unsafe_overshoot,
                    "Boundary deviation ↓": metrics.over_withholding + metrics.unsafe_overshoot,
                }
            )
    return pd.DataFrame(rows)


def _write_tex(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        (
            r"\caption{\textbf{CROP selects a trust boundary closer to the oracle annotated-clean prefix.} "
            r"Boundary deviation is $|M-O|/T$, decomposed into over-withholding of annotated-clean steps "
            r"and unsafe overshoot beyond the first annotated error. Lower values indicate a retained prefix "
            r"closer to the oracle clean boundary.}"
        ),
        r"\label{tab:boundary_deviation}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Object / method & Over-withholding $\downarrow$ & Unsafe overshoot $\downarrow$ & Boundary deviation $\downarrow$ \\",
        r"\midrule",
    ]
    for domain_i, domain in enumerate([DOMAIN_LABELS[item] for item in DOMAINS]):
        if domain_i > 0:
            lines.append(r"\midrule")
        subset = df[df["Dataset"] == domain]
        for row_i, row in enumerate(subset.itertuples(index=False)):
            dataset = rf"\multirow{{4}}{{*}}{{{domain}}}" if row_i == 0 else ""
            method = getattr(row, "_1")
            over = getattr(row, "_2")
            unsafe = getattr(row, "_3")
            deviation = getattr(row, "_4")
            dev_text = _fmt(deviation)
            if method == "CROP prefix":
                dev_text = rf"\textbf{{{dev_text}}}"
            lines.append(f"{dataset} & {method} & {_fmt(over)} & {_fmt(unsafe)} & {dev_text} " + r"\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\begin{minipage}{0.96\linewidth}",
            (
                r"\footnotesize \textit{Note.} Results use trace-level 60/20/20 splits with "
                r"domain-specific calibration at $\alpha=0.05$: Trace features + PRM logistic for "
                r"Arithmetic and Direct PRM for GSM8K."
            ),
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = target_args()
    args.candidate_names = sorted({DEFAULT_SCORE, *DOMAIN_SCORES.values()})
    ctx = _prepare_seed_context(args, "Target", TARGET_TEXT, TARGET_COMBINED, TARGET_QWEN, LOCKED_SEED)
    df = _domain_rows(ctx)
    outputs = ROOT / "outputs"
    tables = ROOT / "tables"
    outputs.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    df.to_csv(outputs / "boundary_deviation_table.csv", index=False)
    _write_tex(df, outputs / "boundary_deviation_table.tex")
    _write_tex(df, tables / "boundary_deviation_table.tex")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
