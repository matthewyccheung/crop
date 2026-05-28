"""Build the fixed-score CROP utility table for reproduction reports.

The CROP rows come from the existing target-domain repeated-split summary.
The external process-supervision rows come from the fixed-score, three-way
train/calibration/test runner in exp15_prefix_aware.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_PATHS = [
    ROOT / "outputs" / "repro" / "tables" / "table_target_domain_mondrian.tex",
]

CROP_SUMMARY = ROOT / "outputs" / "budget_cpcc" / "table_target_domain_mondrian_summary.csv"
CROP_AUROC = ROOT / "outputs" / "budget_cpcc" / "table_target_domain_auroc_summary.csv"
EXTERNAL_ROOT = ROOT / "outputs" / "fixed_score_60_20_20"

ALPHA = 0.05

METHODS = [
    ("Random", "random"),
    ("Token/format", "token_format"),
    ("Trace features", "step_combined"),
    ("Direct PRM", "qwen_prm"),
    ("Trace+PRM", "step_qwen"),
]

CROP_ROWS = [
    ("Arithmetic", "dense step", "arithmetic"),
    ("GSM8K", "dense step", "gsm8k"),
]

EXTERNAL_ROWS = [
    ("ProcessBench", "first-error/process", "processbench"),
    ("Math-Shepherd", "step markup", "math_shepherd"),
    ("PRMBench", "fine-grained", "prmbench"),
    ("PRM800K", "human step", "prm800k"),
]

EXTERNAL_SCORE_MAP = {
    "random": "random",
    "token_format": "artifact_token_formatting_logistic_l2",
    "step_combined": "combined_logistic_l2",
    "qwen_prm": "qwen_prm_error",
    "step_qwen": "step_qwen_combined_logistic_l2",
}


def _format_pct(value: float, bold: bool) -> str:
    text = f"{100.0 * value:.1f}"
    return f"\\textbf{{{text}}}" if bold else text


def _format_auroc(value: float, bold: bool) -> str:
    text = f"{value:.3f}"
    return f"\\textbf{{{text}}}" if bold else text


def _best_flags(values: dict[str, dict[str, float]], metric: str, precision: int) -> dict[str, bool]:
    candidates = {score: row[metric] for score, row in values.items() if score != "random"}
    scale = 100.0 if metric in {"kept", "risk"} else 1.0
    best_rounded = max(round(scale * value, precision) for value in candidates.values())
    return {
        score: score != "random" and round(scale * row[metric], precision) == best_rounded
        for score, row in values.items()
    }


def _crop_values(domain: str, crop: pd.DataFrame, auroc: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for _, score in METHODS:
        metric_subset = crop[(crop["domain"] == domain) & (crop["score"] == score)]
        auroc_subset = auroc[(auroc["domain"] == domain) & (auroc["score"] == score)]
        if metric_subset.empty:
            raise ValueError(f"Missing CROP summary row for domain={domain!r}, score={score!r}")
        if auroc_subset.empty:
            raise ValueError(f"Missing CROP AUROC row for domain={domain!r}, score={score!r}")
        metric_row = metric_subset.iloc[0]
        auroc_row = auroc_subset.iloc[0]
        out[score] = {
            "auroc": float(auroc_row["step_auroc_mean"]),
            "kept": float(metric_row["prefix_kept_mean"]),
            "risk": float(metric_row["prefix_risk_mean"]),
        }
    return out


def _external_values(dataset_dir: str) -> dict[str, dict[str, float]]:
    path = EXTERNAL_ROOT / dataset_dir / "table_prefix_aware_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    summary = pd.read_csv(path)
    alpha_rows = summary[summary["alpha"].round(6) == round(ALPHA, 6)]
    out: dict[str, dict[str, float]] = {}
    for method_score, external_score in EXTERNAL_SCORE_MAP.items():
        subset = alpha_rows[alpha_rows["score"] == external_score]
        if subset.empty:
            raise ValueError(f"Missing external summary row for dataset={dataset_dir!r}, score={external_score!r}")
        row = subset.iloc[0]
        out[method_score] = {
            "auroc": float(row["auroc_mean"]),
            "kept": float(row["prefix_retained_fraction_mean"]),
            "risk": float(row["prefix_contamination_mean"]),
        }
    return out


def _render_group(group: str, labels: str, values: dict[str, dict[str, float]]) -> list[str]:
    bold_auroc = _best_flags(values, "auroc", 3)
    bold_kept = _best_flags(values, "kept", 1)
    rows = []
    for idx, (score_label, score) in enumerate(METHODS):
        dataset_cell = rf"\multirow{{{len(METHODS)}}}{{*}}{{{group}}}" if idx == 0 else ""
        label_cell = rf"\multirow{{{len(METHODS)}}}{{*}}{{{labels}}}" if idx == 0 else ""
        row = values[score]
        rows.append(
            " & ".join(
                [
                    dataset_cell,
                    label_cell,
                    score_label,
                    _format_auroc(row["auroc"], bold_auroc[score]),
                    _format_pct(row["kept"], bold_kept[score]),
                    f"{100.0 * row['risk']:.1f}",
                ]
            )
            + r" \\"
        )
    return rows


def main() -> None:
    crop = pd.read_csv(CROP_SUMMARY)
    crop_auroc = pd.read_csv(CROP_AUROC)
    rows: list[str] = []
    for group, labels, domain in CROP_ROWS:
        rows.extend(_render_group(group, labels, _crop_values(domain, crop, crop_auroc)))
        rows.append(r"\addlinespace[2pt]")
    for group, labels, dataset_dir in EXTERNAL_ROWS:
        rows.extend(_render_group(group, labels, _external_values(dataset_dir)))
        rows.append(r"\addlinespace[2pt]")
    rows = rows[:-1]

    body = "\n".join(rows)
    table = rf"""\begin{{table}}[H]
\centering
\caption{{\textbf{{Fixed-risk prefix utility and step-ranking quality.}} Entries report mean step AUROC, certified prefix kept (\%), and empirical prefix-contamination risk (\%) at $\alpha=0.05$. CROP target-domain rows are averaged over 20 stratified trace-level splits; additional process-supervision rows are averaged over 10 trace-level 60/20/20 splits. Bold marks the best non-random value within each dataset for AUROC and prefix kept separately.}}
\label{{tab:target_domain_mondrian}}
\scriptsize
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{@{{}}lllrrr@{{}}}}
\toprule
Dataset & Labels & Score source & Step AUROC & Prefix kept & Prefix risk \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    for path in TABLE_PATHS:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(table, encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
