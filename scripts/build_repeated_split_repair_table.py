"""Build the repeated-split downstream repair table from Table 3 outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "repeated_split_repair" / "full_repeated_split_table3_60_20_20"
TABLE_PATH = ROOT / "tables" / "downstream_repair_usefulness_priority_full_concise.tex"

MODE_ORDER = ["question_only", "full_trace", "whole_trace_abstention", "cpcc_prefix"]
MODE_LABELS = {
    "question_only": r"\shortstack{Question\\only}",
    "full_trace": r"\shortstack{Full\\trace}",
    "whole_trace_abstention": r"\shortstack{Whole-trace\\abst.}",
    "cpcc_prefix": r"\shortstack{CROP\\prefix}",
}
MODEL_ORDER = ["Gemma 4 8B", "Qwen2.5-7B", "DeepSeek-R1-8B", "Llama3.1-8B"]
DOMAIN_ORDER = [("arithmetic", "Arithmetic"), ("gsm8k", "GSM8K")]


def _fmt(value: float) -> str:
    return f"{value:.1f}"


def _delta_text(row: pd.Series) -> str:
    delta = float(row["mean_delta_pp"])
    lo = float(row["ci95_low_t"])
    hi = float(row["ci95_high_t"])
    text = f"{delta:+.2f} [{lo:.2f}, {hi:.2f}]"
    return rf"\textbf{{{text}}}" if delta > 0 and lo > 0 else text


def main() -> None:
    split_modes = pd.read_csv(RUN_DIR / "split_mode_summary.csv")
    inference = pd.read_csv(RUN_DIR / "crop_vs_best_non_crop_inference.csv")
    accuracy = (
        split_modes.groupby(["domain", "model", "mode"], dropna=False)["final_accuracy"]
        .mean()
        .mul(100.0)
        .to_dict()
    )
    deltas = inference[inference["baseline"].eq("best_non_crop")].set_index(["domain", "model"])

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        (
            r"\caption{\textbf{Repeated-split repair: CROP improves over the best non-CROP input in "
            r"five of eight displayed model--domain settings.} Accuracies are mean percentages over "
            r"20 stratified trace-level 60/20/20 repair splits, each with 300 Arithmetic and 263 GSM8K "
            r"test traces. CROP and whole-trace abstention use domain-specific calibration at "
            r"$\alpha=0.05$, with Trace features + PRM logistic for Arithmetic and Direct PRM for "
            r"GSM8K. $\Delta$ reports CROP-prefix accuracy minus the best deployable non-CROP input "
            r"within each split; brackets give split-seed 95\% confidence intervals. Rows report the "
            r"selected completed full repeated-split repair runs and omit Mistral-7B.}"
        ),
        r"\label{tab:downstream_repair_usefulness}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        "Domain & Repair model & "
        + " & ".join(MODE_LABELS[mode] for mode in MODE_ORDER)
        + r" & \shortstack{CROP $-$ best\\$\Delta$ [95\% CI]} \\",
        r"\midrule",
    ]

    for domain_i, (domain, domain_label) in enumerate(DOMAIN_ORDER):
        if domain_i > 0:
            lines.append(r"\addlinespace[2pt]")
        for model_i, model in enumerate(MODEL_ORDER):
            values = {mode: accuracy[(domain, model, mode)] for mode in MODE_ORDER}
            best = max(round(value, 1) for value in values.values())
            cells = []
            for mode in MODE_ORDER:
                text = _fmt(values[mode])
                if round(values[mode], 1) == best:
                    text = rf"\textbf{{{text}}}"
                cells.append(text)
            delta = _delta_text(deltas.loc[(domain, model)])
            domain_cell = rf"\multirow{{4}}{{*}}{{{domain_label}}}" if model_i == 0 else ""
            lines.append(f"{domain_cell} & {model} & " + " & ".join(cells) + f" & {delta} " + r"\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {TABLE_PATH}")


if __name__ == "__main__":
    main()
