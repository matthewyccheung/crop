"""Fast repeated process-level conformal trace verification experiments.

This runner is intentionally separate from ``exp08_cheap_baselines``.  It loads
the cached process feature files once, splits by trace id, and evaluates all
process-level certificate objects from cached or lightweight detector scores.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import beta

from crop.conformal import fit_lac_threshold, lower_conformal_quantile, predict_lac_sets
from crop.data import StepRecord, TraceRecord, load_many_npz
from crop.experiments.common import ScoreBundle, build_score_bundle
from crop.experiments.exp08_cheap_baselines import _summarize_with_ci
from crop.metrics import (
    ambiguous_rate,
    average_set_size,
    class_conditional_coverage,
    contaminated_full_trace_accept_rate,
    clean_full_trace_accept_rate,
    empty_set_rate,
    error_detection_metrics,
    first_error_diagnostics,
    fpr_at_recall_95,
    full_trace_accept_rate,
    incorrect_singleton_rate,
    prefix_contamination_rate,
    prefix_diagnostics,
    safe_aupr,
    safe_auroc,
    singleton_rate,
    prediction_set_coverage,
)
from crop.models import make_model, fit_verifier, scores_by_trace_from_model
from crop.risk_control import (
    first_error_localization_losses,
    first_error_losses_by_lambda,
    prefix_lengths,
    prefix_losses_by_lambda,
    select_lambda_crc,
    whole_trace_false_accept_losses,
)
from crop.sequence import candidate_first_error_set
from crop.splits import Split, flatten_steps, split_traces
from crop.utils import ensure_dir, write_json


COE_SCORE_COLUMNS = [
    "maxprob_error",
    "ppl_error",
    "entropy_error",
    "tempscl_error",
    "energy_error",
    "coe_r_error",
    "coe_c_error",
    "cotk_error",
]


def _trace_maps(traces: list[TraceRecord]) -> dict[str, TraceRecord]:
    return {trace.trace_id: trace for trace in traces}


def _split_like(reference: Split, traces: list[TraceRecord]) -> Split:
    by_id = _trace_maps(traces)
    return Split(
        train=[by_id[t.trace_id] for t in reference.train],
        cal=[by_id[t.trace_id] for t in reference.cal],
        test=[by_id[t.trace_id] for t in reference.test],
    )


def _copy_with_features(traces: list[TraceRecord], feature_fn: Callable[[TraceRecord, StepRecord], np.ndarray]) -> list[TraceRecord]:
    out: list[TraceRecord] = []
    for trace in traces:
        steps = [replace(step, x=np.asarray(feature_fn(trace, step), dtype=float)) for step in trace.steps]
        out.append(replace(trace, steps=steps))
    return out


def _count_chars(text: str) -> np.ndarray:
    text = text or ""
    return np.asarray(
        [
            len(text),
            len(text.split()),
            sum(ch.isdigit() for ch in text),
            sum(ch.isalpha() for ch in text),
            text.count("="),
            sum(text.count(ch) for ch in "+-*/^"),
            text.count("\n"),
            text.count(":"),
            text.count("(") + text.count(")"),
            text.count("<") + text.count(">"),
        ],
        dtype=float,
    )


def _artifact_views(traces: list[TraceRecord]) -> dict[str, list[TraceRecord]]:
    domains = sorted({trace.domain for trace in traces})
    domain_index = {domain: idx for idx, domain in enumerate(domains)}

    def step_index(trace: TraceRecord, step: StepRecord) -> np.ndarray:
        denom = max(len(trace.steps) - 1, 1)
        return np.asarray([step.step_number / denom, step.step_number, len(trace.steps)], dtype=float)

    def trace_length(trace: TraceRecord, step: StepRecord) -> np.ndarray:
        return np.asarray([len(trace.steps)], dtype=float)

    def dataset_id(trace: TraceRecord, step: StepRecord) -> np.ndarray:
        arr = np.zeros(len(domains), dtype=float)
        arr[domain_index[trace.domain]] = 1.0
        return arr

    def token_formatting(trace: TraceRecord, step: StepRecord) -> np.ndarray:
        original = step.original_expression or ""
        content = step.step_content or ""
        return np.concatenate([_count_chars(content), _count_chars(original), step_index(trace, step)])

    return {
        "artifact_step_index": _copy_with_features(traces, step_index),
        "artifact_trace_length": _copy_with_features(traces, trace_length),
        "artifact_dataset_id": _copy_with_features(traces, dataset_id),
        "artifact_token_formatting": _copy_with_features(traces, token_formatting),
    }


def _probs_from_error_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
    scores = np.clip(scores, 0.0, 1.0)
    return np.column_stack([1.0 - scores, scores])


def _fit_model_bundle(model_name: str, split: Split, seed: int, class_weight: str = "balanced") -> ScoreBundle:
    model = fit_verifier(make_model(model_name, seed=seed, class_weight=class_weight), split.train)
    cal_by_trace = scores_by_trace_from_model(model, split.cal)
    test_by_trace = scores_by_trace_from_model(model, split.test)
    return ScoreBundle(
        name=model_name,
        cal_scores_by_trace=cal_by_trace,
        test_scores_by_trace=test_by_trace,
        cal_step_scores=np.concatenate(cal_by_trace) if cal_by_trace else np.array([]),
        test_step_scores=np.concatenate(test_by_trace) if test_by_trace else np.array([]),
        model=model,
    )


def _trace_group(trace: TraceRecord, group_by: str) -> str:
    if group_by == "domain":
        return trace.domain
    if group_by == "trace_length_bin":
        length = len(trace.steps)
        if length <= 4:
            return "len_01_04"
        if length <= 8:
            return "len_05_08"
        if length <= 12:
            return "len_09_12"
        return "len_13_plus"
    return str(trace.steps[0].metadata.get(group_by, trace.domain)) if trace.steps else "unknown"


def _select_group_lambdas(cal_traces, cal_scores_by_trace, group_by: str, lambdas: np.ndarray, alpha: float, loss_fn):
    pooled_losses = loss_fn(cal_traces, cal_scores_by_trace, lambdas)
    pooled_lambda, pooled_risk = select_lambda_crc(pooled_losses, lambdas, alpha=alpha, direction="increasing")
    groups = [_trace_group(trace, group_by) for trace in cal_traces]
    out: dict[str, tuple[float, float, bool]] = {}
    for group in sorted(set(groups)):
        idx = [i for i, value in enumerate(groups) if value == group]
        losses = loss_fn([cal_traces[i] for i in idx], [cal_scores_by_trace[i] for i in idx], lambdas)
        lam, risk = select_lambda_crc(losses, lambdas, alpha=alpha, direction="increasing")
        out[group] = (lam, risk, False)
    return out, pooled_lambda, pooled_risk


def _whole_trace_losses_by_lambda(traces, scores_by_trace, lambdas: np.ndarray) -> np.ndarray:
    return np.vstack([whole_trace_false_accept_losses(traces, scores_by_trace, lam) for lam in lambdas])


def _conditional_upper_bound(bad: int, total: int, delta: float) -> float:
    if total <= 0:
        return 1.0
    if bad >= total:
        return 1.0
    return float(beta.ppf(1.0 - delta, bad + 1, total - bad))


def _select_lambda_selective_risk(cal_traces, cal_scores_by_trace, lambdas: np.ndarray, alpha: float, delta: float = 0.05):
    per_lambda_delta = delta / max(len(lambdas), 1)
    selected = float(lambdas[0])
    selected_bound = 1.0
    selected_bad = 0
    selected_total = 0
    for lam in lambdas:
        accepted = np.asarray([np.max(scores) <= lam if len(scores) else True for scores in cal_scores_by_trace], dtype=bool)
        bad = int(sum(bool(acc) and trace.has_error for acc, trace in zip(accepted, cal_traces)))
        total = int(np.sum(accepted))
        upper = _conditional_upper_bound(bad, total, per_lambda_delta)
        if upper <= alpha:
            selected = float(lam)
            selected_bound = upper
            selected_bad = bad
            selected_total = total
    return selected, selected_bound, selected_bad, selected_total


def _accepted_error_metrics(traces, scores_by_trace, lambda_hat: float) -> dict[str, float]:
    accepted = np.asarray([np.max(scores) <= lambda_hat if len(scores) else True for scores in scores_by_trace], dtype=bool)
    has_error = np.asarray([trace.has_error for trace in traces], dtype=bool)
    accept_rate = float(np.mean(accepted)) if len(accepted) else float("nan")
    false_accept = float(np.mean(accepted & has_error)) if len(accepted) else float("nan")
    accepted_error = float(np.mean(has_error[accepted])) if np.any(accepted) else float("nan")
    return {
        "marginal_false_accept": false_accept,
        "accept_rate": accept_rate,
        "clean_accept_rate": float(np.mean(accepted[~has_error])) if np.any(~has_error) else float("nan"),
        "incorrect_accept_rate": float(np.mean(accepted[has_error])) if np.any(has_error) else float("nan"),
        "accepted_error_rate": accepted_error,
        "accepted_correct_rate": 1.0 - accepted_error if np.isfinite(accepted_error) else float("nan"),
        "abstain_rate": 1.0 - accept_rate if np.isfinite(accept_rate) else float("nan"),
        "unnecessary_abstain_rate_on_clean": float(np.mean(~accepted[~has_error])) if np.any(~has_error) else float("nan"),
    }


def _evaluate_bundle(
    *,
    score_name: str,
    score_family: str,
    split: Split,
    bundle: ScoreBundle,
    seed: int,
    alphas: list[float],
    lambdas: np.ndarray,
    runtime_seconds: float,
) -> list[dict]:
    _, cal_y, _, _, _ = flatten_steps(split.cal)
    _, test_y, _, _, _ = flatten_steps(split.test)
    cal_probs = _probs_from_error_scores(bundle.cal_step_scores)
    test_probs = _probs_from_error_scores(bundle.test_step_scores)
    ranking = {
        "score": score_name,
        "score_family": score_family,
        "seed": seed,
        "n_train_traces": len(split.train),
        "n_cal_traces": len(split.cal),
        "n_test_traces": len(split.test),
        "n_cal_steps": len(cal_y),
        "n_test_steps": len(test_y),
        "step_error_rate_test": float(np.mean(test_y)) if len(test_y) else float("nan"),
        "trace_error_rate_test": float(np.mean([trace.has_error for trace in split.test])) if split.test else float("nan"),
        "auroc": safe_auroc(test_y, bundle.test_step_scores) if len(test_y) else float("nan"),
        "aupr": safe_aupr(test_y, bundle.test_step_scores) if len(test_y) else float("nan"),
        "fpr_at_recall_95": fpr_at_recall_95(test_y, bundle.test_step_scores) if len(test_y) else float("nan"),
        "runtime_seconds": runtime_seconds,
    }
    rows = []
    prefix_losses = prefix_losses_by_lambda(split.cal, bundle.cal_scores_by_trace, lambdas)
    first_losses = first_error_losses_by_lambda(split.cal, bundle.cal_scores_by_trace, lambdas)
    trace_losses = _whole_trace_losses_by_lambda(split.cal, bundle.cal_scores_by_trace, lambdas)
    totals = np.asarray([len(trace.steps) for trace in split.test], dtype=int)
    for alpha in alphas:
        qhat = fit_lac_threshold(cal_probs, cal_y, alpha)
        pred_sets = predict_lac_sets(test_probs, qhat)
        by_class = class_conditional_coverage(pred_sets, test_y)
        pos_scores = bundle.cal_step_scores[np.asarray(cal_y) == 1]
        det_threshold = lower_conformal_quantile(pos_scores, alpha)
        det = error_detection_metrics(test_y, bundle.test_step_scores, det_threshold)

        prefix_lambda, prefix_cal_risk = select_lambda_crc(prefix_losses, lambdas, alpha=alpha, direction="increasing")
        prefix_test_losses = prefix_losses_by_lambda(split.test, bundle.test_scores_by_trace, np.asarray([prefix_lambda]))[0]
        lengths = prefix_lengths(bundle.test_scores_by_trace, prefix_lambda)

        first_lambda, first_cal_risk = select_lambda_crc(first_losses, lambdas, alpha=alpha, direction="increasing")
        candidate_sets = [
            candidate_first_error_set(scores, first_lambda, include_no_error=True)
            for scores in bundle.test_scores_by_trace
        ]
        first_test_losses = first_error_localization_losses(split.test, bundle.test_scores_by_trace, first_lambda)

        abstain_lambda, abstain_cal_risk = select_lambda_crc(trace_losses, lambdas, alpha=alpha, direction="increasing")
        abstain_test_losses = whole_trace_false_accept_losses(split.test, bundle.test_scores_by_trace, abstain_lambda)
        accepted_lengths = np.asarray(
            [len(trace.steps) if (np.max(scores) <= abstain_lambda if len(scores) else True) else 0
             for trace, scores in zip(split.test, bundle.test_scores_by_trace)],
            dtype=int,
        )
        accepted = _accepted_error_metrics(split.test, bundle.test_scores_by_trace, abstain_lambda)

        selective_lambda, selective_bound, selective_bad, selective_total = _select_lambda_selective_risk(
            split.cal, bundle.cal_scores_by_trace, lambdas, alpha=alpha
        )
        selective_metrics = _accepted_error_metrics(split.test, bundle.test_scores_by_trace, selective_lambda)

        row = dict(ranking)
        row.update(
            {
                "alpha": alpha,
                "step_coverage_all": prediction_set_coverage(pred_sets, test_y),
                "step_coverage_error": by_class[1],
                "step_coverage_correct": by_class[0],
                "step_avg_set_size": average_set_size(pred_sets),
                "step_singleton_rate": singleton_rate(pred_sets),
                "step_ambiguous_rate": ambiguous_rate(pred_sets),
                "step_empty_set_rate": empty_set_rate(pred_sets),
                "step_incorrect_singleton_rate": incorrect_singleton_rate(pred_sets, test_y),
                "high_recall_threshold": det_threshold,
                "high_recall_error_recall": det["error_recall"],
                "high_recall_error_precision": det["precision"],
                "high_recall_error_fpr": det["false_positive_rate"],
                "high_recall_error_fnr": det["missed_error_rate"],
                "high_recall_flagged_fraction": det["flagged_fraction"],
                "prefix_lambda": prefix_lambda,
                "prefix_cal_corrected_risk": prefix_cal_risk,
                "prefix_contamination": prefix_contamination_rate(prefix_test_losses),
                "prefix_retained_steps": float(np.mean(lengths)) if len(lengths) else float("nan"),
                "prefix_retained_fraction": float(np.mean(lengths / np.maximum(totals, 1))) if len(lengths) else float("nan"),
                "prefix_full_trace_rate": full_trace_accept_rate(lengths, totals),
                "prefix_clean_full_trace_rate": clean_full_trace_accept_rate(split.test, lengths),
                "prefix_error_full_trace_rate": contaminated_full_trace_accept_rate(split.test, lengths),
                "first_error_lambda": first_lambda,
                "first_error_cal_corrected_risk": first_cal_risk,
                "first_error_loss": float(np.mean(first_test_losses)) if len(first_test_losses) else float("nan"),
                "trace_abstention_lambda": abstain_lambda,
                "trace_abstention_cal_corrected_risk": abstain_cal_risk,
                "trace_abstention_test_loss": float(np.mean(abstain_test_losses)) if len(abstain_test_losses) else float("nan"),
                "trace_clean_accept_rate_check": clean_full_trace_accept_rate(split.test, accepted_lengths),
                "trace_contaminated_accept_rate_check": contaminated_full_trace_accept_rate(split.test, accepted_lengths),
                "selective_lambda": selective_lambda,
                "selective_cal_upper_bound": selective_bound,
                "selective_cal_bad_accepts": selective_bad,
                "selective_cal_accepts": selective_total,
                "selective_test_accepted_error_rate": selective_metrics["accepted_error_rate"],
                "selective_test_accept_rate": selective_metrics["accept_rate"],
                "selective_test_marginal_false_accept": selective_metrics["marginal_false_accept"],
            }
        )
        row.update(prefix_diagnostics(split.test, lengths))
        row.update(first_error_diagnostics(candidate_sets, bundle.test_scores_by_trace, split.test))
        row.update(accepted)
        rows.append(row)
    return rows


def _evaluate_mondrian(score_name: str, score_family: str, split: Split, bundle: ScoreBundle, seed: int, alphas, lambdas, group_by: str):
    rows = []
    for alpha in alphas:
        group_lambdas, pooled_lambda, pooled_risk = _select_group_lambdas(
            split.cal, bundle.cal_scores_by_trace, group_by, lambdas, alpha, prefix_losses_by_lambda
        )
        lengths = []
        losses = []
        groups = []
        for trace, scores in zip(split.test, bundle.test_scores_by_trace):
            group = _trace_group(trace, group_by)
            lam = group_lambdas.get(group, (pooled_lambda, pooled_risk, True))[0]
            lengths.append(prefix_lengths([scores], lam)[0])
            losses.append(prefix_losses_by_lambda([trace], [scores], np.asarray([lam]))[0, 0])
            groups.append(group)
        totals = np.asarray([len(trace.steps) for trace in split.test], dtype=int)
        row = {
            "score": score_name,
            "score_family": score_family,
            "seed": seed,
            "alpha": alpha,
            "calibration": f"mondrian_{group_by}",
            "group_by": group_by,
            "n_groups": len(group_lambdas),
            "prefix_contamination": prefix_contamination_rate(losses),
            "prefix_retained_fraction": float(np.mean(np.asarray(lengths) / np.maximum(totals, 1))) if len(lengths) else float("nan"),
            "prefix_full_trace_rate": full_trace_accept_rate(lengths, totals),
            "n_test_traces": len(split.test),
        }
        rows.append(row)
        for group in sorted(set(groups)):
            mask = np.asarray([g == group for g in groups], dtype=bool)
            group_lengths = np.asarray(lengths)[mask]
            group_totals = totals[mask]
            group_losses = np.asarray(losses)[mask]
            rows.append(
                {
                    **row,
                    "calibration": f"mondrian_{group_by}_group",
                    "group": group,
                    "prefix_contamination": prefix_contamination_rate(group_losses),
                    "prefix_retained_fraction": float(np.mean(group_lengths / np.maximum(group_totals, 1))),
                    "prefix_full_trace_rate": full_trace_accept_rate(group_lengths, group_totals),
                    "n_test_traces": int(np.sum(mask)),
                }
            )
    return rows


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
        except Exception:
            df.to_pickle(path)
            df.to_csv(path.with_suffix(".csv"), index=False)
    else:
        df.to_csv(path, index=False)


def _build_cache_tables(combined_traces: list[TraceRecord], coe_traces: list[TraceRecord]) -> tuple[pd.DataFrame, pd.DataFrame]:
    coe_by_id = _trace_maps(coe_traces)
    step_rows = []
    trace_rows = []
    for trace in combined_traces:
        coe_trace = coe_by_id.get(trace.trace_id)
        first_error = trace.first_error
        if coe_trace is not None and len(coe_trace.steps) == len(trace.steps):
            coe_matrix = coe_trace.X
        else:
            coe_matrix = np.full((len(trace.steps), len(COE_SCORE_COLUMNS)), np.nan)
        for step, coe_row in zip(trace.steps, coe_matrix):
            row = {
                "trace_id": trace.trace_id,
                "step_id": step.step_number,
                "dataset": trace.domain,
                "split_source": "crop_target",
                "prompt_template": step.before_after,
                "generator_model": "official_crop_annotations",
                "trace_length": len(trace.steps),
                "step_text": step.step_content or "",
                "label_step_error": step.y_error,
                "label_step_correct": int(not step.y_error),
                "first_error_step": -1 if first_error is None else first_error,
                "trace_has_error": int(trace.has_error),
                "final_answer_correct": int(not trace.has_error),
            }
            for name, value in zip(COE_SCORE_COLUMNS, coe_row):
                row[f"score_{name}"] = float(value)
            step_rows.append(row)
        trace_row = {
            "trace_id": trace.trace_id,
            "dataset": trace.domain,
            "prompt_template": trace.steps[0].before_after if trace.steps else "unknown",
            "generator_model": "official_crop_annotations",
            "trace_length": len(trace.steps),
            "has_error": int(trace.has_error),
            "first_error_step": -1 if first_error is None else first_error,
            "final_answer_correct": int(not trace.has_error),
        }
        for idx, name in enumerate(COE_SCORE_COLUMNS):
            values = coe_matrix[:, idx] if len(coe_matrix) else np.asarray([np.nan])
            if np.all(np.isnan(values)):
                trace_row[f"trace_score_max_{name}"] = float("nan")
                trace_row[f"trace_score_mean_{name}"] = float("nan")
            else:
                trace_row[f"trace_score_max_{name}"] = float(np.nanmax(values))
                trace_row[f"trace_score_mean_{name}"] = float(np.nanmean(values))
        trace_rows.append(trace_row)
    return pd.DataFrame(step_rows), pd.DataFrame(trace_rows)


def _write_summary(outdir: Path, process: pd.DataFrame, shift: pd.DataFrame, runtime: pd.DataFrame, config: dict) -> None:
    lines = [
        "# Process-Level Repeated Conformal Trace Verification",
        "",
        "This run uses cached process-labeled text, score-column, and combined features. No graph recomputation was launched.",
        "",
        "## Run Config",
        "",
        f"- Seeds: {config['seeds'][0]}..{config['seeds'][-1]} ({len(config['seeds'])} splits)",
        f"- Alphas: {config['alphas']}",
        f"- Lambda grid size: {config['lambda_grid_size']}",
        "",
    ]
    if not process.empty:
        summary = _summarize_with_ci(process)
        summary.to_csv(outdir / "table_process_main_summary.csv", index=False)
        key = summary[(summary["alpha"] == 0.05) & (~summary["score"].isin(["oracle"]))]
        cols = [
            "score",
            "score_family",
            "auroc_mean",
            "prefix_contamination_mean",
            "prefix_retained_fraction_mean",
            "fe_coverage_error_only_mean",
            "fe_candidate_size_excluding_empty_mean",
            "marginal_false_accept_mean",
            "accept_rate_mean",
            "accepted_error_rate_mean",
        ]
        cols = [c for c in cols if c in key.columns]
        lines.extend(["## Alpha 0.05 Process Summary", "", key[cols].sort_values("prefix_retained_fraction_mean", ascending=False).to_markdown(index=False), ""])
    if not shift.empty:
        _summarize_with_ci(shift).to_csv(outdir / "table_shift_mondrian_summary.csv", index=False)
    if not runtime.empty:
        runtime.groupby(["score", "score_family"], dropna=False).mean(numeric_only=True).reset_index().to_csv(
            outdir / "table_runtime_summary.csv", index=False
        )
    (outdir / "SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_text_features", default="data/cheap_baselines/crop_target_text_steps.npz")
    parser.add_argument("--step_coe_features", default="data/cheap_baselines/crop_target_coe_steps.npz")
    parser.add_argument("--step_combined_features", default="data/strengthened/crop_target_combined_steps.npz")
    parser.add_argument("--output_dir", default="outputs/strengthened/final/process_repeated")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(2806, 2826)))
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.01, 0.02, 0.05, 0.10, 0.20])
    parser.add_argument("--lambda_grid_size", type=int, default=201)
    parser.add_argument("--models", nargs="*", default=["logistic_l2", "hist_gradient_boosting", "random_forest"])
    parser.add_argument("--score_names", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.seeds = args.seeds[:1]
        args.alphas = [0.05]
        args.models = ["logistic_l2"]
        args.lambda_grid_size = min(args.lambda_grid_size, 51)

    outdir = ensure_dir(args.output_dir)
    combined = load_many_npz([args.step_combined_features], ["mixed"])
    text = load_many_npz([args.step_text_features], ["mixed"])
    coe = load_many_npz([args.step_coe_features], ["mixed"])
    artifact = _artifact_views(combined)
    step_cache, trace_cache = _build_cache_tables(combined, coe)
    _save_table(step_cache, outdir / "cached_step_scores.parquet")
    _save_table(trace_cache, outdir / "cached_trace_scores.parquet")

    views = {"combined": combined, "text": text, "coe": coe, **artifact}
    score_specs: list[tuple[str, str, str, str]] = []
    score_specs.extend([("random", "cheap_control", "combined", "random"), ("dummy_prior", "cheap_control", "combined", "dummy_prior"), ("oracle", "diagnostic", "combined", "oracle")])
    for name in COE_SCORE_COLUMNS:
        score_specs.append((name, "hidden_state_or_likelihood", "coe", f"column:{name}"))
    for view_name in ["text", "combined", "artifact_step_index", "artifact_trace_length", "artifact_dataset_id", "artifact_token_formatting"]:
        family = "artifact_control" if view_name.startswith("artifact") else "learned_detector"
        for model in args.models:
            score_specs.append((f"{view_name}_{model}", family, view_name, model))
    if args.score_names:
        wanted = set(args.score_names)
        score_specs = [spec for spec in score_specs if spec[0] in wanted]
        missing = sorted(wanted - {spec[0] for spec in score_specs})
        if missing:
            raise ValueError(f"Requested unknown score_names: {missing}")

    lambdas = np.linspace(0.0, 1.0, args.lambda_grid_size)
    process_rows = []
    shift_rows = []
    runtime_rows = []
    for seed in args.seeds:
        reference = split_traces(combined, seed=seed)
        split_by_view = {name: _split_like(reference, traces) for name, traces in views.items()}
        for score_name, family, view_name, source in score_specs:
            split = split_by_view[view_name]
            started = time.perf_counter()
            if source in {"random", "oracle"} or source.startswith("column:"):
                bundle = build_score_bundle(source, split, seed=seed)
            elif source == "dummy_prior":
                bundle = _fit_model_bundle(source, split, seed=seed)
            else:
                bundle = _fit_model_bundle(source, split, seed=seed)
            elapsed = time.perf_counter() - started
            process_rows.extend(
                _evaluate_bundle(
                    score_name=score_name,
                    score_family=family,
                    split=split,
                    bundle=bundle,
                    seed=seed,
                    alphas=args.alphas,
                    lambdas=lambdas,
                    runtime_seconds=elapsed,
                )
            )
            runtime_rows.append({"score": score_name, "score_family": family, "seed": seed, "runtime_seconds": elapsed})
            if score_name in {"combined_logistic_l2", "text_logistic_l2", "oracle", "random"}:
                for group_by in ("domain", "trace_length_bin"):
                    shift_rows.extend(_evaluate_mondrian(score_name, family, split, bundle, seed, args.alphas, lambdas, group_by))

    process = pd.DataFrame(process_rows)
    process.to_csv(outdir / "table_process_main.csv", index=False)
    process[process["score_family"] == "artifact_control"].to_csv(outdir / "table_artifact_ablation.csv", index=False)
    process[[c for c in process.columns if c.startswith("fe_") or c in {"score", "score_family", "seed", "alpha"}]].to_csv(
        outdir / "table_first_error_diagnostics.csv", index=False
    )
    process.to_csv(outdir / "table_risk_efficiency.csv", index=False)
    shift = pd.DataFrame(shift_rows)
    shift.to_csv(outdir / "table_shift_mondrian.csv", index=False)
    runtime = pd.DataFrame(runtime_rows)
    runtime.to_csv(outdir / "table_runtime.csv", index=False)
    write_json(outdir / "run_config.json", vars(args))
    _write_summary(outdir, process, shift, runtime, vars(args))
    print(f"Wrote {outdir}")


if __name__ == "__main__":
    main()
