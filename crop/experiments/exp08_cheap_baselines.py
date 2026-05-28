"""Experiment 08: conformal CoT verification with cheap verifier scores."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from crop.cheap_baselines import COE_SCORE_COLUMNS, summarize_npz
from crop.conformal import (
    fit_class_conditional_lac,
    fit_lac_threshold,
    lower_conformal_quantile,
    predict_class_conditional_lac,
    predict_lac_sets,
)
from crop.experiments.common import (
    _probs_from_error_scores,
    build_score_bundle,
    evaluate_score_source,
    load_traces_from_args,
    make_split,
    run_error_detection_for_score_source,
    run_first_error_for_score_source,
    run_prefix_crc_for_score_source,
    run_step_cp_for_score_source,
    save_rows,
)
from crop.metrics import (
    average_set_size,
    ambiguous_rate,
    avg_prefix_frac,
    avg_prefix_len,
    class_conditional_coverage,
    clean_full_trace_accept_rate,
    contaminated_full_trace_accept_rate,
    empty_set_rate,
    error_detection_metrics,
    first_error_coverage,
    fpr_at_recall_95,
    full_trace_accept_rate,
    incorrect_singleton_rate,
    mean_nearest_distance,
    median_candidate_set_size,
    median_prefix_frac,
    no_error_coverage,
    prediction_set_coverage,
    prefix_contamination_rate,
    safe_aupr,
    safe_auroc,
    singleton_rate,
    top1_first_error_accuracy,
)
from crop.risk_control import (
    first_error_localization_losses,
    first_error_losses_by_lambda,
    prefix_lengths,
    prefix_losses_by_lambda,
    select_lambda_crc,
    whole_trace_false_accept_losses,
)
from crop.sequence import candidate_first_error_set
from crop.splits import flatten_steps
from crop.utils import ensure_dir, write_json


def _base_args(features: str, alphas: list[float], lambda_grid_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        features=[features],
        domains=["mixed"],
        domain=None,
        complexities=None,
        model="logistic_l2",
        class_weight="balanced",
        calibration=None,
        alphas=alphas,
        seeds=[2806],
        seed=None,
        output_dir=None,
        output=None,
        train_frac=0.6,
        cal_frac=0.2,
        test_frac=0.2,
        split_unit="trace",
        before_after=None,
        truncate_after_first_error=False,
        feature_set="all",
        group_by="domain",
        lambda_grid_size=lambda_grid_size,
        overwrite=True,
        allow_nan=False,
        toy=False,
        toy_n_traces=240,
        toy_min_steps=3,
        toy_max_steps=8,
        toy_n_features=55,
        toy_error_rate=0.15,
    )


def _source_label(source: str) -> str:
    return source.removeprefix("column:")


def _whole_trace_rows(args, traces, seed: int, score_source: str) -> list[dict]:
    split = make_split(args, traces, seed)
    bundle = build_score_bundle(score_source, split, seed, class_weight=args.class_weight)
    lambdas = np.linspace(0.0, 1.0, args.lambda_grid_size)
    losses = np.vstack(
        [whole_trace_false_accept_losses(split.cal, bundle.cal_scores_by_trace, lam) for lam in lambdas]
    )
    rows = []
    _, test_y, _, _, _ = flatten_steps(split.test)
    for alpha in args.alphas:
        lambda_hat, cal_risk = select_lambda_crc(losses, lambdas, alpha=alpha, direction="increasing")
        test_losses = whole_trace_false_accept_losses(split.test, bundle.test_scores_by_trace, lambda_hat)
        accepted = np.asarray([np.max(scores) <= lambda_hat if len(scores) else True for scores in bundle.test_scores_by_trace])
        totals = np.asarray([len(t.steps) for t in split.test])
        rows.append(
            {
                "domain": ",".join(sorted({t.domain for t in traces})),
                "model": score_source,
                "alpha": alpha,
                "seed": seed,
                "lambda_hat": lambda_hat,
                "cal_corrected_risk": cal_risk,
                "false_accept_rate": float(np.mean(test_losses)) if len(test_losses) else float("nan"),
                "accept_rate": float(np.mean(accepted)) if len(accepted) else float("nan"),
                "clean_accept_rate": clean_full_trace_accept_rate(split.test, accepted.astype(int) * totals),
                "contaminated_accept_rate": contaminated_full_trace_accept_rate(split.test, accepted.astype(int) * totals),
                "test_error_rate": float(np.mean(test_y)) if len(test_y) else float("nan"),
                "n_cal_traces": int(len(split.cal)),
                "n_test_traces": int(len(split.test)),
            }
        )
    return rows


def _ranking_by_domain_rows(args, traces, seed: int, score_source: str) -> list[dict]:
    split = make_split(args, traces, seed)
    bundle = build_score_bundle(score_source, split, seed, class_weight=args.class_weight)
    _, test_y, groups, _, _ = flatten_steps(split.test)
    rows = []
    for domain in sorted(set(groups.tolist())):
        mask = groups == domain
        scores = bundle.test_step_scores[mask]
        y_domain = test_y[mask]
        rows.append(
            {
                "domain": domain,
                "model": score_source,
                "seed": seed,
                "n_test_steps": int(mask.sum()),
                "error_rate_test": float(np.mean(y_domain)) if len(y_domain) else float("nan"),
                "auroc": safe_auroc(y_domain, scores) if len(y_domain) else float("nan"),
                "aupr": safe_aupr(y_domain, scores) if len(y_domain) else float("nan"),
                "fpr_at_recall_95": fpr_at_recall_95(y_domain, scores) if len(y_domain) else float("nan"),
            }
        )
    return rows


def _evaluate_from_bundle(traces, split, bundle, seed: int, score_source: str) -> dict:
    _, train_y, _, _, _ = flatten_steps(split.train)
    _, cal_y, _, _, _ = flatten_steps(split.cal)
    _, test_y, _, _, _ = flatten_steps(split.test)
    scores = bundle.test_step_scores
    return {
        "domain": ",".join(sorted({t.domain for t in traces})),
        "model": score_source,
        "seed": seed,
        "n_train_steps": int(len(train_y)),
        "n_cal_steps": int(len(cal_y)),
        "n_test_steps": int(len(test_y)),
        "error_rate_train": float(np.mean(train_y)) if len(train_y) else float("nan"),
        "error_rate_test": float(np.mean(test_y)) if len(test_y) else float("nan"),
        "auroc": safe_auroc(test_y, scores) if len(test_y) else float("nan"),
        "aupr": safe_aupr(test_y, scores) if len(test_y) else float("nan"),
        "fpr_at_recall_95": fpr_at_recall_95(test_y, scores) if len(test_y) else float("nan"),
    }


def _ranking_by_domain_from_bundle(split, bundle, seed: int, score_source: str) -> list[dict]:
    _, test_y, groups, _, _ = flatten_steps(split.test)
    rows = []
    for domain in sorted(set(groups.tolist())):
        mask = groups == domain
        scores = bundle.test_step_scores[mask]
        y_domain = test_y[mask]
        rows.append(
            {
                "domain": domain,
                "model": score_source,
                "seed": seed,
                "n_test_steps": int(mask.sum()),
                "error_rate_test": float(np.mean(y_domain)) if len(y_domain) else float("nan"),
                "auroc": safe_auroc(y_domain, scores) if len(y_domain) else float("nan"),
                "aupr": safe_aupr(y_domain, scores) if len(y_domain) else float("nan"),
                "fpr_at_recall_95": fpr_at_recall_95(y_domain, scores) if len(y_domain) else float("nan"),
            }
        )
    return rows


def _whole_trace_rows_from_bundle(args, traces, split, bundle, seed: int, score_source: str) -> list[dict]:
    lambdas = np.linspace(0.0, 1.0, args.lambda_grid_size)
    losses = np.vstack(
        [whole_trace_false_accept_losses(split.cal, bundle.cal_scores_by_trace, lam) for lam in lambdas]
    )
    rows = []
    _, test_y, _, _, _ = flatten_steps(split.test)
    for alpha in args.alphas:
        lambda_hat, cal_risk = select_lambda_crc(losses, lambdas, alpha=alpha, direction="increasing")
        test_losses = whole_trace_false_accept_losses(split.test, bundle.test_scores_by_trace, lambda_hat)
        accepted = np.asarray([np.max(scores) <= lambda_hat if len(scores) else True for scores in bundle.test_scores_by_trace])
        totals = np.asarray([len(t.steps) for t in split.test])
        rows.append(
            {
                "domain": ",".join(sorted({t.domain for t in traces})),
                "model": score_source,
                "alpha": alpha,
                "seed": seed,
                "lambda_hat": lambda_hat,
                "cal_corrected_risk": cal_risk,
                "false_accept_rate": float(np.mean(test_losses)) if len(test_losses) else float("nan"),
                "accept_rate": float(np.mean(accepted)) if len(accepted) else float("nan"),
                "clean_accept_rate": clean_full_trace_accept_rate(split.test, accepted.astype(int) * totals),
                "contaminated_accept_rate": contaminated_full_trace_accept_rate(split.test, accepted.astype(int) * totals),
                "test_error_rate": float(np.mean(test_y)) if len(test_y) else float("nan"),
                "n_cal_traces": int(len(split.cal)),
                "n_test_traces": int(len(split.test)),
            }
        )
    return rows


def _step_cp_rows_from_bundle(args, traces, split, bundle, seed: int, score_source: str, methods: list[str]) -> list[dict]:
    cal_probs = _probs_from_error_scores(bundle.cal_step_scores)
    test_probs = _probs_from_error_scores(bundle.test_step_scores)
    _, cal_y, _, _, _ = flatten_steps(split.cal)
    _, test_y, _, _, _ = flatten_steps(split.test)
    rows = []
    for alpha in args.alphas:
        for method in methods:
            if method == "lac":
                qhat = fit_lac_threshold(cal_probs, cal_y, alpha)
                pred_sets = predict_lac_sets(test_probs, qhat)
            elif method == "class_conditional_lac":
                thresholds = fit_class_conditional_lac(cal_probs, cal_y, alpha)
                pred_sets = predict_class_conditional_lac(test_probs, thresholds)
                qhat = float("nan")
            else:
                raise ValueError(f"Unknown CP method {method}")
            by_class = class_conditional_coverage(pred_sets, test_y)
            rows.append(
                {
                    "domain": ",".join(sorted({t.domain for t in traces})),
                    "model": score_source,
                    "method": method,
                    "alpha": alpha,
                    "seed": seed,
                    "coverage": prediction_set_coverage(pred_sets, test_y),
                    "coverage_error": by_class[1],
                    "coverage_correct": by_class[0],
                    "avg_set_size": average_set_size(pred_sets),
                    "singleton_rate": singleton_rate(pred_sets),
                    "ambiguous_rate": ambiguous_rate(pred_sets),
                    "empty_set_rate": empty_set_rate(pred_sets),
                    "incorrect_singleton_rate": incorrect_singleton_rate(pred_sets, test_y),
                    "qhat": qhat,
                    "n_cal": int(len(cal_y)),
                    "n_test": int(len(test_y)),
                }
            )
    return rows


def _error_detection_rows_from_bundle(args, traces, split, bundle, seed: int, score_source: str) -> list[dict]:
    _, cal_y, _, _, _ = flatten_steps(split.cal)
    _, test_y, _, _, _ = flatten_steps(split.test)
    pos_scores = bundle.cal_step_scores[cal_y == 1]
    rows = []
    for alpha in args.alphas:
        threshold = lower_conformal_quantile(pos_scores, alpha)
        metrics = error_detection_metrics(test_y, bundle.test_step_scores, threshold)
        rows.append(
            {
                "domain": ",".join(sorted({t.domain for t in traces})),
                "model": score_source,
                "alpha": alpha,
                "seed": seed,
                "threshold": threshold,
                "n_pos_cal": int(len(pos_scores)),
                "auroc": safe_auroc(test_y, bundle.test_step_scores),
                "aupr": safe_aupr(test_y, bundle.test_step_scores),
                **metrics,
            }
        )
    return rows


def _prefix_crc_rows_from_bundle(args, traces, split, bundle, seed: int, score_source: str) -> list[dict]:
    lambdas = np.linspace(0.0, 1.0, args.lambda_grid_size)
    losses = prefix_losses_by_lambda(split.cal, bundle.cal_scores_by_trace, lambdas)
    rows = []
    for alpha in args.alphas:
        lambda_hat, cal_risk = select_lambda_crc(losses, lambdas, alpha=alpha, direction="increasing")
        test_losses = prefix_losses_by_lambda(split.test, bundle.test_scores_by_trace, np.asarray([lambda_hat]))[0]
        lengths = prefix_lengths(bundle.test_scores_by_trace, lambda_hat)
        totals = np.asarray([len(t.steps) for t in split.test], dtype=int)
        _, test_y, _, _, _ = flatten_steps(split.test)
        rows.append(
            {
                "domain": ",".join(sorted({t.domain for t in traces})),
                "model": score_source,
                "alpha": alpha,
                "seed": seed,
                "lambda_hat": lambda_hat,
                "cal_corrected_risk": cal_risk,
                "test_prefix_contamination_rate": prefix_contamination_rate(test_losses),
                "avg_prefix_len": avg_prefix_len(lengths),
                "avg_prefix_frac": avg_prefix_frac(lengths, totals),
                "median_prefix_frac": median_prefix_frac(lengths, totals),
                "full_trace_accept_rate": full_trace_accept_rate(lengths, totals),
                "clean_full_trace_accept_rate": clean_full_trace_accept_rate(split.test, lengths),
                "contaminated_full_trace_accept_rate": contaminated_full_trace_accept_rate(split.test, lengths),
                "n_cal_traces": int(len(split.cal)),
                "n_test_traces": int(len(split.test)),
                "test_error_rate": float(np.mean(test_y)) if len(test_y) else float("nan"),
            }
        )
    return rows


def _first_error_rows_from_bundle(args, traces, split, bundle, seed: int, score_source: str) -> list[dict]:
    lambdas = np.linspace(0.0, 1.0, args.lambda_grid_size)
    losses = first_error_losses_by_lambda(split.cal, bundle.cal_scores_by_trace, lambdas)
    rows = []
    for alpha in args.alphas:
        lambda_hat, cal_risk = select_lambda_crc(losses, lambdas, alpha=alpha, direction="increasing")
        candidate_sets = [candidate_first_error_set(scores, lambda_hat, include_no_error=True) for scores in bundle.test_scores_by_trace]
        test_losses = first_error_localization_losses(split.test, bundle.test_scores_by_trace, lambda_hat)
        rows.append(
            {
                "domain": ",".join(sorted({t.domain for t in traces})),
                "model": score_source,
                "alpha": alpha,
                "seed": seed,
                "lambda_hat": lambda_hat,
                "cal_corrected_risk": cal_risk,
                "coverage": first_error_coverage(candidate_sets, split.test),
                "loss": float(np.mean(test_losses)) if len(test_losses) else float("nan"),
                "avg_candidate_set_size": average_set_size(candidate_sets),
                "median_candidate_set_size": median_candidate_set_size(candidate_sets),
                "top1_accuracy": top1_first_error_accuracy(bundle.test_scores_by_trace, split.test),
                "mean_nearest_distance": mean_nearest_distance(candidate_sets, split.test),
                "no_error_coverage": no_error_coverage(candidate_sets, split.test),
                "n_error_traces_test": int(sum(t.has_error for t in split.test)),
                "n_clean_traces_test": int(sum(not t.has_error for t in split.test)),
            }
        )
    return rows


def _trace_group(trace, group_by: str) -> str:
    if group_by == "domain":
        return str(trace.domain)
    if not trace.steps:
        return "unknown"
    meta = trace.steps[0].metadata
    return str(meta.get(group_by, meta.get("source_dataset", trace.domain)))


def _step_groups(traces, group_by: str) -> np.ndarray:
    return np.asarray([_trace_group(trace, group_by) for trace in traces for _ in trace.steps], dtype=object)


def _with_dataset(rows: list[dict], label: str, score_source: str, calibration: str = "pooled") -> list[dict]:
    for row in rows:
        row["dataset"] = label
        row["score"] = _source_label(score_source)
        row["calibration"] = calibration
    return rows


def _mondrian_step_cp_rows(args, traces, seed: int, score_source: str, group_by: str, split=None, bundle=None) -> list[dict]:
    split = make_split(args, traces, seed) if split is None else split
    bundle = build_score_bundle(score_source, split, seed, class_weight=args.class_weight) if bundle is None else bundle
    cal_probs = _probs_from_error_scores(bundle.cal_step_scores)
    test_probs = _probs_from_error_scores(bundle.test_step_scores)
    _, cal_y, _, _, _ = flatten_steps(split.cal)
    _, test_y, _, _, _ = flatten_steps(split.test)
    cal_groups = _step_groups(split.cal, group_by)
    test_groups = _step_groups(split.test, group_by)
    rows = []
    for alpha in args.alphas:
        pooled_q = fit_lac_threshold(cal_probs, cal_y, alpha)
        thresholds = {}
        for group in sorted(set(test_groups.tolist()) | set(cal_groups.tolist())):
            mask = cal_groups == group
            thresholds[group] = fit_lac_threshold(cal_probs[mask], cal_y[mask], alpha) if np.any(mask) else pooled_q
        pred_sets = []
        for probs, group in zip(test_probs, test_groups):
            pred_sets.extend(predict_lac_sets(probs[None, :], thresholds.get(group, pooled_q)))
        by_class = class_conditional_coverage(pred_sets, test_y)
        rows.append(
            {
                "domain": ",".join(sorted({t.domain for t in traces})),
                "model": score_source,
                "method": "lac",
                "alpha": alpha,
                "seed": seed,
                "group_by": group_by,
                "coverage": prediction_set_coverage(pred_sets, test_y),
                "coverage_error": by_class[1],
                "coverage_correct": by_class[0],
                "avg_set_size": average_set_size(pred_sets),
                "singleton_rate": singleton_rate(pred_sets),
                "n_cal": int(len(cal_y)),
                "n_test": int(len(test_y)),
                "n_groups": int(len(thresholds)),
            }
        )
    return rows


def _select_group_lambdas(cal_traces, cal_scores_by_trace, group_by: str, lambdas: np.ndarray, alpha: float, loss_fn):
    pooled_losses = loss_fn(cal_traces, cal_scores_by_trace, lambdas)
    pooled_lambda, pooled_risk = select_lambda_crc(pooled_losses, lambdas, alpha=alpha, direction="increasing")
    groups = [_trace_group(trace, group_by) for trace in cal_traces]
    out: dict[str, tuple[float, float, bool]] = {}
    for group in sorted(set(groups)):
        idx = [i for i, value in enumerate(groups) if value == group]
        if not idx:
            continue
        losses = loss_fn([cal_traces[i] for i in idx], [cal_scores_by_trace[i] for i in idx], lambdas)
        lam, risk = select_lambda_crc(losses, lambdas, alpha=alpha, direction="increasing")
        out[group] = (lam, risk, False)
    return out, pooled_lambda, pooled_risk


def _mondrian_prefix_crc_rows(args, traces, seed: int, score_source: str, group_by: str, split=None, bundle=None) -> list[dict]:
    split = make_split(args, traces, seed) if split is None else split
    bundle = build_score_bundle(score_source, split, seed, class_weight=args.class_weight) if bundle is None else bundle
    lambdas = np.linspace(0.0, 1.0, args.lambda_grid_size)
    rows = []
    for alpha in args.alphas:
        group_lambdas, pooled_lambda, pooled_risk = _select_group_lambdas(
            split.cal, bundle.cal_scores_by_trace, group_by, lambdas, alpha, prefix_losses_by_lambda
        )
        lengths = []
        losses = []
        totals = []
        fallback = 0
        for trace, scores in zip(split.test, bundle.test_scores_by_trace):
            group = _trace_group(trace, group_by)
            lam, _, used_fallback = group_lambdas.get(group, (pooled_lambda, pooled_risk, True))
            fallback += int(used_fallback)
            lengths.append(prefix_lengths([scores], lam)[0])
            losses.append(prefix_losses_by_lambda([trace], [scores], np.asarray([lam]))[0, 0])
            totals.append(len(trace.steps))
        _, test_y, _, _, _ = flatten_steps(split.test)
        rows.append(
            {
                "domain": ",".join(sorted({t.domain for t in traces})),
                "model": score_source,
                "alpha": alpha,
                "seed": seed,
                "group_by": group_by,
                "lambda_hat": float("nan"),
                "cal_corrected_risk": float(np.nanmean([risk for _, risk, _ in group_lambdas.values()]))
                if group_lambdas
                else pooled_risk,
                "test_prefix_contamination_rate": prefix_contamination_rate(losses),
                "avg_prefix_len": avg_prefix_len(lengths),
                "avg_prefix_frac": avg_prefix_frac(lengths, totals),
                "median_prefix_frac": median_prefix_frac(lengths, totals),
                "full_trace_accept_rate": full_trace_accept_rate(lengths, totals),
                "clean_full_trace_accept_rate": clean_full_trace_accept_rate(split.test, lengths),
                "contaminated_full_trace_accept_rate": contaminated_full_trace_accept_rate(split.test, lengths),
                "n_cal_traces": int(len(split.cal)),
                "n_test_traces": int(len(split.test)),
                "test_error_rate": float(np.mean(test_y)) if len(test_y) else float("nan"),
                "n_groups": int(len(group_lambdas)),
                "fallback_test_traces": int(fallback),
            }
        )
    return rows


def _whole_trace_losses_by_lambda(traces, scores_by_trace, lambdas: np.ndarray) -> np.ndarray:
    return np.vstack([whole_trace_false_accept_losses(traces, scores_by_trace, lam) for lam in lambdas])


def _mondrian_trace_abstention_rows(args, traces, seed: int, score_source: str, group_by: str, split=None, bundle=None) -> list[dict]:
    split = make_split(args, traces, seed) if split is None else split
    bundle = build_score_bundle(score_source, split, seed, class_weight=args.class_weight) if bundle is None else bundle
    lambdas = np.linspace(0.0, 1.0, args.lambda_grid_size)
    rows = []
    for alpha in args.alphas:
        group_lambdas, pooled_lambda, pooled_risk = _select_group_lambdas(
            split.cal, bundle.cal_scores_by_trace, group_by, lambdas, alpha, _whole_trace_losses_by_lambda
        )
        losses = []
        accepted = []
        totals = []
        fallback = 0
        for trace, scores in zip(split.test, bundle.test_scores_by_trace):
            group = _trace_group(trace, group_by)
            lam, _, used_fallback = group_lambdas.get(group, (pooled_lambda, pooled_risk, True))
            fallback += int(used_fallback)
            losses.append(whole_trace_false_accept_losses([trace], [scores], lam)[0])
            accepted.append(np.max(scores) <= lam if len(scores) else True)
            totals.append(len(trace.steps))
        _, test_y, _, _, _ = flatten_steps(split.test)
        accepted_arr = np.asarray(accepted, dtype=bool)
        totals_arr = np.asarray(totals, dtype=int)
        rows.append(
            {
                "domain": ",".join(sorted({t.domain for t in traces})),
                "model": score_source,
                "alpha": alpha,
                "seed": seed,
                "group_by": group_by,
                "lambda_hat": float("nan"),
                "cal_corrected_risk": float(np.nanmean([risk for _, risk, _ in group_lambdas.values()]))
                if group_lambdas
                else pooled_risk,
                "false_accept_rate": float(np.mean(losses)) if losses else float("nan"),
                "accept_rate": float(np.mean(accepted_arr)) if len(accepted_arr) else float("nan"),
                "clean_accept_rate": clean_full_trace_accept_rate(split.test, accepted_arr.astype(int) * totals_arr),
                "contaminated_accept_rate": contaminated_full_trace_accept_rate(
                    split.test, accepted_arr.astype(int) * totals_arr
                ),
                "test_error_rate": float(np.mean(test_y)) if len(test_y) else float("nan"),
                "n_cal_traces": int(len(split.cal)),
                "n_test_traces": int(len(split.test)),
                "n_groups": int(len(group_lambdas)),
                "fallback_test_traces": int(fallback),
            }
        )
    return rows


def _run_feature_file(
    *,
    label: str,
    features: str,
    score_sources: list[str],
    seeds: list[int],
    alphas: list[float],
    lambda_grid_size: int,
    include_sequence: bool,
    include_trace_cp: bool,
    include_mondrian: bool,
    group_by: str,
) -> dict[str, list[dict]]:
    args = _base_args(features, alphas, lambda_grid_size)
    out: dict[str, list[dict]] = {
        "ranking": [],
        "ranking_by_domain": [],
        "step_cp": [],
        "trace_cp": [],
        "error_detection": [],
        "prefix_crc": [],
        "first_error": [],
        "trace_abstention": [],
        "mondrian_step_cp": [],
        "mondrian_prefix_crc": [],
        "mondrian_trace_abstention": [],
        "mondrian_trace_cp": [],
    }
    for seed in seeds:
        traces = load_traces_from_args(args, seed=seed)
        split = make_split(args, traces, seed)
        for source in score_sources:
            bundle = build_score_bundle(source, split, seed, class_weight=args.class_weight)

            ranking = _evaluate_from_bundle(traces, split, bundle, seed, source)
            ranking["dataset"] = label
            ranking["score"] = _source_label(source)
            ranking["calibration"] = "diagnostic"
            out["ranking"].append(ranking)
            for row in _ranking_by_domain_from_bundle(split, bundle, seed, source):
                row["dataset"] = label
                row["score"] = _source_label(source)
                row["calibration"] = "diagnostic"
                out["ranking_by_domain"].append(row)

            whole = _whole_trace_rows_from_bundle(args, traces, split, bundle, seed, source)
            out["trace_abstention"].extend(_with_dataset(whole, label, source, "pooled"))

            if include_trace_cp:
                rows = _step_cp_rows_from_bundle(args, traces, split, bundle, seed, source, ["lac", "class_conditional_lac"])
                out["trace_cp"].extend(_with_dataset(rows, label, source, "pooled"))

            if include_sequence:
                for key, rows in (
                    ("step_cp", _step_cp_rows_from_bundle(args, traces, split, bundle, seed, source, ["lac", "class_conditional_lac"])),
                    ("error_detection", _error_detection_rows_from_bundle(args, traces, split, bundle, seed, source)),
                    ("prefix_crc", _prefix_crc_rows_from_bundle(args, traces, split, bundle, seed, source)),
                    ("first_error", _first_error_rows_from_bundle(args, traces, split, bundle, seed, source)),
                ):
                    out[key].extend(_with_dataset(rows, label, source, "pooled"))

            if include_mondrian:
                calibration = f"mondrian_{group_by}"
                out["mondrian_trace_abstention"].extend(
                    _with_dataset(
                        _mondrian_trace_abstention_rows(args, traces, seed, source, group_by, split=split, bundle=bundle),
                        label,
                        source,
                        calibration,
                    )
                )
                if include_trace_cp:
                    out["mondrian_trace_cp"].extend(
                        _with_dataset(
                            _mondrian_step_cp_rows(args, traces, seed, source, group_by, split=split, bundle=bundle),
                            label,
                            source,
                            calibration,
                        )
                    )
                if include_sequence:
                    out["mondrian_step_cp"].extend(
                        _with_dataset(
                            _mondrian_step_cp_rows(args, traces, seed, source, group_by, split=split, bundle=bundle),
                            label,
                            source,
                            calibration,
                        )
                    )
                    out["mondrian_prefix_crc"].extend(
                        _with_dataset(
                            _mondrian_prefix_crc_rows(args, traces, seed, source, group_by, split=split, bundle=bundle),
                            label,
                            source,
                            calibration,
                        )
                    )
    return out


def _write_summary(
    outdir: Path,
    dataset_summaries: list[dict],
    tables: dict[str, pd.DataFrame],
    runtime_note: str,
) -> None:
    def as_markdown(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except ImportError:
            return df.to_string(index=False)

    lines = [
        "# Cheap Baseline Conformal CoT Verification Summary",
        "",
        "## Dataset Summary",
        "",
        as_markdown(pd.DataFrame(dataset_summaries)),
        "",
        "## GPU/Runtime Summary",
        "",
        runtime_note,
        "",
    ]
    for name, df in tables.items():
        if df.empty:
            continue
        lines.extend([f"## {name.replace('_', ' ').title()}", ""])
        summary = _summarize_with_ci(df)
        cols = [c for c in summary.columns if c not in {"environment"}]
        lines.extend([as_markdown(summary[cols]), ""])
    conclusion = (
        "The most useful verifier is the one that controls the conformal target while minimizing set size, "
        "flagged fraction, discarded prefix length, or whole-trace rejection. Raw confidence, embedding, "
        "kinetic, text, and combined rows are all generic trace error detector instantiations rather than "
        "method-specific guarantees."
    )
    ranking = tables.get("ranking", pd.DataFrame())
    if not ranking.empty and {"dataset", "score", "auroc"}.issubset(ranking.columns):
        deployable = ranking[~ranking["score"].isin(["oracle", "random", "dummy_prior"])].copy()
        if not deployable.empty:
            best = deployable.sort_values("auroc", ascending=False).iloc[0]
            conclusion += (
                f" In this run, the best deployable ranking score by AUROC is "
                f"{best['dataset']}:{best['score']} with AUROC {best['auroc']:.3f}."
            )
    prefix = tables.get("prefix_crc", pd.DataFrame())
    if not prefix.empty and {"dataset", "score", "alpha", "test_prefix_contamination_rate", "avg_prefix_frac"}.issubset(
        prefix.columns
    ):
        deployable = prefix[
            (~prefix["score"].isin(["oracle", "random", "dummy_prior"]))
            & (prefix["test_prefix_contamination_rate"] <= prefix["alpha"])
        ].copy()
        if not deployable.empty:
            strict_alpha = deployable["alpha"].min()
            deployable = deployable[deployable["alpha"] == strict_alpha]
            best = deployable.sort_values("avg_prefix_frac", ascending=False).iloc[0]
            conclusion += (
                f" Among deployable clean-prefix rows that meet the empirical risk target, "
                f"{best['dataset']}:{best['score']} retains the longest average prefix "
                f"({best['avg_prefix_frac']:.3f}) at alpha {best['alpha']:.2f}."
            )
    lines.extend(["## Conclusion", "", conclusion, ""])
    (outdir / "SUMMARY.md").write_text("\n".join(lines))


def _summarize_with_ci(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    group_candidates = ("dataset", "score", "calibration", "domain", "method", "alpha", "group_by")
    group_cols = [col for col in group_candidates if col in df.columns]
    numeric = [col for col in df.columns if col not in set(group_cols) and pd.api.types.is_numeric_dtype(df[col])]
    if not group_cols or not numeric:
        return df
    grouped = df.groupby(group_cols, dropna=False)
    mean = grouped[numeric].mean(numeric_only=True)
    std = grouped[numeric].std(numeric_only=True)
    count = grouped[numeric].count()
    pieces = []
    for col in numeric:
        part = pd.DataFrame(
            {
                f"{col}_mean": mean[col],
                f"{col}_std": std[col].fillna(0.0),
                f"{col}_n": count[col],
                f"{col}_ci95": (1.96 * std[col].fillna(0.0) / np.sqrt(count[col].clip(lower=1))),
            }
        )
        pieces.append(part)
    return pd.concat(pieces, axis=1).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_text_features", default=None)
    parser.add_argument("--trace_text_features", required=True)
    parser.add_argument("--step_coe_features", default=None)
    parser.add_argument("--trace_coe_features", default=None)
    parser.add_argument("--step_combined_features", default=None)
    parser.add_argument("--trace_combined_features", default=None)
    parser.add_argument("--output_dir", default="outputs/cheap_baselines")
    parser.add_argument("--seeds", nargs="*", type=int, default=[2806])
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.05, 0.1])
    parser.add_argument("--text_models", nargs="*", default=["logistic_l2"])
    parser.add_argument("--lambda_grid_size", type=int, default=1001)
    parser.add_argument("--include_mondrian", action="store_true")
    parser.add_argument("--include_trace_cp", action="store_true")
    parser.add_argument("--trace_only", action="store_true")
    parser.add_argument("--group_by", default="domain")
    parser.add_argument("--runtime_note", default="Runtime was not recorded by the experiment driver.")
    args = parser.parse_args()
    if not args.trace_only and not args.step_text_features:
        parser.error("--step_text_features is required unless --trace_only is set")

    outdir = ensure_dir(args.output_dir)
    all_rows = {
        "ranking": [],
        "ranking_by_domain": [],
        "step_cp": [],
        "trace_cp": [],
        "error_detection": [],
        "prefix_crc": [],
        "first_error": [],
        "trace_abstention": [],
        "mondrian_step_cp": [],
        "mondrian_prefix_crc": [],
        "mondrian_trace_abstention": [],
        "mondrian_trace_cp": [],
    }
    dataset_summaries = [summarize_npz(args.trace_text_features)]
    if args.step_text_features and not args.trace_only:
        dataset_summaries.insert(0, summarize_npz(args.step_text_features))

    model_sources = list(dict.fromkeys(args.text_models + ["dummy_prior", "random", "oracle"]))
    configs = [("text_trace", args.trace_text_features, model_sources, False, args.include_trace_cp)]
    if args.step_text_features and not args.trace_only:
        configs.insert(0, ("text_step", args.step_text_features, model_sources, True, False))
    if args.step_coe_features and not args.trace_only:
        dataset_summaries.append(summarize_npz(args.step_coe_features))
        configs.append(
            (
                "coe_step",
                args.step_coe_features,
                [f"column:{name}" for name in COE_SCORE_COLUMNS] + ["random", "oracle"],
                True,
                False,
            )
        )
    if args.trace_coe_features:
        dataset_summaries.append(summarize_npz(args.trace_coe_features))
        configs.append(
            (
                "coe_trace",
                args.trace_coe_features,
                [f"column:{name}" for name in COE_SCORE_COLUMNS] + ["random", "oracle"],
                False,
                args.include_trace_cp,
            )
        )
    if args.step_combined_features and not args.trace_only:
        dataset_summaries.append(summarize_npz(args.step_combined_features))
        configs.append(
            (
                "combined_step",
                args.step_combined_features,
                model_sources,
                True,
                False,
            )
        )
    if args.trace_combined_features:
        dataset_summaries.append(summarize_npz(args.trace_combined_features))
        configs.append(
            (
                "combined_trace",
                args.trace_combined_features,
                model_sources,
                False,
                args.include_trace_cp,
            )
        )

    for label, features, sources, include_sequence, include_trace_cp in configs:
        rows = _run_feature_file(
            label=label,
            features=features,
            score_sources=sources,
            seeds=args.seeds,
            alphas=args.alphas,
            lambda_grid_size=args.lambda_grid_size,
            include_sequence=include_sequence,
            include_trace_cp=include_trace_cp,
            include_mondrian=args.include_mondrian,
            group_by=args.group_by,
        )
        for key, value in rows.items():
            all_rows[key].extend(value)

    tables: dict[str, pd.DataFrame] = {}
    for name, rows in all_rows.items():
        df = save_rows(rows, outdir, f"table_{name}.csv") if rows else pd.DataFrame()
        tables[name] = df
        if not df.empty:
            _summarize_with_ci(df).to_csv(outdir / f"table_{name}_summary.csv", index=False)
    pd.DataFrame(dataset_summaries).to_csv(outdir / "dataset_summary.csv", index=False)
    write_json(outdir / "run_config.json", vars(args))
    _write_summary(outdir, dataset_summaries, tables, args.runtime_note)
    print(f"Wrote {outdir}/SUMMARY.md")


if __name__ == "__main__":
    main()
