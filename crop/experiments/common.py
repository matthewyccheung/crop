"""Shared experiment CLI and evaluation helpers."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from crop.conformal import (
    class_conditional_p_values,
    conformal_quantile,
    fit_aps_threshold,
    fit_class_conditional_lac,
    fit_lac_threshold,
    lower_conformal_quantile,
    predict_aps_sets,
    predict_class_conditional_lac,
    predict_lac_sets,
)
from crop.data import TraceRecord, load_many_npz, make_toy_traces, truncate_after_first_error
from crop.feature_groups import select_feature_set
from crop.metrics import (
    ambiguous_rate,
    average_set_size,
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
    avg_prefix_frac,
    avg_prefix_len,
)
from crop.models import fit_verifier, make_model, predict_probs, scores_by_trace_from_model
from crop.risk_control import (
    corrected_risk,
    first_error_losses_by_lambda,
    first_error_localization_losses,
    prefix_lengths,
    prefix_losses_by_lambda,
    select_lambda_crc,
)
from crop.sequence import candidate_first_error_set
from crop.splits import Split, flatten_steps, split_traces
from crop.utils import ensure_dir, environment_info, parse_optional_ints, set_seed, write_json


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--features", nargs="*", default=None)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--complexities", nargs="*", default=None)
    parser.add_argument("--model", default="gradient_boosting")
    parser.add_argument("--class_weight", default="balanced")
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.1])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--train_frac", type=float, default=0.6)
    parser.add_argument("--cal_frac", type=float, default=0.2)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--split_unit", choices=["trace", "step"], default="trace")
    parser.add_argument("--before_after", default=None)
    parser.add_argument("--truncate_after_first_error", action="store_true")
    parser.add_argument("--feature_set", default="all")
    parser.add_argument("--group_by", default="domain")
    parser.add_argument("--lambda_grid_size", type=int, default=1001)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_nan", action="store_true")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--toy_n_traces", type=int, default=240)
    parser.add_argument("--toy_min_steps", type=int, default=3)
    parser.add_argument("--toy_max_steps", type=int, default=8)
    parser.add_argument("--toy_n_features", type=int, default=55)
    parser.add_argument("--toy_error_rate", type=float, default=0.15)


def seeds_from_args(args) -> list[int]:
    if args.seed is not None:
        return [int(args.seed)]
    return list(args.seeds)


def domains_from_args(args) -> list[str]:
    if args.domains:
        return list(args.domains)
    if args.domain:
        return [args.domain]
    return ["toy"]


def load_traces_from_args(args, seed: int | None = None) -> list[TraceRecord]:
    use_toy = bool(args.toy or not args.features)
    if use_toy:
        domains = domains_from_args(args)
        traces: list[TraceRecord] = []
        for offset, domain in enumerate(domains):
            traces.extend(
                make_toy_traces(
                    n_traces=args.toy_n_traces,
                    min_steps=args.toy_min_steps,
                    max_steps=args.toy_max_steps,
                    n_features=args.toy_n_features,
                    error_rate=args.toy_error_rate,
                    seed=(0 if seed is None else seed) + offset * 1009,
                    domain=domain,
                )
            )
        return select_feature_set(traces, args.feature_set)

    domains = domains_from_args(args)
    if len(domains) == 1 and len(args.features) > 1:
        domains = domains * len(args.features)
    complexities = parse_optional_ints(args.complexities, len(args.features))
    traces = load_many_npz(args.features, domains, complexities, allow_nan=args.allow_nan)
    if args.before_after:
        for trace in traces:
            trace.steps = [s for s in trace.steps if s.before_after == args.before_after]
        traces = [t for t in traces if t.steps]
    if args.truncate_after_first_error:
        traces = truncate_after_first_error(traces)
    return select_feature_set(traces, args.feature_set)


def make_split(args, traces: list[TraceRecord], seed: int) -> Split:
    return split_traces(
        traces,
        train_frac=args.train_frac,
        cal_frac=args.cal_frac,
        test_frac=args.test_frac,
        seed=seed,
    )


class ScoreBundle:
    def __init__(self, name: str, cal_scores_by_trace, test_scores_by_trace, cal_step_scores, test_step_scores, model=None):
        self.name = name
        self.cal_scores_by_trace = cal_scores_by_trace
        self.test_scores_by_trace = test_scores_by_trace
        self.cal_step_scores = cal_step_scores
        self.test_step_scores = test_step_scores
        self.model = model


def _fit_minmax(train_values) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(v, dtype=float).ravel() for v in train_values])
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _apply_minmax(values, lo: float, hi: float):
    out = []
    for value in values:
        arr = np.asarray(value, dtype=float)
        arr = np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo)
        out.append(np.clip((arr - lo) / (hi - lo), 0.0, 1.0))
    return out


def _column_index(trace: TraceRecord, spec: str) -> int:
    if not trace.steps:
        raise ValueError("Cannot infer column index from empty trace")
    text = spec.removeprefix("column:")
    try:
        return int(text)
    except ValueError:
        names = trace.steps[0].metadata.get("_feature_names")
        if not names:
            raise ValueError(f"Feature file does not define feature_names; cannot resolve {spec!r}")
        if text not in names:
            raise ValueError(f"Unknown feature column {text!r}. Available: {names}")
        return int(names.index(text))


def _trace_feature_scores(traces: list[TraceRecord], kind: str) -> list[np.ndarray]:
    out = []
    for trace in traces:
        X = trace.X
        if kind.startswith("column:"):
            idx = _column_index(trace, kind)
            if idx < 0 or idx >= X.shape[1]:
                raise ValueError(f"Column {idx} out of bounds for feature matrix with {X.shape[1]} columns")
            scores = X[:, idx]
        elif kind == "logit_entropy_only":
            scores = X[:, 4] if X.shape[1] > 4 else np.zeros(len(trace.steps))
        elif kind == "logit_maxprob_only":
            scores = 1.0 - (X[:, 3] if X.shape[1] > 3 else np.zeros(len(trace.steps)))
        elif kind == "random":
            raise ValueError("random score source handled separately")
        else:
            raise ValueError(f"Unknown feature score source {kind}")
        out.append(np.asarray(scores, dtype=float))
    return out


def build_score_bundle(score_source: str, split: Split, seed: int, class_weight: str | None = "balanced") -> ScoreBundle:
    """Build error-oriented scores. Larger means more likely erroneous."""

    if score_source == "oracle":
        cal_by_trace = [trace.y_errors.astype(float) for trace in split.cal]
        test_by_trace = [trace.y_errors.astype(float) for trace in split.test]
    elif score_source == "anti_oracle":
        cal_by_trace = [1.0 - trace.y_errors.astype(float) for trace in split.cal]
        test_by_trace = [1.0 - trace.y_errors.astype(float) for trace in split.test]
    elif score_source == "random":
        rng = np.random.default_rng(seed)
        cal_by_trace = [rng.random(len(trace.steps)) for trace in split.cal]
        test_by_trace = [rng.random(len(trace.steps)) for trace in split.test]
    elif score_source.startswith("column:") or score_source in {"logit_entropy_only", "logit_maxprob_only"}:
        train_raw = _trace_feature_scores(split.train, score_source)
        lo, hi = _fit_minmax(train_raw)
        cal_by_trace = _apply_minmax(_trace_feature_scores(split.cal, score_source), lo, hi)
        test_by_trace = _apply_minmax(_trace_feature_scores(split.test, score_source), lo, hi)
    else:
        model = fit_verifier(make_model(score_source, seed=seed, class_weight=class_weight), split.train)
        cal_by_trace = scores_by_trace_from_model(model, split.cal)
        test_by_trace = scores_by_trace_from_model(model, split.test)
        return ScoreBundle(
            name=score_source,
            cal_scores_by_trace=cal_by_trace,
            test_scores_by_trace=test_by_trace,
            cal_step_scores=np.concatenate(cal_by_trace) if cal_by_trace else np.array([]),
            test_step_scores=np.concatenate(test_by_trace) if test_by_trace else np.array([]),
            model=model,
        )

    return ScoreBundle(
        name=score_source,
        cal_scores_by_trace=cal_by_trace,
        test_scores_by_trace=test_by_trace,
        cal_step_scores=np.concatenate(cal_by_trace) if cal_by_trace else np.array([]),
        test_step_scores=np.concatenate(test_by_trace) if test_by_trace else np.array([]),
    )


def fit_model_for_split(args, split: Split, seed: int):
    set_seed(seed)
    model = make_model(args.model, seed=seed, class_weight=args.class_weight, calibration=args.calibration)
    return fit_verifier(model, split.train)


def evaluate_base_model(args, traces: list[TraceRecord], seed: int) -> dict[str, Any]:
    if getattr(args, "split_unit", "trace") == "step":
        return evaluate_base_model_step_split(args, traces, seed)

    split = make_split(args, traces, seed)
    model = fit_model_for_split(args, split, seed)
    X_train, y_train, _, _, _ = flatten_steps(split.train)
    X_cal, y_cal, _, _, _ = flatten_steps(split.cal)
    X_test, y_test, _, _, _ = flatten_steps(split.test)
    scores = model.score_error(X_test) if len(y_test) else np.asarray([])
    return {
        "domain": ",".join(sorted({t.domain for t in traces})),
        "model": model.name,
        "seed": seed,
        "n_train_steps": int(len(y_train)),
        "n_cal_steps": int(len(y_cal)),
        "n_test_steps": int(len(y_test)),
        "error_rate_train": float(np.mean(y_train)) if len(y_train) else float("nan"),
        "error_rate_test": float(np.mean(y_test)) if len(y_test) else float("nan"),
        "auroc": safe_auroc(y_test, scores) if len(y_test) else float("nan"),
        "aupr": safe_aupr(y_test, scores) if len(y_test) else float("nan"),
        "fpr_at_recall_95": fpr_at_recall_95(y_test, scores) if len(y_test) else float("nan"),
        "environment": environment_info(),
    }


def _probs_from_error_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
    scores = np.clip(scores, 0.0, 1.0)
    return np.column_stack([1.0 - scores, scores])


def evaluate_score_source(args, traces: list[TraceRecord], seed: int, score_source: str) -> dict[str, Any]:
    split = make_split(args, traces, seed)
    bundle = build_score_bundle(score_source, split, seed, class_weight=args.class_weight)
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


def _stratify_or_none(y: np.ndarray):
    values, counts = np.unique(y, return_counts=True)
    return y if len(values) > 1 and np.min(counts) >= 2 else None


def evaluate_base_model_step_split(args, traces: list[TraceRecord], seed: int) -> dict[str, Any]:
    X, y, groups, _, _ = flatten_steps(traces)
    if len(y) == 0:
        raise ValueError("Cannot evaluate empty step set")

    holdout_frac = args.cal_frac + args.test_frac
    if holdout_frac <= 0:
        raise ValueError("step split requires cal_frac + test_frac > 0")

    indices = np.arange(len(y))
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=holdout_frac,
        random_state=seed,
        stratify=_stratify_or_none(y),
    )
    if args.cal_frac > 0 and args.test_frac > 0:
        holdout_y = y[holdout_idx]
        relative_test = args.test_frac / holdout_frac
        cal_idx, test_idx = train_test_split(
            holdout_idx,
            test_size=relative_test,
            random_state=seed + 1,
            stratify=_stratify_or_none(holdout_y),
        )
    elif args.test_frac > 0:
        cal_idx = np.asarray([], dtype=int)
        test_idx = holdout_idx
    else:
        cal_idx = holdout_idx
        test_idx = np.asarray([], dtype=int)

    model = make_model(args.model, seed=seed, class_weight=args.class_weight, calibration=args.calibration)
    if len(np.unique(y[train_idx])) < 2:
        model = make_model("dummy_prior", seed=seed)
    model.fit(X[train_idx], y[train_idx])
    scores = model.score_error(X[test_idx]) if len(test_idx) else np.asarray([])
    return {
        "domain": ",".join(sorted(set(groups.tolist()))),
        "model": model.name,
        "seed": seed,
        "split_unit": "step",
        "n_train_steps": int(len(train_idx)),
        "n_cal_steps": int(len(cal_idx)),
        "n_test_steps": int(len(test_idx)),
        "error_rate_train": float(np.mean(y[train_idx])) if len(train_idx) else float("nan"),
        "error_rate_test": float(np.mean(y[test_idx])) if len(test_idx) else float("nan"),
        "auroc": safe_auroc(y[test_idx], scores) if len(test_idx) else float("nan"),
        "aupr": safe_aupr(y[test_idx], scores) if len(test_idx) else float("nan"),
        "fpr_at_recall_95": fpr_at_recall_95(y[test_idx], scores) if len(test_idx) else float("nan"),
        "environment": environment_info(),
    }


def run_step_cp(args, traces: list[TraceRecord], seed: int, methods: list[str]) -> list[dict[str, Any]]:
    split = make_split(args, traces, seed)
    model = fit_model_for_split(args, split, seed)
    cal_probs = predict_probs(model, split.cal)
    test_probs = predict_probs(model, split.test)
    _, cal_y, _, _, _ = flatten_steps(split.cal)
    _, test_y, _, _, _ = flatten_steps(split.test)
    rows = []
    for alpha in args.alphas:
        for method in methods:
            if method == "lac":
                qhat = fit_lac_threshold(cal_probs, cal_y, alpha)
                pred_sets = predict_lac_sets(test_probs, qhat)
            elif method == "aps":
                qhat = fit_aps_threshold(cal_probs, cal_y, alpha)
                pred_sets = predict_aps_sets(test_probs, qhat)
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
                    "model": model.name,
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


def run_step_cp_for_score_source(
    args,
    traces: list[TraceRecord],
    seed: int,
    score_source: str,
    methods: list[str],
) -> list[dict[str, Any]]:
    split = make_split(args, traces, seed)
    bundle = build_score_bundle(score_source, split, seed, class_weight=args.class_weight)
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
            elif method == "aps":
                qhat = fit_aps_threshold(cal_probs, cal_y, alpha)
                pred_sets = predict_aps_sets(test_probs, qhat)
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


def run_error_detection(args, traces: list[TraceRecord], seed: int) -> list[dict[str, Any]]:
    split = make_split(args, traces, seed)
    model = fit_model_for_split(args, split, seed)
    cal_probs = predict_probs(model, split.cal)
    test_probs = predict_probs(model, split.test)
    _, cal_y, _, _, _ = flatten_steps(split.cal)
    _, test_y, _, _, _ = flatten_steps(split.test)
    cal_scores = cal_probs[:, 1]
    test_scores = test_probs[:, 1]
    pos_scores = cal_scores[cal_y == 1]
    if len(pos_scores) < 50:
        print("WARNING: positive calibration set is very small; class-conditional guarantee will be conservative/noisy.")
    rows = []
    for alpha in args.alphas:
        threshold = lower_conformal_quantile(pos_scores, alpha)
        metrics = error_detection_metrics(test_y, test_scores, threshold)
        rows.append(
            {
                "domain": ",".join(sorted({t.domain for t in traces})),
                "model": model.name,
                "alpha": alpha,
                "seed": seed,
                "threshold": threshold,
                "n_pos_cal": int(len(pos_scores)),
                "auroc": safe_auroc(test_y, test_scores),
                "aupr": safe_aupr(test_y, test_scores),
                **metrics,
            }
        )
    return rows


def run_error_detection_for_score_source(args, traces: list[TraceRecord], seed: int, score_source: str) -> list[dict[str, Any]]:
    split = make_split(args, traces, seed)
    bundle = build_score_bundle(score_source, split, seed, class_weight=args.class_weight)
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


def run_prefix_crc_for_score_source(
    args,
    traces: list[TraceRecord],
    seed: int,
    score_source: str,
) -> list[dict[str, Any]]:
    split = make_split(args, traces, seed)
    bundle = build_score_bundle(score_source, split, seed, class_weight=args.class_weight)
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


def run_first_error_for_score_source(args, traces: list[TraceRecord], seed: int, score_source: str) -> list[dict[str, Any]]:
    split = make_split(args, traces, seed)
    bundle = build_score_bundle(score_source, split, seed, class_weight=args.class_weight)
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


def save_rows(rows: list[dict[str, Any]], output_dir: str | Path, csv_name: str) -> pd.DataFrame:
    outdir = ensure_dir(output_dir)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / csv_name, index=False)
    write_json(outdir / "environment.json", environment_info())
    return df


def copy_for_feature_set(args, feature_set: str, seed: int) -> list[TraceRecord]:
    local_args = copy.copy(args)
    local_args.feature_set = feature_set
    return load_traces_from_args(local_args, seed=seed)


def grouped_values_for_trace(trace: TraceRecord, group_by: str) -> str:
    if group_by == "domain":
        return trace.domain
    if group_by in {"complexity", "nt"}:
        return str(trace.complexity)
    if group_by == "has_error":
        return str(int(trace.has_error))
    return "all"
