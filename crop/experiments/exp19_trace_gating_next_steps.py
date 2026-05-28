"""Stabilized trace-gating follow-up experiments.

The experiments in this module are intentionally conservative: each policy is
fit or selected before the final CPCC calibration split, then evaluated only on
the held-out test split.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import shutil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from crop.data import TraceRecord, load_many_npz
from crop.experiments.exp16_adaptive_adapters import (
    ADAPTER_SPECS,
    AdapterBundle,
    AdapterSpec,
    AdaptiveSplit,
    _adaptive_split_like,
    _build_views,
    _concat,
    _fit_hazard_bundle,
    _fit_prefix_bundle,
    _fit_step_bundle,
    _read_qwen_scores,
    _slug,
    _tex,
    split_traces_four_way,
)
from crop.experiments.exp17_adaptive_revision import DATASET_ORDER
from crop.experiments.exp18_trace_conditioned_adaptive_cpcc import (
    ALPHA_MAIN,
    ALL_CANDIDATES,
    CHEAP_CANDIDATES,
    LABELS,
    QWEN_CANDIDATES,
    TIE_ORDER,
    _adapter_distribution,
    _calibrate_gated_index,
    _dataset_configs,
    _entropy,
    _eval_candidate_at_index,
    _evaluate_gated,
    _first_errors,
    _fmt_pct,
    _losses_from_lengths,
    _metrics_from_lengths,
    _select_index_for_candidate,
    _tie_break_candidate,
    _trace_lengths,
    _write_simple_tex_table,
    build_trace_gate_features,
    make_crossfit_gate_labels,
    train_trace_gate,
)
from crop.models import scores_by_trace_from_model
from crop.risk_control import prefix_lengths
from crop.utils import ensure_dir, write_json


TAU_GRID = [0.0, 0.0025, 0.005, 0.010, 0.020, 0.050]
COST_THRESHOLD_GRID = [0.0, 0.0025, 0.005, 0.010, 0.020, 0.050, 0.100]
GROUP_RULES = [
    "cheap_prefix_tercile",
    "trace_length_tercile",
    "token_qwen_disagreement",
    "cheap_prefix_x_disagreement",
]


@dataclass
class AdapterPlus:
    adapter: AdapterBundle
    train_scores_by_trace: list[np.ndarray]
    train_step_scores: np.ndarray


@dataclass
class SeedContext:
    dataset: str
    seed: int
    split: AdaptiveSplit
    raw_scores: dict[str, dict[str, list[np.ndarray]]]
    normalized_scores: dict[str, dict[str, dict[str, list[np.ndarray]]]]
    grids: dict[str, dict[str, np.ndarray]]


def _qwen_scores_for_traces(traces: list[TraceRecord], scores_by_trace_id: dict[str, dict[int, float]]) -> list[np.ndarray]:
    out = []
    for trace in traces:
        trace_scores = scores_by_trace_id.get(trace.trace_id, {})
        out.append(np.asarray([float(trace_scores.get(step.step_number, 0.5)) for step in trace.steps], dtype=float))
    return out


def _fit_adapter_plus(spec: AdapterSpec, split: AdaptiveSplit, seed: int, class_weight: str, scores_by_trace_id) -> AdapterPlus:
    if spec.source == "random":
        rng = np.random.default_rng(seed + 91_337)
        train_scores = [rng.random(len(trace.steps)) for trace in split.train]
        select_scores = [rng.random(len(trace.steps)) for trace in split.select]
        cal_scores = [rng.random(len(trace.steps)) for trace in split.cal]
        test_scores = [rng.random(len(trace.steps)) for trace in split.test]
        adapter = AdapterBundle(
            spec,
            select_scores,
            cal_scores,
            test_scores,
            _concat(select_scores),
            _concat(cal_scores),
            _concat(test_scores),
            0.0,
        )
        return AdapterPlus(adapter, train_scores, _concat(train_scores))
    if spec.source == "qwen":
        if scores_by_trace_id is None:
            raise ValueError("Qwen scores are required for qwen_prm")
        train_scores = _qwen_scores_for_traces(split.train, scores_by_trace_id)
        select_scores = _qwen_scores_for_traces(split.select, scores_by_trace_id)
        cal_scores = _qwen_scores_for_traces(split.cal, scores_by_trace_id)
        test_scores = _qwen_scores_for_traces(split.test, scores_by_trace_id)
        adapter = AdapterBundle(
            spec,
            select_scores,
            cal_scores,
            test_scores,
            _concat(select_scores),
            _concat(cal_scores),
            _concat(test_scores),
            0.0,
        )
        return AdapterPlus(adapter, train_scores, _concat(train_scores))
    if spec.source == "step":
        model, select_scores, cal_scores, test_scores = _fit_step_bundle(split, seed, class_weight)
    elif spec.source == "prefix":
        model, select_scores, cal_scores, test_scores = _fit_prefix_bundle(split, seed, class_weight)
    elif spec.source == "hazard":
        model, select_scores, cal_scores, test_scores = _fit_hazard_bundle(split, seed, class_weight)
    else:
        raise ValueError(f"Unknown adapter source={spec.source!r}")
    train_scores = scores_by_trace_from_model(model, split.train)
    adapter = AdapterBundle(
        spec,
        select_scores,
        cal_scores,
        test_scores,
        _concat(select_scores),
        _concat(cal_scores),
        _concat(test_scores),
        0.0,
    )
    return AdapterPlus(adapter, train_scores, _concat(train_scores))


def _threshold_grids(scores: dict[str, list[np.ndarray]], candidates: list[str], size: int) -> dict[str, np.ndarray]:
    quantiles = np.linspace(0.0, 1.0, size)
    grids: dict[str, np.ndarray] = {}
    for candidate in candidates:
        flat = np.concatenate(scores[candidate]) if scores[candidate] else np.asarray([])
        if len(flat) == 0:
            grid = np.linspace(0.0, 1.0, size)
        else:
            grid = np.quantile(np.asarray(flat, dtype=float), quantiles)
        grids[candidate] = np.maximum.accumulate(np.asarray(grid, dtype=float))
    return grids


def _copy_scores(score_set: dict[str, dict[str, list[np.ndarray]]]) -> dict[str, dict[str, list[np.ndarray]]]:
    return {
        split_name: {candidate: [np.asarray(arr, dtype=float).copy() for arr in traces] for candidate, traces in by_candidate.items()}
        for split_name, by_candidate in score_set.items()
    }


def _cdf_normalized(score_set: dict[str, dict[str, list[np.ndarray]]], candidates: list[str]) -> dict[str, dict[str, list[np.ndarray]]]:
    out = _copy_scores(score_set)
    for candidate in candidates:
        train = np.concatenate(score_set["train"][candidate]) if score_set["train"][candidate] else np.asarray([0.0])
        train = np.sort(np.asarray(train, dtype=float))
        denom = max(len(train), 1)
        for split_name in out:
            out[split_name][candidate] = [
                np.searchsorted(train, np.asarray(scores, dtype=float), side="right").astype(float) / denom
                for scores in score_set[split_name][candidate]
            ]
    return out


def _rank_normalized(score_set: dict[str, dict[str, list[np.ndarray]]], candidates: list[str]) -> dict[str, dict[str, list[np.ndarray]]]:
    out = _copy_scores(score_set)
    for split_name in out:
        for candidate in candidates:
            ranked = []
            for scores in score_set[split_name][candidate]:
                scores = np.asarray(scores, dtype=float)
                if len(scores) == 0:
                    ranked.append(scores.copy())
                    continue
                order = np.argsort(scores, kind="mergesort")
                ranks = np.empty(len(scores), dtype=float)
                ranks[order] = (np.arange(len(scores), dtype=float) + 1.0) / float(len(scores))
                ranked.append(ranks)
            out[split_name][candidate] = ranked
    return out


def _prepare_seed_context(args, dataset: str, text_features: str, combined_features: str, qwen_csv: str, seed: int) -> SeedContext:
    combined = load_many_npz([combined_features], ["mixed"], allow_nan=True)
    text = load_many_npz([text_features], ["mixed"], allow_nan=True)
    scores_by_trace_id = _read_qwen_scores(qwen_csv, args.qwen_score_col)
    views = _build_views(combined, text, scores_by_trace_id)
    reference = split_traces_four_way(
        combined,
        train_frac=args.score_train_frac,
        select_frac=args.gate_select_frac,
        cal_frac=args.cpcc_calibration_frac,
        test_frac=args.test_frac,
        seed=seed,
    )
    split_by_view = {name: _adaptive_split_like(reference, traces) for name, traces in views.items()}
    candidates = list(getattr(args, "candidate_names", ALL_CANDIDATES))
    spec_by_score = {spec.score: spec for spec in ADAPTER_SPECS}
    plus: dict[str, AdapterPlus] = {}
    for candidate in candidates:
        spec = spec_by_score[candidate]
        split = reference if spec.source == "qwen" else split_by_view[spec.view]
        plus[candidate] = _fit_adapter_plus(spec, split, seed, args.class_weight, scores_by_trace_id)
    raw_scores = {
        "train": {candidate: plus[candidate].train_scores_by_trace for candidate in candidates},
        "select": {candidate: plus[candidate].adapter.select_scores_by_trace for candidate in candidates},
        "cal": {candidate: plus[candidate].adapter.cal_scores_by_trace for candidate in candidates},
        "test": {candidate: plus[candidate].adapter.test_scores_by_trace for candidate in candidates},
    }
    normalized_scores = {
        "raw": raw_scores,
        "cdf": _cdf_normalized(raw_scores, candidates),
        "rank": _rank_normalized(raw_scores, candidates),
    }
    grids = {name: _threshold_grids(score_set["train"], candidates, args.threshold_grid_size) for name, score_set in normalized_scores.items()}
    return SeedContext(dataset, seed, reference, raw_scores, normalized_scores, grids)


def _select_adapter_on_selection(
    traces: list[TraceRecord],
    scores: dict[str, list[np.ndarray]],
    grids: dict[str, np.ndarray],
    candidates: list[str],
    alpha: float,
) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    for candidate in candidates:
        idx, risk = _select_index_for_candidate(traces, scores[candidate], grids[candidate], alpha)
        metrics = _eval_candidate_at_index(traces, scores[candidate], grids[candidate], idx, float(candidate in QWEN_CANDIDATES))
        rows.append(
            {
                "candidate": candidate,
                "selection_index": idx,
                "selection_corrected_risk": risk,
                "selection_prefix_kept": metrics["prefix_retained_fraction"],
                "selection_empirical_risk": metrics["prefix_contamination"],
                "selection_feasible": bool(risk <= alpha),
            }
        )
    feasible = [row for row in rows if row["selection_feasible"]]
    pool = feasible if feasible else rows
    selected = sorted(pool, key=lambda row: (-float(row["selection_prefix_kept"]), float(row["selection_corrected_risk"]), TIE_ORDER.get(str(row["candidate"]), 999)))[0]["candidate"]
    return str(selected), rows


def _fixed_eval(
    split: AdaptiveSplit,
    score_set: dict[str, dict[str, list[np.ndarray]]],
    grids: dict[str, np.ndarray],
    candidate: str,
    alpha: float,
) -> tuple[dict[str, float], int, float]:
    idx, risk = _select_index_for_candidate(split.cal, score_set["cal"][candidate], grids[candidate], alpha)
    metrics = _eval_candidate_at_index(split.test, score_set["test"][candidate], grids[candidate], idx, float(candidate in QWEN_CANDIDATES))
    return metrics, idx, risk


def _baseline_rows(ctx: SeedContext, score_set_name: str, alpha: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    score_set = ctx.normalized_scores[score_set_name]
    grids = ctx.grids[score_set_name]
    rows = []
    candidate_indices: dict[str, int] = {}
    fixed_metrics = {}
    for candidate in ALL_CANDIDATES:
        metrics, idx, risk = _fixed_eval(ctx.split, score_set, grids, candidate, alpha)
        candidate_indices[candidate] = idx
        fixed_metrics[candidate] = metrics
        rows.append(_method_row(ctx, alpha, candidate, "fixed", metrics, selected_adapter=candidate, cal_index=idx, cal_risk=risk))
    dataset_adapter, selection_rows = _select_adapter_on_selection(ctx.split.select, score_set["select"], grids, ALL_CANDIDATES, alpha)
    metrics, idx, risk = _fixed_eval(ctx.split, score_set, grids, dataset_adapter, alpha)
    dataset_row = _method_row(ctx, alpha, "dataset_adaptive", "dataset_adaptive", metrics, selected_adapter=dataset_adapter, cal_index=idx, cal_risk=risk)
    rows.append(dataset_row)
    cheap_adapter, _ = _select_adapter_on_selection(ctx.split.select, score_set["select"], grids, CHEAP_CANDIDATES, alpha)
    metrics, idx, risk = _fixed_eval(ctx.split, score_set, grids, cheap_adapter, alpha)
    rows.append(_method_row(ctx, alpha, "best_cheap_adaptive", "diagnostic", metrics, selected_adapter=cheap_adapter, cal_index=idx, cal_risk=risk))
    qwen_adapter, _ = _select_adapter_on_selection(ctx.split.select, score_set["select"], grids, list(QWEN_CANDIDATES), alpha)
    metrics, idx, risk = _fixed_eval(ctx.split, score_set, grids, qwen_adapter, alpha)
    rows.append(_method_row(ctx, alpha, "qwen_backed_adaptive", "diagnostic", metrics, selected_adapter=qwen_adapter, cal_index=idx, cal_risk=risk))
    best_fixed = max(ALL_CANDIDATES, key=lambda candidate: (fixed_metrics[candidate]["prefix_retained_fraction"], -fixed_metrics[candidate]["prefix_contamination"]))
    rows.append(_method_row(ctx, alpha, "best_fixed_adapter", "diagnostic", fixed_metrics[best_fixed], selected_adapter=best_fixed, cal_index=candidate_indices[best_fixed], cal_risk=np.nan))
    hindsight_lengths, hindsight_choices = _hindsight_lengths(ctx.split.test, score_set["test"], grids, candidate_indices, ALL_CANDIDATES)
    metrics = _metrics_from_lengths(
        ctx.split.test,
        hindsight_lengths,
        qwen_call_rate=float(np.mean([choice in QWEN_CANDIDATES for choice in hindsight_choices])),
    )
    rows.append(_method_row(ctx, alpha, "hindsight_per_trace_adapter", "diagnostic", metrics, selected_adapter="per_trace", cal_index=np.nan, cal_risk=np.nan))
    info = {
        "dataset_adapter": dataset_adapter,
        "cheap_adapter": cheap_adapter,
        "qwen_adapter": qwen_adapter,
        "selection_rows": selection_rows,
        "candidate_indices": candidate_indices,
        "dataset_row": dataset_row,
        "fixed_metrics": fixed_metrics,
        "hindsight_choices": hindsight_choices,
        "hindsight_lengths": hindsight_lengths,
    }
    return rows, info


def _method_row(
    ctx: SeedContext,
    alpha: float,
    method: str,
    method_type: str,
    metrics: dict[str, float],
    *,
    selected_adapter: str,
    cal_index: int | float,
    cal_risk: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "dataset": ctx.dataset,
        "seed": ctx.seed,
        "alpha": alpha,
        "method": method,
        "method_type": method_type,
        "selected_adapter": selected_adapter,
        "cal_index": cal_index,
        "cal_corrected_risk": cal_risk,
        **metrics,
    }
    if extra:
        row.update(extra)
    return row


def _hindsight_lengths(
    traces: list[TraceRecord],
    scores_by_candidate: dict[str, list[np.ndarray]],
    grids: dict[str, np.ndarray],
    candidate_indices: dict[str, int],
    candidates: list[str],
) -> tuple[np.ndarray, list[str]]:
    lengths = []
    choices = []
    first_errors = _first_errors(traces)
    totals = _trace_lengths(traces)
    for trace_idx, _trace in enumerate(traces):
        utilities = {}
        candidate_lengths = {}
        for candidate in candidates:
            idx = candidate_indices[candidate]
            length = int(prefix_lengths([scores_by_candidate[candidate][trace_idx]], float(grids[candidate][idx]))[0])
            contaminated = int(length > int(first_errors[trace_idx]))
            utilities[candidate] = float(length / max(totals[trace_idx], 1.0)) - contaminated
            candidate_lengths[candidate] = length
        choice = _tie_break_candidate(utilities, candidates)
        choices.append(choice)
        lengths.append(candidate_lengths[choice])
    return np.asarray(lengths, dtype=int), choices


def _run_trace_gate(
    ctx: SeedContext,
    score_set_name: str,
    alpha: float,
    *,
    method: str,
    feature_mode: str,
    args,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    score_set = ctx.normalized_scores[score_set_name]
    grids = ctx.grids[score_set_name]
    labels = make_crossfit_gate_labels(
        ctx.split.select,
        score_set["select"],
        grids,
        ALL_CANDIDATES,
        alpha=alpha,
        penalty=args.gate_penalty,
        cost_penalty=0.0,
        n_folds=args.n_folds,
        seed=ctx.seed,
    )
    select_features = build_trace_gate_features(ctx.split.select, score_set["select"], mode=feature_mode)
    cal_features = build_trace_gate_features(ctx.split.cal, score_set["cal"], mode=feature_mode)
    test_features = build_trace_gate_features(ctx.split.test, score_set["test"], mode=feature_mode)
    gate = train_trace_gate(select_features, labels, ALL_CANDIDATES, seed=ctx.seed, model_type=args.gate_model, feature_mode=feature_mode)
    cal_pred = gate.model.predict(cal_features).astype(str).tolist()
    test_pred = gate.model.predict(test_features).astype(str).tolist()
    idx, risk = _calibrate_gated_index(ctx.split.cal, cal_pred, score_set["cal"], grids, alpha)
    metrics = _evaluate_gated(ctx.split.test, test_pred, score_set["test"], grids, idx)
    row = _method_row(
        ctx,
        alpha,
        method,
        "trace_gate",
        metrics,
        selected_adapter="trace_conditioned",
        cal_index=idx,
        cal_risk=risk,
        extra={"score_normalization": score_set_name, "selection_entropy": _entropy(_adapter_distribution(test_pred))},
    )
    cal_df = pd.DataFrame({"trace_id": [t.trace_id for t in ctx.split.cal], "predicted_candidate": cal_pred})
    test_df = pd.DataFrame({"trace_id": [t.trace_id for t in ctx.split.test], "predicted_candidate": test_pred})
    return row, cal_df, test_df


def _fit_utility_model(features: list[dict[str, Any]], labels: pd.DataFrame, candidates: list[str], seed: int) -> Pipeline:
    y_cols = [f"utility__{candidate}" for candidate in candidates]
    y = labels[y_cols].to_numpy(float)
    if len(features) < 10:
        reg = MultiOutputRegressor(DummyRegressor(strategy="mean"))
    else:
        reg = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=60,
                learning_rate=0.08,
                max_leaf_nodes=8,
                min_samples_leaf=max(10, min(25, len(features) // 20)),
                random_state=seed,
            )
        )
    pipe = Pipeline([("vec", DictVectorizer(sparse=False)), ("imputer", SimpleImputer(strategy="median")), ("reg", reg)])
    pipe.fit(features, y)
    return pipe


def _predict_utility(pipe: Pipeline, features: list[dict[str, Any]], candidates: list[str]) -> pd.DataFrame:
    pred = np.asarray(pipe.predict(features), dtype=float)
    return pd.DataFrame(pred, columns=candidates)


def _run_confidence_override(ctx: SeedContext, alpha: float, baseline: dict[str, Any], args) -> list[dict[str, Any]]:
    score_set = ctx.normalized_scores["raw"]
    grids = ctx.grids["raw"]
    labels = make_crossfit_gate_labels(
        ctx.split.select,
        score_set["select"],
        grids,
        ALL_CANDIDATES,
        alpha=alpha,
        penalty=args.gate_penalty,
        cost_penalty=0.0,
        n_folds=args.n_folds,
        seed=ctx.seed + 13,
    )
    pipe = _fit_utility_model(build_trace_gate_features(ctx.split.select, score_set["select"], mode="full"), labels, ALL_CANDIDATES, ctx.seed)
    cal_util = _predict_utility(pipe, build_trace_gate_features(ctx.split.cal, score_set["cal"], mode="full"), ALL_CANDIDATES)
    test_util = _predict_utility(pipe, build_trace_gate_features(ctx.split.test, score_set["test"], mode="full"), ALL_CANDIDATES)
    dataset_adapter = str(baseline["dataset_adapter"])
    rows = []
    for tau in args.tau_grid:
        cal_pred = _override_predictions(cal_util, dataset_adapter, tau)
        test_pred = _override_predictions(test_util, dataset_adapter, tau)
        idx, risk = _calibrate_gated_index(ctx.split.cal, cal_pred, score_set["cal"], grids, alpha)
        metrics = _evaluate_gated(ctx.split.test, test_pred, score_set["test"], grids, idx)
        rows.append(
            _method_row(
                ctx,
                alpha,
                "confidence_override",
                "confidence_override",
                metrics,
                selected_adapter=dataset_adapter,
                cal_index=idx,
                cal_risk=risk,
                extra={
                    "tau": tau,
                    "override_rate": float(np.mean([pred != dataset_adapter for pred in test_pred])) if test_pred else float("nan"),
                    "delta_vs_dataset_adaptive": metrics["prefix_retained_fraction"] - float(baseline["dataset_row"]["prefix_retained_fraction"]),
                },
            )
        )
    return rows


def _override_predictions(utilities: pd.DataFrame, dataset_adapter: str, tau: float) -> list[str]:
    out = []
    for row in utilities.itertuples(index=False):
        vals = {candidate: float(getattr(row, candidate)) for candidate in utilities.columns}
        best = _tie_break_candidate(vals, list(utilities.columns))
        margin = vals[best] - vals[dataset_adapter]
        out.append(best if margin > tau else dataset_adapter)
    return out


def _fit_group_thresholds(ctx: SeedContext, score_set: dict[str, dict[str, list[np.ndarray]]]) -> dict[str, Any]:
    train_token = np.concatenate(score_set["train"]["token_format"])
    token_mid = float(np.quantile(train_token, 0.5)) if len(train_token) else 0.5
    select_lengths = np.asarray([len(trace.steps) for trace in ctx.split.select], dtype=float)
    length_bins = _tercile_bins(select_lengths)
    cheap_proxy = _cheap_prefix_proxy(ctx.split.select, score_set["select"], token_mid)
    cheap_bins = _tercile_bins(cheap_proxy)
    disagreement = _token_qwen_disagreement(ctx.split.select, ctx.normalized_scores["cdf"]["select"])
    disagreement_cut = float(np.median(disagreement)) if len(disagreement) else 0.0
    return {
        "token_mid": token_mid,
        "length_bins": length_bins,
        "cheap_bins": cheap_bins,
        "disagreement_cut": disagreement_cut,
    }


def _tercile_bins(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (0.0, 0.0)
    lo, hi = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    return float(lo), float(hi)


def _bin3(value: float, bins: tuple[float, float]) -> str:
    if value <= bins[0]:
        return "short"
    if value <= bins[1]:
        return "medium"
    return "long"


def _cheap_prefix_proxy(traces: list[TraceRecord], scores: dict[str, list[np.ndarray]], threshold: float) -> np.ndarray:
    vals = []
    for trace, token_scores in zip(traces, scores["token_format"]):
        length = prefix_lengths([token_scores], threshold)[0]
        vals.append(float(length / max(len(trace.steps), 1)))
    return np.asarray(vals, dtype=float)


def _token_qwen_disagreement(traces: list[TraceRecord], scores: dict[str, list[np.ndarray]]) -> np.ndarray:
    vals = []
    for token, qwen in zip(scores["token_format"], scores["qwen_prm"]):
        token = np.asarray(token, dtype=float)
        qwen = np.asarray(qwen, dtype=float)
        vals.append(float(np.mean(np.abs(token - qwen))) if len(token) and len(token) == len(qwen) else 0.0)
    return np.asarray(vals, dtype=float)


def _assign_groups(
    traces: list[TraceRecord],
    scores: dict[str, list[np.ndarray]],
    cdf_scores: dict[str, list[np.ndarray]],
    thresholds: dict[str, Any],
    rule: str,
) -> list[str]:
    lengths = np.asarray([len(trace.steps) for trace in traces], dtype=float)
    length_group = [_bin3(float(value), thresholds["length_bins"]) for value in lengths]
    cheap = _cheap_prefix_proxy(traces, scores, float(thresholds["token_mid"]))
    cheap_group = [_bin3(float(value), thresholds["cheap_bins"]) for value in cheap]
    disagreement = _token_qwen_disagreement(traces, cdf_scores)
    dis_group = ["high_disagree" if float(value) > float(thresholds["disagreement_cut"]) else "low_disagree" for value in disagreement]
    if rule == "trace_length_tercile":
        return length_group
    if rule == "cheap_prefix_tercile":
        return cheap_group
    if rule == "token_qwen_disagreement":
        return dis_group
    if rule == "cheap_prefix_x_disagreement":
        return [f"{a}:{b}" for a, b in zip(cheap_group, dis_group)]
    raise ValueError(f"Unknown grouping rule={rule!r}")


def _run_stratified(ctx: SeedContext, alpha: float, baseline: dict[str, Any], args) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    score_set = ctx.normalized_scores["raw"]
    cdf_set = ctx.normalized_scores["cdf"]
    grids = ctx.grids["raw"]
    thresholds = _fit_group_thresholds(ctx, score_set)
    rows = []
    group_rows = []
    for rule in args.group_rules:
        select_groups = _assign_groups(ctx.split.select, score_set["select"], cdf_set["select"], thresholds, rule)
        cal_groups = _assign_groups(ctx.split.cal, score_set["cal"], cdf_set["cal"], thresholds, rule)
        test_groups = _assign_groups(ctx.split.test, score_set["test"], cdf_set["test"], thresholds, rule)
        group_adapter = _select_group_adapters(ctx.split.select, score_set["select"], grids, select_groups, baseline["dataset_adapter"], alpha, args.min_selection_group_n)
        cal_pred = [group_adapter.get(group, baseline["dataset_adapter"]) for group in cal_groups]
        test_pred = [group_adapter.get(group, baseline["dataset_adapter"]) for group in test_groups]
        feature_qwen_rate = 1.0 if "qwen" in rule else 0.0
        idx, risk = _calibrate_gated_index(ctx.split.cal, cal_pred, score_set["cal"], grids, alpha)
        metrics = _evaluate_gated(ctx.split.test, test_pred, score_set["test"], grids, idx)
        metrics["qwen_call_rate"] = max(metrics["qwen_call_rate"], feature_qwen_rate)
        rows.append(
            _method_row(
                ctx,
                alpha,
                "stratified_adaptive",
                "stratified",
                metrics,
                selected_adapter="group_adaptive",
                cal_index=idx,
                cal_risk=risk,
                extra={
                    "grouping_rule": rule,
                    "calibration_type": "pooled",
                    "min_group_n": np.nan,
                    "n_groups": len(set(test_groups)),
                    "fallback_fraction": float(np.mean([group not in group_adapter for group in test_groups])) if test_groups else float("nan"),
                    "delta_vs_dataset_adaptive": metrics["prefix_retained_fraction"] - float(baseline["dataset_row"]["prefix_retained_fraction"]),
                },
            )
        )
        group_rows.extend(_group_rows(ctx, rule, "pooled", group_adapter, test_groups, test_pred, score_set["test"], grids, idx, feature_qwen_rate))
        for min_group_n in args.min_group_n:
            metrics, fallback_fraction = _evaluate_mondrian_groups(ctx, score_set, grids, cal_groups, test_groups, cal_pred, test_pred, group_adapter, alpha, int(min_group_n), idx, feature_qwen_rate)
            rows.append(
                _method_row(
                    ctx,
                    alpha,
                    "stratified_adaptive",
                    "stratified",
                    metrics,
                    selected_adapter="group_adaptive",
                    cal_index=np.nan,
                    cal_risk=np.nan,
                    extra={
                        "grouping_rule": rule,
                        "calibration_type": "mondrian",
                        "min_group_n": int(min_group_n),
                        "n_groups": len(set(test_groups)),
                        "fallback_fraction": fallback_fraction,
                        "delta_vs_dataset_adaptive": metrics["prefix_retained_fraction"] - float(baseline["dataset_row"]["prefix_retained_fraction"]),
                    },
                )
            )
    return rows, group_rows


def _select_group_adapters(
    traces: list[TraceRecord],
    scores: dict[str, list[np.ndarray]],
    grids: dict[str, np.ndarray],
    groups: list[str],
    fallback_adapter: str,
    alpha: float,
    min_selection_group_n: int,
) -> dict[str, str]:
    selected: dict[str, str] = {}
    groups_arr = np.asarray(groups, dtype=object)
    for group in sorted(set(groups)):
        idxs = np.flatnonzero(groups_arr == group)
        if len(idxs) < min_selection_group_n:
            continue
        group_traces = [traces[int(i)] for i in idxs]
        group_scores = {candidate: [scores[candidate][int(i)] for i in idxs] for candidate in ALL_CANDIDATES}
        adapter, _ = _select_adapter_on_selection(group_traces, group_scores, grids, ALL_CANDIDATES, alpha)
        selected[group] = adapter or fallback_adapter
    return selected


def _evaluate_mondrian_groups(
    ctx: SeedContext,
    score_set: dict[str, dict[str, list[np.ndarray]]],
    grids: dict[str, np.ndarray],
    cal_groups: list[str],
    test_groups: list[str],
    cal_pred: list[str],
    test_pred: list[str],
    group_adapter: dict[str, str],
    alpha: float,
    min_group_n: int,
    pooled_idx: int,
    feature_qwen_rate: float,
) -> tuple[dict[str, float], float]:
    cal_groups_arr = np.asarray(cal_groups, dtype=object)
    group_index: dict[str, int] = {}
    fallback_groups = set()
    for group, adapter in group_adapter.items():
        cal_idxs = np.flatnonzero(cal_groups_arr == group)
        if len(cal_idxs) < min_group_n:
            fallback_groups.add(group)
            continue
        group_traces = [ctx.split.cal[int(i)] for i in cal_idxs]
        group_scores = [score_set["cal"][adapter][int(i)] for i in cal_idxs]
        idx, _ = _select_index_for_candidate(group_traces, group_scores, grids[adapter], alpha)
        group_index[group] = idx
    lengths = []
    fallback_count = 0
    for trace_idx, (group, adapter) in enumerate(zip(test_groups, test_pred)):
        use_idx = group_index.get(group, pooled_idx)
        if group not in group_index:
            fallback_count += 1
        length = prefix_lengths([score_set["test"][adapter][trace_idx]], float(grids[adapter][use_idx]))[0]
        lengths.append(int(length))
    metrics = _metrics_from_lengths(ctx.split.test, np.asarray(lengths, dtype=int), qwen_call_rate=float(np.mean([adapter in QWEN_CANDIDATES for adapter in test_pred])))
    metrics["qwen_call_rate"] = max(metrics["qwen_call_rate"], feature_qwen_rate)
    fallback_fraction = float(fallback_count / max(len(test_pred), 1))
    return metrics, fallback_fraction


def _group_rows(
    ctx: SeedContext,
    rule: str,
    calibration_type: str,
    group_adapter: dict[str, str],
    test_groups: list[str],
    test_pred: list[str],
    test_scores: dict[str, list[np.ndarray]],
    grids: dict[str, np.ndarray],
    pooled_idx: int,
    feature_qwen_rate: float,
) -> list[dict[str, Any]]:
    rows = []
    test_groups_arr = np.asarray(test_groups, dtype=object)
    for group in sorted(set(test_groups)):
        idxs = np.flatnonzero(test_groups_arr == group)
        adapter = group_adapter.get(group, "fallback")
        if adapter == "fallback" or len(idxs) == 0:
            continue
        lengths = [int(prefix_lengths([test_scores[adapter][int(i)]], float(grids[adapter][pooled_idx]))[0]) for i in idxs]
        traces = [ctx.split.test[int(i)] for i in idxs]
        metrics = _metrics_from_lengths(traces, np.asarray(lengths, dtype=int), qwen_call_rate=max(float(adapter in QWEN_CANDIDATES), feature_qwen_rate))
        rows.append(
            {
                "dataset": ctx.dataset,
                "seed": ctx.seed,
                "grouping_rule": rule,
                "calibration_type": calibration_type,
                "group": group,
                "size": len(idxs),
                "selected_adapter": adapter,
                **metrics,
            }
        )
    return rows


def _run_cost_only(ctx: SeedContext, alpha: float, baseline: dict[str, Any], args) -> list[dict[str, Any]]:
    score_set = ctx.normalized_scores["raw"]
    grids = ctx.grids["raw"]
    labels = make_crossfit_gate_labels(
        ctx.split.select,
        score_set["select"],
        grids,
        ALL_CANDIDATES,
        alpha=alpha,
        penalty=args.gate_penalty,
        cost_penalty=0.0,
        n_folds=args.n_folds,
        seed=ctx.seed + 29,
    )
    pipe = _fit_utility_model(build_trace_gate_features(ctx.split.select, score_set["select"], mode="cheap"), labels, ALL_CANDIDATES, ctx.seed + 3)
    cal_util = _predict_utility(pipe, build_trace_gate_features(ctx.split.cal, score_set["cal"], mode="cheap"), ALL_CANDIDATES)
    test_util = _predict_utility(pipe, build_trace_gate_features(ctx.split.test, score_set["test"], mode="cheap"), ALL_CANDIDATES)
    rows = []
    for threshold in args.cost_threshold_grid:
        cal_pred = _cost_predictions(cal_util, threshold)
        test_pred = _cost_predictions(test_util, threshold)
        idx, risk = _calibrate_gated_index(ctx.split.cal, cal_pred, score_set["cal"], grids, alpha)
        metrics = _evaluate_gated(ctx.split.test, test_pred, score_set["test"], grids, idx)
        cheap_kept = float(next(row for row in baseline["rows"] if row["method"] == "best_cheap_adaptive")["prefix_retained_fraction"])
        adaptive_kept = float(baseline["dataset_row"]["prefix_retained_fraction"])
        qwen_kept = float(next(row for row in baseline["rows"] if row["method"] == "qwen_backed_adaptive")["prefix_retained_fraction"])
        denom = qwen_kept - cheap_kept
        gain_recovered = np.nan if denom <= 0.02 else (metrics["prefix_retained_fraction"] - cheap_kept) / denom
        rows.append(
            _method_row(
                ctx,
                alpha,
                "cost_only_routing",
                "cost_only",
                metrics,
                selected_adapter="cost_router",
                cal_index=idx,
                cal_risk=risk,
                extra={
                    "cost_threshold": threshold,
                    "delta_vs_dataset_adaptive": metrics["prefix_retained_fraction"] - adaptive_kept,
                    "delta_vs_cheap_only": metrics["prefix_retained_fraction"] - cheap_kept,
                    "gain_recovered": gain_recovered,
                    "gain_recovery_unstable": bool(denom <= 0.02),
                },
            )
        )
    return rows


def _cost_predictions(utilities: pd.DataFrame, threshold: float) -> list[str]:
    out = []
    for row in utilities.itertuples(index=False):
        vals = {candidate: float(getattr(row, candidate)) for candidate in utilities.columns}
        cheap = _tie_break_candidate({candidate: vals[candidate] for candidate in CHEAP_CANDIDATES}, CHEAP_CANDIDATES)
        qwen = _tie_break_candidate({candidate: vals[candidate] for candidate in QWEN_CANDIDATES}, list(QWEN_CANDIDATES))
        out.append(qwen if vals[qwen] - vals[cheap] > threshold else cheap)
    return out


def _run_audit(ctx: SeedContext, alpha: float, baseline: dict[str, Any], trace_gate_row: dict[str, Any], trace_gate_pred: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    score_set = ctx.normalized_scores["raw"]
    grids = ctx.grids["raw"]
    candidate_indices = baseline["candidate_indices"]
    dataset_adapter = baseline["dataset_adapter"]
    gate_selected = trace_gate_pred["predicted_candidate"].astype(str).tolist()
    gate_idx = int(trace_gate_row["cal_index"])
    first_errors = _first_errors(ctx.split.test)
    totals = _trace_lengths(ctx.split.test)
    per_trace = []
    agree = []
    top2 = []
    regrets_h = []
    regrets_d = []
    qwen_needed = []
    qwen_selected = []
    hindsight_choices = []
    for trace_idx, trace in enumerate(ctx.split.test):
        utilities = {}
        lengths = {}
        for candidate in ALL_CANDIDATES:
            idx = candidate_indices[candidate]
            length = int(prefix_lengths([score_set["test"][candidate][trace_idx]], float(grids[candidate][idx]))[0])
            contaminated = int(length > int(first_errors[trace_idx]))
            utilities[candidate] = float(length / max(totals[trace_idx], 1.0)) - contaminated
            lengths[candidate] = length
        sorted_candidates = sorted(ALL_CANDIDATES, key=lambda cand: (-utilities[cand], TIE_ORDER.get(cand, 999)))
        hindsight = sorted_candidates[0]
        selected = gate_selected[trace_idx]
        gate_length = int(prefix_lengths([score_set["test"][selected][trace_idx]], float(grids[selected][gate_idx]))[0])
        dataset_length = lengths[dataset_adapter]
        hindsight_length = lengths[hindsight]
        gate_frac = float(gate_length / max(totals[trace_idx], 1.0))
        dataset_frac = float(dataset_length / max(totals[trace_idx], 1.0))
        hindsight_frac = float(hindsight_length / max(totals[trace_idx], 1.0))
        row = {
            "dataset": ctx.dataset,
            "seed": ctx.seed,
            "alpha": alpha,
            "trace_id": trace.trace_id,
            "trace_gate_adapter": selected,
            "hindsight_best_adapter": hindsight,
            "dataset_adapter": dataset_adapter,
            "trace_gate_retained_fraction": gate_frac,
            "hindsight_retained_fraction": hindsight_frac,
            "dataset_retained_fraction": dataset_frac,
            "regret_vs_hindsight": hindsight_frac - gate_frac,
            "regret_vs_dataset_adaptive": dataset_frac - gate_frac,
            "qwen_needed_by_hindsight": bool(hindsight in QWEN_CANDIDATES),
            "trace_gate_selected_qwen": bool(selected in QWEN_CANDIDATES),
            "top1_agreement": bool(selected == hindsight),
            "top2_agreement": bool(selected in sorted_candidates[:2]),
        }
        per_trace.append(row)
        agree.append(row["top1_agreement"])
        top2.append(row["top2_agreement"])
        regrets_h.append(row["regret_vs_hindsight"])
        regrets_d.append(row["regret_vs_dataset_adaptive"])
        qwen_needed.append(row["qwen_needed_by_hindsight"])
        qwen_selected.append(row["trace_gate_selected_qwen"])
        hindsight_choices.append(hindsight)
    qwen_needed_arr = np.asarray(qwen_needed, dtype=bool)
    qwen_selected_arr = np.asarray(qwen_selected, dtype=bool)
    summary = {
        "dataset": ctx.dataset,
        "seed": ctx.seed,
        "alpha": alpha,
        "top1_agreement": float(np.mean(agree)) if agree else float("nan"),
        "top2_agreement": float(np.mean(top2)) if top2 else float("nan"),
        "mean_regret_vs_hindsight": float(np.mean(regrets_h)) if regrets_h else float("nan"),
        "median_regret_vs_hindsight": float(np.median(regrets_h)) if regrets_h else float("nan"),
        "mean_regret_vs_dataset_adaptive": float(np.mean(regrets_d)) if regrets_d else float("nan"),
        "qwen_needed_misrouted_to_cheap": float(np.mean(~qwen_selected_arr[qwen_needed_arr])) if np.any(qwen_needed_arr) else float("nan"),
        "cheap_sufficient_routed_to_qwen": float(np.mean(qwen_selected_arr[~qwen_needed_arr])) if np.any(~qwen_needed_arr) else float("nan"),
        "modal_selected_adapter": pd.Series(gate_selected).mode().iloc[0] if gate_selected else "",
        "modal_hindsight_best_adapter": pd.Series(hindsight_choices).mode().iloc[0] if hindsight_choices else "",
    }
    return per_trace, summary


def _run_dataset_seed(args, dataset: str, text_features: str, combined_features: str, qwen_csv: str, seed: int) -> dict[str, list[dict[str, Any]]]:
    print(f"Running trace-gating next steps for {dataset} seed={seed}", flush=True)
    ctx = _prepare_seed_context(args, dataset, text_features, combined_features, qwen_csv, seed)
    out: dict[str, list[dict[str, Any]]] = {
        "audit_per_trace": [],
        "audit_summary": [],
        "normalization": [],
        "confidence": [],
        "stratified": [],
        "groups": [],
        "cost": [],
        "raw_methods": [],
    }
    for alpha in args.alpha_grid:
        raw_baseline_rows, baseline = _baseline_rows(ctx, "raw", alpha)
        baseline["rows"] = raw_baseline_rows
        out["raw_methods"].extend(raw_baseline_rows)
        trace_gate_row, _cal_pred, test_pred = _run_trace_gate(ctx, "raw", alpha, method="current_trace_gate", feature_mode="full", args=args)
        cdf_gate_row, _, _ = _run_trace_gate(ctx, "cdf", alpha, method="cdf_normalized_trace_gate", feature_mode="full", args=args)
        rank_gate_row, _, _ = _run_trace_gate(ctx, "rank", alpha, method="rank_normalized_trace_gate", feature_mode="full", args=args)
        out["normalization"].extend(
            [
                _normalization_row(raw_baseline_rows, "dataset_adaptive"),
                _normalization_row([trace_gate_row], "current_trace_gate", baseline["dataset_row"]),
                _normalization_row([cdf_gate_row], "cdf_normalized_trace_gate", baseline["dataset_row"]),
                _normalization_row([rank_gate_row], "rank_normalized_trace_gate", baseline["dataset_row"]),
                _normalization_row(raw_baseline_rows, "hindsight_per_trace_adapter", baseline["dataset_row"]),
            ]
        )
        per_trace, audit_summary = _run_audit(ctx, alpha, baseline, trace_gate_row, test_pred)
        out["audit_per_trace"].extend(per_trace)
        out["audit_summary"].append(audit_summary)
        out["confidence"].extend(_run_confidence_override(ctx, alpha, baseline, args))
        stratified, groups = _run_stratified(ctx, alpha, baseline, args)
        out["stratified"].extend(stratified)
        out["groups"].extend(groups)
        out["cost"].extend(_run_cost_only(ctx, alpha, baseline, args))
    return out


def _normalization_row(rows: list[dict[str, Any]], method: str, dataset_row: dict[str, Any] | None = None) -> dict[str, Any]:
    row = next(row for row in rows if row["method"] == method)
    out = dict(row)
    out["delta_vs_dataset_adaptive"] = (
        out["prefix_retained_fraction"] - float(dataset_row["prefix_retained_fraction"])
        if dataset_row is not None
        else 0.0
    )
    return out


def _aggregate_mean(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    numeric = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in group_cols]
    nonnumeric = [col for col in df.columns if col not in numeric and col not in group_cols]
    agg = {col: "mean" for col in numeric}
    agg.update({col: "first" for col in nonnumeric})
    return df.groupby(group_cols, dropna=False).agg(agg).reset_index()


def _write_df_table(df: pd.DataFrame, csv_path: Path, md_path: Path | None, tex_path: Path | None, *, caption: str, label: str, copy_tex_name: str | None, output_root: Path) -> None:
    ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=False)
    if md_path is not None:
        df.to_markdown(md_path, index=False)
    if tex_path is not None:
        tex = df.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.4f}", caption=caption, label=label)
        tex_path.write_text(tex)
        if copy_tex_name and output_root.resolve() == Path("outputs").resolve():
            shutil.copyfile(tex_path, ensure_dir("tables") / copy_tex_name)


def _write_outputs(output_root: Path, all_rows: dict[str, list[dict[str, Any]]], *, smoke: bool) -> dict[str, bool]:
    output_root = ensure_dir(output_root)
    audit_dir = ensure_dir(output_root / "trace_gating_audit")
    norm_dir = ensure_dir(output_root / "trace_gating_normalization")
    conf_dir = ensure_dir(output_root / "confidence_override")
    strat_dir = ensure_dir(output_root / "stratified_adaptive")
    cost_dir = ensure_dir(output_root / "cost_only_routing")

    per_trace = pd.DataFrame(all_rows["audit_per_trace"])
    audit = _aggregate_mean(pd.DataFrame(all_rows["audit_summary"]), ["dataset", "alpha"])
    _write_df_table(audit, audit_dir / "table_gate_failure_audit.csv", audit_dir / "table_gate_failure_audit.md", audit_dir / "table_gate_failure_audit.tex", caption="Trace-gate failure audit.", label="tab:gate_failure_audit", copy_tex_name="table_gate_failure_audit.tex", output_root=output_root)
    per_trace.to_csv(audit_dir / "per_trace_gate_audit.csv", index=False)

    norm = _aggregate_mean(pd.DataFrame(all_rows["normalization"]), ["dataset", "alpha", "method"])
    _write_df_table(norm, norm_dir / "table_score_normalized_trace_gate.csv", norm_dir / "table_score_normalized_trace_gate.md", norm_dir / "table_score_normalized_trace_gate.tex", caption="Score-normalized trace gates.", label="tab:score_normalized_trace_gate", copy_tex_name="table_score_normalized_trace_gate.tex", output_root=output_root)

    conf = _aggregate_mean(pd.DataFrame(all_rows["confidence"]), ["dataset", "alpha", "tau"])
    _write_df_table(conf, conf_dir / "table_confidence_override.csv", conf_dir / "table_confidence_override.md", conf_dir / "table_confidence_override.tex", caption="Confidence-thresholded override gate.", label="tab:confidence_override", copy_tex_name="table_confidence_override.tex", output_root=output_root)
    _plot_frontier(conf, conf_dir / "fig_confidence_override_frontier.pdf", x_col="qwen_call_rate", y_col="prefix_retained_fraction", label_col="tau", title="Confidence override")
    if output_root.resolve() == Path("outputs").resolve():
        shutil.copyfile(conf_dir / "fig_confidence_override_frontier.pdf", ensure_dir("figures") / "fig_confidence_override_frontier.pdf")

    strat = _aggregate_mean(pd.DataFrame(all_rows["stratified"]), ["dataset", "alpha", "grouping_rule", "calibration_type", "min_group_n"])
    _write_df_table(strat, strat_dir / "table_stratified_adaptive.csv", strat_dir / "table_stratified_adaptive.md", strat_dir / "table_stratified_adaptive.tex", caption="Stratified adaptive CPCC.", label="tab:stratified_adaptive", copy_tex_name="table_stratified_adaptive.tex", output_root=output_root)
    groups = _aggregate_mean(pd.DataFrame(all_rows["groups"]), ["dataset", "grouping_rule", "calibration_type", "group", "selected_adapter"])
    _write_df_table(groups, strat_dir / "table_group_selected_adapters.csv", None, strat_dir / "table_group_selected_adapters.tex", caption="Group-level selected adapters.", label="tab:group_selected_adapters", copy_tex_name="table_group_selected_adapters.tex", output_root=output_root)

    cost = _aggregate_mean(pd.DataFrame(all_rows["cost"]), ["dataset", "alpha", "cost_threshold"])
    _write_df_table(cost, cost_dir / "table_cost_only_routing.csv", cost_dir / "table_cost_only_routing.md", cost_dir / "table_cost_only_routing.tex", caption="Cost-only trace routing.", label="tab:cost_only_routing", copy_tex_name="table_cost_only_routing.tex", output_root=output_root)
    _plot_frontier(cost, cost_dir / "fig_cost_only_routing_frontier.pdf", x_col="qwen_call_rate", y_col="prefix_retained_fraction", label_col="cost_threshold", title="Cost-only routing")
    if output_root.resolve() == Path("outputs").resolve():
        shutil.copyfile(cost_dir / "fig_cost_only_routing_frontier.pdf", ensure_dir("figures") / "fig_cost_only_routing_frontier.pdf")

    decision = _smoke_decision(norm, conf, strat, cost) if smoke else {"run_full": False, "normalization_promising": False, "confidence_promising": False, "stratified_promising": False, "cost_promising": False}
    _write_smoke_decision(output_root / "SMOKE_DECISION.md", decision, norm, conf, strat, cost)
    write_json(output_root / "run_summary.json", {key: int(len(value)) for key, value in all_rows.items()} | {"smoke": smoke, **decision})
    return decision


def merge_part_roots(part_roots: list[str], output_root: Path) -> None:
    output_root = ensure_dir(output_root)
    table_specs = [
        (
            "trace_gating_audit",
            "table_gate_failure_audit.csv",
            "table_gate_failure_audit.md",
            "table_gate_failure_audit.tex",
            "Trace-gate failure audit.",
            "tab:gate_failure_audit",
            "table_gate_failure_audit.tex",
        ),
        (
            "trace_gating_normalization",
            "table_score_normalized_trace_gate.csv",
            "table_score_normalized_trace_gate.md",
            "table_score_normalized_trace_gate.tex",
            "Score-normalized trace gates.",
            "tab:score_normalized_trace_gate",
            "table_score_normalized_trace_gate.tex",
        ),
        (
            "confidence_override",
            "table_confidence_override.csv",
            "table_confidence_override.md",
            "table_confidence_override.tex",
            "Confidence-thresholded override gate.",
            "tab:confidence_override",
            "table_confidence_override.tex",
        ),
        (
            "stratified_adaptive",
            "table_stratified_adaptive.csv",
            "table_stratified_adaptive.md",
            "table_stratified_adaptive.tex",
            "Stratified adaptive CPCC.",
            "tab:stratified_adaptive",
            "table_stratified_adaptive.tex",
        ),
        (
            "stratified_adaptive",
            "table_group_selected_adapters.csv",
            None,
            "table_group_selected_adapters.tex",
            "Group-level selected adapters.",
            "tab:group_selected_adapters",
            "table_group_selected_adapters.tex",
        ),
        (
            "cost_only_routing",
            "table_cost_only_routing.csv",
            "table_cost_only_routing.md",
            "table_cost_only_routing.tex",
            "Cost-only trace routing.",
            "tab:cost_only_routing",
            "table_cost_only_routing.tex",
        ),
    ]
    merged: dict[str, pd.DataFrame] = {}
    for subdir, csv_name, md_name, tex_name, caption, label, copy_name in table_specs:
        frames = []
        for part in part_roots:
            path = Path(part) / subdir / csv_name
            if path.exists():
                frames.append(pd.read_csv(path))
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        sort_cols = [col for col in ["dataset", "alpha", "method", "tau", "grouping_rule", "calibration_type", "min_group_n", "cost_threshold", "group"] if col in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        dest = ensure_dir(output_root / subdir)
        _write_df_table(
            df,
            dest / csv_name,
            dest / md_name if md_name else None,
            dest / tex_name,
            caption=caption,
            label=label,
            copy_tex_name=copy_name,
            output_root=output_root,
        )
        merged[f"{subdir}/{csv_name}"] = df
    per_trace_frames = []
    for part in part_roots:
        path = Path(part) / "trace_gating_audit" / "per_trace_gate_audit.csv"
        if path.exists():
            per_trace_frames.append(pd.read_csv(path))
    if per_trace_frames:
        pd.concat(per_trace_frames, ignore_index=True).to_csv(ensure_dir(output_root / "trace_gating_audit") / "per_trace_gate_audit.csv", index=False)
    conf = merged.get("confidence_override/table_confidence_override.csv", pd.DataFrame())
    if not conf.empty:
        _plot_frontier(conf, ensure_dir(output_root / "confidence_override") / "fig_confidence_override_frontier.pdf", x_col="qwen_call_rate", y_col="prefix_retained_fraction", label_col="tau", title="Confidence override")
        if output_root.resolve() == Path("outputs").resolve():
            shutil.copyfile(output_root / "confidence_override" / "fig_confidence_override_frontier.pdf", ensure_dir("figures") / "fig_confidence_override_frontier.pdf")
    cost = merged.get("cost_only_routing/table_cost_only_routing.csv", pd.DataFrame())
    if not cost.empty:
        _plot_frontier(cost, ensure_dir(output_root / "cost_only_routing") / "fig_cost_only_routing_frontier.pdf", x_col="qwen_call_rate", y_col="prefix_retained_fraction", label_col="cost_threshold", title="Cost-only routing")
        if output_root.resolve() == Path("outputs").resolve():
            shutil.copyfile(output_root / "cost_only_routing" / "fig_cost_only_routing_frontier.pdf", ensure_dir("figures") / "fig_cost_only_routing_frontier.pdf")
    _write_full_analysis(
        output_root / "TRACE_GATING_NEXT_STEPS_ANALYSIS.md",
        merged.get("trace_gating_audit/table_gate_failure_audit.csv", pd.DataFrame()),
        merged.get("trace_gating_normalization/table_score_normalized_trace_gate.csv", pd.DataFrame()),
        conf,
        merged.get("stratified_adaptive/table_stratified_adaptive.csv", pd.DataFrame()),
        cost,
    )
    write_json(output_root / "trace_gating_next_steps_merged_parts.json", {"part_roots": [str(Path(p)) for p in part_roots]})


def _plot_frontier(df: pd.DataFrame, path: Path, *, x_col: str, y_col: str, label_col: str, title: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for dataset, sub in df.groupby("dataset", dropna=False):
        sub = sub.sort_values(x_col)
        ax.plot(100.0 * sub[x_col], 100.0 * sub[y_col], marker="o", label=str(dataset))
        for row in sub.itertuples(index=False):
            ax.text(100.0 * getattr(row, x_col), 100.0 * getattr(row, y_col), f"{getattr(row, label_col):g}", fontsize=7)
    ax.set_xlabel("Qwen call rate (%)")
    ax.set_ylabel("Prefix kept (%)")
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _smoke_decision(norm: pd.DataFrame, conf: pd.DataFrame, strat: pd.DataFrame, cost: pd.DataFrame) -> dict[str, bool]:
    normalization_promising = False
    if not norm.empty:
        raw = norm[norm["method"] == "current_trace_gate"].set_index("dataset")
        for method in ["cdf_normalized_trace_gate", "rank_normalized_trace_gate"]:
            sub = norm[norm["method"] == method].set_index("dataset")
            shared = sorted(set(raw.index) & set(sub.index))
            if shared:
                improved = any(float(sub.loc[d, "prefix_retained_fraction"]) > float(raw.loc[d, "prefix_retained_fraction"]) + 0.01 for d in shared if d in {"ProcessBench", "PRMBench"})
                losses = [float(sub.loc[d, "delta_vs_dataset_adaptive"]) for d in shared]
                normalization_promising = normalization_promising or (improved and np.nanmin(losses) >= -0.05)
    confidence_promising = _override_like_promising(conf)
    stratified_promising = _override_like_promising(strat)
    cost_promising = False
    if not cost.empty:
        for dataset in ["Target", "PRMBench"]:
            sub = cost[cost["dataset"] == dataset]
            if sub.empty:
                continue
            if bool(((sub["delta_vs_dataset_adaptive"] >= -0.02) & (sub["qwen_call_rate"] <= 0.70)).any()):
                cost_promising = True
    return {
        "normalization_promising": bool(normalization_promising),
        "confidence_promising": bool(confidence_promising),
        "stratified_promising": bool(stratified_promising),
        "cost_promising": bool(cost_promising),
        "run_full": bool(normalization_promising or confidence_promising or stratified_promising or cost_promising),
    }


def _override_like_promising(df: pd.DataFrame) -> bool:
    if df.empty or "delta_vs_dataset_adaptive" not in df:
        return False
    grouping = [col for col in ["tau", "grouping_rule", "calibration_type", "min_group_n"] if col in df.columns]
    for _, sub in df.groupby(grouping, dropna=False) if grouping else [(None, df)]:
        by_dataset = {row.dataset: row for row in sub.itertuples(index=False)}
        guard_ok = all(dataset not in by_dataset or getattr(by_dataset[dataset], "delta_vs_dataset_adaptive") >= -0.01 for dataset in ["Target", "Math-Shepherd", "PRM800K"])
        utility_gain = any(dataset in by_dataset and getattr(by_dataset[dataset], "delta_vs_dataset_adaptive") >= 0.01 for dataset in ["ProcessBench", "PRMBench"])
        cost_gain = any(dataset in by_dataset and getattr(by_dataset[dataset], "delta_vs_dataset_adaptive") >= -0.02 and getattr(by_dataset[dataset], "qwen_call_rate") <= 0.70 for dataset in ["Target", "PRMBench"])
        nontrivial = any(getattr(row, "override_rate", 1.0) > 0.01 for row in by_dataset.values()) if "override_rate" in sub.columns else True
        if guard_ok and nontrivial and (utility_gain or cost_gain):
            return True
    return False


def _write_smoke_decision(path: Path, decision: dict[str, bool], norm: pd.DataFrame, conf: pd.DataFrame, strat: pd.DataFrame, cost: pd.DataFrame) -> None:
    lines = ["# Trace-Gating Next-Steps Smoke Decision", ""]
    lines.append(f"Run full grid: **{decision.get('run_full', False)}**")
    lines.append("")
    for key in ["normalization_promising", "confidence_promising", "stratified_promising", "cost_promising"]:
        lines.append(f"- {key}: {decision.get(key, False)}")
    lines.append("")
    lines.append("## Best Smoke Rows")
    lines.append("")
    for name, df in [("normalization", norm), ("confidence", conf), ("stratified", strat), ("cost", cost)]:
        lines.append(f"### {name}")
        if df.empty:
            lines.append("")
            continue
        cols = [col for col in ["dataset", "method", "tau", "grouping_rule", "calibration_type", "cost_threshold", "prefix_retained_fraction", "prefix_contamination", "qwen_call_rate", "delta_vs_dataset_adaptive"] if col in df.columns]
        show = df.sort_values(["dataset", "prefix_retained_fraction"], ascending=[True, False]).groupby("dataset", dropna=False).head(3)[cols].copy()
        for col in ["prefix_retained_fraction", "prefix_contamination", "qwen_call_rate", "delta_vs_dataset_adaptive"]:
            if col in show:
                show[col] = (100.0 * show[col]).round(2)
        lines.append(show.to_markdown(index=False))
        lines.append("")
    path.write_text("\n".join(lines))


def _write_full_analysis(path: Path, audit: pd.DataFrame, norm: pd.DataFrame, conf: pd.DataFrame, strat: pd.DataFrame, cost: pd.DataFrame) -> None:
    lines = ["# Trace-Gating Next-Steps Full Analysis", ""]
    lines.append("## Decision")
    lines.append("")
    lines.append("The full grid does not justify promoting unrestricted trace-level adaptation. Dataset-level adaptive CPCC remains the main method.")
    lines.append("")
    lines.append("The strongest follow-up signal is stratified adaptive CPCC as a cost-aware deployment variant on Target: cheap-prefix-tercile grouping with pooled calibration keeps essentially the same prefix utility while reducing Qwen calls.")
    lines.append("")
    lines.append("## Gate Failure Audit")
    lines.append("")
    if not audit.empty:
        show = audit[[
            "dataset",
            "top1_agreement",
            "top2_agreement",
            "mean_regret_vs_hindsight",
            "mean_regret_vs_dataset_adaptive",
            "qwen_needed_misrouted_to_cheap",
            "cheap_sufficient_routed_to_qwen",
            "modal_selected_adapter",
            "modal_hindsight_best_adapter",
        ]].copy()
        for col in [
            "top1_agreement",
            "top2_agreement",
            "mean_regret_vs_hindsight",
            "mean_regret_vs_dataset_adaptive",
            "qwen_needed_misrouted_to_cheap",
            "cheap_sufficient_routed_to_qwen",
        ]:
            show[col] = (100.0 * show[col]).round(2)
        lines.append(show.to_markdown(index=False))
    lines.append("")
    lines.append("## Normalization")
    lines.append("")
    lines.append("CDF normalization does not fix trace gating; rank normalization is worse. Current and normalized trace gates remain far below dataset-adaptive on Target, ProcessBench, Math-Shepherd, and PRMBench.")
    lines.append("")
    _append_best_rows(lines, norm, ["dataset", "method", "prefix_retained_fraction", "prefix_contamination", "qwen_call_rate", "delta_vs_dataset_adaptive"], n=5)
    lines.append("")
    lines.append("## Confidence Override")
    lines.append("")
    lines.append("Confidence-thresholded override preserves Target only at high thresholds, but then gives little cost reduction and still loses on ProcessBench and PRMBench.")
    lines.append("")
    _append_best_rows(lines, conf, ["dataset", "tau", "prefix_retained_fraction", "prefix_contamination", "qwen_call_rate", "delta_vs_dataset_adaptive", "override_rate"], n=3)
    lines.append("")
    lines.append("## Stratified Adaptive")
    lines.append("")
    lines.append("Stratification is stable but not a utility win: no external dataset improves by the required 1 retained-prefix point. The Target cheap-prefix-tercile pooled row is the useful deployment-style result.")
    lines.append("")
    _append_best_rows(lines, strat, ["dataset", "grouping_rule", "calibration_type", "min_group_n", "prefix_retained_fraction", "prefix_contamination", "qwen_call_rate", "delta_vs_dataset_adaptive", "fallback_fraction"], n=5)
    lines.append("")
    lines.append("## Cost-Only Routing")
    lines.append("")
    lines.append("Cost-only routing does not preserve dataset-adaptive utility on Target or PRMBench in the full run. It is not competitive with the earlier target cost-aware cascade.")
    lines.append("")
    _append_best_rows(lines, cost, ["dataset", "cost_threshold", "prefix_retained_fraction", "prefix_contamination", "qwen_call_rate", "delta_vs_dataset_adaptive", "gain_recovered"], n=3)
    path.write_text("\n".join(lines))


def _append_best_rows(lines: list[str], df: pd.DataFrame, cols: list[str], *, n: int) -> None:
    if df.empty:
        return
    present = [col for col in cols if col in df.columns]
    show = df.sort_values(["dataset", "prefix_retained_fraction"], ascending=[True, False]).groupby("dataset", dropna=False).head(n)[present].copy()
    for col in ["prefix_retained_fraction", "prefix_contamination", "qwen_call_rate", "delta_vs_dataset_adaptive", "override_rate", "fallback_fraction", "gain_recovered"]:
        if col in show:
            show[col] = (100.0 * show[col]).round(2)
    lines.append(show.to_markdown(index=False))


def run(args) -> dict[str, bool]:
    total = args.score_train_frac + args.gate_select_frac + args.cpcc_calibration_frac + args.test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")
    all_rows: dict[str, list[dict[str, Any]]] = {
        "audit_per_trace": [],
        "audit_summary": [],
        "normalization": [],
        "confidence": [],
        "stratified": [],
        "groups": [],
        "cost": [],
        "raw_methods": [],
    }
    for dataset, text_features, combined_features, qwen_csv, seeds in _dataset_configs(args):
        for seed in seeds:
            seed_rows = _run_dataset_seed(args, dataset, text_features, combined_features, qwen_csv, seed)
            for key, rows in seed_rows.items():
                all_rows[key].extend(rows)
    return _write_outputs(Path(args.output_root), all_rows, smoke=args.smoke)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="outputs/trace_gating_next_steps_smoke")
    parser.add_argument("--merge_part_roots", nargs="*", default=None)
    parser.add_argument("--qwen_score_col", default="qwen_prm_error")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--target_seeds", nargs="*", type=int, default=list(range(2806, 2826)))
    parser.add_argument("--external_seeds", nargs="*", type=int, default=list(range(2806, 2816)))
    parser.add_argument("--alpha_grid", nargs="*", type=float, default=[ALPHA_MAIN])
    parser.add_argument("--tau_grid", nargs="*", type=float, default=TAU_GRID)
    parser.add_argument("--cost_threshold_grid", nargs="*", type=float, default=COST_THRESHOLD_GRID)
    parser.add_argument("--group_rules", nargs="*", default=GROUP_RULES)
    parser.add_argument("--min_group_n", nargs="*", type=int, default=[100, 200])
    parser.add_argument("--min_selection_group_n", type=int, default=25)
    parser.add_argument("--threshold_grid_size", type=int, default=201)
    parser.add_argument("--score_train_frac", type=float, default=0.40)
    parser.add_argument("--gate_select_frac", type=float, default=0.20)
    parser.add_argument("--cpcc_calibration_frac", type=float, default=0.20)
    parser.add_argument("--test_frac", type=float, default=0.20)
    parser.add_argument("--class_weight", default="balanced")
    parser.add_argument("--gate_model", choices=["hgb", "logistic"], default="hgb")
    parser.add_argument("--gate_penalty", type=float, default=1.0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.merge_part_roots:
        merge_part_roots(args.merge_part_roots, Path(args.output_root))
        print(f"Merged trace-gating next-step outputs to {args.output_root}", flush=True)
        return
    if args.quick:
        args.smoke = True
        args.target_seeds = args.target_seeds[:1]
        args.external_seeds = args.external_seeds[:1]
        args.threshold_grid_size = min(args.threshold_grid_size, 51)
        args.tau_grid = [0.0, 0.005, 0.02, 0.05]
        args.cost_threshold_grid = [0.0, 0.005, 0.02, 0.05, 0.10]
        args.min_group_n = [25]
        args.min_selection_group_n = 10
    decision = run(args)
    print(f"Wrote trace-gating next-step outputs to {args.output_root}", flush=True)
    if args.smoke:
        print(f"Smoke full-run decision: {decision['run_full']}", flush=True)


if __name__ == "__main__":
    main()
