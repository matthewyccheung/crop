"""Trace-conditioned adaptive CPCC experiments.

This experiment learns a trace-level gate over existing adaptive CPCC score
adapters, then calibrates the composed gated policy with a shared threshold
index over adapter-specific threshold grids.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
import pickle
from pathlib import Path
import shutil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

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
    _slug,
    _summarize,
    _tex,
    split_traces_four_way,
)
from crop.experiments.exp17_adaptive_revision import DATASET_ORDER
from crop.metrics import full_trace_accept_rate, prefix_contamination_rate
from crop.risk_control import prefix_lengths, select_lambda_crc
from crop.utils import ensure_dir, write_json


ALPHA_GRID = [0.025, 0.05, 0.075, 0.10]
ALPHA_MAIN = 0.05
QWEN_CANDIDATES = {"qwen_prm", "step_qwen", "prefix_qwen", "hazard_qwen"}
CHEAP_CANDIDATES = ["token_format", "step_combined", "prefix_combined", "hazard_combined"]
ALL_CANDIDATES = CHEAP_CANDIDATES + ["qwen_prm", "step_qwen", "prefix_qwen", "hazard_qwen"]
CANDIDATE_SETS = {
    "all": ALL_CANDIDATES,
    "cheap_only": CHEAP_CANDIDATES,
    "qwen_backed_only": ["qwen_prm", "step_qwen", "prefix_qwen", "hazard_qwen"],
    "no_hazard": ["token_format", "step_combined", "prefix_combined", "qwen_prm", "step_qwen", "prefix_qwen"],
    "no_prefix": ["token_format", "step_combined", "hazard_combined", "qwen_prm", "step_qwen", "hazard_qwen"],
    "no_step": ["token_format", "prefix_combined", "hazard_combined", "qwen_prm", "prefix_qwen", "hazard_qwen"],
}
TIE_ORDER = {
    "token_format": 0,
    "step_combined": 1,
    "prefix_combined": 2,
    "hazard_combined": 3,
    "qwen_prm": 4,
    "step_qwen": 5,
    "prefix_qwen": 6,
    "hazard_qwen": 7,
}
LABELS = {spec.score: spec.label for spec in ADAPTER_SPECS}


@dataclass
class GateFit:
    model: Pipeline
    candidates: list[str]
    model_type: str
    feature_mode: str


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
    path.write_text(
        "\n".join(
            [
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
        )
    )


def _trace_ids(traces: list[TraceRecord]) -> list[str]:
    return [trace.trace_id for trace in traces]


def _trace_lengths(traces: list[TraceRecord]) -> np.ndarray:
    return np.asarray([len(trace.steps) for trace in traces], dtype=float)


def _first_errors(traces: list[TraceRecord]) -> np.ndarray:
    return np.asarray([len(trace.steps) + 1 if trace.first_error is None else int(trace.first_error) for trace in traces], dtype=int)


def _losses_from_lengths(traces: list[TraceRecord], lengths: np.ndarray) -> np.ndarray:
    return (np.asarray(lengths, dtype=int) > _first_errors(traces)).astype(int)


def _lengths_by_grid(scores_by_trace: list[np.ndarray], grid: np.ndarray) -> np.ndarray:
    out = np.zeros((len(grid), len(scores_by_trace)), dtype=int)
    for col, scores in enumerate(scores_by_trace):
        scores = np.asarray(scores, dtype=float)
        if len(scores) == 0:
            continue
        prefix_max = np.maximum.accumulate(scores)
        out[:, col] = np.searchsorted(prefix_max, grid, side="right")
    return out


def _candidate_threshold_grids(adapters: dict[str, AdapterBundle], candidates: list[str], j_grid: int) -> dict[str, np.ndarray]:
    grids: dict[str, np.ndarray] = {}
    quantiles = np.linspace(0.0, 1.0, j_grid)
    for candidate in candidates:
        scores = np.concatenate(adapters[candidate].select_scores_by_trace)
        if len(scores) == 0:
            grids[candidate] = np.linspace(0.0, 1.0, j_grid)
            continue
        grid = np.quantile(np.asarray(scores, dtype=float), quantiles)
        grid = np.maximum.accumulate(grid)
        grids[candidate] = grid
    return grids


def _metrics_from_lengths(traces: list[TraceRecord], lengths: np.ndarray, qwen_call_rate: float) -> dict[str, float]:
    lengths = np.asarray(lengths, dtype=int)
    totals = _trace_lengths(traces)
    losses = _losses_from_lengths(traces, lengths)
    full_accept = lengths == totals.astype(int)
    marginal_false_accept = np.asarray([trace.has_error for trace in traces], dtype=bool) & full_accept
    accepted_error_rate = float(np.mean([trace.has_error for trace, acc in zip(traces, full_accept) if acc])) if np.any(full_accept) else 0.0
    return {
        "prefix_contamination": prefix_contamination_rate(losses),
        "prefix_retained_fraction": float(np.mean(lengths / np.maximum(totals, 1.0))) if len(lengths) else float("nan"),
        "prefix_retained_steps": float(np.mean(lengths)) if len(lengths) else float("nan"),
        "prefix_full_trace_rate": full_trace_accept_rate(lengths, totals),
        "accepted_error_rate": accepted_error_rate,
        "marginal_full_trace_false_accept": float(np.mean(marginal_false_accept)) if len(lengths) else float("nan"),
        "qwen_call_rate": qwen_call_rate,
    }


def _eval_candidate_at_index(
    traces: list[TraceRecord],
    scores_by_trace: list[np.ndarray],
    grid: np.ndarray,
    idx: int,
    qwen_call_rate: float,
) -> dict[str, float]:
    lengths = prefix_lengths(scores_by_trace, float(grid[idx]))
    return _metrics_from_lengths(traces, lengths, qwen_call_rate=qwen_call_rate)


def _select_index_for_candidate(
    traces: list[TraceRecord],
    scores_by_trace: list[np.ndarray],
    grid: np.ndarray,
    alpha: float,
) -> tuple[int, float]:
    lengths = _lengths_by_grid(scores_by_trace, grid)
    losses = np.vstack([_losses_from_lengths(traces, row) for row in lengths])
    _, risk = select_lambda_crc(losses, np.arange(len(grid), dtype=float), alpha=alpha, direction="increasing")
    valid = np.flatnonzero(np.asarray([(1.0 + np.sum(row)) / (len(row) + 1.0) for row in losses]) <= alpha)
    idx = int(valid[-1]) if len(valid) else 0
    return idx, risk


def _calibrate_gated_index(
    traces: list[TraceRecord],
    predictions: list[str],
    scores_by_candidate: dict[str, list[np.ndarray]],
    grids: dict[str, np.ndarray],
    alpha: float,
) -> tuple[int, float]:
    j_grid = len(next(iter(grids.values())))
    losses = []
    for idx in range(j_grid):
        lengths = []
        for trace_idx, candidate in enumerate(predictions):
            score = scores_by_candidate[candidate][trace_idx]
            lengths.append(prefix_lengths([score], float(grids[candidate][idx]))[0])
        losses.append(_losses_from_lengths(traces, np.asarray(lengths, dtype=int)))
    losses_by_index = np.vstack(losses)
    _, risk = select_lambda_crc(losses_by_index, np.arange(j_grid, dtype=float), alpha=alpha, direction="increasing")
    risks = np.asarray([(1.0 + np.sum(row)) / (len(row) + 1.0) for row in losses_by_index])
    valid = np.flatnonzero(risks <= alpha)
    idx = int(valid[-1]) if len(valid) else 0
    return idx, risk


def _evaluate_gated(
    traces: list[TraceRecord],
    predictions: list[str],
    scores_by_candidate: dict[str, list[np.ndarray]],
    grids: dict[str, np.ndarray],
    idx: int,
) -> dict[str, float]:
    lengths = []
    for trace_idx, candidate in enumerate(predictions):
        score = scores_by_candidate[candidate][trace_idx]
        lengths.append(prefix_lengths([score], float(grids[candidate][idx]))[0])
    qwen_call_rate = float(np.mean([candidate in QWEN_CANDIDATES for candidate in predictions])) if predictions else float("nan")
    return _metrics_from_lengths(traces, np.asarray(lengths, dtype=int), qwen_call_rate=qwen_call_rate)


def _score_summary(scores: np.ndarray, prefix: str) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return {f"{prefix}_{name}": 0.0 for name in ["mean", "max", "min", "std", "last", "early_max", "slope", "first_spike_frac"]}
    early = scores[: max(1, int(math.ceil(0.25 * len(scores))))]
    threshold = float(np.quantile(scores, 0.75))
    spike = np.flatnonzero(scores >= threshold)
    return {
        f"{prefix}_mean": float(np.mean(scores)),
        f"{prefix}_max": float(np.max(scores)),
        f"{prefix}_min": float(np.min(scores)),
        f"{prefix}_std": float(np.std(scores)),
        f"{prefix}_last": float(scores[-1]),
        f"{prefix}_early_max": float(np.max(early)),
        f"{prefix}_slope": float(scores[-1] - scores[0]) if len(scores) > 1 else 0.0,
        f"{prefix}_first_spike_frac": float(spike[0] / max(len(scores) - 1, 1)) if len(spike) else 1.0,
    }


def _text_features(trace: TraceRecord) -> dict[str, float | str]:
    contents = [str(step.step_content or "") for step in trace.steps]
    joined = " ".join(contents)
    lengths = np.asarray([len(text) for text in contents], dtype=float)
    digit_count = sum(ch.isdigit() for ch in joined)
    token_count = max(len(joined.split()), 1)
    x = trace.X
    out: dict[str, float | str] = {
        "domain": trace.domain,
        "complexity": -1.0 if trace.complexity is None else float(trace.complexity),
        "n_steps": float(len(trace.steps)),
        "total_chars": float(len(joined)),
        "mean_step_chars": float(np.mean(lengths)) if len(lengths) else 0.0,
        "max_step_chars": float(np.max(lengths)) if len(lengths) else 0.0,
        "std_step_chars": float(np.std(lengths)) if len(lengths) else 0.0,
        "digit_fraction": float(digit_count / max(len(joined), 1)),
        "numeric_token_fraction": float(sum(any(ch.isdigit() for ch in tok) for tok in joined.split()) / token_count),
        "equals_count": float(joined.count("=")),
        "ineq_count": float(joined.count("<") + joined.count(">")),
        "operator_count": float(sum(joined.count(ch) for ch in "+-*/^")),
        "format_count": float(sum(joined.count(ch) for ch in "[](){}")),
        "feature_mean": float(np.nanmean(x)),
        "feature_std": float(np.nanstd(x)),
        "feature_max": float(np.nanmax(x)),
        "feature_min": float(np.nanmin(x)),
    }
    for idx in range(min(8, x.shape[1])):
        out[f"x{idx}_mean"] = float(np.nanmean(x[:, idx]))
        out[f"x{idx}_max"] = float(np.nanmax(x[:, idx]))
    return out


def build_trace_gate_features(
    traces: list[TraceRecord],
    scores_by_candidate: dict[str, list[np.ndarray]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    allowed = CHEAP_CANDIDATES if mode == "cheap" else ALL_CANDIDATES
    rows = []
    for idx, trace in enumerate(traces):
        row = _text_features(trace)
        for candidate in allowed:
            if candidate in scores_by_candidate:
                row.update(_score_summary(scores_by_candidate[candidate][idx], candidate))
        if mode == "full" and "qwen_prm" in scores_by_candidate:
            qwen = np.asarray(scores_by_candidate["qwen_prm"][idx], dtype=float)
            token_scores = scores_by_candidate.get("token_format")
            token = np.asarray(token_scores[idx] if token_scores is not None else qwen, dtype=float)
            if len(qwen) == len(token) and len(qwen) > 1 and np.std(qwen) > 0 and np.std(token) > 0:
                corr = np.corrcoef(qwen, token)[0, 1]
                row["qwen_token_corr"] = 0.0 if not np.isfinite(corr) else float(corr)
                row["qwen_token_max_gap"] = float(np.max(qwen - token))
            else:
                row["qwen_token_corr"] = 0.0
                row["qwen_token_max_gap"] = 0.0
        rows.append(row)
    return rows


def _utility_columns(candidates: list[str]) -> list[str]:
    return [f"utility__{candidate}" for candidate in candidates]


def _tie_break_candidate(utilities: dict[str, float], candidates: list[str]) -> str:
    best = max(utilities.values())
    near = [candidate for candidate in candidates if utilities[candidate] >= best - 0.0025]
    non_qwen = [candidate for candidate in near if candidate not in QWEN_CANDIDATES]
    pool = non_qwen if non_qwen else near
    return sorted(pool, key=lambda candidate: (TIE_ORDER.get(candidate, 999), candidate))[0]


def make_crossfit_gate_labels(
    traces: list[TraceRecord],
    scores_by_candidate: dict[str, list[np.ndarray]],
    grids: dict[str, np.ndarray],
    candidates: list[str],
    *,
    alpha: float,
    penalty: float,
    cost_penalty: float,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    n = len(traces)
    y_strata = np.asarray([f"{trace.domain}:{int(trace.has_error)}" for trace in traces])
    if n < max(2, n_folds):
        fold_ids = np.arange(n) % max(1, min(n, n_folds))
    else:
        try:
            splitter = StratifiedKFold(n_splits=min(n_folds, n), shuffle=True, random_state=seed)
            fold_ids = np.zeros(n, dtype=int)
            for fold, (_, val_idx) in enumerate(splitter.split(np.zeros(n), y_strata)):
                fold_ids[val_idx] = fold
        except ValueError:
            rng = np.random.default_rng(seed)
            fold_ids = rng.permutation(np.arange(n) % min(n_folds, n))
    rows = []
    for heldout_fold in sorted(set(fold_ids.tolist())):
        train_idx = np.flatnonzero(fold_ids != heldout_fold)
        val_idx = np.flatnonzero(fold_ids == heldout_fold)
        if len(train_idx) == 0:
            train_idx = val_idx
        fold_thresholds: dict[str, int] = {}
        for candidate in candidates:
            train_scores = [scores_by_candidate[candidate][int(i)] for i in train_idx]
            train_traces = [traces[int(i)] for i in train_idx]
            idx, _ = _select_index_for_candidate(train_traces, train_scores, grids[candidate], alpha)
            fold_thresholds[candidate] = idx
        for i in val_idx:
            trace = traces[int(i)]
            utilities: dict[str, float] = {}
            kept_by_candidate: dict[str, float] = {}
            contaminated_by_candidate: dict[str, int] = {}
            for candidate in candidates:
                scores = scores_by_candidate[candidate][int(i)]
                length = prefix_lengths([scores], float(grids[candidate][fold_thresholds[candidate]]))[0]
                kept = float(length / max(len(trace.steps), 1))
                contaminated = int(length > (len(trace.steps) + 1 if trace.first_error is None else int(trace.first_error)))
                utility = kept - penalty * contaminated - cost_penalty * float(candidate in QWEN_CANDIDATES)
                utilities[candidate] = utility
                kept_by_candidate[candidate] = kept
                contaminated_by_candidate[candidate] = contaminated
            label = _tie_break_candidate(utilities, candidates)
            row: dict[str, Any] = {
                "trace_id": trace.trace_id,
                "best_candidate_label": label,
                "cost_penalty": cost_penalty,
                "penalty": penalty,
            }
            for candidate in candidates:
                row[f"utility__{candidate}"] = utilities[candidate]
                row[f"kept__{candidate}"] = kept_by_candidate[candidate]
                row[f"contaminated__{candidate}"] = contaminated_by_candidate[candidate]
            rows.append(row)
    return pd.DataFrame(rows)


def _prediction_utility(y_pred: np.ndarray, labels: pd.DataFrame) -> float:
    vals = []
    indexed = labels.reset_index(drop=True)
    for idx, candidate in enumerate(y_pred):
        col = f"utility__{candidate}"
        vals.append(float(indexed.loc[idx, col]) if col in indexed else -1.0)
    return float(np.mean(vals)) if vals else float("-inf")


def train_trace_gate(
    features: list[dict[str, Any]],
    labels: pd.DataFrame,
    candidates: list[str],
    *,
    seed: int,
    model_type: str,
    feature_mode: str,
) -> GateFit:
    y = labels["best_candidate_label"].astype(str).to_numpy()
    if len(set(y.tolist())) < 2:
        clf = DummyClassifier(strategy="constant", constant=y[0] if len(y) else candidates[0])
        pipe = Pipeline([("vec", DictVectorizer(sparse=False)), ("imputer", SimpleImputer(strategy="median")), ("clf", clf)])
        pipe.fit(features, y if len(y) else np.asarray([candidates[0]]))
        return GateFit(pipe, candidates, model_type="constant", feature_mode=feature_mode)
    configs = []
    if model_type == "logistic":
        configs = [("logistic", LogisticRegression(max_iter=500, C=c, multi_class="auto", random_state=seed)) for c in [0.3, 1.0, 3.0]]
    else:
        for depth in [2, 3]:
            for lr in [0.05, 0.1]:
                configs.append(
                    (
                        f"hgb_d{depth}_lr{lr}",
                        HistGradientBoostingClassifier(
                            max_iter=80,
                            learning_rate=lr,
                            max_leaf_nodes=2**depth,
                            min_samples_leaf=15,
                            random_state=seed,
                        ),
                    )
                )
    if len(y) < 15 or min(Counter(y).values()) < 2:
        best_clf = configs[0][1]
    else:
        splitter = StratifiedKFold(n_splits=min(3, min(Counter(y).values())), shuffle=True, random_state=seed)
        best_score = float("-inf")
        best_clf = configs[0][1]
        for _, clf in configs:
            fold_scores = []
            for train_idx, val_idx in splitter.split(np.zeros(len(y)), y):
                pipe = Pipeline([("vec", DictVectorizer(sparse=False)), ("imputer", SimpleImputer(strategy="median")), ("clf", clf)])
                train_features = [features[int(i)] for i in train_idx]
                val_features = [features[int(i)] for i in val_idx]
                pipe.fit(train_features, y[train_idx])
                pred = pipe.predict(val_features)
                fold_scores.append(_prediction_utility(pred, labels.iloc[val_idx].reset_index(drop=True)))
            score = float(np.mean(fold_scores))
            if score > best_score:
                best_score = score
                best_clf = clf
    pipe = Pipeline([("vec", DictVectorizer(sparse=False)), ("imputer", SimpleImputer(strategy="median")), ("clf", best_clf)])
    pipe.fit(features, y)
    return GateFit(pipe, candidates, model_type=model_type, feature_mode=feature_mode)


def _predict_gate(gate: GateFit, features: list[dict[str, Any]], trace_ids: list[str]) -> pd.DataFrame:
    pred = gate.model.predict(features).astype(str)
    rows = []
    probs = None
    classes = []
    if hasattr(gate.model[-1], "predict_proba"):
        try:
            probs = gate.model.predict_proba(features)
            classes = [str(cls) for cls in gate.model[-1].classes_]
        except Exception:
            probs = None
    for i, trace_id in enumerate(trace_ids):
        row: dict[str, Any] = {
            "trace_id": trace_id,
            "predicted_candidate": pred[i],
            "uses_qwen": bool(pred[i] in QWEN_CANDIDATES),
        }
        if probs is not None:
            for col, cls in enumerate(classes):
                row[f"prob__{cls}"] = float(probs[i, col])
        rows.append(row)
    return pd.DataFrame(rows)


def _scores_for_split(adapters: dict[str, AdapterBundle], split_name: str, candidates: list[str]) -> dict[str, list[np.ndarray]]:
    attr = {
        "select": "select_scores_by_trace",
        "cal": "cal_scores_by_trace",
        "test": "test_scores_by_trace",
    }[split_name]
    return {candidate: getattr(adapters[candidate], attr) for candidate in candidates}


def _write_score_matrix(path: Path, traces_by_split: dict[str, list[TraceRecord]], scores_by_split: dict[str, dict[str, list[np.ndarray]]], candidates: list[str]) -> None:
    rows = []
    for split_name, traces in traces_by_split.items():
        for candidate in candidates:
            for trace, scores in zip(traces, scores_by_split[split_name][candidate]):
                for step_idx, score in enumerate(scores):
                    rows.append(
                        {
                            "split": split_name,
                            "trace_id": trace.trace_id,
                            "step_index": step_idx,
                            "candidate_name": candidate,
                            "score": float(score),
                            "uses_qwen": bool(candidate in QWEN_CANDIDATES),
                        }
                    )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _adapter_distribution(predictions: list[str]) -> dict[str, float]:
    if not predictions:
        return {}
    counts = Counter(predictions)
    total = float(sum(counts.values()))
    return {candidate: counts.get(candidate, 0) / total for candidate in ALL_CANDIDATES}


def _entropy(distribution: dict[str, float]) -> float:
    vals = np.asarray([value for value in distribution.values() if value > 0], dtype=float)
    return float(-np.sum(vals * np.log2(vals))) if len(vals) else 0.0


def _top_distribution(distribution: dict[str, float], n: int = 3) -> str:
    items = [(LABELS.get(candidate, candidate), frac) for candidate, frac in distribution.items() if frac > 0]
    items = sorted(items, key=lambda item: item[1], reverse=True)[:n]
    return ", ".join(f"{name} {100.0 * frac:.0f}%" for name, frac in items)


def _fixed_and_dataset_rows(
    dataset: str,
    seed: int,
    split: AdaptiveSplit,
    adapters: dict[str, AdapterBundle],
    grids: dict[str, np.ndarray],
    candidates: list[str],
    alphas: list[float],
) -> tuple[list[dict[str, Any]], dict[tuple[float, str], int]]:
    rows = []
    cal_indices: dict[tuple[float, str], int] = {}
    for alpha in alphas:
        selection_values = []
        for candidate in candidates:
            sel_idx, sel_risk = _select_index_for_candidate(split.select, adapters[candidate].select_scores_by_trace, grids[candidate], alpha)
            sel_metrics = _eval_candidate_at_index(split.select, adapters[candidate].select_scores_by_trace, grids[candidate], sel_idx, float(candidate in QWEN_CANDIDATES))
            cal_idx, cal_risk = _select_index_for_candidate(split.cal, adapters[candidate].cal_scores_by_trace, grids[candidate], alpha)
            cal_indices[(alpha, candidate)] = cal_idx
            metrics = _eval_candidate_at_index(split.test, adapters[candidate].test_scores_by_trace, grids[candidate], cal_idx, float(candidate in QWEN_CANDIDATES))
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "alpha": alpha,
                    "method": candidate,
                    "method_type": "fixed",
                    "gate_type": "",
                    "candidate_set": "fixed",
                    "cost_penalty": np.nan,
                    "selected_adapter": candidate,
                    "selected_adapter_distribution": json.dumps({candidate: 1.0}),
                    "selection_entropy": 0.0,
                    "cal_index": cal_idx,
                    "cal_corrected_risk": cal_risk,
                    **metrics,
                }
            )
            selection_values.append((candidate, sel_metrics["prefix_retained_fraction"], sel_risk))
        feasible = [item for item in selection_values if item[2] <= alpha]
        pool = feasible if feasible else selection_values
        selected = sorted(pool, key=lambda item: (-item[1], item[2], TIE_ORDER.get(item[0], 999)))[0][0]
        cal_idx, cal_risk = _select_index_for_candidate(split.cal, adapters[selected].cal_scores_by_trace, grids[selected], alpha)
        metrics = _eval_candidate_at_index(split.test, adapters[selected].test_scores_by_trace, grids[selected], cal_idx, float(selected in QWEN_CANDIDATES))
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "alpha": alpha,
                "method": "dataset_adaptive",
                "method_type": "dataset_adaptive",
                "gate_type": "",
                "candidate_set": "all",
                "cost_penalty": np.nan,
                "selected_adapter": selected,
                "selected_adapter_distribution": json.dumps({selected: 1.0}),
                "selection_entropy": 0.0,
                "cal_index": cal_idx,
                "cal_corrected_risk": cal_risk,
                **metrics,
            }
        )
    return rows, cal_indices


def _diagnostic_rows(
    dataset: str,
    seed: int,
    split: AdaptiveSplit,
    adapters: dict[str, AdapterBundle],
    grids: dict[str, np.ndarray],
    candidates: list[str],
    fixed_rows: list[dict[str, Any]],
    cal_indices: dict[tuple[float, str], int],
    alphas: list[float],
) -> list[dict[str, Any]]:
    rows = []
    fixed_by_alpha = {alpha: [row for row in fixed_rows if row["alpha"] == alpha and row["method_type"] == "fixed"] for alpha in alphas}
    for alpha in alphas:
        fixed = fixed_by_alpha[alpha]
        best = max(fixed, key=lambda row: (row["prefix_retained_fraction"], -row["prefix_contamination"]))
        best_cheap = max([row for row in fixed if row["method"] in CHEAP_CANDIDATES], key=lambda row: (row["prefix_retained_fraction"], -row["prefix_contamination"]))
        for name, src in [("best_fixed_adapter", best), ("best_cheap_adapter", best_cheap)]:
            row = dict(src)
            row["method"] = name
            row["method_type"] = "diagnostic"
            rows.append(row)
        lengths = []
        chosen = []
        for trace_idx, trace in enumerate(split.test):
            utilities: dict[str, float] = {}
            lengths_by_candidate: dict[str, int] = {}
            for candidate in candidates:
                idx = cal_indices[(alpha, candidate)]
                score = adapters[candidate].test_scores_by_trace[trace_idx]
                length = prefix_lengths([score], float(grids[candidate][idx]))[0]
                contaminated = int(length > (len(trace.steps) + 1 if trace.first_error is None else int(trace.first_error)))
                utilities[candidate] = float(length / max(len(trace.steps), 1)) - contaminated
                lengths_by_candidate[candidate] = int(length)
            candidate = _tie_break_candidate(utilities, candidates)
            chosen.append(candidate)
            lengths.append(lengths_by_candidate[candidate])
        distribution = _adapter_distribution(chosen)
        metrics = _metrics_from_lengths(split.test, np.asarray(lengths, dtype=int), qwen_call_rate=float(np.mean([c in QWEN_CANDIDATES for c in chosen])))
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "alpha": alpha,
                "method": "hindsight_per_trace_adapter",
                "method_type": "diagnostic",
                "gate_type": "",
                "candidate_set": "all",
                "cost_penalty": np.nan,
                "selected_adapter": "per_trace",
                "selected_adapter_distribution": json.dumps(distribution),
                "selection_entropy": _entropy(distribution),
                "cal_index": np.nan,
                "cal_corrected_risk": np.nan,
                **metrics,
            }
        )
    return rows


def _random_gate_predictions(trace_ids: list[str], candidates: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 811)
    pred = rng.choice(candidates, size=len(trace_ids), replace=True)
    return pd.DataFrame({"trace_id": trace_ids, "predicted_candidate": pred, "uses_qwen": [p in QWEN_CANDIDATES for p in pred]})


def _run_gate_method(
    dataset: str,
    seed: int,
    split: AdaptiveSplit,
    adapters: dict[str, AdapterBundle],
    grids: dict[str, np.ndarray],
    candidates: list[str],
    *,
    gate_name: str,
    feature_mode: str,
    cost_penalty: float,
    penalty: float,
    args,
    outdir: Path,
    alphas: list[float],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    split_scores = {
        "select": _scores_for_split(adapters, "select", candidates),
        "cal": _scores_for_split(adapters, "cal", candidates),
        "test": _scores_for_split(adapters, "test", candidates),
    }
    select_features = build_trace_gate_features(split.select, split_scores["select"], mode=feature_mode)
    cal_features = build_trace_gate_features(split.cal, split_scores["cal"], mode=feature_mode)
    test_features = build_trace_gate_features(split.test, split_scores["test"], mode=feature_mode)
    if gate_name.startswith("random_gate"):
        cal_pred = _random_gate_predictions(_trace_ids(split.cal), candidates, seed)
        test_pred = _random_gate_predictions(_trace_ids(split.test), candidates, seed + 1)
        gate = None
        labels = pd.DataFrame()
    else:
        labels = make_crossfit_gate_labels(
            split.select,
            split_scores["select"],
            grids,
            candidates,
            alpha=ALPHA_MAIN,
            penalty=penalty,
            cost_penalty=cost_penalty,
            n_folds=args.n_folds,
            seed=seed,
        )
        gate = train_trace_gate(select_features, labels, candidates, seed=seed, model_type=args.gate_model, feature_mode=feature_mode)
        cal_pred = _predict_gate(gate, cal_features, _trace_ids(split.cal))
        test_pred = _predict_gate(gate, test_features, _trace_ids(split.test))
    gate_dir = ensure_dir(outdir / "gates" / _slug(dataset) / str(seed))
    labels_path = gate_dir / f"{gate_name}_labels.csv"
    if not labels.empty:
        labels.to_csv(labels_path, index=False)
    if gate is not None:
        with (gate_dir / f"{gate_name}.pkl").open("wb") as f:
            pickle.dump(gate, f)
    cal_pred.to_csv(gate_dir / f"{gate_name}_cal_predictions.csv", index=False)
    test_pred.to_csv(gate_dir / f"{gate_name}_test_predictions.csv", index=False)

    rows = []
    pred_list_cal = cal_pred["predicted_candidate"].astype(str).tolist()
    pred_list_test = test_pred["predicted_candidate"].astype(str).tolist()
    distribution = _adapter_distribution(pred_list_test)
    for alpha in alphas:
        idx, cal_risk = _calibrate_gated_index(split.cal, pred_list_cal, split_scores["cal"], grids, alpha)
        metrics = _evaluate_gated(split.test, pred_list_test, split_scores["test"], grids, idx)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "alpha": alpha,
                "method": gate_name,
                "method_type": "trace_gate",
                "gate_type": gate_name,
                "candidate_set": "_".join(candidates),
                "cost_penalty": cost_penalty,
                "selected_adapter": "trace_conditioned",
                "selected_adapter_distribution": json.dumps(distribution),
                "selection_entropy": _entropy(distribution),
                "cal_index": idx,
                "cal_corrected_risk": cal_risk,
                **metrics,
            }
        )
    assignments = []
    for split_name, pred_df in [("cal", cal_pred), ("test", test_pred)]:
        tmp = pred_df.copy()
        tmp["dataset"] = dataset
        tmp["seed"] = seed
        tmp["split"] = split_name
        tmp["gate_type"] = gate_name
        tmp["feature_mode"] = feature_mode
        tmp["cost_penalty"] = cost_penalty
        assignments.append(tmp)
    return rows, assignments


def _dataset_run(args, dataset: str, text_features: str, combined_features: str, qwen_csv: str, seeds: list[int], outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = load_many_npz([combined_features], ["mixed"], allow_nan=True)
    text = load_many_npz([text_features], ["mixed"], allow_nan=True)
    scores_by_trace_id = _read_qwen_scores(qwen_csv, args.qwen_score_col)
    views = _build_views(combined, text, scores_by_trace_id)
    all_rows = []
    assignment_frames = []
    spec_by_score = {spec.score: spec for spec in ADAPTER_SPECS}
    specs = [spec_by_score[name] for name in ["random", *ALL_CANDIDATES]]
    for seed in seeds:
        print(f"Running trace-conditioned gates for {dataset} seed={seed}", flush=True)
        reference = split_traces_four_way(
            combined,
            train_frac=args.score_train_frac,
            select_frac=args.gate_select_frac,
            cal_frac=args.cpcc_calibration_frac,
            test_frac=args.test_frac,
            seed=seed,
        )
        split_by_view = {name: _adaptive_split_like(reference, traces) for name, traces in views.items()}
        adapters: dict[str, AdapterBundle] = {}
        for spec in specs:
            split = reference if spec.source == "qwen" else split_by_view[spec.view]
            adapters[spec.score] = _fit_adapter(spec, split, seed, args.class_weight, scores_by_trace_id)
        candidates = list(ALL_CANDIDATES)
        grids = _candidate_threshold_grids(adapters, candidates, args.threshold_grid_size)
        fixed_rows, cal_indices = _fixed_and_dataset_rows(dataset, seed, reference, adapters, grids, candidates, args.alpha_grid)
        all_rows.extend(fixed_rows)
        all_rows.extend(_diagnostic_rows(dataset, seed, reference, adapters, grids, candidates, fixed_rows, cal_indices, args.alpha_grid))
        gate_specs: list[tuple[str, list[str], str, float]] = [
            ("trace_cheap_all", CANDIDATE_SETS["all"], "cheap", 0.0),
            ("trace_full_all", CANDIDATE_SETS["all"], "full", 0.0),
            ("trace_cheap_cheap_only", CANDIDATE_SETS["cheap_only"], "cheap", 0.0),
            ("trace_cheap_qwen_backed_only", CANDIDATE_SETS["qwen_backed_only"], "full", 0.0),
            ("trace_cheap_no_hazard", CANDIDATE_SETS["no_hazard"], "cheap", 0.0),
            ("trace_cheap_no_prefix", CANDIDATE_SETS["no_prefix"], "cheap", 0.0),
            ("trace_cheap_no_step", CANDIDATE_SETS["no_step"], "cheap", 0.0),
            ("random_gate_all", CANDIDATE_SETS["all"], "cheap", 0.0),
        ]
        for cost in args.cost_grid:
            gate_specs.append((f"trace_cost_c{cost:.2f}_all", CANDIDATE_SETS["all"], "cheap", float(cost)))
        if args.only_gate_methods:
            allowed_gate_methods = set(args.only_gate_methods)
            gate_specs = [spec for spec in gate_specs if spec[0] in allowed_gate_methods]
        seen = set()
        for gate_name, cand_set, feature_mode, cost in gate_specs:
            key = (gate_name, tuple(cand_set), feature_mode, cost)
            if key in seen:
                continue
            seen.add(key)
            grids_subset = {candidate: grids[candidate] for candidate in cand_set}
            rows, assignments = _run_gate_method(
                dataset,
                seed,
                reference,
                adapters,
                grids_subset,
                cand_set,
                gate_name=gate_name,
                feature_mode=feature_mode,
                cost_penalty=cost,
                penalty=args.gate_penalty,
                args=args,
                outdir=outdir,
                alphas=args.alpha_grid,
            )
            all_rows.extend(rows)
            assignment_frames.extend(assignments)
        if args.write_score_matrices:
            scores_dir = ensure_dir(outdir / "scores" / _slug(dataset) / str(seed))
            split_scores = {
                "select": _scores_for_split(adapters, "select", candidates),
                "cal": _scores_for_split(adapters, "cal", candidates),
                "test": _scores_for_split(adapters, "test", candidates),
            }
            _write_score_matrix(
                scores_dir / "candidate_scores.parquet",
                {"select": reference.select, "cal": reference.cal, "test": reference.test},
                split_scores,
                candidates,
            )
    metrics = pd.DataFrame(all_rows)
    assignments = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()
    return metrics, assignments


def _add_external_cascade_baseline(metrics: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    path = Path("outputs/adaptive_adapters_extensions/cost_aware_cascades/selected_cost_frontier.csv")
    if not path.exists():
        return metrics
    cascade = pd.read_csv(path)
    cascade = cascade[np.isclose(cascade["cost_lambda"], 0.10)].copy()
    if cascade.empty:
        return metrics
    rows = []
    for row in cascade.itertuples(index=False):
        rows.append(
            {
                "dataset": row.dataset,
                "seed": int(row.seed),
                "alpha": float(row.alpha),
                "method": "existing_cost_aware_cascade_c0.10",
                "method_type": "external_baseline",
                "gate_type": "",
                "candidate_set": "",
                "cost_penalty": 0.10,
                "selected_adapter": "cascade",
                "selected_adapter_distribution": "{}",
                "selection_entropy": np.nan,
                "cal_index": np.nan,
                "cal_corrected_risk": row.prefix_cal_corrected_risk,
                "prefix_contamination": row.prefix_contamination,
                "prefix_retained_fraction": row.prefix_retained_fraction,
                "prefix_retained_steps": row.prefix_retained_steps,
                "prefix_full_trace_rate": row.prefix_full_trace_rate,
                "accepted_error_rate": np.nan,
                "marginal_full_trace_false_accept": np.nan,
                "qwen_call_rate": row.qwen_call_rate,
            }
        )
    return pd.concat([metrics, pd.DataFrame(rows)], ignore_index=True)


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, mean, mean
    half = float(1.96 * np.std(values, ddof=1) / math.sqrt(len(values)))
    return mean, mean - half, mean + half


PAIRED_DELTA_COLUMNS = [
    "dataset",
    "alpha",
    "comparison",
    "method_a",
    "method_b",
    "n_paired_splits",
    "delta_kept_mean",
    "delta_kept_ci_low",
    "delta_kept_ci_high",
    "delta_risk_mean",
    "delta_risk_ci_low",
    "delta_risk_ci_high",
]


def _paired_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("trace_full_all", "dataset_adaptive", "Trace-gated full - dataset-adaptive"),
        ("trace_cheap_all", "dataset_adaptive", "Trace-gated cheap - dataset-adaptive"),
        ("trace_cost_c0.10_all", "dataset_adaptive", "Trace-gated cost-aware - dataset-adaptive"),
        ("trace_full_all", "best_fixed_adapter", "Trace-gated full - best fixed adapter"),
        ("trace_cost_c0.10_all", "best_cheap_adapter", "Trace-gated cost-aware - best cheap"),
        ("trace_cost_c0.10_all", "qwen_prm", "Trace-gated cost-aware - raw Qwen"),
        ("trace_cost_c0.10_all", "existing_cost_aware_cascade_c0.10", "Trace-gated cost-aware - existing cascade"),
    ]
    rows = []
    for (dataset, alpha), group in metrics.groupby(["dataset", "alpha"], dropna=False):
        by_method = {method: sub.set_index("seed") for method, sub in group.groupby("method")}
        for left, right, label in comparisons:
            if left not in by_method or right not in by_method:
                continue
            seeds = sorted(set(by_method[left].index) & set(by_method[right].index))
            if not seeds:
                continue
            kept = by_method[left].loc[seeds, "prefix_retained_fraction"].to_numpy(float) - by_method[right].loc[seeds, "prefix_retained_fraction"].to_numpy(float)
            risk = by_method[left].loc[seeds, "prefix_contamination"].to_numpy(float) - by_method[right].loc[seeds, "prefix_contamination"].to_numpy(float)
            kept_mean, kept_lo, kept_hi = _mean_ci(kept)
            risk_mean, risk_lo, risk_hi = _mean_ci(risk)
            rows.append(
                {
                    "dataset": dataset,
                    "alpha": alpha,
                    "comparison": label,
                    "method_a": left,
                    "method_b": right,
                    "n_paired_splits": len(seeds),
                    "delta_kept_mean": kept_mean,
                    "delta_kept_ci_low": kept_lo,
                    "delta_kept_ci_high": kept_hi,
                    "delta_risk_mean": risk_mean,
                    "delta_risk_ci_low": risk_lo,
                    "delta_risk_ci_high": risk_hi,
                }
            )
    return pd.DataFrame(rows, columns=PAIRED_DELTA_COLUMNS)


def _summary_row(summary: pd.DataFrame, dataset: str, method: str, alpha: float = ALPHA_MAIN) -> pd.Series | None:
    sub = summary[(summary["dataset"] == dataset) & (summary["method"] == method) & np.isclose(summary["alpha"], alpha)]
    if sub.empty:
        return None
    return sub.iloc[0]


def summarize_outputs(outdir: Path) -> None:
    tables_dir = ensure_dir(outdir / "tables")
    figures_dir = ensure_dir(outdir / "figures")
    metrics = pd.read_csv(outdir / "raw_split_metrics.csv")
    metrics = _add_external_cascade_baseline(metrics, outdir)
    metrics.to_csv(outdir / "raw_split_metrics_with_external_baselines.csv", index=False)
    assignments = pd.read_csv(outdir / "gate_assignments.csv") if (outdir / "gate_assignments.csv").exists() else pd.DataFrame()
    summary = _summarize(metrics, ["dataset", "alpha", "method", "method_type", "gate_type"])
    summary.to_csv(tables_dir / "table_trace_gated_summary.csv", index=False)
    headline_rows = []
    for dataset in DATASET_ORDER:
        da = _summary_row(summary, dataset, "dataset_adaptive")
        cheap = _summary_row(summary, dataset, "trace_cheap_all")
        full = _summary_row(summary, dataset, "trace_full_all")
        cost = _summary_row(summary, dataset, "trace_cost_c0.10_all")
        best = _summary_row(summary, dataset, "best_fixed_adapter")
        hindsight = _summary_row(summary, dataset, "hindsight_per_trace_adapter")
        qwen = _summary_row(summary, dataset, "qwen_prm")
        best_cheap = _summary_row(summary, dataset, "best_cheap_adapter")
        if da is None:
            continue
        headline_rows.append(
            {
                "dataset": dataset,
                "dataset_adaptive_kept": da.prefix_retained_fraction_mean,
                "dataset_adaptive_risk": da.prefix_contamination_mean,
                "trace_gated_cheap_kept": cheap.prefix_retained_fraction_mean if cheap is not None else np.nan,
                "trace_gated_cheap_risk": cheap.prefix_contamination_mean if cheap is not None else np.nan,
                "trace_gated_cheap_qwen_call": cheap.qwen_call_rate_mean if cheap is not None else np.nan,
                "trace_gated_full_kept": full.prefix_retained_fraction_mean if full is not None else np.nan,
                "trace_gated_full_risk": full.prefix_contamination_mean if full is not None else np.nan,
                "trace_gated_full_qwen_call": full.qwen_call_rate_mean if full is not None else np.nan,
                "trace_gated_cost_kept": cost.prefix_retained_fraction_mean if cost is not None else np.nan,
                "trace_gated_cost_risk": cost.prefix_contamination_mean if cost is not None else np.nan,
                "trace_gated_cost_qwen_call": cost.qwen_call_rate_mean if cost is not None else np.nan,
                "best_fixed_kept": best.prefix_retained_fraction_mean if best is not None else np.nan,
                "best_fixed_risk": best.prefix_contamination_mean if best is not None else np.nan,
                "hindsight_per_trace_kept": hindsight.prefix_retained_fraction_mean if hindsight is not None else np.nan,
                "hindsight_per_trace_risk": hindsight.prefix_contamination_mean if hindsight is not None else np.nan,
                "qwen_prm_kept": qwen.prefix_retained_fraction_mean if qwen is not None else np.nan,
                "qwen_prm_risk": qwen.prefix_contamination_mean if qwen is not None else np.nan,
                "best_cheap_kept": best_cheap.prefix_retained_fraction_mean if best_cheap is not None else np.nan,
                "best_cheap_risk": best_cheap.prefix_contamination_mean if best_cheap is not None else np.nan,
                "main_selected_families": _main_distribution(assignments, dataset, "trace_cost_c0.10_all"),
            }
        )
    headline = pd.DataFrame(headline_rows)
    headline.to_csv(tables_dir / "table_trace_gated_headline.csv", index=False)
    headline.to_markdown(tables_dir / "table_trace_gated_headline.md", index=False)
    tex_rows = []
    for row in headline.itertuples(index=False):
        gap = row.best_fixed_kept - row.trace_gated_full_kept
        tex_rows.append(
            f"{_tex(row.dataset)} & {_fmt_pct(row.dataset_adaptive_kept)} ({_fmt_pct(row.dataset_adaptive_risk)}) & "
            f"{_fmt_pct(row.trace_gated_full_kept)} ({_fmt_pct(row.trace_gated_full_risk)}) & "
            f"{_fmt_pct(row.trace_gated_cost_kept)} ({_fmt_pct(row.trace_gated_cost_risk)}) & "
            f"{_fmt_pct(row.trace_gated_cost_qwen_call)} & {_fmt_pct(row.best_fixed_kept)} & "
            f"{100.0 * gap:.1f} & {_tex(row.main_selected_families)} \\\\"
        )
    _write_simple_tex_table(
        tables_dir / "table_trace_gated_headline.tex",
        "Trace-conditioned adaptive CPCC at $\\alpha=0.05$. Entries report prefix kept with empirical prefix risk in parentheses. Gap is relative to the hindsight best fixed adapter.",
        "tab:trace_gated_headline",
        "lrrrrrrl",
        r"Dataset & Dataset-adapt & Trace-gated & Trace-gated cost & Qwen calls & Best fixed & Gap & Main selected \\",
        tex_rows,
    )
    shutil.copyfile(tables_dir / "table_trace_gated_headline.tex", ensure_dir("tables") / "table_trace_gated_headline.tex")
    deltas = _paired_deltas(metrics)
    deltas.to_csv(tables_dir / "table_trace_gated_paired_deltas.csv", index=False)
    tex_rows = []
    focus = deltas[np.isclose(deltas["alpha"], ALPHA_MAIN)].copy()
    for row in focus.itertuples(index=False):
        tex_rows.append(
            f"{_tex(row.dataset)} & {_tex(row.comparison)} & {100.0 * row.delta_kept_mean:+.1f} & "
            f"[{100.0 * row.delta_kept_ci_low:+.1f}, {100.0 * row.delta_kept_ci_high:+.1f}] & {100.0 * row.delta_risk_mean:+.1f} \\\\"
        )
    _write_simple_tex_table(
        tables_dir / "table_trace_gated_paired_deltas.tex",
        "Paired split-level deltas for trace-conditioned CPCC at $\\alpha=0.05$.",
        "tab:trace_gated_paired_deltas",
        "llrrr",
        r"Dataset & Comparison & $\Delta$ kept & 95\% CI & $\Delta$ risk \\",
        tex_rows,
    )
    _selection_distribution(assignments, tables_dir)
    _gate_ablation(summary, tables_dir)
    _cost_frontier(metrics, summary, tables_dir)
    _figures(headline, metrics, assignments, figures_dir)
    _write_analysis(outdir, headline, deltas, metrics)


def _main_distribution(assignments: pd.DataFrame, dataset: str, gate_type: str) -> str:
    if assignments.empty:
        return ""
    sub = assignments[(assignments["dataset"] == dataset) & (assignments["gate_type"] == gate_type) & (assignments["split"] == "test")]
    if sub.empty:
        return ""
    return _top_distribution(_adapter_distribution(sub["predicted_candidate"].astype(str).tolist()))


def _selection_distribution(assignments: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    rows = []
    if not assignments.empty:
        for (dataset, gate_type), group in assignments[assignments["split"] == "test"].groupby(["dataset", "gate_type"], dropna=False):
            dist = _adapter_distribution(group["predicted_candidate"].astype(str).tolist())
            row: dict[str, Any] = {"dataset": dataset, "gate_type": gate_type, "entropy": _entropy(dist)}
            for candidate in ALL_CANDIDATES:
                row[f"frac_{candidate}"] = dist.get(candidate, 0.0)
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(tables_dir / "table_gate_selection_distribution.csv", index=False)
    focus = table[table["gate_type"].isin(["trace_cheap_all", "trace_full_all", "trace_cost_c0.10_all"])].copy() if not table.empty else table
    tex_rows = []
    for row in focus.itertuples(index=False):
        tex_rows.append(
            f"{_tex(row.dataset)} & {_tex(row.gate_type)} & {_fmt_pct(getattr(row, 'frac_token_format', 0.0))} & "
            f"{_fmt_pct(getattr(row, 'frac_step_combined', 0.0))} & {_fmt_pct(getattr(row, 'frac_prefix_combined', 0.0))} & "
            f"{_fmt_pct(getattr(row, 'frac_hazard_combined', 0.0))} & {_fmt_pct(getattr(row, 'frac_qwen_prm', 0.0))} & "
            f"{_fmt_pct(getattr(row, 'frac_step_qwen', 0.0))} & {_fmt_pct(getattr(row, 'frac_prefix_qwen', 0.0))} & "
            f"{_fmt_pct(getattr(row, 'frac_hazard_qwen', 0.0))} & {row.entropy:.2f} \\\\"
        )
    _write_simple_tex_table(
        tables_dir / "table_gate_selection_distribution.tex",
        "Trace-gate selected adapter distributions on held-out test traces.",
        "tab:gate_selection_distribution",
        "llrrrrrrrrr",
        r"Dataset & Gate & Token & Step & Prefix & Hazard & Qwen & Step+Q & Prefix+Q & Hazard+Q & Ent. \\",
        tex_rows,
    )
    return table


def _gate_ablation(summary: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    methods = [
        "dataset_adaptive",
        "trace_cheap_all",
        "trace_full_all",
        "trace_cheap_cheap_only",
        "trace_cheap_qwen_backed_only",
        "trace_cheap_no_hazard",
        "trace_cheap_no_prefix",
        "trace_cheap_no_step",
        "random_gate_all",
    ]
    rows = []
    for dataset in DATASET_ORDER:
        best = _summary_row(summary, dataset, "best_fixed_adapter")
        hindsight = _summary_row(summary, dataset, "hindsight_per_trace_adapter")
        for method in methods:
            row = _summary_row(summary, dataset, method)
            if row is None:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "prefix_contamination": row.prefix_contamination_mean,
                    "prefix_retained_fraction": row.prefix_retained_fraction_mean,
                    "qwen_call_rate": row.qwen_call_rate_mean,
                    "selection_entropy": row.selection_entropy_mean,
                    "gap_to_best_fixed": (best.prefix_retained_fraction_mean - row.prefix_retained_fraction_mean) if best is not None else np.nan,
                    "gap_to_hindsight_per_trace": (hindsight.prefix_retained_fraction_mean - row.prefix_retained_fraction_mean) if hindsight is not None else np.nan,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(tables_dir / "table_gate_ablation.csv", index=False)
    tex_rows = [
        f"{_tex(row.dataset)} & {_tex(row.method)} & {_fmt_pct(row.prefix_contamination)} & {_fmt_pct(row.prefix_retained_fraction)} & "
        f"{_fmt_pct(row.qwen_call_rate)} & {row.selection_entropy:.2f} & {100.0 * row.gap_to_best_fixed:.1f} & "
        f"{100.0 * row.gap_to_hindsight_per_trace:.1f} \\\\"
        for row in table.itertuples(index=False)
    ]
    _write_simple_tex_table(
        tables_dir / "table_gate_ablation.tex",
        "Trace-gate ablations at $\\alpha=0.05$.",
        "tab:gate_ablation",
        "llrrrrrr",
        r"Dataset & Method & Risk & Kept & Qwen calls & Ent. & Gap fixed & Gap trace \\",
        tex_rows,
    )
    return table


def _cost_frontier(metrics: pd.DataFrame, summary: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    rows = []
    for dataset in DATASET_ORDER:
        best_cheap = _summary_row(summary, dataset, "best_cheap_adapter")
        qwen_methods = [_summary_row(summary, dataset, method) for method in QWEN_CANDIDATES]
        qwen_methods = [row for row in qwen_methods if row is not None]
        strong_kept = max([row.prefix_retained_fraction_mean for row in qwen_methods], default=np.nan)
        cheap_kept = best_cheap.prefix_retained_fraction_mean if best_cheap is not None else np.nan
        denom = strong_kept - cheap_kept
        for method, group in summary[(summary["dataset"] == dataset) & (summary["method"].str.startswith("trace_cost_c")) & np.isclose(summary["alpha"], ALPHA_MAIN)].groupby("method"):
            row = group.iloc[0]
            invalid = (not np.isfinite(denom)) or denom <= 0.02
            rows.append(
                {
                    "dataset": dataset,
                    "cost_penalty": row.cost_penalty_mean if "cost_penalty_mean" in row else _method_cost(method),
                    "method": method,
                    "prefix_contamination": row.prefix_contamination_mean,
                    "prefix_retained_fraction": row.prefix_retained_fraction_mean,
                    "qwen_call_rate": row.qwen_call_rate_mean,
                    "prefix_full_trace_rate": row.prefix_full_trace_rate_mean,
                    "gain_recovery_denominator": denom,
                    "gain_recovery_unstable": invalid,
                    "gain_recovered": np.nan if invalid else (row.prefix_retained_fraction_mean - cheap_kept) / denom,
                }
            )
    table = pd.DataFrame(rows).sort_values(["dataset", "cost_penalty"])
    table.to_csv(tables_dir / "table_trace_gated_cost_frontier.csv", index=False)
    tex_rows = []
    for row in table.itertuples(index=False):
        gain = "--" if row.gain_recovery_unstable else _fmt_pct(row.gain_recovered)
        tex_rows.append(
            f"{_tex(row.dataset)} & {row.cost_penalty:.2f} & {_fmt_pct(row.prefix_contamination)} & {_fmt_pct(row.prefix_retained_fraction)} & "
            f"{_fmt_pct(row.qwen_call_rate)} & {_fmt_pct(row.prefix_full_trace_rate)} & {gain} \\\\"
        )
    _write_simple_tex_table(
        tables_dir / "table_trace_gated_cost_frontier.tex",
        "Cost-aware trace-gated CPCC frontier at $\\alpha=0.05$.",
        "tab:trace_gated_cost_frontier",
        "lrrrrrr",
        r"Dataset & Cost & Risk & Kept & Qwen calls & Full accept & Gain rec. \\",
        tex_rows,
    )
    return table


def _method_cost(method: str) -> float:
    try:
        return float(method.split("_c", 1)[1].split("_", 1)[0])
    except Exception:
        return float("nan")


def _figures(headline: pd.DataFrame, metrics: pd.DataFrame, assignments: pd.DataFrame, figures_dir: Path) -> None:
    cost = metrics[(metrics["method"].str.startswith("trace_cost_c")) & np.isclose(metrics["alpha"], ALPHA_MAIN)].copy()
    cost["cost_penalty"] = cost["method"].map(_method_cost)
    summary = _summarize(cost, ["dataset", "cost_penalty"])
    for target_only, filename, datasets in [
        (True, "fig_trace_gated_cost_frontier_target.pdf", ["Target"]),
        (False, "fig_trace_gated_cost_frontier_external.pdf", [d for d in DATASET_ORDER if d != "Target"]),
    ]:
        fig, ax = plt.subplots(figsize=(6.2 if target_only else 7.5, 3.7))
        for dataset in datasets:
            sub = summary[summary["dataset"] == dataset].sort_values("cost_penalty")
            if sub.empty:
                continue
            ax.plot(100.0 * sub["qwen_call_rate_mean"], 100.0 * sub["prefix_retained_fraction_mean"], marker="o", label=dataset)
            if target_only:
                for row in sub.itertuples(index=False):
                    ax.text(100.0 * row.qwen_call_rate_mean, 100.0 * row.prefix_retained_fraction_mean, f"{row.cost_penalty:.2g}", fontsize=7)
        ax.set_xlabel("Qwen call rate (%)")
        ax.set_ylabel("Prefix kept (%)")
        ax.grid(True, alpha=0.25, linewidth=0.7)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(figures_dir / filename)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    methods = [
        ("dataset_adaptive_kept", "Dataset-adapt"),
        ("trace_gated_cheap_kept", "Gate cheap"),
        ("trace_gated_full_kept", "Gate full"),
        ("trace_gated_cost_kept", "Gate cost"),
    ]
    x = np.arange(len(headline))
    width = 0.18
    for idx, (col, label) in enumerate(methods):
        gaps = 100.0 * (headline["best_fixed_kept"] - headline[col])
        ax.bar(x + (idx - 1.5) * width, gaps, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(headline["dataset"].astype(str), rotation=20, ha="right")
    ax.set_ylabel("Gap to best fixed (pp)")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_trace_gated_gap_to_best_fixed.pdf")
    plt.close(fig)
    if not assignments.empty:
        heat = []
        for dataset in DATASET_ORDER:
            sub = assignments[(assignments["dataset"] == dataset) & (assignments["gate_type"] == "trace_cost_c0.10_all") & (assignments["split"] == "test")]
            heat.append([_adapter_distribution(sub["predicted_candidate"].astype(str).tolist()).get(candidate, 0.0) for candidate in ALL_CANDIDATES])
        fig, ax = plt.subplots(figsize=(8.2, 3.8))
        im = ax.imshow(np.asarray(heat), aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_yticks(np.arange(len(DATASET_ORDER)))
        ax.set_yticklabels(DATASET_ORDER)
        ax.set_xticks(np.arange(len(ALL_CANDIDATES)))
        ax.set_xticklabels([LABELS.get(c, c) for c in ALL_CANDIDATES], rotation=30, ha="right")
        fig.colorbar(im, ax=ax, label="Fraction assigned")
        fig.tight_layout()
        fig.savefig(figures_dir / "fig_gate_selection_heatmap.pdf")
        plt.close(fig)


def _write_analysis(outdir: Path, headline: pd.DataFrame, deltas: pd.DataFrame, metrics: pd.DataFrame) -> None:
    lines = ["# Trace-Conditioned Adaptive CPCC", ""]
    lines.append("## Headline")
    lines.append("")
    show = headline.copy()
    for col in show.columns:
        if col != "dataset" and pd.api.types.is_numeric_dtype(show[col]):
            show[col] = (100.0 * show[col]).round(2) if "kept" in col or "risk" in col or "qwen_call" in col else show[col]
    lines.append(show.to_markdown(index=False))
    lines.append("")
    focus = deltas[(deltas["comparison"].isin(["Trace-gated full - dataset-adaptive", "Trace-gated cost-aware - dataset-adaptive"])) & np.isclose(deltas["alpha"], ALPHA_MAIN)]
    lines.append("## Paired Deltas vs Dataset-Adaptive")
    lines.append("")
    if not focus.empty:
        table = focus[["dataset", "comparison", "delta_kept_mean", "delta_kept_ci_low", "delta_kept_ci_high", "delta_risk_mean"]].copy()
        for col in ["delta_kept_mean", "delta_kept_ci_low", "delta_kept_ci_high", "delta_risk_mean"]:
            table[col] = (100.0 * table[col]).round(2)
        lines.append(table.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Promote trace-conditioned adaptive CPCC only if it meets the utility, cost, or heterogeneity criteria in the implementation spec. Otherwise, keep it as an appendix diagnostic and retain dataset-level adaptive CPCC as the main method.")
    (outdir / "ANALYSIS.md").write_text("\n".join(lines))


def run(args) -> None:
    outdir = ensure_dir(args.output_dir)
    all_metrics = []
    all_assignments = []
    for dataset, text_features, combined_features, qwen_csv, seeds in _dataset_configs(args):
        metrics, assignments = _dataset_run(args, dataset, text_features, combined_features, qwen_csv, seeds, outdir)
        dataset_dir = ensure_dir(outdir / _slug(dataset))
        metrics.to_csv(dataset_dir / "raw_split_metrics.csv", index=False)
        assignments.to_csv(dataset_dir / "gate_assignments.csv", index=False)
        all_metrics.append(metrics)
        all_assignments.append(assignments)
    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    assignments = pd.concat(all_assignments, ignore_index=True) if all_assignments else pd.DataFrame()
    metrics.to_csv(outdir / "raw_split_metrics.csv", index=False)
    assignments.to_csv(outdir / "gate_assignments.csv", index=False)
    write_json(outdir / "run_config.json", vars(args))
    if not getattr(args, "skip_summary", False):
        summarize_outputs(outdir)


def merge_part_dirs(part_dirs: list[str], outdir: Path) -> None:
    ensure_dir(outdir)
    metrics_frames = []
    assignment_frames = []
    for raw_part in part_dirs:
        part = Path(raw_part)
        metrics_path = part / "raw_split_metrics.csv"
        assignments_path = part / "gate_assignments.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing {metrics_path}")
        metrics_frames.append(pd.read_csv(metrics_path))
        if assignments_path.exists():
            assignment_frames.append(pd.read_csv(assignments_path))
        for child in part.iterdir():
            if child.is_dir() and child.name not in {"tables", "figures"}:
                shutil.copytree(child, outdir / child.name, dirs_exist_ok=True)
    metrics = pd.concat(metrics_frames, ignore_index=True)
    assignments = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()
    metrics.to_csv(outdir / "raw_split_metrics.csv", index=False)
    assignments.to_csv(outdir / "gate_assignments.csv", index=False)
    write_json(outdir / "merged_parts.json", {"part_dirs": [str(Path(p)) for p in part_dirs]})
    summarize_outputs(outdir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/trace_conditioned_adaptive_cpcc")
    parser.add_argument("--qwen_score_col", default="qwen_prm_error")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--target_seeds", nargs="*", type=int, default=list(range(2806, 2826)))
    parser.add_argument("--external_seeds", nargs="*", type=int, default=list(range(2806, 2816)))
    parser.add_argument("--alpha_grid", nargs="*", type=float, default=ALPHA_GRID)
    parser.add_argument("--cost_grid", nargs="*", type=float, default=[0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50])
    parser.add_argument("--threshold_grid_size", type=int, default=201)
    parser.add_argument("--score_train_frac", type=float, default=0.40)
    parser.add_argument("--gate_select_frac", type=float, default=0.20)
    parser.add_argument("--cpcc_calibration_frac", type=float, default=0.20)
    parser.add_argument("--test_frac", type=float, default=0.20)
    parser.add_argument("--class_weight", default="balanced")
    parser.add_argument("--gate_model", choices=["hgb", "logistic"], default="hgb")
    parser.add_argument("--gate_penalty", type=float, default=1.0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--write_score_matrices", action="store_true")
    parser.add_argument("--merge_part_dirs", nargs="*", default=None)
    parser.add_argument("--only_gate_methods", nargs="*", default=None)
    parser.add_argument("--skip_summary", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.merge_part_dirs:
        merge_part_dirs(args.merge_part_dirs, ensure_dir(args.output_dir))
        print(f"Wrote trace-conditioned outputs to {args.output_dir}", flush=True)
        return
    if args.quick:
        args.target_seeds = args.target_seeds[:1]
        args.external_seeds = args.external_seeds[:1]
        args.threshold_grid_size = min(args.threshold_grid_size, 51)
        args.cost_grid = [0.0, 0.10, 0.30]
    total = args.score_train_frac + args.gate_select_frac + args.cpcc_calibration_frac + args.test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")
    run(args)
    print(f"Wrote trace-conditioned outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
