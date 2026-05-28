"""Annotation-protocol adaptive CPCC score-adapter experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from crop.data import TraceRecord, load_many_npz
from crop.experiments.common import ScoreBundle, build_score_bundle
from crop.experiments.exp09_process_repeated import (
    COE_SCORE_COLUMNS,
    _artifact_views,
    _fit_model_bundle,
)
from crop.experiments.exp15_prefix_aware import (
    _qwen_bundle,
    _read_qwen_scores,
    _score_map_coverage,
    _summarize_prefix_aware,
)
from crop.metrics import full_trace_accept_rate, prefix_contamination_rate, safe_aupr, safe_auroc
from crop.models import fit_verifier, make_model, scores_by_trace_from_model
from crop.prefix_aware import (
    append_trace_score_feature,
    augment_with_prefix_features,
    flatten_hazard_labels,
    flatten_prefix_labels,
    select_named_feature_columns,
    traces_with_hazard_targets,
    traces_with_prefix_targets,
)
from crop.risk_control import corrected_risk, prefix_lengths, prefix_losses_by_lambda, select_lambda_crc
from crop.splits import Split, flatten_steps
from crop.utils import ensure_dir, write_json


warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class AdaptiveSplit:
    train: list[TraceRecord]
    select: list[TraceRecord]
    cal: list[TraceRecord]
    test: list[TraceRecord]


@dataclass(frozen=True)
class AdapterSpec:
    score: str
    label: str
    target: str
    family: str
    view: str
    source: str


@dataclass
class AdapterBundle:
    spec: AdapterSpec
    select_scores_by_trace: list[np.ndarray]
    cal_scores_by_trace: list[np.ndarray]
    test_scores_by_trace: list[np.ndarray]
    select_step_scores: np.ndarray
    cal_step_scores: np.ndarray
    test_step_scores: np.ndarray
    fit_seconds: float


ADAPTER_SPECS = [
    AdapterSpec("random", "Random", "random", "frozen", "combined", "random"),
    AdapterSpec("token_format", "Token/format", "artifact", "artifact", "artifact_token_formatting", "step"),
    AdapterSpec("qwen_prm", "Qwen PRM", "external PRM", "frozen", "combined", "qwen"),
    AdapterSpec("step_combined", "Step combined", "$Y_t$", "cheap", "combined", "step"),
    AdapterSpec("prefix_combined", "Prefix combined", "$C_t$", "cheap", "prefix_combined", "prefix"),
    AdapterSpec("hazard_combined", "Hazard combined", "$H_t$", "cheap", "prefix_combined", "hazard"),
    AdapterSpec("step_qwen", "Step+Qwen", "$Y_t$ + Qwen", "qwen_adapter", "prefix_qwen_combined", "step"),
    AdapterSpec("prefix_qwen", "Prefix+Qwen", "$C_t$ + Qwen", "qwen_adapter", "prefix_qwen_combined", "prefix"),
    AdapterSpec("hazard_qwen", "Hazard+Qwen", "$H_t$ + Qwen", "qwen_adapter", "prefix_qwen_combined", "hazard"),
]


PLOT_COLORS = {
    "token_format": "#CC79A7",
    "qwen_prm": "#0072B2",
    "step_qwen": "#D55E00",
    "hazard_qwen": "#009E73",
    "adaptive_max_feasible": "#5B2A86",
    "adaptive_utility_lcb": "#117733",
}


CASCADE_CHEAP_SCORES = ["step_combined", "prefix_combined", "hazard_combined", "token_format"]
CASCADE_STRONG_SCORES = ["qwen_prm", "step_qwen", "hazard_qwen"]
CASCADE_ROUTE_RULES = [
    "cheap_prefix_fraction_below_tau",
    "cheap_max_score_above_tau",
    "cheap_prefix_nonempty_below_tau",
]
CASCADE_TAUS = [0.25, 0.40, 0.50, 0.60, 0.75, 0.90]


def _strat_key(trace: TraceRecord, by_domain: bool = True, by_has_error: bool = True) -> tuple:
    key: list[object] = []
    if by_domain:
        key.append(trace.domain)
    if by_has_error:
        key.append(int(trace.has_error))
    return tuple(key or ["all"])


def _split_counts(n: int, fracs: list[float]) -> np.ndarray:
    raw = np.asarray(fracs, dtype=float) * n
    counts = np.asarray([int(round(x)) for x in raw], dtype=int)
    positive = np.asarray([frac > 0 for frac in fracs], dtype=bool)
    min_counts = positive.astype(int)
    if n < int(min_counts.sum()):
        counts[:] = 0
        for idx in np.argsort(raw)[::-1][:n]:
            counts[int(idx)] = 1
    else:
        counts = np.maximum(counts, min_counts)
    while int(counts.sum()) > n:
        reducible = np.where(counts > min_counts)[0]
        if len(reducible) == 0:
            break
        idx = int(reducible[np.argmax(counts[reducible] - raw[reducible])])
        counts[idx] -= 1
    while int(counts.sum()) < n:
        idx = int(np.argmax(raw - counts))
        counts[idx] += 1
    return counts


def split_traces_four_way(
    traces: list[TraceRecord],
    *,
    train_frac: float = 0.5,
    select_frac: float = 0.15,
    cal_frac: float = 0.15,
    test_frac: float = 0.2,
    seed: int = 0,
) -> AdaptiveSplit:
    total = train_frac + select_frac + cal_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1, got {total}")
    rng = np.random.default_rng(seed)
    grouped: dict[tuple, list[TraceRecord]] = {}
    for trace in traces:
        grouped.setdefault(_strat_key(trace), []).append(trace)

    train: list[TraceRecord] = []
    select: list[TraceRecord] = []
    cal: list[TraceRecord] = []
    test: list[TraceRecord] = []
    for group in grouped.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_train, n_select, n_cal, _ = _split_counts(
            len(shuffled), [train_frac, select_frac, cal_frac, test_frac]
        )
        train.extend(shuffled[:n_train])
        select.extend(shuffled[n_train : n_train + n_select])
        cal.extend(shuffled[n_train + n_select : n_train + n_select + n_cal])
        test.extend(shuffled[n_train + n_select + n_cal :])

    rng.shuffle(train)
    rng.shuffle(select)
    rng.shuffle(cal)
    rng.shuffle(test)
    _assert_four_way_no_leakage(train, select, cal, test)
    return AdaptiveSplit(train=train, select=select, cal=cal, test=test)


def _assert_four_way_no_leakage(*parts: list[TraceRecord]) -> None:
    sets = [{trace.trace_id for trace in part} for part in parts]
    for i, left in enumerate(sets):
        for right in sets[i + 1 :]:
            if left & right:
                raise AssertionError("Trace leakage detected across adaptive split partitions")


def _trace_maps(traces: list[TraceRecord]) -> dict[str, TraceRecord]:
    return {trace.trace_id: trace for trace in traces}


def _adaptive_split_like(reference: AdaptiveSplit, traces: list[TraceRecord]) -> AdaptiveSplit:
    by_id = _trace_maps(traces)
    return AdaptiveSplit(
        train=[by_id[t.trace_id] for t in reference.train],
        select=[by_id[t.trace_id] for t in reference.select],
        cal=[by_id[t.trace_id] for t in reference.cal],
        test=[by_id[t.trace_id] for t in reference.test],
    )


def _as_three_way(split: AdaptiveSplit, *, test_part: str) -> Split:
    if test_part == "select":
        test = split.select
    elif test_part == "test":
        test = split.test
    else:
        raise ValueError(f"Unknown test_part={test_part!r}")
    return Split(train=split.train, cal=split.cal, test=test)


def _concat(scores_by_trace: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(scores_by_trace) if scores_by_trace else np.asarray([])


def _fit_prefix_bundle(split: AdaptiveSplit, seed: int, class_weight: str) -> tuple[object, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    train = traces_with_prefix_targets(split.train)
    model = fit_verifier(make_model("logistic_l2", seed=seed, class_weight=class_weight), train)
    return (
        model,
        scores_by_trace_from_model(model, split.select),
        scores_by_trace_from_model(model, split.cal),
        scores_by_trace_from_model(model, split.test),
    )


def _fit_hazard_bundle(split: AdaptiveSplit, seed: int, class_weight: str) -> tuple[object, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    train = traces_with_hazard_targets(split.train)
    model = fit_verifier(make_model("logistic_l2", seed=seed, class_weight=class_weight), train)
    return (
        model,
        scores_by_trace_from_model(model, split.select),
        scores_by_trace_from_model(model, split.cal),
        scores_by_trace_from_model(model, split.test),
    )


def _fit_step_bundle(split: AdaptiveSplit, seed: int, class_weight: str) -> tuple[object, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    model = fit_verifier(make_model("logistic_l2", seed=seed, class_weight=class_weight), split.train)
    return (
        model,
        scores_by_trace_from_model(model, split.select),
        scores_by_trace_from_model(model, split.cal),
        scores_by_trace_from_model(model, split.test),
    )


def _random_bundle(spec: AdapterSpec, split: AdaptiveSplit, seed: int) -> AdapterBundle:
    rng = np.random.default_rng(seed + 91_337)
    select_scores = [rng.random(len(trace.steps)) for trace in split.select]
    cal_scores = [rng.random(len(trace.steps)) for trace in split.cal]
    test_scores = [rng.random(len(trace.steps)) for trace in split.test]
    return AdapterBundle(spec, select_scores, cal_scores, test_scores, _concat(select_scores), _concat(cal_scores), _concat(test_scores), 0.0)


def _qwen_adapter_bundle(
    spec: AdapterSpec,
    split: AdaptiveSplit,
    scores_by_trace_id: dict[str, dict[int, float]],
) -> AdapterBundle:
    three = Split(train=split.train, cal=split.cal, test=split.test)
    qwen = _qwen_bundle(three, scores_by_trace_id)
    select_scores = []
    for trace in split.select:
        trace_scores = scores_by_trace_id.get(trace.trace_id, {})
        select_scores.append(np.asarray([float(trace_scores.get(step.step_number, 0.5)) for step in trace.steps]))
    return AdapterBundle(
        spec,
        select_scores,
        qwen.cal_scores_by_trace,
        qwen.test_scores_by_trace,
        _concat(select_scores),
        qwen.cal_step_scores,
        qwen.test_step_scores,
        0.0,
    )


def _fit_adapter(spec: AdapterSpec, split: AdaptiveSplit, seed: int, class_weight: str, scores_by_trace_id) -> AdapterBundle:
    started = time.perf_counter()
    if spec.source == "random":
        return _random_bundle(spec, split, seed)
    if spec.source == "qwen":
        if scores_by_trace_id is None:
            raise ValueError("Qwen scores are required for qwen_prm")
        return _qwen_adapter_bundle(spec, split, scores_by_trace_id)
    if spec.source == "step":
        _, select_scores, cal_scores, test_scores = _fit_step_bundle(split, seed, class_weight)
    elif spec.source == "prefix":
        _, select_scores, cal_scores, test_scores = _fit_prefix_bundle(split, seed, class_weight)
    elif spec.source == "hazard":
        _, select_scores, cal_scores, test_scores = _fit_hazard_bundle(split, seed, class_weight)
    else:
        raise ValueError(f"Unknown adapter source={spec.source!r}")
    return AdapterBundle(
        spec,
        select_scores,
        cal_scores,
        test_scores,
        _concat(select_scores),
        _concat(cal_scores),
        _concat(test_scores),
        time.perf_counter() - started,
    )


def _build_views(combined: list[TraceRecord], text: list[TraceRecord], scores_by_trace_id) -> dict[str, list[TraceRecord]]:
    views: dict[str, list[TraceRecord]] = {
        "combined": combined,
        "text": text,
        **_artifact_views(combined),
        "prefix_combined": augment_with_prefix_features(combined, include_position_features=True),
    }
    try:
        no_artifact = select_named_feature_columns(combined, keep_names=set(COE_SCORE_COLUMNS))
        views["prefix_no_artifact"] = augment_with_prefix_features(no_artifact, include_position_features=False)
    except ValueError:
        pass
    if scores_by_trace_id is not None:
        qwen_combined = append_trace_score_feature(combined, scores_by_trace_id)
        views["prefix_qwen_combined"] = augment_with_prefix_features(qwen_combined, include_position_features=True)
    return views


def _prefix_metrics(traces: list[TraceRecord], scores_by_trace: list[np.ndarray], lambda_: float) -> dict[str, float]:
    losses = prefix_losses_by_lambda(traces, scores_by_trace, np.asarray([lambda_]))[0]
    lengths = prefix_lengths(scores_by_trace, lambda_)
    totals = np.asarray([len(trace.steps) for trace in traces], dtype=float)
    return {
        "prefix_contamination": prefix_contamination_rate(losses),
        "prefix_retained_steps": float(np.mean(lengths)) if len(lengths) else float("nan"),
        "prefix_retained_fraction": float(np.mean(lengths / np.maximum(totals, 1.0))) if len(lengths) else float("nan"),
        "prefix_full_trace_rate": full_trace_accept_rate(lengths, totals),
    }


def _retained_fractions(traces: list[TraceRecord], scores_by_trace: list[np.ndarray], lambda_: float) -> np.ndarray:
    lengths = prefix_lengths(scores_by_trace, lambda_)
    totals = np.asarray([len(trace.steps) for trace in traces], dtype=float)
    return lengths / np.maximum(totals, 1.0)


def _calibrate_and_eval(
    traces_cal: list[TraceRecord],
    cal_scores: list[np.ndarray],
    traces_test: list[TraceRecord],
    test_scores: list[np.ndarray],
    *,
    alpha: float,
    lambdas: np.ndarray,
) -> dict[str, float]:
    losses = prefix_losses_by_lambda(traces_cal, cal_scores, lambdas)
    lambda_hat, cal_risk = select_lambda_crc(losses, lambdas, alpha=alpha, direction="increasing")
    out = _prefix_metrics(traces_test, test_scores, lambda_hat)
    out.update({"prefix_lambda": lambda_hat, "prefix_cal_corrected_risk": cal_risk})
    return out


def _selection_stats(split: AdaptiveSplit, adapter: AdapterBundle, alpha: float, lambdas: np.ndarray) -> dict[str, float]:
    losses = prefix_losses_by_lambda(split.select, adapter.select_scores_by_trace, lambdas)
    lambda_hat, corrected = select_lambda_crc(losses, lambdas, alpha=alpha, direction="increasing")
    metrics = _prefix_metrics(split.select, adapter.select_scores_by_trace, lambda_hat)
    fractions = _retained_fractions(split.select, adapter.select_scores_by_trace, lambda_hat)
    se = float(np.std(fractions, ddof=1) / np.sqrt(len(fractions))) if len(fractions) > 1 else 0.0
    feasible = corrected <= alpha
    return {
        "selection_lambda": lambda_hat,
        "selection_corrected_risk": corrected,
        "selection_empirical_risk": metrics["prefix_contamination"],
        "selection_prefix_kept": metrics["prefix_retained_fraction"],
        "selection_prefix_kept_lcb": max(0.0, metrics["prefix_retained_fraction"] - 1.96 * se),
        "selection_full_accept": metrics["prefix_full_trace_rate"],
        "selection_feasible": bool(feasible),
    }


def _choose_adapter(
    selection_rows: list[dict],
    *,
    rule: str,
    alpha: float,
    gamma: float,
) -> dict:
    candidates = [row for row in selection_rows if row["selection_rule"] == rule]
    if rule == "max_feasible":
        feasible = [row for row in candidates if bool(row["selection_feasible"])]
        pool = feasible if feasible else candidates
        return max(pool, key=lambda r: (float(r["selection_prefix_kept"]), -float(r["selection_corrected_risk"])))
    if rule == "utility_lcb":
        feasible = [row for row in candidates if bool(row["selection_feasible"])]
        pool = feasible if feasible else candidates
        return max(pool, key=lambda r: (float(r["selection_value"]), float(r["selection_prefix_kept"])))
    if rule == "penalized":
        return max(
            candidates,
            key=lambda r: float(r["selection_prefix_kept"])
            - gamma * max(0.0, float(r["selection_corrected_risk"]) - alpha),
        )
    raise ValueError(f"Unknown selection rule={rule!r}")


def _fixed_adapter_rows(
    dataset: str,
    seed: int,
    split: AdaptiveSplit,
    adapters: dict[str, AdapterBundle],
    alphas: list[float],
    lambdas: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    _, test_y, _, _, _ = flatten_steps(split.test)
    prefix_y = flatten_prefix_labels(split.test)
    hazard_y = flatten_hazard_labels(split.test)
    for alpha in alphas:
        for adapter in adapters.values():
            metrics = _calibrate_and_eval(
                split.cal,
                adapter.cal_scores_by_trace,
                split.test,
                adapter.test_scores_by_trace,
                alpha=alpha,
                lambdas=lambdas,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "alpha": alpha,
                    "row_type": "fixed",
                    "score": adapter.spec.score,
                    "label": adapter.spec.label,
                    "target": adapter.spec.target,
                    "score_family": adapter.spec.family,
                    "selected_adapter": adapter.spec.score,
                    "selection_rule": "fixed",
                    "selection_prefix_kept": np.nan,
                    "selection_corrected_risk": np.nan,
                    "selection_feasible": True,
                    "fit_seconds": adapter.fit_seconds,
                    "auroc": safe_auroc(test_y, adapter.test_step_scores) if len(test_y) else float("nan"),
                    "prefix_auroc": safe_auroc(prefix_y, adapter.test_step_scores) if len(prefix_y) else float("nan"),
                    "first_error_aupr": safe_aupr(hazard_y, adapter.test_step_scores) if len(hazard_y) else float("nan"),
                    **metrics,
                }
            )
    return rows


def _adaptive_rows(
    dataset: str,
    seed: int,
    split: AdaptiveSplit,
    adapters: dict[str, AdapterBundle],
    alphas: list[float],
    lambdas: np.ndarray,
    gamma: float,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    selection_detail: list[dict] = []
    _, test_y, _, _, _ = flatten_steps(split.test)
    prefix_y = flatten_prefix_labels(split.test)
    hazard_y = flatten_hazard_labels(split.test)
    for alpha in alphas:
        base_selection: list[dict] = []
        for adapter in adapters.values():
            stats = _selection_stats(split, adapter, alpha, lambdas)
            for rule in ("max_feasible", "penalized", "utility_lcb"):
                value = float(stats["selection_prefix_kept"])
                if rule == "penalized":
                    value -= gamma * max(0.0, float(stats["selection_corrected_risk"]) - alpha)
                elif rule == "utility_lcb":
                    value = float(stats["selection_prefix_kept_lcb"])
                base_selection.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "alpha": alpha,
                        "selection_rule": rule,
                        "score": adapter.spec.score,
                        "label": adapter.spec.label,
                        "target": adapter.spec.target,
                        "selection_value": value,
                        **stats,
                    }
                )
        selection_detail.extend(base_selection)
        for rule in ("max_feasible", "penalized", "utility_lcb"):
            selected = _choose_adapter(base_selection, rule=rule, alpha=alpha, gamma=gamma)
            adapter = adapters[str(selected["score"])]
            metrics = _calibrate_and_eval(
                split.cal,
                adapter.cal_scores_by_trace,
                split.test,
                adapter.test_scores_by_trace,
                alpha=alpha,
                lambdas=lambdas,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "alpha": alpha,
                    "row_type": "adaptive",
                    "score": f"adaptive_{rule}",
                    "label": _adaptive_rule_label(rule),
                    "target": "selected on held-out split",
                    "score_family": "adaptive",
                    "selected_adapter": adapter.spec.score,
                    "selected_label": adapter.spec.label,
                    "selection_rule": rule,
                    "selection_prefix_kept": selected["selection_prefix_kept"],
                    "selection_corrected_risk": selected["selection_corrected_risk"],
                    "selection_empirical_risk": selected["selection_empirical_risk"],
                    "selection_feasible": selected["selection_feasible"],
                    "selection_value": selected["selection_value"],
                    "fit_seconds": adapter.fit_seconds,
                    "auroc": safe_auroc(test_y, adapter.test_step_scores) if len(test_y) else float("nan"),
                    "prefix_auroc": safe_auroc(prefix_y, adapter.test_step_scores) if len(prefix_y) else float("nan"),
                    "first_error_aupr": safe_aupr(hazard_y, adapter.test_step_scores) if len(hazard_y) else float("nan"),
                    **metrics,
                }
            )
    return rows, selection_detail


def _adaptive_rule_label(rule: str) -> str:
    labels = {
        "max_feasible": "Adaptive selected adapter",
        "penalized": "Adaptive penalized",
        "utility_lcb": "Adaptive utility LCB",
    }
    return labels.get(rule, f"Adaptive {rule}")


def _best_fixed_rows(df: pd.DataFrame) -> pd.DataFrame:
    fixed = df[df["row_type"] == "fixed"].copy()
    rows = []
    for keys, group in fixed.groupby(["dataset", "seed", "alpha"], dropna=False):
        group = group.sort_values(["prefix_retained_fraction", "prefix_contamination"], ascending=[False, True])
        best = group.iloc[0].to_dict()
        best.update(
            {
                "row_type": "diagnostic",
                "score": "best_fixed_adapter",
                "label": "Best fixed adapter on test",
                "target": "test-selected diagnostic",
                "score_family": "diagnostic",
                "selected_adapter": best["score"],
                "selected_label": best["label"],
                "selection_rule": "test_oracle_diagnostic",
            }
        )
        rows.append(best)
    return pd.DataFrame(rows)


def _summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    numeric = [col for col in df.columns if col not in set(group_cols) and pd.api.types.is_numeric_dtype(df[col])]
    grouped = df.groupby(group_cols, dropna=False, observed=False)
    mean = grouped[numeric].mean(numeric_only=True)
    std = grouped[numeric].std(numeric_only=True).fillna(0.0)
    count = grouped[numeric].count()
    pieces = []
    for col in numeric:
        pieces.append(
            pd.DataFrame(
                {
                    f"{col}_mean": mean[col],
                    f"{col}_std": std[col],
                    f"{col}_n": count[col],
                    f"{col}_ci95": 1.96 * std[col] / np.sqrt(count[col].clip(lower=1)),
                }
            )
        )
    return pd.concat(pieces, axis=1).reset_index()


def _paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("adaptive_max_feasible", "qwen_prm", "Adaptive - Qwen"),
        ("adaptive_max_feasible", "token_format", "Adaptive - Token/format"),
        ("adaptive_max_feasible", "step_qwen", "Adaptive - Step+Qwen"),
        ("adaptive_max_feasible", "hazard_qwen", "Adaptive - Hazard+Qwen"),
        ("adaptive_max_feasible", "best_fixed_adapter", "Adaptive - best fixed adapter on test"),
        ("adaptive_utility_lcb", "adaptive_max_feasible", "Rule C - Rule A"),
        ("adaptive_utility_lcb", "qwen_prm", "Rule C - Qwen"),
        ("adaptive_utility_lcb", "step_qwen", "Rule C - Step+Qwen"),
        ("adaptive_utility_lcb", "hazard_qwen", "Rule C - Hazard+Qwen"),
        ("hazard_qwen", "step_qwen", "Hazard+Qwen - Step+Qwen"),
        ("step_qwen", "qwen_prm", "Step+Qwen - Qwen"),
        ("prefix_qwen", "step_qwen", "Prefix+Qwen - Step+Qwen"),
    ]
    rows = []
    for (dataset, alpha), group in df.groupby(["dataset", "alpha"], dropna=False):
        by_score = {score: sub.set_index("seed") for score, sub in group.groupby("score")}
        for score_a, score_b, comparison in comparisons:
            if score_a not in by_score or score_b not in by_score:
                continue
            a = by_score[score_a]
            b = by_score[score_b]
            seeds = sorted(set(a.index) & set(b.index))
            if not seeds:
                continue
            kept = a.loc[seeds, "prefix_retained_fraction"].to_numpy(float) - b.loc[seeds, "prefix_retained_fraction"].to_numpy(float)
            risk = a.loc[seeds, "prefix_contamination"].to_numpy(float) - b.loc[seeds, "prefix_contamination"].to_numpy(float)
            rows.append(
                {
                    "dataset": dataset,
                    "alpha": alpha,
                    "comparison": comparison,
                    "score_a": score_a,
                    "score_b": score_b,
                    "n_paired_splits": len(seeds),
                    "delta_kept_mean": float(np.mean(kept)),
                    "delta_kept_ci_low": float(np.percentile(kept, 2.5)),
                    "delta_kept_ci_high": float(np.percentile(kept, 97.5)),
                    "delta_risk_mean": float(np.mean(risk)),
                    "delta_risk_ci_low": float(np.percentile(risk, 2.5)),
                    "delta_risk_ci_high": float(np.percentile(risk, 97.5)),
                    "interpretation": _interpret_delta(kept),
                }
            )
    return pd.DataFrame(rows)


def _interpret_delta(values: np.ndarray) -> str:
    lo = float(np.percentile(values, 2.5))
    hi = float(np.percentile(values, 97.5))
    mean = float(np.mean(values))
    if lo > 0:
        return "positive"
    if hi < 0:
        return "negative"
    if abs(mean) < 0.01:
        return "similar"
    return "mixed"


def _length_losses_from_traces(traces: list[TraceRecord], lengths: np.ndarray) -> np.ndarray:
    losses = []
    for trace, length in zip(traces, np.asarray(lengths, dtype=int)):
        losses.append(bool(length > 0 and np.any(trace.y_errors[:length] > 0)))
    return np.asarray(losses, dtype=int)


def _route_mask(scores_by_trace: list[np.ndarray], rule: str, tau: float) -> np.ndarray:
    totals = np.asarray([len(scores) for scores in scores_by_trace], dtype=float)
    tau_lengths = prefix_lengths(scores_by_trace, tau)
    tau_fractions = tau_lengths / np.maximum(totals, 1.0)
    max_scores = np.asarray([float(np.max(scores)) if len(scores) else 0.0 for scores in scores_by_trace])
    if rule == "cheap_prefix_fraction_below_tau":
        return tau_fractions < tau
    if rule == "cheap_max_score_above_tau":
        return max_scores > tau
    if rule == "cheap_prefix_nonempty_below_tau":
        return (tau_lengths > 0) & (tau_fractions < tau)
    raise ValueError(f"Unknown cascade route rule={rule!r}")


def _cascade_lengths(
    cheap_scores_by_trace: list[np.ndarray],
    strong_scores_by_trace: list[np.ndarray],
    route: np.ndarray,
    lambda_: float,
) -> np.ndarray:
    cheap_lengths = prefix_lengths(cheap_scores_by_trace, lambda_)
    strong_lengths = prefix_lengths(strong_scores_by_trace, lambda_)
    return np.where(route, strong_lengths, cheap_lengths)


def _cascade_losses_by_lambda(
    traces: list[TraceRecord],
    cheap_scores_by_trace: list[np.ndarray],
    strong_scores_by_trace: list[np.ndarray],
    route: np.ndarray,
    lambdas: np.ndarray,
) -> np.ndarray:
    losses = []
    for lambda_ in lambdas:
        lengths = _cascade_lengths(cheap_scores_by_trace, strong_scores_by_trace, route, float(lambda_))
        losses.append(_length_losses_from_traces(traces, lengths))
    return np.vstack(losses)


def _cascade_metrics_from_lengths(
    traces: list[TraceRecord],
    lengths: np.ndarray,
    route: np.ndarray,
) -> dict[str, float]:
    totals = np.asarray([len(trace.steps) for trace in traces], dtype=float)
    losses = _length_losses_from_traces(traces, lengths)
    suffix = np.maximum(totals - lengths, 0.0)
    return {
        "prefix_contamination": prefix_contamination_rate(losses),
        "prefix_retained_steps": float(np.mean(lengths)) if len(lengths) else float("nan"),
        "prefix_retained_fraction": float(np.mean(lengths / np.maximum(totals, 1.0))) if len(lengths) else float("nan"),
        "prefix_full_trace_rate": full_trace_accept_rate(lengths, totals),
        "qwen_call_rate": float(np.mean(route)) if len(route) else float("nan"),
        "review_steps_routed": float(np.mean(suffix)) if len(suffix) else float("nan"),
        "review_steps_routed_fraction": float(np.mean(suffix / np.maximum(totals, 1.0))) if len(suffix) else float("nan"),
    }


def _cascade_selection_stats(
    split: AdaptiveSplit,
    cheap: AdapterBundle,
    strong: AdapterBundle,
    *,
    route_rule: str,
    tau: float,
    alpha: float,
    lambdas: np.ndarray,
) -> dict[str, float | bool]:
    route = _route_mask(cheap.select_scores_by_trace, route_rule, tau)
    losses = _cascade_losses_by_lambda(split.select, cheap.select_scores_by_trace, strong.select_scores_by_trace, route, lambdas)
    lambda_hat, corrected = select_lambda_crc(losses, lambdas, alpha=alpha, direction="increasing")
    lengths = _cascade_lengths(cheap.select_scores_by_trace, strong.select_scores_by_trace, route, lambda_hat)
    metrics = _cascade_metrics_from_lengths(split.select, lengths, route)
    feasible = corrected <= alpha
    return {
        "selection_lambda": lambda_hat,
        "selection_corrected_risk": corrected,
        "selection_empirical_risk": metrics["prefix_contamination"],
        "selection_prefix_kept": metrics["prefix_retained_fraction"],
        "selection_full_accept": metrics["prefix_full_trace_rate"],
        "selection_qwen_call_rate": metrics["qwen_call_rate"],
        "selection_feasible": bool(feasible),
    }


def _calibrate_and_eval_cascade(
    split: AdaptiveSplit,
    cheap: AdapterBundle,
    strong: AdapterBundle,
    *,
    route_rule: str,
    tau: float,
    alpha: float,
    lambdas: np.ndarray,
) -> dict[str, float]:
    route_cal = _route_mask(cheap.cal_scores_by_trace, route_rule, tau)
    losses = _cascade_losses_by_lambda(split.cal, cheap.cal_scores_by_trace, strong.cal_scores_by_trace, route_cal, lambdas)
    lambda_hat, cal_risk = select_lambda_crc(losses, lambdas, alpha=alpha, direction="increasing")
    route_test = _route_mask(cheap.test_scores_by_trace, route_rule, tau)
    lengths = _cascade_lengths(cheap.test_scores_by_trace, strong.test_scores_by_trace, route_test, lambda_hat)
    metrics = _cascade_metrics_from_lengths(split.test, lengths, route_test)
    metrics.update({"prefix_lambda": lambda_hat, "prefix_cal_corrected_risk": cal_risk})
    return metrics


def _gain_recovered(cascade_kept: float, cheap_kept: float, strong_kept: float) -> float:
    denom = strong_kept - cheap_kept
    if not np.isfinite(denom) or denom <= 1e-9:
        return float("nan")
    return float((cascade_kept - cheap_kept) / denom)


def _direct_cascade_rows(
    dataset: str,
    seed: int,
    split: AdaptiveSplit,
    adapters: dict[str, AdapterBundle],
    alphas: list[float],
    lambdas: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    selection_rows: list[dict] = []
    available_cheap = [score for score in CASCADE_CHEAP_SCORES if score in adapters]
    available_strong = [score for score in CASCADE_STRONG_SCORES if score in adapters]
    baseline_cache: dict[tuple[str, float], dict[str, float]] = {}

    def baseline(score: str, alpha: float) -> dict[str, float]:
        key = (score, alpha)
        if key not in baseline_cache:
            adapter = adapters[score]
            baseline_cache[key] = _calibrate_and_eval(
                split.cal,
                adapter.cal_scores_by_trace,
                split.test,
                adapter.test_scores_by_trace,
                alpha=alpha,
                lambdas=lambdas,
            )
        return baseline_cache[key]

    for alpha in alphas:
        candidates: list[dict] = []
        for cheap_score in available_cheap:
            cheap = adapters[cheap_score]
            for strong_score in available_strong:
                strong = adapters[strong_score]
                for route_rule in CASCADE_ROUTE_RULES:
                    for tau in CASCADE_TAUS:
                        stats = _cascade_selection_stats(
                            split,
                            cheap,
                            strong,
                            route_rule=route_rule,
                            tau=tau,
                            alpha=alpha,
                            lambdas=lambdas,
                        )
                        value = float(stats["selection_prefix_kept"])
                        if float(stats["selection_qwen_call_rate"]) > 0.4:
                            value -= 0.25 * (float(stats["selection_qwen_call_rate"]) - 0.4)
                        candidate = {
                            "dataset": dataset,
                            "seed": seed,
                            "alpha": alpha,
                            "cheap_score": cheap_score,
                            "cheap_label": cheap.spec.label,
                            "strong_score": strong_score,
                            "strong_label": strong.spec.label,
                            "route_rule": route_rule,
                            "tau": tau,
                            "selection_value": value,
                            **stats,
                        }
                        candidates.append(candidate)
        selection_rows.extend(candidates)
        feasible = [row for row in candidates if bool(row["selection_feasible"])]
        pool = feasible if feasible else candidates
        selected = max(
            pool,
            key=lambda r: (
                float(r["selection_value"]),
                -float(r["selection_corrected_risk"]),
                -float(r["selection_qwen_call_rate"]),
            ),
        )
        cheap = adapters[str(selected["cheap_score"])]
        strong = adapters[str(selected["strong_score"])]
        metrics = _calibrate_and_eval_cascade(
            split,
            cheap,
            strong,
            route_rule=str(selected["route_rule"]),
            tau=float(selected["tau"]),
            alpha=alpha,
            lambdas=lambdas,
        )
        cheap_metrics = baseline(str(selected["cheap_score"]), alpha)
        strong_metrics = baseline(str(selected["strong_score"]), alpha)
        recovered = _gain_recovered(
            metrics["prefix_retained_fraction"],
            cheap_metrics["prefix_retained_fraction"],
            strong_metrics["prefix_retained_fraction"],
        )
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "alpha": alpha,
                "row_type": "cascade_adaptive",
                "score": "direct_cascade_adaptive",
                "policy": _cascade_policy_label(selected),
                "cheap_score": selected["cheap_score"],
                "cheap_label": selected["cheap_label"],
                "strong_score": selected["strong_score"],
                "strong_label": selected["strong_label"],
                "route_rule": selected["route_rule"],
                "tau": selected["tau"],
                "selection_prefix_kept": selected["selection_prefix_kept"],
                "selection_corrected_risk": selected["selection_corrected_risk"],
                "selection_qwen_call_rate": selected["selection_qwen_call_rate"],
                "selection_feasible": selected["selection_feasible"],
                "cheap_only_kept": cheap_metrics["prefix_retained_fraction"],
                "strong_only_kept": strong_metrics["prefix_retained_fraction"],
                "gain_recovered": recovered,
                "compelling": bool(metrics["qwen_call_rate"] <= 0.40 and np.isfinite(recovered) and recovered >= 0.70),
                **metrics,
            }
        )
    return rows, selection_rows


def _cascade_policy_label(row: dict) -> str:
    return (
        f"{row['cheap_label']} -> {row['strong_label']}; "
        f"{str(row['route_rule']).replace('_', ' ')} @ {float(row['tau']):.2f}"
    )


def _hazard_threshold_crossing_rows(
    dataset: str,
    seed: int,
    split: AdaptiveSplit,
    adapters: dict[str, AdapterBundle],
    alphas: list[float],
    lambdas: np.ndarray,
) -> list[dict]:
    if dataset not in {"ProcessBench", "PRMBench"}:
        return []
    focus_scores = ["qwen_prm", "step_qwen", "prefix_qwen", "hazard_qwen"]
    rows: list[dict] = []
    for score in focus_scores:
        if score not in adapters:
            continue
        adapter = adapters[score]
        for alpha in alphas:
            losses = prefix_losses_by_lambda(split.cal, adapter.cal_scores_by_trace, lambdas)
            lambda_hat, cal_risk = select_lambda_crc(losses, lambdas, alpha=alpha, direction="increasing")
            crossing_steps = []
            crossing_offsets = []
            before_flags = []
            near_flags = []
            no_crossing_flags = []
            scores_before = []
            scores_at = []
            for trace, scores in zip(split.test, adapter.test_scores_by_trace):
                first_error = trace.first_error
                if first_error is None or first_error >= len(scores):
                    continue
                scores = np.asarray(scores, dtype=float)
                crossing = np.flatnonzero(scores > lambda_hat)
                if len(crossing):
                    first_crossing = int(crossing[0])
                    crossing_steps.append(first_crossing + 1)
                    offset = first_crossing - int(first_error)
                    crossing_offsets.append(offset)
                    before_flags.append(first_crossing < int(first_error))
                    near_flags.append(abs(offset) <= 1)
                    no_crossing_flags.append(False)
                else:
                    before_flags.append(False)
                    near_flags.append(False)
                    no_crossing_flags.append(True)
                if first_error > 0:
                    scores_before.append(float(scores[first_error - 1]))
                scores_at.append(float(scores[first_error]))
            n_error = len(before_flags)
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "alpha": alpha,
                    "score": score,
                    "label": adapter.spec.label,
                    "threshold": lambda_hat,
                    "cal_corrected_risk": cal_risk,
                    "n_error_traces": n_error,
                    "median_first_threshold_crossing": float(np.median(crossing_steps)) if crossing_steps else float("nan"),
                    "median_crossing_minus_first_error": float(np.median(crossing_offsets)) if crossing_offsets else float("nan"),
                    "fraction_crossing_before_first_error": float(np.mean(before_flags)) if n_error else float("nan"),
                    "fraction_crossing_at_or_near_first_error": float(np.mean(near_flags)) if n_error else float("nan"),
                    "fraction_no_threshold_crossing": float(np.mean(no_crossing_flags)) if n_error else float("nan"),
                    "median_score_before_first_error": float(np.median(scores_before)) if scores_before else float("nan"),
                    "median_score_at_first_error": float(np.median(scores_at)) if scores_at else float("nan"),
                }
            )
    return rows


def _selection_frequencies(df: pd.DataFrame, adaptive_score: str = "adaptive_max_feasible") -> pd.DataFrame:
    adaptive = df[(df["row_type"] == "adaptive") & (df["score"] == adaptive_score)].copy()
    rows = []
    categories = [
        ("step_qwen", "Step+Qwen selected"),
        ("prefix_qwen", "Prefix+Qwen selected"),
        ("hazard_qwen", "Hazard+Qwen selected"),
        ("qwen_prm", "Qwen selected"),
        ("token_format", "Token selected"),
    ]
    for (dataset, alpha), group in adaptive.groupby(["dataset", "alpha"], dropna=False):
        total = len(group)
        row = {"dataset": dataset, "alpha": alpha, "n_splits": total}
        selected = group["selected_adapter"].astype(str)
        for name, label in categories:
            row[label] = float(np.mean(selected == name)) if total else float("nan")
        cheap = selected.isin(["step_combined", "prefix_combined", "hazard_combined"])
        row["Cheap selected"] = float(np.mean(cheap)) if total else float("nan")
        other = float(
            1.0
            - sum(row[label] for _, label in categories)
            - row["Cheap selected"]
        )
        row["Other selected"] = min(1.0, max(0.0, other))
        row["mode_adapter"] = selected.mode().iloc[0] if total else ""
        row["mode_fraction"] = float(np.mean(selected == row["mode_adapter"])) if total else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _cell(kept: float, risk: float) -> str:
    return f"{100.0 * kept:.1f} ({100.0 * risk:.1f})"


def _pct_or_dash(value: float) -> str:
    return "--" if not np.isfinite(value) else f"{100.0 * value:.1f}"


def _write_tables(df: pd.DataFrame, deltas: pd.DataFrame, freqs: pd.DataFrame, outdir: Path) -> None:
    alpha = 0.05
    summary = _summarize(
        df[df["alpha"].round(4) == round(alpha, 4)],
        ["dataset", "score", "label", "target", "row_type", "score_family"],
    )
    summary.to_csv(outdir / "table_all_methods_summary.csv", index=False)

    target_rows = ["token_format", "qwen_prm", "step_qwen", "prefix_qwen", "hazard_qwen", "adaptive_max_feasible", "best_fixed_adapter"]
    target = summary[(summary["dataset"] == "Target") & (summary["score"].isin(target_rows))].copy()
    token = target[target["score"] == "token_format"]
    if not token.empty:
        token_kept = float(token["prefix_retained_fraction_mean"].iloc[0])
        target["delta_token"] = 100.0 * (target["prefix_retained_fraction_mean"] - token_kept)
    else:
        target["delta_token"] = np.nan
    target.to_csv(outdir / "table_target_adaptive.csv", index=False)

    external_rows = []
    label_lookup = {spec.score: spec.label for spec in ADAPTER_SPECS}
    for dataset, group in summary[summary["dataset"] != "Target"].groupby("dataset", dropna=False):
        row = {"dataset": dataset}
        freq = freqs[(freqs["dataset"] == dataset) & (freqs["alpha"].round(4) == round(alpha, 4))]
        if not freq.empty:
            mode = str(freq.iloc[0]["mode_adapter"])
            row["selected_adapter"] = f"{label_lookup.get(mode, mode)} ({100.0 * freq.iloc[0]['mode_fraction']:.0f}%)"
        for score in ["adaptive_max_feasible", "token_format", "qwen_prm", "step_qwen", "hazard_qwen", "best_fixed_adapter"]:
            sub = group[group["score"] == score]
            if sub.empty:
                continue
            r = sub.iloc[0]
            row[f"{score}_kept_risk"] = _cell(r["prefix_retained_fraction_mean"], r["prefix_contamination_mean"])
            row[f"{score}_kept"] = r["prefix_retained_fraction_mean"]
            row[f"{score}_risk"] = r["prefix_contamination_mean"]
        external_rows.append(row)
    pd.DataFrame(external_rows).to_csv(outdir / "table_external_adaptive.csv", index=False)

    freq05 = freqs[freqs["alpha"].round(4) == round(alpha, 4)].copy()
    freq05.to_csv(outdir / "table_selection_frequencies.csv", index=False)
    rule_c_freqs = _selection_frequencies(df, adaptive_score="adaptive_utility_lcb")
    rule_c_freqs[rule_c_freqs["alpha"].round(4) == round(alpha, 4)].to_csv(
        outdir / "table_rule_c_selection_frequencies.csv", index=False
    )
    deltas[deltas["alpha"].round(4) == round(alpha, 4)].to_csv(outdir / "table_paired_deltas_alpha05.csv", index=False)
    _write_tex_tables(outdir, target, pd.DataFrame(external_rows), freq05, deltas[deltas["alpha"].round(4) == round(alpha, 4)])


def _tex(text: object) -> str:
    return str(text).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def _tex_target(text: object) -> str:
    text = str(text).replace("&", r"\&").replace("%", r"\%")
    return text if "$" in text else text.replace("_", r"\_")


def _write_tex_tables(outdir: Path, target: pd.DataFrame, external: pd.DataFrame, freqs: pd.DataFrame, deltas: pd.DataFrame) -> None:
    tables = Path("tables")
    tables.mkdir(exist_ok=True)
    order = ["token_format", "qwen_prm", "step_qwen", "prefix_qwen", "hazard_qwen", "adaptive_max_feasible", "best_fixed_adapter"]
    target = target.set_index("score").reindex(order).dropna(how="all").reset_index()
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Adaptive CPCC score adaptation on the target benchmark at $\alpha=0.05$. The adaptive row selects a score adapter on a held-out selection split, then recalibrates CPCC on a separate calibration split before evaluating on test traces.}",
        r"\label{tab:target_adaptive_cpcc}",
        r"\footnotesize",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Score source & Target & Prefix risk & Prefix kept & Full accept & $\Delta$Token \\",
        r"\midrule",
    ]
    for _, row in target.iterrows():
        lines.append(
            f"{_tex(row['label'])} & {_tex_target(row['target'])} & "
            f"{100.0 * row['prefix_contamination_mean']:.1f} & "
            f"{100.0 * row['prefix_retained_fraction_mean']:.1f} & "
            f"{100.0 * row['prefix_full_trace_rate_mean']:.1f} & "
            f"{row['delta_token']:+.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (tables / "target_adaptive_cpcc.tex").write_text("\n".join(lines))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{External adaptive score utility with in-domain recalibration. Each cell reports retained-prefix percentage with empirical prefix risk in parentheses. All thresholds are recalibrated in-domain using trace-level calibration splits; these rows do not claim cross-dataset conformal validity.}",
        r"\label{tab:external_adaptive_cpcc}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Dataset & Selected adapter & Adaptive & Token/format & Qwen PRM & Step+Qwen & Hazard+Qwen \\",
        r"\midrule",
    ]
    for _, row in external.iterrows():
        lines.append(
            f"{_tex(row['dataset'])} & {_tex(row.get('selected_adapter', ''))} & "
            f"{row.get('adaptive_max_feasible_kept_risk', '--')} & "
            f"{row.get('token_format_kept_risk', '--')} & "
            f"{row.get('qwen_prm_kept_risk', '--')} & "
            f"{row.get('step_qwen_kept_risk', '--')} & "
            f"{row.get('hazard_qwen_kept_risk', '--')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (tables / "external_adaptive_cpcc.tex").write_text("\n".join(lines))

    cols = [
        "Step+Qwen selected",
        "Prefix+Qwen selected",
        "Hazard+Qwen selected",
        "Cheap selected",
        "Token selected",
        "Qwen selected",
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Adapter selection frequencies across repeated trace-level splits at $\alpha=0.05$.}",
        r"\label{tab:selection_frequencies}",
        r"\footnotesize",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & Step+Qwen & Prefix+Qwen & Hazard+Qwen & Cheap & Token & Qwen \\",
        r"\midrule",
    ]
    for _, row in freqs.iterrows():
        vals = [100.0 * float(row[col]) for col in cols]
        lines.append(f"{_tex(row['dataset'])} & " + " & ".join(f"{v:.0f}" for v in vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (tables / "selection_frequencies.tex").write_text("\n".join(lines))

    focus = deltas[deltas["comparison"].isin(["Adaptive - Qwen", "Adaptive - Step+Qwen", "Adaptive - Hazard+Qwen", "Adaptive - best fixed adapter on test"])]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paired retained-prefix deltas for adaptive CPCC score selection at $\alpha=0.05$. Intervals are empirical paired split percentiles over repeated trace-level splits.}",
        r"\label{tab:paired_adaptive_deltas}",
        r"\footnotesize",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Comparison & Mean $\Delta$ kept & 95\% CI & Mean $\Delta$ risk \\",
        r"\midrule",
    ]
    for _, row in focus.iterrows():
        lines.append(
            f"{_tex(row['dataset'])} & {_tex(row['comparison'])} & "
            f"{100.0 * row['delta_kept_mean']:+.1f} & "
            f"[{100.0 * row['delta_kept_ci_low']:+.1f}, {100.0 * row['delta_kept_ci_high']:+.1f}] & "
            f"{100.0 * row['delta_risk_mean']:+.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (tables / "paired_adaptive_deltas.tex").write_text("\n".join(lines))


def _make_extension_tables(cascade: pd.DataFrame, hazard: pd.DataFrame, outdir: Path) -> None:
    tables = Path("tables")
    tables.mkdir(exist_ok=True)
    alpha = 0.05
    if not cascade.empty:
        selected = cascade[
            (cascade["row_type"] == "cascade_adaptive") & (cascade["alpha"].round(4) == round(alpha, 4))
        ].copy()
        selected.to_csv(outdir / "table_direct_cascade_selected_alpha05.csv", index=False)
        summary = _summarize(
            selected,
            ["dataset", "row_type"],
        )
        modes = []
        for dataset, group in selected.groupby("dataset", dropna=False):
            mode = group["policy"].mode().iloc[0] if len(group) else ""
            modes.append(
                {
                    "dataset": dataset,
                    "mode_policy": mode,
                    "mode_fraction": float(np.mean(group["policy"] == mode)) if len(group) else float("nan"),
                    "compelling_fraction": float(np.mean(group["compelling"].astype(bool))) if len(group) else float("nan"),
                }
            )
        mode_df = pd.DataFrame(modes)
        summary = summary.merge(mode_df, on="dataset", how="left")
        summary.to_csv(outdir / "table_direct_cascade_selected_summary.csv", index=False)

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Directly calibrated cascade policies at $\alpha=0.05$. The route policy is selected on a held-out selection split, then the final returned-prefix loss is calibrated directly on the calibration split.}",
            r"\label{tab:direct_cascade_cpcc}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{2.5pt}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Dataset & Prefix risk & Prefix kept & Qwen calls & Gain rec. & Compelling \\",
            r"\midrule",
        ]
        for _, row in summary.iterrows():
            lines.append(
                f"{_tex(row['dataset'])} & "
                f"{_pct_or_dash(row['prefix_contamination_mean'])} & "
                f"{_pct_or_dash(row['prefix_retained_fraction_mean'])} & "
                f"{_pct_or_dash(row['qwen_call_rate_mean'])} & "
                f"{_pct_or_dash(row['gain_recovered_mean'])} & "
                f"{100.0 * row['compelling_fraction']:.0f} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
        (tables / "direct_cascade_cpcc.tex").write_text("\n".join(lines))

    if not hazard.empty:
        focus = hazard[hazard["alpha"].round(4) == round(alpha, 4)].copy()
        focus.to_csv(outdir / "table_hazard_threshold_crossing_alpha05.csv", index=False)
        summary = _summarize(focus, ["dataset", "score", "label"])
        summary.to_csv(outdir / "table_hazard_threshold_crossing_summary.csv", index=False)
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Threshold-crossing diagnostics on external first-error-style datasets at $\alpha=0.05$. Crossing offsets are measured relative to the first annotated error; negative values mean the score crosses before the first error.}",
            r"\label{tab:hazard_threshold_crossing}",
            r"\footnotesize",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Dataset & Score & Median offset & Before FE & Near FE & No crossing \\",
            r"\midrule",
        ]
        for _, row in summary.iterrows():
            lines.append(
                f"{_tex(row['dataset'])} & {_tex(row['label'])} & "
                f"{row['median_crossing_minus_first_error_mean']:.1f} & "
                f"{_pct_or_dash(row['fraction_crossing_before_first_error_mean'])} & "
                f"{_pct_or_dash(row['fraction_crossing_at_or_near_first_error_mean'])} & "
                f"{_pct_or_dash(row['fraction_no_threshold_crossing_mean'])} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
        (tables / "hazard_threshold_crossing.tex").write_text("\n".join(lines))


def _make_figures(df: pd.DataFrame, freqs: pd.DataFrame, outdir: Path) -> None:
    figures = Path("figures")
    figures.mkdir(exist_ok=True)
    freq05 = freqs[freqs["alpha"].round(4) == 0.05].copy()
    datasets = ["Target", "ProcessBench", "Math-Shepherd", "PRMBench", "PRM800K"]
    freq05["dataset"] = pd.Categorical(freq05["dataset"], datasets, ordered=True)
    freq05 = freq05.sort_values("dataset")
    cols = [
        ("Step+Qwen selected", "#D55E00"),
        ("Prefix+Qwen selected", "#5B2A86"),
        ("Hazard+Qwen selected", "#009E73"),
        ("Cheap selected", "#E69F00"),
        ("Token selected", "#CC79A7"),
        ("Qwen selected", "#0072B2"),
        ("Other selected", "#777777"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bottom = np.zeros(len(freq05))
    x = np.arange(len(freq05))
    for col, color in cols:
        vals = freq05[col].to_numpy(float)
        ax.bar(x, vals, bottom=bottom, label=col.replace(" selected", ""), color=color, width=0.72)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(freq05["dataset"].astype(str), rotation=20, ha="right")
    ax.set_ylabel("Fraction of splits")
    ax.set_ylim(0.0, 1.0)
    ax.legend(ncol=3, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "figure_selection_frequencies.pdf")
    fig.savefig(figures / "selection_frequencies.pdf")
    plt.close(fig)

    focus_scores = [
        ("token_format", "Token/format"),
        ("qwen_prm", "Qwen PRM"),
        ("step_qwen", "Step+Qwen"),
        ("hazard_qwen", "Hazard+Qwen"),
        ("adaptive_max_feasible", "Adaptive selected"),
    ]
    target = df[df["dataset"] == "Target"].copy()
    summary = _summarize(target, ["score", "label", "alpha"])
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    for score, label in focus_scores:
        sub = summary[summary["score"] == score].sort_values("alpha")
        if sub.empty:
            continue
        ax.plot(
            100.0 * sub["prefix_contamination_mean"],
            100.0 * sub["prefix_retained_fraction_mean"],
            marker="o",
            linewidth=1.8,
            label=label,
            color=PLOT_COLORS.get(score, None),
        )
    ax.axvline(5.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Empirical prefix risk (%)")
    ax.set_ylabel("Prefix kept (%)")
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(fontsize=7.5, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "figure_risk_utility_adaptive.pdf")
    fig.savefig(figures / "risk_utility_adaptive.pdf")
    plt.close(fig)


def _write_readme(outdir: Path, df: pd.DataFrame, deltas: pd.DataFrame, freqs: pd.DataFrame) -> None:
    alpha_df = df[df["alpha"].round(4) == 0.05]
    summary = _summarize(alpha_df, ["dataset", "score", "label", "row_type"])
    lines = [
        "# Adaptive CPCC Score Adapters",
        "",
        "This directory contains the annotation-protocol adaptive CPCC score-selection experiment.",
        "",
        "Protocol: trace-level train/select/calibration/test splits; adapters fit on train; adapter selected on select; final CPCC threshold selected on calibration; metrics reported on test.",
        "",
        "## Alpha 0.05 Snapshot",
        "",
    ]
    focus = summary[summary["score"].isin(["token_format", "qwen_prm", "step_qwen", "hazard_qwen", "adaptive_max_feasible", "best_fixed_adapter"])]
    if not focus.empty:
        table = focus[["dataset", "label", "prefix_contamination_mean", "prefix_retained_fraction_mean", "prefix_full_trace_rate_mean"]].copy()
        for col in ["prefix_contamination_mean", "prefix_retained_fraction_mean", "prefix_full_trace_rate_mean"]:
            table[col] = (100.0 * table[col]).round(2)
        lines.append(table.to_markdown(index=False))
        lines.append("")
    lines.extend(
        [
            "## Selection Frequencies",
            "",
            freqs[freqs["alpha"].round(4) == 0.05].to_markdown(index=False),
            "",
            "## Key Paired Deltas",
            "",
        ]
    )
    key = deltas[(deltas["alpha"].round(4) == 0.05) & (deltas["comparison"].str.startswith("Adaptive"))].copy()
    if not key.empty:
        for col in ["delta_kept_mean", "delta_kept_ci_low", "delta_kept_ci_high", "delta_risk_mean"]:
            key[col] = (100.0 * key[col]).round(2)
        lines.append(key[["dataset", "comparison", "delta_kept_mean", "delta_kept_ci_low", "delta_kept_ci_high", "delta_risk_mean", "interpretation"]].to_markdown(index=False))
    (outdir / "README.md").write_text("\n".join(lines))


def _run_dataset(args, dataset_name: str, text_features: str, combined_features: str, qwen_csv: str, seeds: list[int]) -> dict[str, pd.DataFrame]:
    combined = load_many_npz([combined_features], ["mixed"], allow_nan=True)
    text = load_many_npz([text_features], ["mixed"], allow_nan=True)
    scores_by_trace_id = _read_qwen_scores(qwen_csv, args.qwen_score_col)
    views = _build_views(combined, text, scores_by_trace_id)
    required_views = {spec.view for spec in ADAPTER_SPECS if spec.source != "qwen"}
    missing = sorted(view for view in required_views if view not in views)
    if missing:
        raise ValueError(f"{dataset_name} is missing required feature views: {missing}")

    lambdas = np.linspace(0.0, 1.0, args.lambda_grid_size)
    all_rows: list[dict] = []
    selection_rows: list[dict] = []
    cascade_rows: list[dict] = []
    cascade_selection_rows: list[dict] = []
    hazard_rows: list[dict] = []
    coverage_rows: list[dict] = []
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
        coverage_rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "train_qwen_step_coverage": _score_map_coverage(reference.train, scores_by_trace_id),
                "select_qwen_step_coverage": _score_map_coverage(reference.select, scores_by_trace_id),
                "cal_qwen_step_coverage": _score_map_coverage(reference.cal, scores_by_trace_id),
                "test_qwen_step_coverage": _score_map_coverage(reference.test, scores_by_trace_id),
                "n_train_traces": len(reference.train),
                "n_select_traces": len(reference.select),
                "n_cal_traces": len(reference.cal),
                "n_test_traces": len(reference.test),
            }
        )
        adapters = {}
        for spec in ADAPTER_SPECS:
            split = reference if spec.source == "qwen" else split_by_view[spec.view]
            adapters[spec.score] = _fit_adapter(spec, split, seed, args.class_weight, scores_by_trace_id)
        fixed_rows = _fixed_adapter_rows(dataset_name, seed, reference, adapters, args.alphas, lambdas)
        adaptive_rows, selection_detail = _adaptive_rows(
            dataset_name, seed, reference, adapters, args.alphas, lambdas, args.penalty_gamma
        )
        if args.run_direct_cascades:
            selected_cascades, cascade_selection = _direct_cascade_rows(
                dataset_name, seed, reference, adapters, args.alphas, lambdas
            )
            cascade_rows.extend(selected_cascades)
            cascade_selection_rows.extend(cascade_selection)
        if args.run_hazard_diagnostic:
            hazard_rows.extend(
                _hazard_threshold_crossing_rows(dataset_name, seed, reference, adapters, args.alphas, lambdas)
            )
        split_rows = pd.DataFrame(fixed_rows + adaptive_rows)
        best = _best_fixed_rows(split_rows)
        all_rows.extend(fixed_rows)
        all_rows.extend(adaptive_rows)
        all_rows.extend(best.to_dict("records"))
        selection_rows.extend(selection_detail)
    dataset_out = ensure_dir(Path(args.output_dir) / _slug(dataset_name))
    pd.DataFrame(selection_rows).to_csv(dataset_out / "table_selection_detail.csv", index=False)
    pd.DataFrame(cascade_rows).to_csv(dataset_out / "table_direct_cascade_selected.csv", index=False)
    pd.DataFrame(cascade_selection_rows).to_csv(dataset_out / "table_direct_cascade_selection_detail.csv", index=False)
    pd.DataFrame(hazard_rows).to_csv(dataset_out / "table_hazard_threshold_crossing.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(dataset_out / "qwen_score_coverage.csv", index=False)
    return {
        "raw": pd.DataFrame(all_rows),
        "cascade": pd.DataFrame(cascade_rows),
        "cascade_selection": pd.DataFrame(cascade_selection_rows),
        "hazard": pd.DataFrame(hazard_rows),
    }


def _slug(text: str) -> str:
    return text.lower().replace("-", "_").replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/adaptive_adapters")
    parser.add_argument("--qwen_score_col", default="qwen_prm_error")
    parser.add_argument("--target_seeds", nargs="*", type=int, default=list(range(2806, 2826)))
    parser.add_argument("--external_seeds", nargs="*", type=int, default=list(range(2806, 2816)))
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.025, 0.05, 0.075, 0.10])
    parser.add_argument("--lambda_grid_size", type=int, default=101)
    parser.add_argument("--class_weight", default="balanced")
    parser.add_argument("--penalty_gamma", type=float, default=1.0)
    parser.add_argument("--train_frac", type=float, default=0.5)
    parser.add_argument("--select_frac", type=float, default=0.15)
    parser.add_argument("--cal_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names/slugs to run, e.g. Target processbench prm800k.",
    )
    parser.add_argument("--run_direct_cascades", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_hazard_diagnostic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.target_seeds = args.target_seeds[:1]
        args.external_seeds = args.external_seeds[:1]
        args.alphas = [0.05]
        args.lambda_grid_size = min(args.lambda_grid_size, 51)

    outdir = ensure_dir(args.output_dir)
    datasets = [
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
        datasets = [dataset for dataset in datasets if _slug(dataset[0]) in wanted]
        if not datasets:
            raise ValueError(f"No datasets matched --datasets {args.datasets!r}")
    rows = []
    cascade_frames = []
    cascade_selection_frames = []
    hazard_frames = []
    for dataset_name, text_features, combined_features, qwen_csv, seeds in datasets:
        print(f"Running adaptive adapters for {dataset_name}", flush=True)
        result = _run_dataset(args, dataset_name, text_features, combined_features, qwen_csv, seeds)
        rows.append(result["raw"])
        cascade_frames.append(result["cascade"])
        cascade_selection_frames.append(result["cascade_selection"])
        hazard_frames.append(result["hazard"])
    raw = pd.concat(rows, ignore_index=True)
    raw.to_csv(outdir / "table_adaptive_all.csv", index=False)
    summary = _summarize(raw, ["dataset", "score", "label", "target", "row_type", "score_family", "alpha"])
    summary.to_csv(outdir / "table_adaptive_all_summary.csv", index=False)
    deltas = _paired_deltas(raw)
    deltas.to_csv(outdir / "table_paired_deltas.csv", index=False)
    freqs = _selection_frequencies(raw)
    freqs.to_csv(outdir / "table_selection_frequencies_all_alpha.csv", index=False)
    _write_tables(raw, deltas, freqs, outdir)
    _make_figures(raw, freqs, outdir)
    cascade = pd.concat(cascade_frames, ignore_index=True) if cascade_frames else pd.DataFrame()
    cascade_selection = pd.concat(cascade_selection_frames, ignore_index=True) if cascade_selection_frames else pd.DataFrame()
    hazard = pd.concat(hazard_frames, ignore_index=True) if hazard_frames else pd.DataFrame()
    cascade.to_csv(outdir / "table_direct_cascade_selected.csv", index=False)
    cascade_selection.to_csv(outdir / "table_direct_cascade_selection_detail.csv", index=False)
    hazard.to_csv(outdir / "table_hazard_threshold_crossing.csv", index=False)
    _make_extension_tables(cascade, hazard, outdir)
    _write_readme(outdir, raw, deltas, freqs)
    write_json(outdir / "run_config.json", vars(args))
    print(f"Wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
