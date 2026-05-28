"""Shared configuration and metrics for paper-result reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable

import numpy as np


ALPHA = 0.05
TARGET_SEEDS = list(range(2806, 2826))
EXTERNAL_SEEDS = list(range(2806, 2816))

TARGET_TEXT = "data/cheap_baselines/crop_target_text_steps.npz"
TARGET_COMBINED = "data/strengthened/crop_target_combined_steps.npz"
TARGET_QWEN = "outputs/strengthened/final/process_repeated_qwen_prm/qwen_prm_scores.csv"


def target_args() -> SimpleNamespace:
    return SimpleNamespace(
        qwen_score_col="qwen_prm_error",
        target_seeds=TARGET_SEEDS,
        external_seeds=[],
        datasets=["Target"],
        threshold_grid_size=201,
        score_train_frac=0.60,
        gate_select_frac=0.0,
        cpcc_calibration_frac=0.20,
        test_frac=0.20,
        class_weight="balanced",
    )


@dataclass(frozen=True)
class BoundaryMetrics:
    prefix_risk: float
    prefix_kept: float
    clean_prefix_recovery: float
    over_withholding: float
    unsafe_overshoot: float
    review_burden: float


def oracle_lengths(traces: Iterable[object]) -> np.ndarray:
    lengths: list[int] = []
    for trace in traces:
        total = len(trace.steps)
        if trace.first_error is None:
            lengths.append(total)
        else:
            lengths.append(max(0, min(int(trace.first_error), total)))
    return np.asarray(lengths, dtype=int)


def trace_totals(traces: Iterable[object]) -> np.ndarray:
    return np.asarray([len(trace.steps) for trace in traces], dtype=int)


def boundary_metrics(traces: list[object], retained: np.ndarray) -> BoundaryMetrics:
    totals = trace_totals(traces)
    retained = np.asarray(retained, dtype=int)
    keep = totals > 0
    if not np.all(keep):
        traces = [trace for trace, ok in zip(traces, keep) if ok]
        totals = totals[keep]
        retained = retained[keep]
    if len(totals) == 0:
        nan = float("nan")
        return BoundaryMetrics(nan, nan, nan, nan, nan, nan)

    oracle = oracle_lengths(traces)
    retained = np.clip(retained, 0, totals)
    totals_f = totals.astype(float)
    oracle_f = oracle.astype(float)
    retained_f = retained.astype(float)
    clean_den = np.maximum(oracle_f, 1.0)

    return BoundaryMetrics(
        prefix_risk=float(np.mean(retained > oracle)),
        prefix_kept=float(np.mean(retained_f / totals_f)),
        clean_prefix_recovery=float(np.mean(np.minimum(retained_f, oracle_f) / clean_den)),
        over_withholding=float(np.mean(np.maximum(oracle_f - retained_f, 0.0) / totals_f)),
        unsafe_overshoot=float(np.mean(np.maximum(retained_f - oracle_f, 0.0) / totals_f)),
        review_burden=float(np.mean(1.0 - retained_f / totals_f)),
    )
