"""Prefix-aware score-learning helpers for CPCC experiments."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from crop.data import StepRecord, TraceRecord


def prefix_contamination_labels(step_errors: Iterable[int]) -> np.ndarray:
    """Return C_t = max_{j <= t} Y_j for a trace."""

    y = np.asarray(list(step_errors), dtype=int)
    if y.size == 0:
        return np.asarray([], dtype=int)
    y = np.asarray(y > 0, dtype=int)
    return np.maximum.accumulate(y)


def first_error_hazard_labels(step_errors: Iterable[int]) -> np.ndarray:
    """Return H_t = 1{t = F}, where F is the first annotated error."""

    y = np.asarray(list(step_errors), dtype=int)
    if y.size == 0:
        return np.asarray([], dtype=int)
    y = np.asarray(y > 0, dtype=int)
    hazard = np.zeros_like(y, dtype=int)
    hits = np.flatnonzero(y)
    if len(hits):
        hazard[int(hits[0])] = 1
    return hazard


def first_error_risk_set_mask(step_errors: Iterable[int]) -> np.ndarray:
    """Return the discrete-time hazard risk set t <= F.

    Clean traces keep every step. Erroneous traces keep pre-error steps and the
    first error step, censoring post-error steps for hazard training.
    """

    y = np.asarray(list(step_errors), dtype=int)
    if y.size == 0:
        return np.asarray([], dtype=bool)
    hits = np.flatnonzero(y > 0)
    mask = np.ones_like(y, dtype=bool)
    if len(hits):
        mask[int(hits[0]) + 1 :] = False
    return mask


def prefix_labels_by_trace(traces: Iterable[TraceRecord]) -> list[np.ndarray]:
    return [prefix_contamination_labels(trace.y_errors) for trace in traces]


def hazard_labels_by_trace(traces: Iterable[TraceRecord]) -> list[np.ndarray]:
    return [first_error_hazard_labels(trace.y_errors) for trace in traces]


def flatten_prefix_labels(traces: Iterable[TraceRecord]) -> np.ndarray:
    labels = prefix_labels_by_trace(traces)
    if not labels:
        return np.asarray([], dtype=int)
    return np.concatenate(labels).astype(int)


def flatten_hazard_labels(traces: Iterable[TraceRecord]) -> np.ndarray:
    labels = hazard_labels_by_trace(traces)
    if not labels:
        return np.asarray([], dtype=int)
    return np.concatenate(labels).astype(int)


def traces_with_prefix_targets(traces: Iterable[TraceRecord]) -> list[TraceRecord]:
    """Copy traces with step labels replaced by derived prefix-contamination labels.

    This is used only for score-model training. Calibration and evaluation should
    keep the original local step labels so CPCC losses still measure whether the
    returned prefix contains any annotated step error.
    """

    out: list[TraceRecord] = []
    for trace in traces:
        prefix_y = prefix_contamination_labels(trace.y_errors)
        steps: list[StepRecord] = []
        for step, label in zip(trace.steps, prefix_y):
            meta = dict(step.metadata)
            meta["prefix_contamination_label"] = int(label)
            steps.append(
                replace(
                    step,
                    y_error=int(label),
                    is_correct=not bool(label),
                    metadata=meta,
                )
            )
        out.append(replace(trace, steps=steps))
    return out


def traces_with_hazard_targets(traces: Iterable[TraceRecord]) -> list[TraceRecord]:
    """Copy traces with labels replaced by first-error hazard labels.

    Post-error steps are removed from erroneous training traces so the fitted
    classifier sees only the discrete-time risk set t <= F. Calibration and
    evaluation callers should continue to use the original uncensored traces.
    """

    out: list[TraceRecord] = []
    for trace in traces:
        hazard_y = first_error_hazard_labels(trace.y_errors)
        risk_set = first_error_risk_set_mask(trace.y_errors)
        steps: list[StepRecord] = []
        for step, label, keep in zip(trace.steps, hazard_y, risk_set):
            if not keep:
                continue
            meta = dict(step.metadata)
            meta["first_error_hazard_label"] = int(label)
            meta["hazard_risk_set"] = True
            steps.append(
                replace(
                    step,
                    y_error=int(label),
                    is_correct=not bool(label),
                    metadata=meta,
                )
            )
        out.append(replace(trace, steps=steps))
    return out


def _clean_matrix(trace: TraceRecord) -> np.ndarray:
    return np.nan_to_num(trace.X, nan=0.0, posinf=0.0, neginf=0.0)


def augment_with_prefix_features(
    traces: Iterable[TraceRecord],
    *,
    include_position_features: bool = True,
) -> list[TraceRecord]:
    """Append label-free prefix summaries to each step feature vector.

    The augmented vector contains the current features, cumulative mean and max
    of previous/current feature values, previous-step features, and optionally
    position/length features. All features are deterministic functions of the
    completed trace and cached non-label score/text features.
    """

    out: list[TraceRecord] = []
    for trace in traces:
        X = _clean_matrix(trace)
        if X.size == 0:
            out.append(trace)
            continue
        n_steps = X.shape[0]
        denom = np.arange(1, n_steps + 1, dtype=float)[:, None]
        cumulative_mean = np.cumsum(X, axis=0) / denom
        cumulative_max = np.maximum.accumulate(X, axis=0)
        previous = np.vstack([np.zeros((1, X.shape[1]), dtype=float), X[:-1]])
        extra_parts = [X, cumulative_mean, cumulative_max, previous]
        if include_position_features:
            idx = np.arange(n_steps, dtype=float)
            rel_pos = idx / max(n_steps - 1, 1)
            extra_parts.append(
                np.column_stack(
                    [
                        idx,
                        rel_pos,
                        np.full(n_steps, float(n_steps)),
                    ]
                )
            )
        features = np.hstack(extra_parts)
        steps = [replace(step, x=features[i]) for i, step in enumerate(trace.steps)]
        out.append(replace(trace, steps=steps))
    return out


def select_named_feature_columns(
    traces: Iterable[TraceRecord],
    *,
    keep_names: set[str],
) -> list[TraceRecord]:
    """Keep cached feature columns whose metadata names are in ``keep_names``."""

    traces = list(traces)
    if not traces:
        return []
    names = traces[0].steps[0].metadata.get("_feature_names")
    if not names:
        raise ValueError("Feature names are required for named-column selection")
    idx = [i for i, name in enumerate(names) if str(name) in keep_names]
    if not idx:
        raise ValueError(f"No requested feature columns found. Requested: {sorted(keep_names)}")
    out: list[TraceRecord] = []
    for trace in traces:
        steps = [replace(step, x=np.asarray(step.x, dtype=float)[idx]) for step in trace.steps]
        out.append(replace(trace, steps=steps))
    return out


def append_trace_score_feature(
    traces: Iterable[TraceRecord],
    scores_by_trace_id: dict[str, dict[int, float]],
    *,
    missing_value: float = 0.5,
) -> list[TraceRecord]:
    """Append one cached per-step score column, keyed by trace id and step index."""

    out: list[TraceRecord] = []
    for trace in traces:
        trace_scores = scores_by_trace_id.get(trace.trace_id, {})
        steps = []
        for step in trace.steps:
            value = float(trace_scores.get(step.step_number, missing_value))
            x = np.concatenate([np.asarray(step.x, dtype=float), np.asarray([value], dtype=float)])
            steps.append(replace(step, x=x))
        out.append(replace(trace, steps=steps))
    return out
