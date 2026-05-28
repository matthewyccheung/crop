"""Metrics for verifier scores and conformal objects."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .sequence import first_error_index


def safe_auroc(y_true, scores) -> float:
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def safe_aupr(y_true, scores) -> float:
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, scores))


def fpr_at_recall_95(y_true, scores) -> float:
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = scores[y == 1]
    negatives = scores[y == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    k = int(math.ceil(0.95 * len(positives)))
    threshold = np.sort(positives)[::-1][k - 1]
    return float(np.mean(negatives >= threshold))


def balanced_accuracy_at_threshold(y_true, scores, threshold: float = 0.5) -> float:
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    return float(balanced_accuracy_score(y_true, y_pred))


def precision_recall_f1_at_threshold(y_true, scores, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def prediction_set_coverage(sets: Iterable[set[int]], y) -> float:
    sets = list(sets)
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return float("nan")
    return float(np.mean([int(label in pred_set) for pred_set, label in zip(sets, y)]))


def class_conditional_coverage(sets: Iterable[set[int]], y) -> dict[int, float]:
    sets = list(sets)
    y = np.asarray(y, dtype=int)
    out: dict[int, float] = {}
    for label in (0, 1):
        mask = y == label
        out[label] = (
            float(np.mean([int(y_i in s) for s, y_i, keep in zip(sets, y, mask) if keep]))
            if np.any(mask)
            else float("nan")
        )
    return out


def average_set_size(sets: Iterable[set]) -> float:
    sets = list(sets)
    return float(np.mean([len(s) for s in sets])) if sets else float("nan")


def singleton_rate(sets: Iterable[set]) -> float:
    sets = list(sets)
    return float(np.mean([len(s) == 1 for s in sets])) if sets else float("nan")


def empty_set_rate(sets: Iterable[set]) -> float:
    sets = list(sets)
    return float(np.mean([len(s) == 0 for s in sets])) if sets else float("nan")


def ambiguous_rate(sets: Iterable[set]) -> float:
    sets = list(sets)
    return float(np.mean([len(s) == 2 for s in sets])) if sets else float("nan")


def incorrect_singleton_rate(sets: Iterable[set[int]], y) -> float:
    sets = list(sets)
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return float("nan")
    return float(np.mean([len(s) == 1 and label not in s for s, label in zip(sets, y)]))


def error_detection_metrics(y_true, scores, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    flagged = np.asarray(scores, dtype=float) >= threshold
    error_mask = y == 1
    correct_mask = y == 0
    recall = float(np.mean(flagged[error_mask])) if np.any(error_mask) else float("nan")
    fpr = float(np.mean(flagged[correct_mask])) if np.any(correct_mask) else float("nan")
    precision = float(np.mean(y[flagged])) if np.any(flagged) else 0.0
    return {
        "error_recall": recall,
        "missed_error_rate": float(1.0 - recall) if not math.isnan(recall) else float("nan"),
        "false_positive_rate": fpr,
        "precision": precision,
        "flagged_fraction": float(np.mean(flagged)) if len(y) else float("nan"),
    }


def prefix_contamination_rate(losses) -> float:
    losses = np.asarray(losses, dtype=float)
    return float(np.mean(losses)) if len(losses) else float("nan")


def avg_prefix_len(lengths) -> float:
    lengths = np.asarray(lengths, dtype=float)
    return float(np.mean(lengths)) if len(lengths) else float("nan")


def avg_prefix_frac(lengths, totals) -> float:
    lengths = np.asarray(lengths, dtype=float)
    totals = np.asarray(totals, dtype=float)
    return float(np.mean(lengths / np.maximum(totals, 1))) if len(lengths) else float("nan")


def median_prefix_frac(lengths, totals) -> float:
    lengths = np.asarray(lengths, dtype=float)
    totals = np.asarray(totals, dtype=float)
    return float(np.median(lengths / np.maximum(totals, 1))) if len(lengths) else float("nan")


def full_trace_accept_rate(lengths, totals) -> float:
    lengths = np.asarray(lengths)
    totals = np.asarray(totals)
    return float(np.mean(lengths == totals)) if len(lengths) else float("nan")


def clean_full_trace_accept_rate(traces, lengths) -> float:
    clean = np.asarray([not t.has_error for t in traces], dtype=bool)
    if not np.any(clean):
        return float("nan")
    totals = np.asarray([len(t.steps) for t in traces])
    lengths = np.asarray(lengths)
    return float(np.mean(lengths[clean] == totals[clean]))


def contaminated_full_trace_accept_rate(traces, lengths) -> float:
    contaminated = np.asarray([t.has_error for t in traces], dtype=bool)
    if not np.any(contaminated):
        return float("nan")
    totals = np.asarray([len(t.steps) for t in traces])
    lengths = np.asarray(lengths)
    return float(np.mean(lengths[contaminated] == totals[contaminated]))


def first_error_coverage(candidate_sets: Iterable[set[int | None]], traces) -> float:
    candidate_sets = list(candidate_sets)
    if not candidate_sets:
        return float("nan")
    truths = [t.first_error for t in traces]
    return float(np.mean([truth in s for truth, s in zip(truths, candidate_sets)]))


def avg_candidate_set_size(candidate_sets: Iterable[set]) -> float:
    return average_set_size(candidate_sets)


def median_candidate_set_size(candidate_sets: Iterable[set]) -> float:
    sets = list(candidate_sets)
    return float(np.median([len(s) for s in sets])) if sets else float("nan")


def no_error_coverage(candidate_sets: Iterable[set[int | None]], traces) -> float:
    candidate_sets = list(candidate_sets)
    clean = [i for i, t in enumerate(traces) if not t.has_error]
    if not clean:
        return float("nan")
    return float(np.mean([None in candidate_sets[i] for i in clean]))


def mean_nearest_distance(candidate_sets: Iterable[set[int | None]], traces) -> float:
    distances: list[float] = []
    for candidate_set, trace in zip(candidate_sets, traces):
        truth = trace.first_error
        if truth is None:
            continue
        numeric = [c for c in candidate_set if c is not None]
        if not numeric:
            distances.append(float(len(trace.steps)))
        else:
            distances.append(float(min(abs(int(c) - truth) for c in numeric)))
    return float(np.mean(distances)) if distances else float("nan")


def top1_first_error_accuracy(scores_by_trace, traces) -> float:
    hits = []
    for scores, trace in zip(scores_by_trace, traces):
        truth = trace.first_error
        if truth is None:
            continue
        pred = int(np.argmax(scores))
        hits.append(pred == truth)
    return float(np.mean(hits)) if hits else float("nan")


def first_error_diagnostics(candidate_sets: Iterable[set[int | None]], scores_by_trace, traces) -> dict[str, float]:
    """Detailed first-error metrics that do not hide behind clean traces.

    Candidate sets may include ``None`` to denote the no-error option.  Metrics
    with ``error_only`` are computed only over traces with a true first error.
    """

    candidate_sets = list(candidate_sets)
    traces = list(traces)
    if not candidate_sets:
        return {
            "fe_coverage_all": float("nan"),
            "fe_coverage_error_only": float("nan"),
            "fe_candidate_size_all": float("nan"),
            "fe_candidate_size_error_only": float("nan"),
            "fe_candidate_size_excluding_empty": float("nan"),
            "fe_top1_accuracy_error_only": float("nan"),
            "fe_top1_mean_abs_distance_error_only": float("nan"),
            "fe_top1_median_abs_distance_error_only": float("nan"),
            "fe_within1_error_only": float("nan"),
            "fe_within2_error_only": float("nan"),
            "fe_mean_nearest_distance_error_only": float("nan"),
            "fe_candidate_before_first_error_rate": float("nan"),
            "fe_candidate_after_first_error_rate": float("nan"),
            "fe_empty_included_rate": float("nan"),
            "fe_only_empty_rate": float("nan"),
            "clean_trace_empty_precision": float("nan"),
            "clean_trace_false_alarm_rate": float("nan"),
            "false_localization_on_clean": float("nan"),
        }

    is_error = np.asarray([trace.has_error for trace in traces], dtype=bool)
    is_clean = ~is_error
    numeric_sizes = np.asarray([sum(candidate is not None for candidate in s) for s in candidate_sets], dtype=float)
    sizes = np.asarray([len(s) for s in candidate_sets], dtype=float)
    covers = np.asarray([trace.first_error in s for trace, s in zip(traces, candidate_sets)], dtype=bool)
    empty_included = np.asarray([None in s for s in candidate_sets], dtype=bool)
    only_empty = np.asarray([s == {None} for s in candidate_sets], dtype=bool)
    clean_empty_denom = only_empty | empty_included
    false_clean = np.asarray(
        [trace.first_error is None and any(candidate is not None for candidate in s) for trace, s in zip(traces, candidate_sets)],
        dtype=bool,
    )
    nearest_distances = []
    top1_distances = []
    before_flags = []
    after_flags = []
    for trace, candidate_set in zip(traces, candidate_sets):
        truth = trace.first_error
        if truth is None:
            continue
        numeric = [int(candidate) for candidate in candidate_set if candidate is not None]
        if numeric:
            nearest_distances.append(float(min(abs(candidate - truth) for candidate in numeric)))
        else:
            nearest_distances.append(float(len(trace.steps)))
        before_flags.append(any(candidate < truth for candidate in numeric))
        after_flags.append(any(candidate > truth for candidate in numeric))
    for scores, trace in zip(scores_by_trace, traces):
        truth = trace.first_error
        if truth is None:
            continue
        if len(scores):
            top1_distances.append(float(abs(int(np.argmax(scores)) - truth)))
        else:
            top1_distances.append(float(len(trace.steps)))
    nearest = np.asarray(nearest_distances, dtype=float)
    top1_distance = np.asarray(top1_distances, dtype=float)

    return {
        "fe_coverage_all": float(np.mean(covers)),
        "fe_coverage_error_only": float(np.mean(covers[is_error])) if np.any(is_error) else float("nan"),
        "fe_candidate_size_all": float(np.mean(sizes)),
        "fe_candidate_size_error_only": float(np.mean(sizes[is_error])) if np.any(is_error) else float("nan"),
        "fe_candidate_size_excluding_empty": float(np.mean(numeric_sizes)),
        "fe_top1_accuracy_error_only": top1_first_error_accuracy(scores_by_trace, traces),
        "fe_top1_mean_abs_distance_error_only": float(np.mean(top1_distance)) if len(top1_distance) else float("nan"),
        "fe_top1_median_abs_distance_error_only": float(np.median(top1_distance)) if len(top1_distance) else float("nan"),
        "fe_within1_error_only": float(np.mean(nearest <= 1.0)) if len(nearest) else float("nan"),
        "fe_within2_error_only": float(np.mean(nearest <= 2.0)) if len(nearest) else float("nan"),
        "fe_mean_nearest_distance_error_only": float(np.mean(nearest)) if len(nearest) else float("nan"),
        "fe_candidate_before_first_error_rate": float(np.mean(before_flags)) if before_flags else float("nan"),
        "fe_candidate_after_first_error_rate": float(np.mean(after_flags)) if after_flags else float("nan"),
        "fe_empty_included_rate": float(np.mean(empty_included)),
        "fe_only_empty_rate": float(np.mean(only_empty)),
        "clean_trace_empty_precision": float(np.mean(is_clean[clean_empty_denom])) if np.any(clean_empty_denom) else float("nan"),
        "clean_trace_false_alarm_rate": float(np.mean(false_clean[is_clean])) if np.any(is_clean) else float("nan"),
        "false_localization_on_clean": float(np.mean(false_clean[is_clean])) if np.any(is_clean) else float("nan"),
    }


def prefix_diagnostics(traces, lengths) -> dict[str, float]:
    """Detailed prefix-retention diagnostics."""

    traces = list(traces)
    lengths = np.asarray(lengths, dtype=int)
    if len(traces) == 0:
        return {
            "prefix_nonempty_rate": float("nan"),
            "prefix_retained_fraction_on_clean": float("nan"),
            "prefix_retained_fraction_on_error": float("nan"),
            "prefix_stops_before_first_error_rate": float("nan"),
            "prefix_stops_at_or_before_first_error_rate": float("nan"),
            "prefix_overruns_first_error_rate": float("nan"),
        }
    totals = np.asarray([len(trace.steps) for trace in traces], dtype=float)
    fracs = lengths / np.maximum(totals, 1.0)
    is_error = np.asarray([trace.has_error for trace in traces], dtype=bool)
    is_clean = ~is_error
    first_errors = np.asarray([trace.first_error if trace.first_error is not None else -1 for trace in traces], dtype=int)
    err_lengths = lengths[is_error]
    err_first = first_errors[is_error]
    return {
        "prefix_nonempty_rate": float(np.mean(lengths > 0)),
        "prefix_retained_fraction_on_clean": float(np.mean(fracs[is_clean])) if np.any(is_clean) else float("nan"),
        "prefix_retained_fraction_on_error": float(np.mean(fracs[is_error])) if np.any(is_error) else float("nan"),
        "prefix_stops_before_first_error_rate": float(np.mean(err_lengths < err_first)) if len(err_lengths) else float("nan"),
        "prefix_stops_at_or_before_first_error_rate": float(np.mean(err_lengths <= err_first)) if len(err_lengths) else float("nan"),
        "prefix_overruns_first_error_rate": float(np.mean(err_lengths > err_first)) if len(err_lengths) else float("nan"),
    }
