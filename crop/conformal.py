"""Conformal prediction sets and p-values."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .scores import aps_scores, lac_scores


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split conformal quantile for upper-tail scores."""

    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return np.inf
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    n = len(scores)
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return np.inf
    k = max(k, 1)
    return float(np.sort(scores)[k - 1])


def lower_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Lower-tail class-conditional threshold with finite-sample correction."""

    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return np.inf
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    n = len(scores)
    k = int(math.floor((n + 1) * alpha))
    if k < 1:
        return -np.inf
    k = min(k, n)
    return float(np.sort(scores)[k - 1])


def fit_lac_threshold(cal_probs: np.ndarray, cal_y: np.ndarray, alpha: float) -> float:
    return conformal_quantile(lac_scores(cal_probs, cal_y), alpha)


def predict_lac_sets(test_probs: np.ndarray, qhat: float) -> list[set[int]]:
    probs = np.asarray(test_probs, dtype=float)
    sets: list[set[int]] = []
    for row in probs:
        labels = {int(label) for label in (0, 1) if 1.0 - row[label] <= qhat}
        if not labels:
            labels.add(int(np.argmax(row)))
        sets.append(labels)
    return sets


def fit_aps_threshold(cal_probs: np.ndarray, cal_y: np.ndarray, alpha: float, randomized: bool = False, seed=None) -> float:
    return conformal_quantile(aps_scores(cal_probs, cal_y, randomized=randomized, seed=seed), alpha)


def predict_aps_sets(test_probs: np.ndarray, qhat: float) -> list[set[int]]:
    probs = np.asarray(test_probs, dtype=float)
    sets: list[set[int]] = []
    for row in probs:
        order = np.argsort(-row)
        cumulative = 0.0
        labels: set[int] = set()
        for label in order:
            cumulative += float(row[label])
            labels.add(int(label))
            if cumulative >= qhat:
                break
        if not labels:
            labels.add(int(order[0]))
        sets.append(labels)
    return sets


def fit_class_conditional_lac(
    cal_probs: np.ndarray,
    cal_y: np.ndarray,
    alpha_by_class: Mapping[int, float] | float,
) -> dict[int, float]:
    if isinstance(alpha_by_class, (float, int)):
        alpha_by_class = {0: float(alpha_by_class), 1: float(alpha_by_class)}
    scores = lac_scores(cal_probs, cal_y)
    thresholds: dict[int, float] = {}
    for label in (0, 1):
        alpha = float(alpha_by_class[label])
        thresholds[label] = conformal_quantile(scores[np.asarray(cal_y) == label], alpha)
    return thresholds


def predict_class_conditional_lac(test_probs: np.ndarray, thresholds: Mapping[int, float]) -> list[set[int]]:
    probs = np.asarray(test_probs, dtype=float)
    sets: list[set[int]] = []
    for row in probs:
        labels = {int(label) for label in (0, 1) if 1.0 - row[label] <= thresholds[label]}
        if not labels:
            labels.add(int(np.argmax(row)))
        sets.append(labels)
    return sets


def class_conditional_p_values(cal_probs: np.ndarray, cal_y: np.ndarray, test_probs: np.ndarray) -> np.ndarray:
    """Return columns ``[p_correct, p_error]`` using class-conditional LAC scores."""

    cal_y = np.asarray(cal_y, dtype=int)
    cal_probs = np.asarray(cal_probs, dtype=float)
    test_probs = np.asarray(test_probs, dtype=float)
    pvals = np.empty((len(test_probs), 2), dtype=float)
    for label in (0, 1):
        cal_scores = 1.0 - cal_probs[cal_y == label, label]
        n = len(cal_scores)
        if n == 0:
            pvals[:, label] = 1.0
            continue
        test_scores = 1.0 - test_probs[:, label]
        pvals[:, label] = (1.0 + np.sum(cal_scores[None, :] >= test_scores[:, None], axis=1)) / (n + 1.0)
    return pvals
