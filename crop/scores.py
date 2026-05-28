"""Score functions used by conformal procedures."""

from __future__ import annotations

import numpy as np


def _check_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise ValueError(f"Expected binary probabilities with shape [n,2], got {probs.shape}")
    if np.any(probs < -1e-8) or np.any(probs > 1 + 1e-8):
        raise ValueError("Probabilities must lie in [0,1]")
    if not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Rows of probs must sum to 1")
    return probs


def _check_y(y_true) -> np.ndarray:
    y = np.asarray(y_true, dtype=int)
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("y_true must contain only 0/1 labels")
    return y


def lac_scores(probs: np.ndarray, y_true) -> np.ndarray:
    probs = _check_probs(probs)
    y = _check_y(y_true)
    return 1.0 - probs[np.arange(len(y)), y]


def aps_scores(probs: np.ndarray, y_true, randomized: bool = False, seed: int | None = None) -> np.ndarray:
    probs = _check_probs(probs)
    y = _check_y(y_true)
    rng = np.random.default_rng(seed)
    scores = np.empty(len(y), dtype=float)
    for i, label in enumerate(y):
        order = np.argsort(-probs[i])
        rank = int(np.where(order == label)[0][0])
        cumulative = float(np.sum(probs[i, order[: rank + 1]]))
        u = float(rng.random()) if randomized else 0.0
        scores[i] = cumulative - u * probs[i, label]
    return scores


def error_scores(probs: np.ndarray) -> np.ndarray:
    probs = _check_probs(probs)
    return probs[:, 1]


def negative_error_scores(probs: np.ndarray) -> np.ndarray:
    return -error_scores(probs)
