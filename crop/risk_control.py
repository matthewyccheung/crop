"""Conformal risk-control utilities for trace-level certificates."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import beta

from .sequence import candidate_first_error_set, prefix_contains_error


def corrected_risk(losses: np.ndarray, correction: Literal["plus_one"] = "plus_one") -> float:
    losses = np.asarray(losses, dtype=float)
    if len(losses) == 0:
        return float("inf")
    if correction == "plus_one":
        return float((1.0 + np.sum(losses)) / (len(losses) + 1.0))
    raise ValueError(f"Unknown correction={correction!r}")


def select_lambda_crc(
    losses_by_lambda: np.ndarray,
    lambdas: np.ndarray,
    alpha: float,
    direction: Literal["increasing", "decreasing"],
    correction: Literal["plus_one"] = "plus_one",
) -> tuple[float, float]:
    """Select a threshold using conservative conformal risk control.

    Returns ``(lambda_hat, corrected_cal_risk)``.
    """

    losses_by_lambda = np.asarray(losses_by_lambda, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    if losses_by_lambda.shape != (len(lambdas), losses_by_lambda.shape[1]):
        raise ValueError("losses_by_lambda must have shape [n_lambdas, n_examples]")
    risks = np.asarray([corrected_risk(row, correction=correction) for row in losses_by_lambda])
    valid = np.flatnonzero(risks <= alpha)
    if len(valid) == 0:
        idx = int(np.argmin(lambdas)) if direction == "increasing" else int(np.argmax(lambdas))
        return float(lambdas[idx]), float(risks[idx])
    if direction == "increasing":
        idx = valid[np.argmax(lambdas[valid])]
    elif direction == "decreasing":
        idx = valid[np.argmin(lambdas[valid])]
    else:
        raise ValueError("direction must be 'increasing' or 'decreasing'")
    return float(lambdas[idx]), float(risks[idx])


def prefix_length(scores: np.ndarray, lambda_: float) -> int:
    """Largest prefix length whose scores are all at most ``lambda_``."""

    scores = np.asarray(scores, dtype=float)
    for idx, score in enumerate(scores):
        if score > lambda_:
            return int(idx)
    return int(len(scores))


def prefix_lengths(scores_by_trace, lambda_: float) -> np.ndarray:
    return np.asarray([prefix_length(scores, lambda_) for scores in scores_by_trace], dtype=int)


def prefix_contamination_losses(traces, scores_by_trace, lambda_: float) -> np.ndarray:
    losses = []
    for trace, scores in zip(traces, scores_by_trace):
        m = prefix_length(scores, lambda_)
        losses.append(prefix_contains_error(trace.y_errors, m))
    return np.asarray(losses, dtype=int)


def prefix_losses_by_lambda(traces, scores_by_trace, lambdas: np.ndarray) -> np.ndarray:
    return np.vstack([prefix_contamination_losses(traces, scores_by_trace, lam) for lam in lambdas])


def whole_trace_false_accept_losses(traces, scores_by_trace, lambda_: float) -> np.ndarray:
    losses = []
    for trace, scores in zip(traces, scores_by_trace):
        accepted = bool(np.max(scores) <= lambda_) if len(scores) else True
        losses.append(accepted and trace.has_error)
    return np.asarray(losses, dtype=int)


def first_error_localization_losses(traces, scores_by_trace, lambda_: float) -> np.ndarray:
    losses = []
    for trace, scores in zip(traces, scores_by_trace):
        candidates = candidate_first_error_set(scores, lambda_, include_no_error=True)
        losses.append(trace.first_error not in candidates)
    return np.asarray(losses, dtype=int)


def first_error_losses_by_lambda(traces, scores_by_trace, lambdas: np.ndarray) -> np.ndarray:
    return np.vstack([first_error_localization_losses(traces, scores_by_trace, lam) for lam in lambdas])


def first_error_error_only_localization_losses(traces, scores_by_trace, lambda_: float) -> np.ndarray:
    """First-error miss losses on traces known to contain an error.

    Unlike ``first_error_localization_losses``, the candidate set does not
    include the no-error option. Callers should pass only erroneous traces.
    """

    losses = []
    for trace, scores in zip(traces, scores_by_trace):
        if trace.first_error is None:
            continue
        candidates = set(np.flatnonzero(np.asarray(scores, dtype=float) >= lambda_).astype(int).tolist())
        losses.append(trace.first_error not in candidates)
    return np.asarray(losses, dtype=int)


def first_error_error_only_losses_by_lambda(traces, scores_by_trace, lambdas: np.ndarray) -> np.ndarray:
    return np.vstack([first_error_error_only_localization_losses(traces, scores_by_trace, lam) for lam in lambdas])


def clopper_pearson_upper(bad: int, total: int, delta: float) -> float:
    """One-sided Clopper-Pearson upper confidence bound for a binomial rate."""

    if total <= 0:
        return 0.0
    if bad >= total:
        return 1.0
    return float(beta.ppf(1.0 - delta, bad + 1, total - bad))


def calibrate_selective_acceptance(
    traces,
    scores_by_trace,
    lambdas: np.ndarray,
    beta_level: float,
    delta: float,
) -> dict[str, float | int | bool]:
    """Select a trace-acceptance threshold with finite-grid selective risk.

    The selected threshold is the largest grid value whose one-sided
    Clopper-Pearson upper bound on accepted-error rate is at most
    ``beta_level`` after a union correction over the finite grid. If no
    nonempty acceptance threshold is feasible, a reject-all sentinel is
    returned.
    """

    lambdas = np.asarray(lambdas, dtype=float)
    per_lambda_delta = delta / max(len(lambdas), 1)
    selected: dict[str, float | int | bool] | None = None
    for lambda_ in lambdas:
        accepted = np.asarray([np.max(scores) <= lambda_ if len(scores) else True for scores in scores_by_trace], dtype=bool)
        bad = int(sum(bool(acc) and trace.has_error for acc, trace in zip(accepted, traces)))
        total = int(np.sum(accepted))
        upper = clopper_pearson_upper(bad, total, per_lambda_delta)
        if total > 0 and upper <= beta_level:
            selected = {
                "lambda": float(lambda_),
                "cp_upper": upper,
                "bad_accepts": bad,
                "accepted": total,
                "feasible": True,
            }
    if selected is not None:
        return selected
    return {
        "lambda": float("-inf"),
        "cp_upper": 0.0,
        "bad_accepts": 0,
        "accepted": 0,
        "feasible": False,
    }
