"""Small, stable API for calibrating and applying CROP certificates.

The experiment modules expose every paper ablation.  This module is the
recommended entry point for users who already have one error-risk score per
reasoning step and only want the core CROP operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .risk_control import prefix_lengths, prefix_losses_by_lambda, select_lambda_crc


def _score_arrays(scores_by_trace: Iterable[Sequence[float]]) -> list[np.ndarray]:
    arrays = [np.asarray(scores, dtype=float).reshape(-1) for scores in scores_by_trace]
    if any(not np.all(np.isfinite(scores)) for scores in arrays):
        raise ValueError("scores must contain only finite values")
    return arrays


def _check_alignment(traces: Sequence[object], scores_by_trace: Sequence[np.ndarray]) -> None:
    if len(traces) != len(scores_by_trace):
        raise ValueError("traces and scores_by_trace must have the same length")
    for index, (trace, scores) in enumerate(zip(traces, scores_by_trace)):
        if len(trace.steps) != len(scores):
            raise ValueError(
                f"trace {index} has {len(trace.steps)} steps but {len(scores)} scores"
            )


@dataclass(frozen=True)
class PrefixCalibrator:
    """A fitted CROP threshold.

    ``alpha`` is the requested marginal prefix-contamination risk and
    ``corrected_calibration_risk`` is the finite-sample corrected risk at the
    selected threshold.  ``feasible`` is false when the calibration set is too
    small for the requested alpha (in particular, when ``1 / (n + 1) > alpha``).
    """

    alpha: float
    threshold: float
    corrected_calibration_risk: float
    n_calibration_traces: int
    feasible: bool

    def predict(self, scores_by_trace: Iterable[Sequence[float]]) -> np.ndarray:
        """Return the number of certified steps for each trace."""

        return prefix_lengths(_score_arrays(scores_by_trace), self.threshold)


def fit_prefix_calibrator(
    calibration_traces: Sequence[object],
    calibration_scores: Iterable[Sequence[float]],
    *,
    alpha: float = 0.05,
    thresholds: Sequence[float] | None = None,
    grid_size: int = 1001,
) -> PrefixCalibrator:
    """Fit CROP from labeled calibration traces and step-level error scores.

    Scores must be oriented so that larger values mean greater error risk.
    The calibration unit is a complete trace, not an individual step.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    traces = list(calibration_traces)
    scores = _score_arrays(calibration_scores)
    if not traces:
        raise ValueError("at least one calibration trace is required")
    _check_alignment(traces, scores)

    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if thresholds is None:
        observed = np.unique(np.concatenate(scores)) if scores else np.asarray([])
        if len(observed) > grid_size:
            observed = np.unique(
                np.quantile(observed, np.linspace(0.0, 1.0, grid_size))
            )
        # -inf represents the valid reject-all/empty-prefix option.
        grid = np.concatenate(([float("-inf")], observed))
    else:
        grid = np.asarray(thresholds, dtype=float).reshape(-1)
        if len(grid) == 0 or np.isnan(grid).any():
            raise ValueError("thresholds must be a nonempty array without NaNs")
        grid = np.unique(grid)
        if not np.isneginf(grid[0]):
            grid = np.concatenate(([float("-inf")], grid))

    losses = prefix_losses_by_lambda(traces, scores, grid)
    threshold, corrected = select_lambda_crc(
        losses, grid, alpha=alpha, direction="increasing"
    )
    return PrefixCalibrator(
        alpha=float(alpha),
        threshold=threshold,
        corrected_calibration_risk=corrected,
        n_calibration_traces=len(traces),
        feasible=bool(corrected <= alpha),
    )
