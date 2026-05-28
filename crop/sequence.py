"""Sequence-level utilities for reasoning traces."""

from __future__ import annotations

from typing import Optional

import numpy as np


NO_ERROR = None


def first_error_index(y_errors) -> Optional[int]:
    """Return the 0-indexed first error step, or ``None`` for clean traces."""

    y = np.asarray(y_errors, dtype=int)
    hits = np.flatnonzero(y == 1)
    if len(hits) == 0:
        return None
    return int(hits[0])


def has_error(y_errors) -> bool:
    return first_error_index(y_errors) is not None


def prefix_contains_error(y_errors, m: int) -> bool:
    """Whether the first ``m`` steps contain at least one error."""

    if m <= 0:
        return False
    y = np.asarray(y_errors, dtype=int)
    return bool(np.any(y[:m] == 1))


def candidate_first_error_set(
    scores,
    lambda_: float,
    include_no_error: bool = True,
) -> set[int | None]:
    """Return candidate first-error indices where error score crosses threshold."""

    scores = np.asarray(scores, dtype=float)
    candidates: set[int | None] = set(np.flatnonzero(scores >= lambda_).astype(int).tolist())
    if include_no_error:
        candidates.add(NO_ERROR)
    return candidates
