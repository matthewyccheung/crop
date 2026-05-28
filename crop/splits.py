"""Trace-level splitting and flattening helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .data import StepRecord, TraceRecord


@dataclass
class Split:
    train: list[TraceRecord]
    cal: list[TraceRecord]
    test: list[TraceRecord]


def _strat_key(trace: TraceRecord, by_domain: bool, by_has_error: bool) -> tuple:
    key = []
    if by_domain:
        key.append(trace.domain)
    if by_has_error:
        key.append(int(trace.has_error))
    if not key:
        key.append("all")
    return tuple(key)


def _split_indices(n: int, train_frac: float, cal_frac: float, test_frac: float) -> tuple[int, int]:
    raw = np.asarray([train_frac, cal_frac, test_frac], dtype=float) * n
    counts = np.asarray([int(round(x)) for x in raw], dtype=int)
    positive = np.asarray([train_frac > 0, cal_frac > 0, test_frac > 0], dtype=bool)

    min_counts = positive.astype(int)
    if n < int(min_counts.sum()):
        counts[:] = 0
        for idx in np.argsort(raw)[::-1][:n]:
            counts[idx] = 1
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

    return int(counts[0]), int(counts[1])


def split_traces(
    traces: list[TraceRecord],
    train_frac: float = 0.6,
    cal_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 0,
    stratify_by_domain: bool = True,
    stratify_by_has_error: bool = True,
) -> Split:
    """Split traces into train/cal/test partitions without step leakage."""

    if not traces:
        raise ValueError("Cannot split empty trace list")
    total = train_frac + cal_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1, got {total}")

    rng = np.random.default_rng(seed)
    grouped: dict[tuple, list[TraceRecord]] = {}
    for trace in traces:
        grouped.setdefault(_strat_key(trace, stratify_by_domain, stratify_by_has_error), []).append(trace)

    train: list[TraceRecord] = []
    cal: list[TraceRecord] = []
    test: list[TraceRecord] = []
    for group in grouped.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_train, n_cal = _split_indices(len(shuffled), train_frac, cal_frac, test_frac)
        train.extend(shuffled[:n_train])
        cal.extend(shuffled[n_train : n_train + n_cal])
        test.extend(shuffled[n_train + n_cal :])

    rng.shuffle(train)
    rng.shuffle(cal)
    rng.shuffle(test)
    _assert_no_leakage(train, cal, test)
    return Split(train=train, cal=cal, test=test)


def _assert_no_leakage(train, cal, test) -> None:
    train_ids = {t.trace_id for t in train}
    cal_ids = {t.trace_id for t in cal}
    test_ids = {t.trace_id for t in test}
    if train_ids & cal_ids or train_ids & test_ids or cal_ids & test_ids:
        raise AssertionError("Trace leakage detected across split partitions")


def flatten_steps(traces: Iterable[TraceRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten traces into step arrays after splitting."""

    steps: list[StepRecord] = [step for trace in traces for step in trace.steps]
    if not steps:
        return (
            np.empty((0, 0), dtype=float),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=int),
        )
    X = np.vstack([s.x for s in steps])
    y = np.asarray([s.y_error for s in steps], dtype=int)
    groups = np.asarray([s.domain for s in steps], dtype=object)
    trace_ids = np.asarray([s.trace_id for s in steps], dtype=object)
    step_numbers = np.asarray([s.step_number for s in steps], dtype=int)
    return X, y, groups, trace_ids, step_numbers


def split_summary(split: Split) -> dict[str, dict[str, int]]:
    """Return compact class/domain/error-trace counts for logging."""

    summary: dict[str, dict[str, int]] = {}
    for name, traces in (("train", split.train), ("cal", split.cal), ("test", split.test)):
        _, y, groups, _, _ = flatten_steps(traces)
        counts: dict[str, int] = {
            "traces": len(traces),
            "steps": int(len(y)),
            "error_steps": int(y.sum()) if len(y) else 0,
            "error_traces": int(sum(t.has_error for t in traces)),
        }
        for group in sorted(set(groups.tolist())) if len(groups) else []:
            counts[f"domain_{group}"] = int(np.sum(groups == group))
        summary[name] = counts
    return summary
