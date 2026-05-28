"""Data structures and loaders for CROP feature files."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Optional

import numpy as np

from .labels import extract_step_label, metadata_get, normalize_step_label
from .sequence import first_error_index


@dataclass
class StepRecord:
    trace_id: str
    domain: str
    complexity: Optional[int]
    step_number: int
    total_steps: Optional[int]
    before_after: str
    x: np.ndarray
    y_error: int
    is_correct: bool
    original_expression: Optional[str]
    step_content: Optional[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceRecord:
    trace_id: str
    domain: str
    complexity: Optional[int]
    steps: list[StepRecord]

    @property
    def y_errors(self) -> np.ndarray:
        return np.asarray([s.y_error for s in self.steps], dtype=int)

    @property
    def X(self) -> np.ndarray:
        return np.vstack([s.x for s in self.steps])

    @property
    def first_error(self) -> Optional[int]:
        return first_error_index(self.y_errors)

    @property
    def has_error(self) -> bool:
        return self.first_error is not None


def _as_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, np.ndarray) and value.shape == ():
        return _as_metadata(value.item())
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw_metadata": value}
        if isinstance(parsed, dict):
            return parsed
        return {"raw_metadata": parsed}
    return {"raw_metadata": value}


def _metadata_rows(metadata: Any, n: int) -> list[dict[str, Any]]:
    if metadata is None:
        return [{} for _ in range(n)]
    if isinstance(metadata, np.ndarray):
        if metadata.shape == ():
            metadata = metadata.item()
        else:
            metadata = metadata.tolist()
    if isinstance(metadata, dict):
        return [metadata for _ in range(n)]
    if isinstance(metadata, list):
        if len(metadata) != n:
            raise ValueError(f"metadata length {len(metadata)} does not match features rows {n}")
        return [_as_metadata(row) for row in metadata]
    raise ValueError(f"Unsupported metadata container: {type(metadata)!r}")


def _infer_source_key(path: Path, metadata: dict[str, Any]) -> Optional[str]:
    for key in ("source_dataset", "source_file", "dataset", "file"):
        value = metadata_get(metadata, key)
        if value is not None:
            value_path = Path(str(value))
            return value_path.stem or str(value)

    parts = list(path.parts)
    if "repro_graph_shards" in parts:
        idx = parts.index("repro_graph_shards")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    match = re.search(r"(arith_nt\d+|bool_nt\d+|gsm8k)", path.stem)
    if match:
        return match.group(1)
    return None


def _infer_complexity_from_path(path: Path) -> Optional[int]:
    match = re.search(r"nt(\d+)", str(path))
    return int(match.group(1)) if match else None


def _infer_trace_key(
    metadata: dict[str, Any],
    domain: str,
    row_index: int,
    source_key: Optional[str] = None,
    complexity: Optional[int] = None,
) -> str:
    keys = [
        "expr_id",
        "problem_id",
        "question_id",
        "trace_id",
        "id",
        "idx",
        "example_id",
    ]
    for key in keys:
        value = metadata_get(metadata, key)
        if value is not None:
            text = str(value)
            if text.startswith(f"{domain}:"):
                return text
            namespace = source_key or (f"nt{complexity}" if complexity is not None else None)
            if namespace:
                return f"{domain}:{namespace}:{text}"
            return f"{domain}:{text}"
    original = metadata_get(metadata, "original_expression")
    if original is not None:
        namespace = source_key or (f"nt{complexity}" if complexity is not None else "expr")
        return f"{domain}:{namespace}:expr:{hash(str(original))}"
    namespace = source_key or (f"nt{complexity}" if complexity is not None else "row")
    return f"{domain}:{namespace}:row:{row_index}"


def _infer_complexity(metadata: dict[str, Any], default: Optional[int]) -> Optional[int]:
    for key in ("complexity", "nt", "num_ops", "n_ops", "difficulty"):
        value = metadata_get(metadata, key)
        if value is None:
            continue
        try:
            if isinstance(value, float) and math.isnan(value):
                continue
            return int(value)
        except (TypeError, ValueError):
            continue
    return default


def _infer_step_number(metadata: dict[str, Any], row_index: int) -> int:
    value = metadata_get(metadata, "step_number")
    if value is None:
        value = metadata_get(metadata, "step_idx")
    if value is None:
        return row_index
    try:
        step = int(value)
    except (TypeError, ValueError):
        return row_index
    return max(step - 1, 0) if step >= 1 else step


def _pick_features_key(npz: np.lib.npyio.NpzFile) -> str:
    for key in ("features", "X", "x"):
        if key in npz.files:
            return key
    raise KeyError(f"Could not find features key in npz. Keys: {npz.files}")


def load_crop_npz(
    path: str | Path,
    domain: str,
    complexity: Optional[int] = None,
    allow_nan: bool = False,
) -> list[StepRecord]:
    """Load CROP ``feature_extraction.py`` output into step records."""

    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        features_key = _pick_features_key(data)
        features = np.asarray(data[features_key], dtype=float)
        metadata = data["metadata"] if "metadata" in data.files else None
        feature_names = data["feature_names"].tolist() if "feature_names" in data.files else None

    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")
    if not allow_nan and np.isnan(features).any():
        raise ValueError(f"NaNs found in {path}; use allow_nan and train-time imputation")

    rows = _metadata_rows(metadata, features.shape[0])
    records: list[StepRecord] = []
    for i, meta in enumerate(rows):
        meta = dict(meta)
        if feature_names is not None:
            meta["_feature_names"] = [str(name) for name in feature_names]
        label = extract_step_label(meta)
        is_correct, y_error = normalize_step_label(label)
        row_complexity = _infer_complexity(meta, complexity if complexity is not None else _infer_complexity_from_path(path))
        row_domain = str(metadata_get(meta, "domain", domain))
        source_key = _infer_source_key(path, meta)
        trace_id = _infer_trace_key(meta, row_domain, i, source_key=source_key, complexity=row_complexity)
        step_number = _infer_step_number(meta, i)
        records.append(
            StepRecord(
                trace_id=trace_id,
                domain=row_domain,
                complexity=row_complexity,
                step_number=step_number,
                total_steps=None,
                before_after=str(metadata_get(meta, "before_after", "unknown")),
                x=np.asarray(features[i], dtype=float),
                y_error=y_error,
                is_correct=is_correct,
                original_expression=metadata_get(meta, "original_expression"),
                step_content=metadata_get(meta, "step_content"),
                metadata=meta,
            )
        )
    return records


def records_to_traces(records: Iterable[StepRecord]) -> list[TraceRecord]:
    """Group step records into trace records without crossing trace IDs."""

    grouped: dict[str, list[StepRecord]] = {}
    for record in records:
        grouped.setdefault(record.trace_id, []).append(record)

    traces: list[TraceRecord] = []
    for trace_id, steps in grouped.items():
        steps = sorted(steps, key=lambda s: s.step_number)
        total = len(steps)
        for idx, step in enumerate(steps):
            step.step_number = idx
            step.total_steps = total
        domain = steps[0].domain
        complexity = steps[0].complexity
        traces.append(TraceRecord(trace_id=trace_id, domain=domain, complexity=complexity, steps=steps))
    return sorted(traces, key=lambda t: t.trace_id)


def truncate_after_first_error(traces: Iterable[TraceRecord]) -> list[TraceRecord]:
    """Keep steps only through the first incorrect step, matching the CROP protocol protocol."""

    out: list[TraceRecord] = []
    for trace in traces:
        first_error = trace.first_error
        steps = list(trace.steps if first_error is None else trace.steps[: first_error + 1])
        total = len(steps)
        for idx, step in enumerate(steps):
            step.step_number = idx
            step.total_steps = total
        out.append(TraceRecord(trace_id=trace.trace_id, domain=trace.domain, complexity=trace.complexity, steps=steps))
    return out


def load_many_npz(
    paths: list[str | Path],
    domains: list[str],
    complexities: Optional[list[Optional[int]]] = None,
    allow_nan: bool = False,
) -> list[TraceRecord]:
    if len(paths) != len(domains):
        raise ValueError("paths and domains must have the same length")
    if complexities is None:
        complexities = [None] * len(paths)
    if len(paths) != len(complexities):
        raise ValueError("paths and complexities must have the same length")

    records: list[StepRecord] = []
    for path, domain, complexity in zip(paths, domains, complexities):
        records.extend(load_crop_npz(path, domain, complexity, allow_nan=allow_nan))
    return records_to_traces(records)


def make_toy_traces(
    n_traces: int = 200,
    min_steps: int = 3,
    max_steps: int = 8,
    n_features: int = 55,
    error_rate: float = 0.15,
    seed: int = 0,
    domain: str = "toy",
    complexity: Optional[int] = None,
) -> list[TraceRecord]:
    """Generate a toy dataset with graph-like features and a known error signal."""

    rng = np.random.default_rng(seed)
    traces: list[TraceRecord] = []
    force_clean = max(1, n_traces // 10)
    force_error = max(1, n_traces // 10)
    for i in range(n_traces):
        total_steps = int(rng.integers(min_steps, max_steps + 1))
        y = (rng.random(total_steps) < error_rate).astype(int)
        if i < force_clean:
            y[:] = 0
        elif i < force_clean + force_error and not np.any(y):
            y[int(rng.integers(0, total_steps))] = 1

        steps: list[StepRecord] = []
        trace_id = f"{domain}:toy_{i:05d}"
        for t in range(total_steps):
            x = rng.normal(0.0, 1.0, size=n_features)
            if y[t] == 1:
                x[0] += 2.0
                if n_features > 8:
                    x[8] += 1.0
                if n_features >= 4:
                    x[-4] -= 1.0
            metadata = {
                "expr_id": f"toy_{i:05d}",
                "step_number": t + 1,
                "before_after": "after",
                "step_labels": {
                    "step_number": t + 1,
                    "step_content": f"toy step {t + 1}",
                    "step_label": bool(y[t] == 0),
                },
                "original_expression": f"toy expression {i}",
            }
            steps.append(
                StepRecord(
                    trace_id=trace_id,
                    domain=domain,
                    complexity=complexity,
                    step_number=t,
                    total_steps=total_steps,
                    before_after="after",
                    x=x,
                    y_error=int(y[t]),
                    is_correct=bool(y[t] == 0),
                    original_expression=f"toy expression {i}",
                    step_content=f"toy step {t + 1}",
                    metadata=metadata,
                )
            )
        traces.append(TraceRecord(trace_id=trace_id, domain=domain, complexity=complexity, steps=steps))
    return traces
