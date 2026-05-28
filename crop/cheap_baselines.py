"""Utilities for cheap verifier baselines on CROP reasoning traces.

This module intentionally avoids attribution-graph construction.  It exports
official CROP annotations into the JSONL format expected by the Chain-of-
Embedding baseline code and imports its pickle outputs back into the local NPZ
format consumed by the conformal experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from crop.labels import normalize_step_label
from crop.scripts.annotated_json_to_npz import infer_complexity, infer_domain, make_features


COE_SCORE_COLUMNS = [
    "maxprob_error",
    "ppl_error",
    "entropy_error",
    "tempscl_error",
    "energy_error",
    "coe_r_error",
    "coe_c_error",
    "cotk_error",
]

TEXT_FEATURE_COLUMNS = [f"text_feature_{i}" for i in range(55)]
ANSWER_TEXT_FEATURE_COLUMNS = [f"answer_text_feature_{i}" for i in range(55)]


@dataclass(frozen=True)
class CheapTrace:
    trace_id: str
    domain: str
    source_file: str
    source_stem: str
    complexity: int | None
    expression_id: str
    original_expression: str
    correct_value: str
    predicted_value: str
    steps: tuple[dict[str, Any], ...]

    @property
    def has_error(self) -> bool:
        return any(not bool(step.get("step_label")) for step in self.steps)


def safe_inverse(value: Any, eps: float = 1e-8) -> float:
    """Return a finite reciprocal with an epsilon guard around zero."""

    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(x):
        return float("nan")
    if abs(x) < eps:
        x = eps if x >= 0 else -eps
    return float(1.0 / x)


def oriented_error_scores(output: dict[str, Any], coe: dict[str, Any], cotk: dict[str, Any]) -> dict[str, float]:
    """Orient baseline scores so larger means more likely erroneous."""

    return {
        "maxprob_error": safe_inverse(output.get("maxprob")),
        "ppl_error": float(output.get("ppl", float("nan"))),
        "entropy_error": float(output.get("entropy", float("nan"))),
        "tempscl_error": safe_inverse(output.get("tempscl")),
        "energy_error": float(output.get("energy", float("nan"))),
        "coe_r_error": safe_inverse(coe.get("R")),
        "coe_c_error": safe_inverse(coe.get("C")),
        "cotk_error": safe_inverse(cotk.get("CoTK")),
    }


def _stable_code(value: Any, modulus: int = 1009) -> float:
    text = "" if value is None else str(value)
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    return float(int(digest[:8], 16) % modulus)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "correct"}


def _text_stats(text: Any) -> list[float]:
    value = "" if text is None else str(text)
    words = re.findall(r"\w+", value)
    digits = re.findall(r"\d", value)
    numbers = re.findall(r"[-+]?\d*\.?\d+", value.replace(",", ""))
    letters = [ch for ch in value if ch.isalpha()]
    upper = [ch for ch in letters if ch.isupper()]
    lines = value.splitlines()
    return [
        float(len(value)),
        float(len(words)),
        float(len(digits)),
        float(len(numbers)),
        float(len(lines)),
        float(sum(ch in ".,;:!?" for ch in value)),
        float(value.count("$")),
        float(value.count("=")),
        float(value.count("\\")),
        float(value.count("{") + value.count("}")),
        float(len(upper) / max(len(letters), 1)),
        float(len(value) / max(len(words), 1)),
    ]


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def load_crop_traces(paths: Iterable[str | Path]) -> list[CheapTrace]:
    traces: list[CheapTrace] = []
    for raw_path in paths:
        path = Path(raw_path)
        domain = infer_domain(path)
        complexity = infer_complexity(path)
        source_stem = path.stem
        for expr in _load_json(path):
            expression_id = str(expr.get("expression_id"))
            trace_id = f"{domain}:{source_stem}:{expression_id}"
            steps = tuple(dict(step) for step in expr.get("step_expressions", []))
            if not steps:
                continue
            traces.append(
                CheapTrace(
                    trace_id=trace_id,
                    domain=domain,
                    source_file=str(path),
                    source_stem=source_stem,
                    complexity=complexity,
                    expression_id=expression_id,
                    original_expression=str(expr.get("original_expression", "")),
                    correct_value=str(expr.get("correct_value", "")),
                    predicted_value=str(expr.get("predicted_value", "")),
                    steps=steps,
                )
            )
    return traces


def _sample_group(traces: list[CheapTrace], n: int, rng: np.random.Generator) -> list[CheapTrace]:
    if n <= 0 or len(traces) <= n:
        return list(traces)
    idx = rng.choice(len(traces), size=n, replace=False)
    return [traces[int(i)] for i in np.sort(idx)]


def select_target_traces(
    root: str | Path,
    arithmetic_n: int = 1500,
    boolean_n: int = 1500,
    include_all_gsm8k: bool = True,
    seed: int = 2806,
) -> list[CheapTrace]:
    """Select a reproducible natural subset for cheap baseline experiments."""

    root = Path(root)
    groups = {
        "arithmetic": sorted((root / "arithmetic_expressions").glob("arith.nt*.annotated.json")),
        "boolean": sorted((root / "boolean_expressions").glob("bool.nt*.annotated.json")),
        "gsm8k": sorted((root / "gsm8k_expressions").glob("gsm8k.annotated.json")),
    }
    rng = np.random.default_rng(seed)
    selected: list[CheapTrace] = []
    for domain, n_total in (("arithmetic", arithmetic_n), ("boolean", boolean_n)):
        per_file = int(math.ceil(n_total / max(len(groups[domain]), 1)))
        domain_selected: list[CheapTrace] = []
        for path in groups[domain]:
            domain_selected.extend(_sample_group(load_crop_traces([path]), per_file, rng))
        selected.extend(_sample_group(domain_selected, n_total, rng))
    if include_all_gsm8k:
        selected.extend(load_crop_traces(groups["gsm8k"]))
    else:
        selected.extend(_sample_group(load_crop_traces(groups["gsm8k"]), arithmetic_n, rng))
    return sorted(selected, key=lambda t: t.trace_id)


def _prior_steps_text(trace: CheapTrace, step_idx: int) -> str:
    prior = [str(step.get("step_content", "")).strip() for step in trace.steps[:step_idx]]
    prior = [item for item in prior if item]
    if not prior:
        return "(none)"
    return "\n".join(f"Step {i + 1}: {text}" for i, text in enumerate(prior))


def step_prompt(trace: CheapTrace, step_idx: int) -> str:
    return (
        "Problem:\n"
        f"{trace.original_expression}\n\n"
        "Reasoning so far:\n"
        f"{_prior_steps_text(trace, step_idx)}\n\n"
        "Continue with the next reasoning step."
    )


def trace_prompt(trace: CheapTrace) -> str:
    return f"Problem:\n{trace.original_expression}\n\nSolve the problem step by step."


def trace_cot(trace: CheapTrace) -> str:
    return "\n".join(str(step.get("step_content", "")).strip() for step in trace.steps if step.get("step_content"))


def _manifest_row(
    *,
    dataset: str,
    dataset_index: int,
    granularity: Literal["step", "trace"],
    trace: CheapTrace,
    step_idx: int | None,
) -> dict[str, Any]:
    if step_idx is None:
        step_content = trace_cot(trace)
        step_label = not trace.has_error
        step_number = 0
    else:
        step = trace.steps[step_idx]
        step_content = str(step.get("step_content", ""))
        step_label = step.get("step_label")
        step_number = int(step.get("step_number", step_idx))
    is_correct, y_error = normalize_step_label(step_label)
    return {
        "dataset": dataset,
        "dataset_index": dataset_index,
        "granularity": granularity,
        "trace_id": trace.trace_id,
        "domain": trace.domain,
        "complexity": "" if trace.complexity is None else trace.complexity,
        "source_file": trace.source_file,
        "source_stem": trace.source_stem,
        "expression_id": trace.expression_id,
        "step_number": step_number,
        "total_steps": len(trace.steps),
        "step_label": bool(is_correct),
        "y_error": int(y_error),
        "original_expression": trace.original_expression,
        "correct_value": trace.correct_value,
        "predicted_value": trace.predicted_value,
        "step_content": step_content,
    }


def export_coe_jsonl(
    traces: list[CheapTrace],
    output_dir: str | Path,
    dataset_prefix: str = "crop_target",
) -> dict[str, Path]:
    """Write step and trace JSONL files plus manifests for CoE."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "step_jsonl": output_dir / f"{dataset_prefix}_steps.jsonl",
        "trace_jsonl": output_dir / f"{dataset_prefix}_traces.jsonl",
        "step_manifest": output_dir / f"{dataset_prefix}_steps_manifest.csv",
        "trace_manifest": output_dir / f"{dataset_prefix}_traces_manifest.csv",
    }
    fieldnames = list(_manifest_row(dataset="x", dataset_index=0, granularity="trace", trace=traces[0], step_idx=None))

    with paths["step_jsonl"].open("w") as step_f, paths["step_manifest"].open("w", newline="") as step_m:
        writer = csv.DictWriter(step_m, fieldnames=fieldnames)
        writer.writeheader()
        idx = 0
        dataset = paths["step_jsonl"].stem
        for trace in traces:
            for step_idx, step in enumerate(trace.steps):
                row = _manifest_row(
                    dataset=dataset,
                    dataset_index=idx,
                    granularity="step",
                    trace=trace,
                    step_idx=step_idx,
                )
                item = {
                    "id": idx,
                    "en": step_prompt(trace, step_idx),
                    "answer": trace.correct_value,
                    "cached_output": str(step.get("step_content", "")),
                    **row,
                }
                step_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                writer.writerow(row)
                idx += 1

    with paths["trace_jsonl"].open("w") as trace_f, paths["trace_manifest"].open("w", newline="") as trace_m:
        writer = csv.DictWriter(trace_m, fieldnames=fieldnames)
        writer.writeheader()
        dataset = paths["trace_jsonl"].stem
        for idx, trace in enumerate(traces):
            row = _manifest_row(
                dataset=dataset,
                dataset_index=idx,
                granularity="trace",
                trace=trace,
                step_idx=None,
            )
            item = {
                "id": idx,
                "en": trace_prompt(trace),
                "answer": trace.correct_value,
                "cached_output": trace_cot(trace),
                **row,
            }
            trace_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            writer.writerow(row)
    return paths


def export_text_npz(
    traces: list[CheapTrace],
    output: str | Path,
    granularity: Literal["step", "trace"] = "step",
) -> Path:
    """Export graph-free text/metadata features for the selected traces."""

    output = Path(output)
    features: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    if granularity == "step":
        for trace in traces:
            expr = {
                "original_expression": trace.original_expression,
                "total_steps": len(trace.steps),
            }
            for step_idx, step in enumerate(trace.steps):
                features.append(make_features(expr, step, trace.complexity))
                row = _manifest_row(
                    dataset=output.stem,
                    dataset_index=len(metadata),
                    granularity="step",
                    trace=trace,
                    step_idx=step_idx,
                )
                row["step_labels"] = {
                    "step_number": row["step_number"],
                    "step_content": row["step_content"],
                    "step_label": row["step_label"],
                }
                row["expr_id"] = trace.trace_id
                row["before_after"] = "after"
                metadata.append(row)
    else:
        for trace in traces:
            row = _manifest_row(
                dataset=output.stem,
                dataset_index=len(metadata),
                granularity="trace",
                trace=trace,
                step_idx=None,
            )
            row["step_labels"] = {
                "step_number": 0,
                "step_content": row["step_content"],
                "step_label": row["step_label"],
            }
            row["expr_id"] = trace.trace_id
            row["before_after"] = "trace"
            metadata.append(row)
            domain_code = {"arithmetic": 0.0, "boolean": 1.0, "gsm8k": 2.0}.get(trace.domain, -1.0)
            features.append(
                np.asarray(
                    [
                        len(trace.original_expression),
                        len(trace_cot(trace)),
                        len(trace.steps),
                        -1 if trace.complexity is None else trace.complexity,
                        domain_code,
                    ]
                    + [0.0] * 50,
                    dtype=float,
                )
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        features=np.vstack(features) if features else np.empty((0, 55), dtype=float),
        metadata=np.asarray(metadata, dtype=object),
        feature_names=np.asarray(TEXT_FEATURE_COLUMNS, dtype=object),
    )
    return output


def _npz_alignment_key(meta: dict[str, Any], granularity: Literal["step", "trace"] | None = None) -> tuple:
    trace_id = str(meta.get("trace_id", meta.get("expr_id", meta.get("id", ""))))
    if not trace_id:
        raise ValueError(f"Cannot align row without trace_id/expr_id: {meta}")
    inferred = granularity or ("trace" if str(meta.get("before_after", meta.get("granularity", ""))) == "trace" else "step")
    if inferred == "trace":
        return (trace_id,)
    step_number = meta.get("step_number")
    if step_number is None and isinstance(meta.get("step_labels"), dict):
        step_number = meta["step_labels"].get("step_number")
    return (trace_id, _to_int(step_number))


def combine_npz_features(
    left_npz: str | Path,
    right_npz: str | Path,
    output_npz: str | Path,
    *,
    granularity: Literal["step", "trace"] | None = None,
    left_prefix: str = "",
    right_prefix: str = "",
) -> Path:
    """Horizontally align two feature files by trace id and optional step number."""

    left_npz = Path(left_npz)
    right_npz = Path(right_npz)
    output_npz = Path(output_npz)
    with np.load(left_npz, allow_pickle=True) as left:
        left_features = np.asarray(left["features"], dtype=float)
        left_meta = [dict(row) for row in left["metadata"]]
        left_names = [str(name) for name in left.get("feature_names", [])]
    with np.load(right_npz, allow_pickle=True) as right:
        right_features = np.asarray(right["features"], dtype=float)
        right_meta = [dict(row) for row in right["metadata"]]
        right_names = [str(name) for name in right.get("feature_names", [])]

    if left_features.shape[0] != len(left_meta) or right_features.shape[0] != len(right_meta):
        raise ValueError("Feature row counts and metadata lengths must match")

    right_index: dict[tuple, int] = {}
    for idx, meta in enumerate(right_meta):
        key = _npz_alignment_key(meta, granularity)
        if key in right_index:
            raise ValueError(f"Duplicate alignment key in {right_npz}: {key}")
        right_index[key] = idx

    aligned_right: list[np.ndarray] = []
    missing: list[tuple] = []
    for meta in left_meta:
        key = _npz_alignment_key(meta, granularity)
        idx = right_index.get(key)
        if idx is None:
            missing.append(key)
            continue
        aligned_right.append(right_features[idx])
    if missing:
        preview = ", ".join(map(str, missing[:5]))
        raise ValueError(f"{len(missing)} rows in {left_npz} were missing from {right_npz}: {preview}")

    names: list[str] = []
    seen: set[str] = set()
    for prefix, source_names, width in (
        (left_prefix, left_names, left_features.shape[1]),
        (right_prefix, right_names, right_features.shape[1]),
    ):
        source_names = source_names or [f"feature_{i}" for i in range(width)]
        for name in source_names:
            candidate = f"{prefix}{name}"
            if candidate in seen:
                candidate = f"{prefix or 'right_'}{name}"
            if candidate in seen:
                raise ValueError(f"Duplicate combined feature name {candidate!r}")
            seen.add(candidate)
            names.append(candidate)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        features=np.hstack([left_features, np.vstack(aligned_right)]),
        metadata=np.asarray(left_meta, dtype=object),
        feature_names=np.asarray(names, dtype=object),
    )
    return output_npz


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _stratified_sample_indices(
    rows: list[dict[str, Any]],
    n: int,
    keys: tuple[str, ...],
    rng: np.random.Generator,
) -> list[int]:
    if n <= 0 or len(rows) <= n:
        return list(range(len(rows)))
    groups: dict[tuple, list[int]] = {}
    for idx, row in enumerate(rows):
        group = tuple(str(row.get(key, "")) for key in keys)
        groups.setdefault(group, []).append(idx)
    allocation = {group: max(1, int(round(n * len(indices) / len(rows)))) for group, indices in groups.items()}
    while sum(allocation.values()) > n:
        group = max(allocation, key=lambda g: allocation[g] - n * len(groups[g]) / len(rows))
        if allocation[group] > 1:
            allocation[group] -= 1
        else:
            break
    while sum(allocation.values()) < n:
        group = max(allocation, key=lambda g: n * len(groups[g]) / len(rows) - allocation[g])
        allocation[group] += 1
    selected: list[int] = []
    for group, indices in groups.items():
        take = min(allocation[group], len(indices))
        chosen = rng.choice(indices, size=take, replace=False)
        selected.extend(int(i) for i in chosen)
    if len(selected) > n:
        selected = [int(i) for i in rng.choice(selected, size=n, replace=False)]
    return sorted(selected)


def export_coe_answer_subset(
    coe_data_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset_prefix: str = "strength",
    seed: int = 2806,
    math_n: int = 1000,
    datasets: Iterable[str] = ("math", "mgsm", "theoremqa", "commonsenseqa", "belebele"),
) -> dict[str, Path]:
    """Export targeted final-answer datasets in CoE JSONL format plus manifests."""

    coe_data_dir = Path(coe_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    outputs: dict[str, Path] = {}

    for source_dataset in datasets:
        rows = _read_jsonl(coe_data_dir / f"{source_dataset}.jsonl")
        if source_dataset == "math":
            selected_indices = _stratified_sample_indices(rows, math_n, ("domain", "level"), rng)
        else:
            selected_indices = list(range(len(rows)))
        dataset = f"{dataset_prefix}_{source_dataset}"
        jsonl_path = output_dir / f"{dataset}.jsonl"
        manifest_path = output_dir / f"{dataset}_manifest.csv"
        fieldnames = [
            "dataset",
            "dataset_index",
            "source_dataset",
            "source_id",
            "trace_id",
            "domain",
            "subdomain",
            "level",
            "answer_type",
            "answer",
            "question",
        ]
        manifest_rows = []
        with jsonl_path.open("w") as jf:
            for local_idx, source_idx in enumerate(selected_indices):
                row = dict(rows[source_idx])
                domain_value = row.get("domain", source_dataset)
                if isinstance(domain_value, list):
                    domain = source_dataset
                    subdomain = " / ".join(str(x) for x in domain_value)
                else:
                    domain = str(domain_value or source_dataset)
                    subdomain = str(row.get("level", ""))
                trace_id = f"{source_dataset}:{row.get('id', source_idx)}"
                item = dict(row)
                item.update(
                    {
                        "id": local_idx,
                        "dataset": dataset,
                        "dataset_index": local_idx,
                        "source_dataset": source_dataset,
                        "source_id": row.get("id", source_idx),
                        "trace_id": trace_id,
                    }
                )
                jf.write(json.dumps(item, ensure_ascii=False) + "\n")
                manifest_rows.append(
                    {
                        "dataset": dataset,
                        "dataset_index": local_idx,
                        "source_dataset": source_dataset,
                        "source_id": row.get("id", source_idx),
                        "trace_id": trace_id,
                        "domain": domain,
                        "subdomain": subdomain,
                        "level": row.get("level", ""),
                        "answer_type": row.get("answer_type", ""),
                        "answer": row.get("answer", ""),
                        "question": row.get("en", ""),
                    }
                )
        with manifest_path.open("w", newline="") as mf:
            writer = csv.DictWriter(mf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
        outputs[f"{source_dataset}_jsonl"] = jsonl_path
        outputs[f"{source_dataset}_manifest"] = manifest_path
    return outputs


def _load_answer_parser(project_path: Path):
    if str(project_path) not in sys.path:
        sys.path.insert(0, str(project_path))
    evaluation_path = project_path / "Evaluation"
    if str(evaluation_path) not in sys.path:
        sys.path.insert(0, str(evaluation_path))
    from Evaluation.match import AnswerParsing  # type: ignore

    return AnswerParsing


def _answer_text_features(row: dict[str, Any], output: dict[str, Any], extracted_answer: Any) -> np.ndarray:
    question_stats = _text_stats(row.get("question", ""))
    output_stats = _text_stats(output.get("output_seq", ""))
    input_stats = _text_stats(output.get("input_seq", ""))
    source_dataset = row.get("source_dataset", row.get("dataset", ""))
    level_text = str(row.get("level", ""))
    level_numbers = re.findall(r"\d+", level_text)
    level = float(level_numbers[0]) if level_numbers else -1.0
    generated = str(output.get("output_seq", ""))
    extras = [
        _stable_code(source_dataset),
        _stable_code(row.get("domain", "")),
        _stable_code(row.get("subdomain", "")),
        _stable_code(row.get("answer_type", "")),
        level,
        float("\\boxed" in generated),
        float("Answer" in generated or "answer" in generated),
        float("Incomplete" == str(extracted_answer)),
        float("<|eot_id|>" in generated or "</s>" in generated or "<|im_end|>" in generated),
        float(len(generated) / max(len(str(row.get("question", ""))), 1)),
    ]
    values = question_stats + output_stats + input_stats + extras
    if len(values) < len(ANSWER_TEXT_FEATURE_COLUMNS):
        values.extend([0.0] * (len(ANSWER_TEXT_FEATURE_COLUMNS) - len(values)))
    return np.asarray(values[: len(ANSWER_TEXT_FEATURE_COLUMNS)], dtype=float)


def import_coe_answer_outputs(
    project_path: str | Path,
    manifest_paths: Iterable[str | Path],
    score_npz: str | Path,
    *,
    text_npz: str | Path | None = None,
    model_name: str = "Llama-3.1-8B-Instruct",
    language: str = "en",
    token_aggregation: str = "average",
    strict: bool = True,
) -> Path:
    """Import CoE final-answer outputs as one trace-level row per example."""

    project_path = Path(project_path)
    score_npz = Path(score_npz)
    answer_parser_cls = _load_answer_parser(project_path)
    rows: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        rows.extend(read_manifest(manifest_path))

    score_features: list[list[float]] = []
    text_features: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for row in rows:
        dataset = str(row["dataset"])
        idx = int(row["dataset_index"])
        output_path = project_path / "OutputInfo" / language / "Output" / model_name / dataset / f"{dataset}_{idx}.pkl"
        coe_path = project_path / "OutputInfo" / language / "CoE" / model_name / dataset / f"{dataset}.{token_aggregation}_{idx}.pkl"
        cotk_path = project_path / "OutputInfo" / language / "CoTK" / model_name / dataset / f"{dataset}.{token_aggregation}_{idx}.pkl"
        missing = [str(p) for p in (output_path, coe_path, cotk_path) if not p.exists()]
        if missing:
            if strict:
                raise FileNotFoundError(f"Missing CoE outputs for {dataset}:{idx}: {missing}")
            continue
        output = _load_pickle(output_path)
        coe = _load_pickle(coe_path)
        cotk = _load_pickle(cotk_path)
        source_dataset = str(row.get("source_dataset", dataset))
        try:
            extracted, correct = answer_parser_cls(source_dataset).dataset_parse(
                str(output.get("output_seq", "")),
                str(row.get("answer", "")),
                row,
            )
        except Exception as exc:  # pragma: no cover - defensive around third-party parsers
            extracted, correct = f"parse_error:{type(exc).__name__}", False
        scores = oriented_error_scores(output, coe, cotk)
        y_error = int(not bool(correct))
        meta = dict(row)
        meta.update(
            {
                "granularity": "trace",
                "before_after": "trace",
                "expr_id": row.get("trace_id", f"{dataset}:{idx}"),
                "step_number": 0,
                "total_steps": 1,
                "step_content": str(output.get("output_seq", "")),
                "extracted_answer": extracted,
                "correct": bool(correct),
                "y_error": y_error,
                "step_labels": {
                    "step_number": 0,
                    "step_content": str(output.get("output_seq", "")),
                    "step_label": bool(correct),
                },
            }
        )
        score_features.append([scores[name] for name in COE_SCORE_COLUMNS])
        text_features.append(_answer_text_features(row, output, extracted))
        metadata.append(meta)

    score_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        score_npz,
        features=np.asarray(score_features, dtype=float),
        metadata=np.asarray(metadata, dtype=object),
        feature_names=np.asarray(COE_SCORE_COLUMNS, dtype=object),
    )
    if text_npz is not None:
        text_npz = Path(text_npz)
        text_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            text_npz,
            features=np.asarray(text_features, dtype=float),
            metadata=np.asarray(metadata, dtype=object),
            feature_names=np.asarray(ANSWER_TEXT_FEATURE_COLUMNS, dtype=object),
        )
    return score_npz


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        value = pickle.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected dict pickle at {path}")
    return value


def import_coe_outputs(
    project_path: str | Path,
    manifest_path: str | Path,
    output_npz: str | Path,
    model_name: str = "Llama-3.1-8B-Instruct",
    language: str = "en",
    token_aggregation: str = "average",
    strict: bool = True,
) -> Path:
    """Convert CoE OutputInfo pickles to local score-column NPZ format."""

    project_path = Path(project_path)
    output_npz = Path(output_npz)
    rows = read_manifest(manifest_path)
    features: list[list[float]] = []
    metadata: list[dict[str, Any]] = []
    for row in rows:
        dataset = str(row["dataset"])
        idx = int(row["dataset_index"])
        output_path = project_path / "OutputInfo" / language / "Output" / model_name / dataset / f"{dataset}_{idx}.pkl"
        coe_path = project_path / "OutputInfo" / language / "CoE" / model_name / dataset / f"{dataset}.{token_aggregation}_{idx}.pkl"
        cotk_path = project_path / "OutputInfo" / language / "CoTK" / model_name / dataset / f"{dataset}.{token_aggregation}_{idx}.pkl"
        missing = [str(p) for p in (output_path, coe_path, cotk_path) if not p.exists()]
        if missing:
            if strict:
                raise FileNotFoundError(f"Missing CoE outputs for row {idx}: {missing}")
            continue
        scores = oriented_error_scores(_load_pickle(output_path), _load_pickle(coe_path), _load_pickle(cotk_path))
        features.append([scores[name] for name in COE_SCORE_COLUMNS])
        meta = dict(row)
        meta["step_labels"] = {
            "step_number": int(row.get("step_number", 0)),
            "step_content": row.get("step_content", ""),
            "step_label": row.get("step_label") in {"True", "true", "1", "yes", "correct", True},
        }
        meta["expr_id"] = row["trace_id"]
        meta["before_after"] = row.get("granularity", "step")
        metadata.append(meta)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        features=np.asarray(features, dtype=float),
        metadata=np.asarray(metadata, dtype=object),
        feature_names=np.asarray(COE_SCORE_COLUMNS, dtype=object),
    )
    return output_npz


def merge_npz_files(paths: Iterable[str | Path], output_npz: str | Path) -> Path:
    """Merge compatible feature NPZ files while preserving row metadata."""

    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("At least one input NPZ path is required")

    feature_blocks: list[np.ndarray] = []
    metadata: list[Any] = []
    feature_names: np.ndarray | None = None
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            features = np.asarray(data["features"], dtype=float)
            names = np.asarray(data.get("feature_names", []), dtype=object)
            rows = list(data["metadata"])
        if feature_names is None:
            feature_names = names
        elif list(feature_names) != list(names):
            raise ValueError(f"Feature names in {path} do not match the first shard")
        feature_blocks.append(features)
        metadata.extend(rows)

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        features=np.vstack(feature_blocks),
        metadata=np.asarray(metadata, dtype=object),
        feature_names=np.asarray(feature_names if feature_names is not None else [], dtype=object),
    )
    return output_npz


def summarize_npz(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        features = np.asarray(data["features"], dtype=float)
        metadata = list(data["metadata"])
    y = np.asarray([int(m.get("y_error", int(not bool(m["step_labels"]["step_label"])))) for m in metadata], dtype=int)
    domains = sorted({str(m.get("domain", "unknown")) for m in metadata})
    traces = sorted({str(m.get("trace_id", m.get("expr_id", i))) for i, m in enumerate(metadata)})
    return {
        "path": str(path),
        "rows": int(len(metadata)),
        "traces": int(len(traces)),
        "features": int(features.shape[1]) if features.ndim == 2 else 0,
        "errors": int(y.sum()) if len(y) else 0,
        "error_rate": float(y.mean()) if len(y) else float("nan"),
        "domains": ",".join(domains),
    }
