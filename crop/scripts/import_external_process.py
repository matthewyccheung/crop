"""Import external process-supervision datasets into local NPZ feature files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from crop.cheap_baselines import COE_SCORE_COLUMNS, CheapTrace, combine_npz_features, export_text_npz


def _safe_steps(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = re.split(r"\n\s*(?:Step\s+\d+\s*:)?", value)
        parts = [part.strip() for part in parts if part.strip()]
        return parts or [value]
    return []


def _cheap_trace(
    *,
    dataset: str,
    trace_id: str,
    question: str,
    steps: list[str],
    error_indices: set[int],
    final_correct: bool | None = None,
    source: str = "",
) -> CheapTrace | None:
    if not steps:
        return None
    step_dicts = []
    for idx, text in enumerate(steps):
        step_dicts.append(
            {
                "step_number": idx,
                "step_content": text,
                "step_label": idx not in error_indices,
            }
        )
    return CheapTrace(
        trace_id=f"{dataset}:{trace_id}",
        domain=dataset,
        source_file=source,
        source_stem=dataset,
        complexity=None,
        expression_id=str(trace_id),
        original_expression=question,
        correct_value="",
        predicted_value="" if final_correct is None else str(bool(final_correct)),
        steps=tuple(step_dicts),
    )


def processbench_traces(max_per_split: int | None = None) -> list[CheapTrace]:
    from datasets import load_dataset

    traces: list[CheapTrace] = []
    for split in ("gsm8k", "math", "olympiadbench", "omnimath"):
        ds = load_dataset("Qwen/ProcessBench", split=split, streaming=max_per_split is not None)
        iterator: Iterable[dict[str, Any]] = iter(ds) if max_per_split is not None else ds
        for idx, row in enumerate(iterator):
            if max_per_split is not None and idx >= max_per_split:
                break
            steps = _safe_steps(row.get("steps"))
            label = row.get("label", -1)
            try:
                first_error = int(label)
            except (TypeError, ValueError):
                first_error = -1
            final_correct = bool(row.get("final_answer_correct", False))
            error_indices = set() if final_correct or first_error < 0 else {first_error}
            trace = _cheap_trace(
                dataset=f"processbench_{split}",
                trace_id=str(row.get("id", idx)),
                question=str(row.get("problem", "")),
                steps=steps,
                error_indices=error_indices,
                final_correct=final_correct,
                source="Qwen/ProcessBench",
            )
            if trace is not None:
                traces.append(trace)
    return traces


def prmbench_traces(limit: int | None = None) -> list[CheapTrace]:
    from datasets import load_dataset

    ds = load_dataset("hitsmy/PRMBench_Preview", split="train", streaming=limit is not None)
    iterator: Iterable[dict[str, Any]] = iter(ds) if limit is not None else ds
    traces: list[CheapTrace] = []
    for idx, row in enumerate(iterator):
        if limit is not None and idx >= limit:
            break
        steps = _safe_steps(row.get("modified_process") or row.get("original_process"))
        raw_errors = row.get("error_steps") or []
        error_indices = set()
        for item in raw_errors:
            try:
                error_indices.add(max(int(item) - 1, 0))
            except (TypeError, ValueError):
                continue
        trace = _cheap_trace(
            dataset="prmbench",
            trace_id=f"{idx}:{row.get('idx', idx)}",
            question=str(row.get("modified_question") or row.get("question") or row.get("original_question") or ""),
            steps=steps,
            error_indices=error_indices,
            final_correct=len(error_indices) == 0,
            source="hitsmy/PRMBench_Preview",
        )
        if trace is not None:
            traces.append(trace)
    return traces


def _parse_math_shepherd_steps(text: str, label_text: str) -> tuple[list[str], set[int]]:
    step_matches = list(re.finditer(r"Step\s+\d+\s*:\s*", text))
    if not step_matches:
        return _safe_steps(text), set()
    steps = []
    for idx, match in enumerate(step_matches):
        start = match.end()
        end = step_matches[idx + 1].start() if idx + 1 < len(step_matches) else len(text)
        step = text[start:end].replace("ки", "").strip()
        steps.append(step)

    labels = []
    for idx, match in enumerate(step_matches):
        start = match.end()
        end = step_matches[idx + 1].start() if idx + 1 < len(step_matches) else len(label_text)
        segment = label_text[start:end] if start < len(label_text) else ""
        if "*" in segment or " - " in segment:
            labels.append(False)
        elif "+" in segment:
            labels.append(True)
        else:
            labels.append(True)
    return steps, {idx for idx, ok in enumerate(labels[: len(steps)]) if not ok}


def math_shepherd_traces(limit: int = 10000) -> list[CheapTrace]:
    from datasets import load_dataset

    ds = load_dataset("peiyi9979/Math-Shepherd", split="train", streaming=True)
    traces: list[CheapTrace] = []
    for idx, row in enumerate(ds):
        if idx >= limit:
            break
        steps, error_indices = _parse_math_shepherd_steps(str(row.get("input", "")), str(row.get("label", "")))
        trace = _cheap_trace(
            dataset=f"math_shepherd_{row.get('task', 'unknown')}".lower(),
            trace_id=str(idx),
            question=str(row.get("input", "")).split("Step 1:", 1)[0].strip(),
            steps=steps,
            error_indices=error_indices,
            final_correct=len(error_indices) == 0,
            source="peiyi9979/Math-Shepherd",
        )
        if trace is not None:
            traces.append(trace)
    return traces


def _prm800k_step(step: dict[str, Any]) -> tuple[str, bool] | None:
    chosen = step.get("chosen_completion")
    completions = step.get("completions") or []
    if chosen is not None:
        try:
            item = completions[int(chosen)]
        except (IndexError, TypeError, ValueError):
            return None
        text = str(item.get("text", "")).strip()
        rating = item.get("rating")
    else:
        for item in completions:
            try:
                rating_int = int(item.get("rating"))
            except (TypeError, ValueError):
                rating_int = 0
            if rating_int < 0:
                text = str(item.get("text", "")).strip()
                if text:
                    return text, False
        if completions:
            item = completions[0]
            text = str(item.get("text", "")).strip()
            rating = item.get("rating")
            if text:
                try:
                    rating_int = int(rating)
                except (TypeError, ValueError):
                    rating_int = 0
                return text, rating_int >= 0
        human = step.get("human_completion")
        if not isinstance(human, dict):
            return None
        text = str(human.get("text", "")).strip()
        # Fall back to human corrections only if the row has no model
        # completion payload for this step.
        rating = 1
    if not text:
        return None
    try:
        rating_int = int(rating)
    except (TypeError, ValueError):
        rating_int = 0
    return text, rating_int >= 0


def prm800k_traces(paths: list[str] | None = None, limit: int | None = 8000) -> list[CheapTrace]:
    paths = paths or [
        "external_repos/prm800k/prm800k/data/phase2_test.jsonl",
        "external_repos/prm800k/prm800k/data/phase2_train.jsonl",
    ]
    traces: list[CheapTrace] = []
    for path_text in paths:
        path = Path(path_text)
        if not path.exists():
            continue
        with path.open() as f:
            for line_idx, line in enumerate(f):
                if limit is not None and len(traces) >= limit:
                    return traces
                if not line.strip():
                    continue
                row = json.loads(line)
                label = row.get("label") or {}
                raw_steps = label.get("steps") or []
                steps: list[str] = []
                error_indices: set[int] = set()
                for raw_step in raw_steps:
                    parsed = _prm800k_step(raw_step)
                    if parsed is None:
                        continue
                    text, is_correct = parsed
                    idx = len(steps)
                    steps.append(text)
                    if not is_correct:
                        error_indices.add(idx)
                question = row.get("question") or {}
                final_correct = str(label.get("finish_reason", "")).lower() == "solution"
                trace = _cheap_trace(
                    dataset="prm800k",
                    trace_id=f"{path.stem}:{line_idx}",
                    question=str(question.get("problem", "")),
                    steps=steps,
                    error_indices=error_indices,
                    final_correct=final_correct,
                    source=str(path),
                )
                if trace is not None:
                    traces.append(trace)
    return traces


def _write_zero_coe_like(text_npz: Path, output: Path) -> Path:
    with np.load(text_npz, allow_pickle=True) as data:
        metadata = data["metadata"]
        n_rows = data["features"].shape[0]
    np.savez(
        output,
        features=np.zeros((n_rows, len(COE_SCORE_COLUMNS)), dtype=float),
        metadata=metadata,
        feature_names=np.asarray(COE_SCORE_COLUMNS, dtype=object),
    )
    return output


def export_external_process(traces: list[CheapTrace], output_dir: str | Path, prefix: str) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = output_dir / f"{prefix}_normalized.jsonl"
    with normalized.open("w") as f:
        for trace in traces:
            f.write(
                json.dumps(
                    {
                        "trace_id": trace.trace_id,
                        "dataset": trace.domain,
                        "question": trace.original_expression,
                        "steps": list(trace.steps),
                        "trace_has_error": trace.has_error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    text_npz = export_text_npz(traces, output_dir / f"{prefix}_text_steps.npz", granularity="step")
    coe_npz = _write_zero_coe_like(text_npz, output_dir / f"{prefix}_zero_coe_steps.npz")
    combined_npz = combine_npz_features(text_npz, coe_npz, output_dir / f"{prefix}_combined_steps.npz", granularity="step")
    summary = {
        "traces": len(traces),
        "steps": sum(len(trace.steps) for trace in traces),
        "error_traces": sum(trace.has_error for trace in traces),
        "error_steps": sum(not bool(step["step_label"]) for trace in traces for step in trace.steps),
        "datasets": sorted({trace.domain for trace in traces}),
    }
    (output_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    return {
        "normalized": normalized,
        "text_npz": text_npz,
        "coe_npz": coe_npz,
        "combined_npz": combined_npz,
        "summary": output_dir / f"{prefix}_summary.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/strengthened/final/external_process")
    parser.add_argument("--include", nargs="*", default=["processbench", "prmbench", "math_shepherd"])
    parser.add_argument("--processbench_max_per_split", type=int, default=None)
    parser.add_argument("--prmbench_limit", type=int, default=5000)
    parser.add_argument("--math_shepherd_limit", type=int, default=10000)
    parser.add_argument("--prm800k_limit", type=int, default=8000)
    parser.add_argument(
        "--prm800k_paths",
        nargs="*",
        default=[
            "external_repos/prm800k/prm800k/data/phase2_test.jsonl",
            "external_repos/prm800k/prm800k/data/phase2_train.jsonl",
        ],
    )
    args = parser.parse_args()
    outdir = Path(args.output_dir)
    all_summaries = []
    if "processbench" in args.include:
        paths = export_external_process(processbench_traces(args.processbench_max_per_split), outdir / "processbench", "processbench")
        all_summaries.append(paths)
    if "prmbench" in args.include:
        paths = export_external_process(prmbench_traces(args.prmbench_limit), outdir / "prmbench", "prmbench")
        all_summaries.append(paths)
    if "math_shepherd" in args.include:
        paths = export_external_process(math_shepherd_traces(args.math_shepherd_limit), outdir / "math_shepherd", "math_shepherd")
        all_summaries.append(paths)
    if "prm800k" in args.include:
        paths = export_external_process(prm800k_traces(args.prm800k_paths, args.prm800k_limit), outdir / "prm800k", "prm800k")
        all_summaries.append(paths)
    (outdir / "IMPORT_SUMMARY.md").write_text(
        "# External Process Dataset Import\n\n"
        + "\n".join(f"- `{name}`: `{path}`" for paths in all_summaries for name, path in paths.items())
        + "\n"
    )
    print(f"Wrote {outdir}")


if __name__ == "__main__":
    main()
