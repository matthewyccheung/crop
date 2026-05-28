"""Convert official CROP annotated JSON files to graph-free baseline npz files.

This is a fallback for environments where attribution graph generation is not
available.  It uses only prompt/text/position metadata, never ``step_label``, to
build fixed-width baseline features over the official real CROP annotations.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


TOKEN_PATTERNS = [
    r"\d",
    r"[A-Za-z]",
    r"\s",
    r"\+",
    r"-",
    r"\*",
    r"/",
    r"=",
    r"\(",
    r"\)",
    r"\bTrue\b",
    r"\bFalse\b",
    r"\band\b",
    r"\bor\b",
    r"\bnot\b",
    r"\.",
    r",",
    r":",
]


def infer_domain(path: Path) -> str:
    text = str(path)
    if "arith" in text or "arithmetic" in text:
        return "arithmetic"
    if "bool" in text or "boolean" in text:
        return "boolean"
    if "gsm8k" in text:
        return "gsm8k"
    return "unknown"


def infer_complexity(path: Path):
    match = re.search(r"nt(\d+)", path.name)
    return int(match.group(1)) if match else None


def _counts(text: str) -> list[float]:
    return [float(len(re.findall(pattern, text))) for pattern in TOKEN_PATTERNS]


def make_features(expr: dict, step: dict, complexity) -> np.ndarray:
    original = str(expr.get("original_expression", ""))
    step_content = str(step.get("step_content", ""))
    before = str(step.get("assistant_content_before", ""))
    after = str(step.get("assistant_content_after", ""))
    formatted_before = str(step.get("formatted_assistant_content_before", ""))
    formatted_after = str(step.get("formatted_assistant_content_after", ""))
    step_number = int(step.get("step_number", 0))
    total_steps = int(expr.get("total_steps", 0)) or 1
    delta = max(len(after) - len(before), 0)

    base = [
        float(step_number),
        float(total_steps),
        float(step_number / max(total_steps - 1, 1)),
        float(complexity if complexity is not None else -1),
        float(len(original)),
        float(len(step_content)),
        float(len(before)),
        float(len(after)),
        float(delta),
        math.log1p(len(original)),
        math.log1p(len(step_content)),
        math.log1p(len(before)),
        math.log1p(len(after)),
        math.log1p(delta),
        float(len(formatted_before)),
        float(len(formatted_after)),
        math.log1p(len(formatted_before)),
        math.log1p(len(formatted_after)),
        float(step_content.count("\n")),
    ]
    values = base + _counts(original) + _counts(step_content)
    if len(values) < 55:
        values.extend([0.0] * (55 - len(values)))
    return np.asarray(values[:55], dtype=float)


def convert(paths: list[Path], output: Path, domain: str | None = None) -> None:
    features = []
    metadata = []
    for path in paths:
        inferred_domain = domain or infer_domain(path)
        complexity = infer_complexity(path)
        with path.open() as f:
            data = json.load(f)
        source_id = path.stem
        for expr in data:
            expr_id = f"{source_id}:{expr.get('expression_id')}"
            for step in expr.get("step_expressions", []):
                features.append(make_features(expr, step, complexity))
                metadata.append(
                    {
                        "expr_id": expr_id,
                        "expression_id": expr_id,
                        "source_file": str(path),
                        "domain": inferred_domain,
                        "complexity": complexity,
                        "step_number": int(step.get("step_number", 0)),
                        "total_steps": int(expr.get("total_steps", 0)),
                        "before_after": "after",
                        "step_labels": {
                            "step_number": int(step.get("step_number", 0)),
                            "step_content": step.get("step_content"),
                            "step_label": step.get("step_label"),
                        },
                        "original_expression": expr.get("original_expression"),
                    }
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, features=np.vstack(features), metadata=np.asarray(metadata, dtype=object))
    y_error = np.asarray([int(not bool(m["step_labels"]["step_label"])) for m in metadata], dtype=int)
    print(
        f"wrote {output} rows={len(metadata)} features={len(features[0]) if features else 0} "
        f"errors={int(y_error.sum())} error_rate={float(y_error.mean()) if len(y_error) else float('nan'):.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--domain", default=None)
    args = parser.parse_args()
    convert([Path(p) for p in args.inputs], Path(args.output), domain=args.domain)


if __name__ == "__main__":
    main()
