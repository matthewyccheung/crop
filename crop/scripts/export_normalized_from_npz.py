"""Export a local NPZ trace cache to normalized JSONL for external scorers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crop.data import load_many_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--domain", default="mixed")
    parser.add_argument("--output_jsonl", required=True)
    args = parser.parse_args()

    traces = load_many_npz([args.features], [args.domain])
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for trace in traces:
            question = trace.steps[0].original_expression if trace.steps else trace.trace_id
            row = {
                "trace_id": trace.trace_id,
                "dataset": trace.domain,
                "question": question or trace.trace_id,
                "steps": [
                    {
                        "step_number": step.step_number,
                        "step_content": step.step_content or "",
                        "step_label": bool(not step.y_error),
                    }
                    for step in trace.steps
                ],
                "trace_has_error": trace.has_error,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
