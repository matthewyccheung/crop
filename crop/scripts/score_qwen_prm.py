"""Score normalized process traces with Qwen2.5-Math-PRM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _step_rewards(logits: Any, token_masks: Any) -> list[list[float]]:
    import torch.nn.functional as F

    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)
    out: list[list[float]] = []
    for sample in probabilities:
        positive = sample[sample != 0].view(-1, 2)[:, 1]
        out.append([float(x) for x in positive.detach().cpu().tolist()])
    return out


def _iter_traces(path: Path, start: int, max_traces: int | None):
    yielded = 0
    with path.open() as f:
        for idx, line in enumerate(f):
            if idx < start:
                continue
            if max_traces is not None and yielded >= max_traces:
                break
            if not line.strip():
                continue
            yielded += 1
            yield idx, json.loads(line)


def _conversation(tokenizer, question: str, steps: list[str]) -> str:
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_traces", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--flush_every", type=int, default=25)
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Qwen PRM scoring requires optional GPU dependencies: torch and transformers."
        ) from exc

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_name,
        device_map=args.device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    model.config.use_cache = False
    sep_id = tokenizer.encode("<extra_0>")[0]

    done: set[tuple[str, int]] = set()
    if output_path.exists():
        with output_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add((row["trace_id"], int(row["step_id"])))

    fields = ["trace_id", "dataset", "step_id", "qwen_prm_reward", "qwen_prm_error", "n_steps", "source_row"]
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        written_since_flush = 0
        for source_row, trace in _iter_traces(input_path, args.start, args.max_traces):
            steps = [str(step.get("step_content", "")) for step in trace.get("steps", [])]
            if not steps:
                continue
            if all((trace["trace_id"], idx) in done for idx in range(len(steps))):
                continue
            text = _conversation(tokenizer, str(trace.get("question", "")), steps)
            input_ids = tokenizer.encode(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            ).to(model.device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, use_cache=False)
            token_masks = input_ids == sep_id
            rewards = _step_rewards(outputs[0], token_masks)[0]
            if len(rewards) < len(steps):
                rewards.extend([float("nan")] * (len(steps) - len(rewards)))
            rewards = rewards[: len(steps)]
            for step_id, reward in enumerate(rewards):
                if (trace["trace_id"], step_id) in done:
                    continue
                error = 1.0 - reward if reward == reward else float("nan")
                writer.writerow(
                    {
                        "trace_id": trace["trace_id"],
                        "dataset": trace.get("dataset", ""),
                        "step_id": step_id,
                        "qwen_prm_reward": reward,
                        "qwen_prm_error": error,
                        "n_steps": len(steps),
                        "source_row": source_row,
                    }
                )
                written_since_flush += 1
            if written_since_flush >= args.flush_every:
                f.flush()
                written_since_flush = 0
        f.flush()


if __name__ == "__main__":
    main()
