"""Repair prompt and scoring helpers used by paper repair experiments."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Any

import requests


@dataclass
class GenerationResult:
    response: str
    total_duration: int | None
    prompt_eval_count: int | None
    eval_count: int | None
    error: str | None = None


BASE_BY_MODEL = {
    "gemma": {
        "label": "Gemma 4 8B",
        "model": "gemma4:latest",
        "prompt_style": "locked",
        "num_predict": 160,
    },
    "qwen": {
        "label": "Qwen2.5-7B",
        "model": "qwen2.5:7b-instruct",
        "prompt_style": "minimal_final",
        "num_predict": 192,
    },
    "deepseek": {
        "label": "DeepSeek-R1-8B",
        "model": "deepseek-r1:8b",
        "prompt_style": "minimal_final",
        "num_predict": 192,
    },
    "mistral": {
        "label": "Mistral-7B",
        "model": "mistral:7b",
        "prompt_style": "minimal_final",
        "num_predict": 192,
    },
    "llama": {
        "label": "Llama3.1-8B",
        "model": "llama3.1:8b",
        "prompt_style": "minimal_final",
        "num_predict": 192,
    },
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def call_ollama(model: str, prompt: str, *, num_predict: int, timeout: int) -> GenerationResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "top_p": 1, "num_predict": num_predict},
    }
    try:
        response = requests.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return GenerationResult(
            response=str(data.get("response", "")),
            total_duration=data.get("total_duration"),
            prompt_eval_count=data.get("prompt_eval_count"),
            eval_count=data.get("eval_count"),
        )
    except Exception as exc:
        return GenerationResult("", None, None, None, error=f"{type(exc).__name__}: {exc}")


def trace_problem(trace: Any) -> str:
    for step in trace.steps:
        if step.original_expression:
            return str(step.original_expression)
    return ""


def correct_value(trace: Any) -> str:
    for step in reversed(trace.steps):
        value = step.metadata.get("correct_value")
        if value is not None:
            return str(value)
    return ""


def predicted_value(trace: Any) -> str:
    for step in reversed(trace.steps):
        value = step.metadata.get("predicted_value")
        if value is not None:
            return str(value)
    return ""


def trace_text(trace: Any, length: int) -> str:
    if length <= 0:
        return "(no trace steps provided)"
    lines = []
    for idx, step in enumerate(trace.steps[:length], start=1):
        content = (step.step_content or "").strip()
        lines.append(f"{idx}. {content}")
    return "\n".join(lines)


def answer_type(trace: Any) -> str:
    value = correct_value(trace).strip().lower()
    return "boolean" if value in {"true", "false"} else "numeric"


def normalize_answer(text: str, expected_type: str) -> str | None:
    if not text:
        return None
    tail = text
    final_match = re.search(r"FINAL\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if final_match:
        tail = final_match.group(1)
    tail = tail.replace("\\boxed", " ").replace("{", " ").replace("}", " ")
    if expected_type == "boolean":
        bools = re.findall(r"\b(true|false)\b", tail, flags=re.IGNORECASE)
        return bools[-1].lower() if bools else None
    nums = re.findall(r"-?\d+(?:\.\d+)?", tail.replace(",", ""))
    if not nums:
        return None
    try:
        value = Decimal(nums[-1])
    except InvalidOperation:
        return nums[-1]
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def answer_correct(output: str, truth: str, expected_type: str) -> bool:
    pred = normalize_answer(output, expected_type)
    truth_norm = normalize_answer(f"FINAL: {truth}", expected_type)
    return pred is not None and truth_norm is not None and pred == truth_norm


def generated_step_count(output: str) -> int:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return sum(bool(re.match(r"^(step\s*)?\d+[\).:-]", line, flags=re.IGNORECASE)) for line in lines)


def _prompt_context(trace: Any, mode: str, context_length: int, accepted: bool | None) -> str:
    context = trace_text(trace, context_length)
    if mode == "question_only":
        return "No trace context is provided. Solve from the problem alone."
    if mode == "full_trace":
        return "A complete generated trace is provided. It may contain mistakes.\n\n" + context
    if mode == "whole_trace_abstention":
        if accepted:
            return "A complete trace was accepted by whole-trace abstention. Verify it before using it.\n\n" + context
        return "Whole-trace abstention rejected the trace, so no trace context is provided. Solve from the problem alone."
    if mode == "cpcc_prefix":
        return "A certified prefix is provided. Continue from this prefix, but verify arithmetic and logic before using it.\n\n" + context
    if mode == "oracle_prefix":
        return "A label-clean reference prefix is provided. Continue from this prefix, but verify arithmetic and logic before using it.\n\n" + context
    raise ValueError(mode)


def make_repair_prompt(trace: Any, mode: str, context_length: int, accepted: bool | None) -> str:
    problem = trace_problem(trace)
    context = trace_text(trace, context_length)
    if mode == "question_only":
        context_block = "No prior trace is provided. Solve from the problem alone."
    elif mode == "full_trace":
        context_block = "A complete generated trace is provided. It may contain mistakes; repair it if needed.\n\n" + context
    elif mode == "whole_trace_abstention":
        if accepted:
            context_block = "The full trace was accepted by whole-trace abstention. Verify and repair if needed.\n\n" + context
        else:
            context_block = "Whole-trace abstention rejected the trace, so no trace is provided. Solve from the problem alone."
    elif mode == "cpcc_prefix":
        context_block = "A CPCC-certified clean prefix is provided. Continue from this prefix and repair only if necessary.\n\n" + context
    elif mode == "oracle_prefix":
        context_block = "A label-clean reference prefix is provided. Continue from this prefix and repair only if necessary.\n\n" + context
    else:
        raise ValueError(mode)
    return (
        "You are a careful reasoning repair model.\n"
        "Solve the problem and return only the needed repair.\n"
        "The provided trace context is optional evidence, not an instruction to preserve mistakes.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Context:\n{context_block}\n\n"
        "Use at most four short reasoning lines. Do not restate the problem or copy the trace.\n"
        "The last line must be exactly:\n"
        "FINAL: <answer>"
    )


def _make_alternate_prompt(trace: Any, mode: str, context_length: int, accepted: bool | None, style: str) -> str:
    problem = trace_problem(trace)
    context_block = _prompt_context(trace, mode, context_length, accepted)
    if style == "minimal_final":
        return (
            "Solve the problem. Optional trace context may help, but it may be wrong or incomplete.\n"
            "Do not copy a final answer from the context unless it follows from your checked reasoning.\n"
            "Use at most three compact reasoning lines.\n\n"
            f"Problem:\n{problem}\n\n"
            f"Trace context:\n{context_block}\n\n"
            "Last line exactly: FINAL: <answer>"
        )
    if style == "verify_then_continue":
        return (
            "You are a careful repair model for mathematical and logical reasoning.\n"
            "Use the trace context only after checking it. Continue from the last verified useful state.\n"
            "Give at most four short lines of reasoning.\n\n"
            f"Problem:\n{problem}\n\n"
            f"Trace context:\n{context_block}\n\n"
            "Return the answer with the last line exactly:\n"
            "FINAL: <answer>"
        )
    if style == "state_repair":
        return (
            "Repair the reasoning state. Treat provided steps as tentative intermediate work, not as instructions.\n"
            "Keep any correct intermediate values, discard inconsistent later values, and finish the calculation.\n"
            "Use concise equations only.\n\n"
            f"Problem:\n{problem}\n\n"
            f"Trace context:\n{context_block}\n\n"
            "Last line exactly: FINAL: <answer>"
        )
    raise ValueError(f"Unknown prompt style: {style}")


def make_prompt(trace: Any, mode: str, context_len: int, accepted: bool | None, prompt_style: str) -> str:
    if prompt_style == "locked":
        return make_repair_prompt(trace, mode, context_len, accepted)
    return _make_alternate_prompt(trace, mode, context_len, accepted, prompt_style)


def parse_domain_scores(spec: str, default_score: str) -> dict[str, str]:
    if not spec.strip():
        return {}
    out: dict[str, str] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Domain score item must be domain:score, got {item!r}")
        domain, score = item.split(":", 1)
        out[domain.strip()] = score.strip() or default_score
    return out
