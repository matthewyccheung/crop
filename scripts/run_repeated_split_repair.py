#!/usr/bin/env python
"""Repeated-split downstream repair experiment.

This is the expensive validation path for Table 3-style repair claims.  Each
split refits any trainable score adapters, recalibrates CROP and whole-trace
abstention on a fresh calibration split, evaluates all repair input modes on
the same test traces, and summarizes paired CROP deltas across split seeds.

The runner is deliberately resumable.  Repair generations are keyed by the
exact prompt hash and Ollama model, so repeated splits reuse identical
question-only, full-trace, abstention, or CROP-prefix prompts when possible.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crop.experiments.exp18_trace_conditioned_adaptive_cpcc import (  # noqa: E402
    ALPHA_MAIN,
    _select_index_for_candidate,
)
from crop.experiments.exp19_trace_gating_next_steps import _prepare_seed_context  # noqa: E402
from crop.paper_repro import TARGET_COMBINED, TARGET_QWEN, TARGET_TEXT, target_args  # noqa: E402
from crop.repair_utils import (  # noqa: E402
    BASE_BY_MODEL,
    answer_correct,
    answer_type,
    call_ollama,
    correct_value,
    generated_step_count,
    make_prompt,
    normalize_answer,
    parse_domain_scores,
    predicted_value,
    safe_name,
)
from crop.risk_control import (  # noqa: E402
    prefix_lengths,
    select_lambda_crc,
    whole_trace_false_accept_losses,
)


OUT_ROOT = ROOT / "outputs" / "repeated_split_repair"
DEPLOYABLE_MODES = ["question_only", "full_trace", "whole_trace_abstention", "cpcc_prefix"]
NON_CROP_MODES = ["question_only", "full_trace", "whole_trace_abstention"]


@dataclass(frozen=True)
class DomainCalibration:
    domain: str
    score: str
    prefix_lambda: float
    prefix_cal_corrected_risk: float
    whole_lambda: float
    whole_cal_corrected_risk: float


def parse_seeds(spec: str) -> list[int]:
    """Parse comma-separated seeds and inclusive-exclusive ranges like 2806:2826."""

    seeds: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            parts = item.split(":")
            if len(parts) not in {2, 3}:
                raise ValueError(f"Bad seed range: {item!r}")
            start = int(parts[0])
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 else 1
            seeds.extend(range(start, stop, step))
        else:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def clean_error(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def cache_key(model: str, prompt_sha256: str) -> tuple[str, str]:
    return (str(model), str(prompt_sha256))


def response_fields_from_row(row: pd.Series) -> dict[str, Any]:
    fields = [
        "model_answer",
        "final_correct",
        "recovered_original_wrong",
        "degraded_original_correct",
        "output_tokens",
        "prompt_tokens",
        "generated_steps",
        "seconds",
        "error",
        "response",
    ]
    return {field: row.get(field, np.nan) for field in fields}


def read_prompt_for_row(row: pd.Series) -> str | None:
    prompt_path = row.get("prompt_path")
    if prompt_path is None or (isinstance(prompt_path, float) and math.isnan(prompt_path)):
        return None
    path = Path(str(prompt_path))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    return path.read_text()


def load_generation_cache(paths: list[Path], reuse_errors: bool = False) -> dict[tuple[str, str], pd.Series]:
    cache: dict[tuple[str, str], pd.Series] = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "prompt_sha256" not in df.columns:
            hashes: list[str | None] = []
            for _, row in df.iterrows():
                prompt = read_prompt_for_row(row)
                hashes.append(prompt_hash(prompt) if prompt is not None else None)
            df = df.copy()
            df["prompt_sha256"] = hashes
        for _, row in df.iterrows():
            if not reuse_errors and clean_error(row.get("error")):
                continue
            sha = row.get("prompt_sha256")
            model = row.get("model")
            if sha is None or model is None or (isinstance(sha, float) and math.isnan(sha)):
                continue
            cache[cache_key(str(model), str(sha))] = row
    return cache


def default_existing_generation_paths() -> list[Path]:
    paths: list[Path] = []
    for base in [
        ROOT / "outputs" / "repair_robustness_mondrian",
        ROOT / "outputs" / "repair_robustness",
        ROOT / "outputs" / "priority_experiments",
        ROOT / "outputs" / "repeated_split_repair",
    ]:
        if not base.exists():
            continue
        paths.extend(sorted(base.glob("*/generations.csv")))
        paths.extend(sorted(base.glob("*/downstream_repair_generations.csv")))
    return paths


def calibrate_domains(
    ctx,
    default_score: str,
    domain_scores: dict[str, str],
    domains: set[str],
) -> tuple[dict[str, int], dict[str, bool], dict[str, DomainCalibration]]:
    score_set = ctx.normalized_scores["raw"]
    lengths_by_trace: dict[str, int] = {}
    accepted_by_trace: dict[str, bool] = {}
    info: dict[str, DomainCalibration] = {}
    cal_domains = np.asarray([trace.domain for trace in ctx.split.cal], dtype=object)
    test_domains = np.asarray([trace.domain for trace in ctx.split.test], dtype=object)

    for domain in sorted(domains):
        cal_idx = np.flatnonzero(cal_domains == domain)
        test_idx = np.flatnonzero(test_domains == domain)
        if len(cal_idx) == 0 or len(test_idx) == 0:
            continue
        score = domain_scores.get(domain, default_score)
        grid = ctx.grids["raw"][score]
        cal_traces = [ctx.split.cal[int(i)] for i in cal_idx]
        test_traces = [ctx.split.test[int(i)] for i in test_idx]
        cal_scores = [score_set["cal"][score][int(i)] for i in cal_idx]
        test_scores = [score_set["test"][score][int(i)] for i in test_idx]

        prefix_idx, prefix_risk = _select_index_for_candidate(cal_traces, cal_scores, grid, ALPHA_MAIN)
        prefix_lambda = float(grid[prefix_idx])
        prefix_len = prefix_lengths(test_scores, prefix_lambda)

        whole_losses = np.vstack([whole_trace_false_accept_losses(cal_traces, cal_scores, lam) for lam in grid])
        whole_lambda, whole_risk = select_lambda_crc(whole_losses, grid, alpha=ALPHA_MAIN, direction="increasing")
        whole_lambda = float(whole_lambda)
        whole_accept = [bool(len(scores) and np.max(scores) <= whole_lambda) for scores in test_scores]

        for trace, length, accepted in zip(test_traces, prefix_len, whole_accept):
            lengths_by_trace[trace.trace_id] = int(length)
            accepted_by_trace[trace.trace_id] = bool(accepted)
        info[domain] = DomainCalibration(
            domain=domain,
            score=score,
            prefix_lambda=prefix_lambda,
            prefix_cal_corrected_risk=float(prefix_risk),
            whole_lambda=whole_lambda,
            whole_cal_corrected_risk=float(whole_risk),
        )
    return lengths_by_trace, accepted_by_trace, info


def selected_test_traces(ctx, domains: set[str], per_domain_limit: int | None, seed: int) -> list[Any]:
    traces = [trace for trace in ctx.split.test if trace.domain in domains]
    if per_domain_limit is None or per_domain_limit <= 0:
        return traces
    rng = np.random.default_rng(seed + 8_177)
    out: list[Any] = []
    for domain in sorted(domains):
        domain_traces = [trace for trace in traces if trace.domain == domain]
        order = np.arange(len(domain_traces))
        rng.shuffle(order)
        out.extend([domain_traces[int(i)] for i in order[:per_domain_limit]])
    return out


def build_prompt_job(
    *,
    split_seed: int,
    split_index: int,
    trace: Any,
    mode: str,
    context_len: int,
    accepted: bool | None,
    cfg: dict[str, Any],
    domain_info: DomainCalibration,
    prompt_dir: Path,
) -> dict[str, Any]:
    truth = correct_value(trace)
    expected_type = answer_type(trace)
    original_correct = answer_correct(f"FINAL: {predicted_value(trace)}", truth, expected_type)
    prompt = make_prompt(trace, mode, context_len, accepted, str(cfg["prompt_style"]))
    sha = prompt_hash(prompt)
    prompt_path = prompt_dir / f"seed{split_seed}__{safe_name(cfg['model'])}__{safe_name(trace.trace_id)}__{mode}__{context_len}__{sha[:12]}.txt"
    prompt_path.write_text(prompt)
    total_steps = len(trace.steps)
    return {
        "split_seed": int(split_seed),
        "split_index": int(split_index),
        "model": cfg["model"],
        "model_label": cfg["label"],
        "prompt_style": cfg["prompt_style"],
        "calibration": "repeated_split_mondrian_domain",
        "score": domain_info.score,
        "domain": trace.domain,
        "trace_id": trace.trace_id,
        "mode": mode,
        "truth": truth,
        "original_predicted": predicted_value(trace),
        "original_correct": original_correct,
        "has_annotated_error": bool(trace.has_error),
        "first_error": -1 if trace.first_error is None else int(trace.first_error),
        "total_steps": total_steps,
        "context_steps": int(context_len),
        "review_burden": (total_steps - int(context_len)) / max(total_steps, 1),
        "full_trace_available": int(context_len) == total_steps,
        "whole_trace_accepted": accepted if mode == "whole_trace_abstention" else np.nan,
        "prefix_lambda": domain_info.prefix_lambda,
        "prefix_cal_corrected_risk": domain_info.prefix_cal_corrected_risk,
        "whole_lambda": domain_info.whole_lambda,
        "whole_cal_corrected_risk": domain_info.whole_cal_corrected_risk,
        "expected_type": expected_type,
        "prompt": prompt,
        "prompt_sha256": sha,
        "prompt_path": str(prompt_path.relative_to(ROOT)),
    }


def row_from_job_and_response(job: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    row = {key: value for key, value in job.items() if key not in {"expected_type", "prompt"}}
    row.update(response)
    return row


def run_generation(args: argparse.Namespace, job: dict[str, Any], num_predict: int) -> dict[str, Any]:
    started = time.time()
    gen = call_ollama(str(job["model"]), str(job["prompt"]), num_predict=num_predict, timeout=args.timeout)
    correct = answer_correct(gen.response, str(job["truth"]), str(job["expected_type"]))
    row = {key: value for key, value in job.items() if key not in {"expected_type", "prompt"}}
    row.update(
        {
            "model_answer": normalize_answer(gen.response, str(job["expected_type"])),
            "final_correct": correct,
            "recovered_original_wrong": bool((not bool(job["original_correct"])) and correct),
            "degraded_original_correct": bool(bool(job["original_correct"]) and not correct),
            "output_tokens": gen.eval_count,
            "prompt_tokens": gen.prompt_eval_count,
            "generated_steps": generated_step_count(gen.response),
            "seconds": time.time() - started,
            "error": gen.error,
            "response": gen.response,
        }
    )
    return row


def summarize_split_modes(raw: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = raw[raw["error"].isna() | raw["error"].fillna("").astype(str).eq("")].copy()
    mode_rows = []
    for (seed, model, domain, mode), group in valid.groupby(["split_seed", "model_label", "domain", "mode"], sort=True):
        originally_wrong = group[~group["original_correct"].astype(bool)]
        originally_correct = group[group["original_correct"].astype(bool)]
        mode_rows.append(
            {
                "split_seed": int(seed),
                "model": model,
                "domain": domain,
                "mode": mode,
                "n": int(len(group)),
                "final_accuracy": float(group["final_correct"].astype(bool).mean()) if len(group) else math.nan,
                "recovery_rate": float(originally_wrong["final_correct"].astype(bool).mean()) if len(originally_wrong) else math.nan,
                "degradation_rate": float((~originally_correct["final_correct"].astype(bool)).mean()) if len(originally_correct) else math.nan,
                "review_burden": float(pd.to_numeric(group["review_burden"], errors="coerce").mean()) if len(group) else math.nan,
            }
        )
    split_modes = pd.DataFrame(mode_rows)
    split_modes.to_csv(out_dir / "split_mode_summary.csv", index=False)

    delta_rows = []
    for (seed, model, domain), group in valid.groupby(["split_seed", "model_label", "domain"], sort=True):
        piv = group.pivot_table(index="trace_id", columns="mode", values="final_correct", aggfunc="first")
        if "cpcc_prefix" not in piv:
            continue
        mode_acc = {
            mode: float(piv[mode].astype(float).mean())
            for mode in DEPLOYABLE_MODES
            if mode in piv
        }
        non_crop = [mode for mode in NON_CROP_MODES if mode in mode_acc]
        if not non_crop:
            continue
        best = max(non_crop, key=lambda mode: mode_acc[mode])
        for baseline in non_crop + ["best_non_crop"]:
            compare_mode = best if baseline == "best_non_crop" else baseline
            paired = (piv["cpcc_prefix"].astype(float) - piv[compare_mode].astype(float)).dropna()
            delta_rows.append(
                {
                    "split_seed": int(seed),
                    "model": model,
                    "domain": domain,
                    "baseline": baseline,
                    "best_non_crop_mode": best,
                    "n": int(len(paired)),
                    "crop_accuracy": mode_acc["cpcc_prefix"],
                    "baseline_accuracy": mode_acc[compare_mode],
                    "delta_accuracy": float(paired.mean()) if len(paired) else math.nan,
                    "delta_pp": 100.0 * float(paired.mean()) if len(paired) else math.nan,
                }
            )
    split_deltas = pd.DataFrame(delta_rows)
    split_deltas.to_csv(out_dir / "split_delta_summary.csv", index=False)

    inference_rows = []
    rng = np.random.default_rng(20260515)
    for (model, domain, baseline), group in split_deltas.groupby(["model", "domain", "baseline"], sort=True):
        vals = pd.to_numeric(group["delta_pp"], errors="coerce").dropna().to_numpy(dtype=float)
        n = len(vals)
        if n == 0:
            continue
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 1 else math.nan
        if n > 1:
            try:
                from scipy import stats

                tcrit = float(stats.t.ppf(0.975, df=n - 1))
                p_value = float(stats.ttest_1samp(vals, popmean=0.0).pvalue)
            except Exception:
                tcrit = 1.96
                p_value = math.nan
            ci_low_t = mean - tcrit * se
            ci_high_t = mean + tcrit * se
            boots = vals[rng.integers(0, n, size=(20_000, n))].mean(axis=1)
            ci_low_boot, ci_high_boot = np.percentile(boots, [2.5, 97.5])
        else:
            ci_low_t = ci_high_t = ci_low_boot = ci_high_boot = p_value = math.nan
        inference_rows.append(
            {
                "model": model,
                "domain": domain,
                "baseline": baseline,
                "n_splits": int(n),
                "mean_delta_pp": mean,
                "sd_delta_pp": sd,
                "se_delta_pp": se,
                "ci95_low_t": float(ci_low_t),
                "ci95_high_t": float(ci_high_t),
                "ci95_low_bootstrap": float(ci_low_boot),
                "ci95_high_bootstrap": float(ci_high_boot),
                "p_value_t_two_sided": p_value,
                "splits_positive": int(np.sum(vals > 0)),
            }
        )
    inference = pd.DataFrame(inference_rows)
    inference.to_csv(out_dir / "paired_split_inference.csv", index=False)
    return split_modes, split_deltas, inference


def write_compact_tables(split_modes: pd.DataFrame, inference: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for (model, domain, mode), group in split_modes.groupby(["model", "domain", "mode"], sort=True):
        vals = pd.to_numeric(group["final_accuracy"], errors="coerce").dropna().to_numpy(dtype=float) * 100.0
        if len(vals) == 0:
            continue
        rows.append(
            {
                "model": model,
                "domain": domain,
                "mode": mode,
                "n_splits": len(vals),
                "mean_accuracy": float(np.mean(vals)),
                "sd_accuracy": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "accuracy_by_model_domain_mode.csv", index=False)

    best = inference[inference["baseline"].eq("best_non_crop")].copy()
    best.to_csv(out_dir / "crop_vs_best_non_crop_inference.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="2806:2826", help="Comma list and/or ranges, e.g. 2806:2826")
    parser.add_argument("--domains", default="arithmetic,gsm8k")
    parser.add_argument("--repair_score", default="step_qwen")
    parser.add_argument("--domain_scores", default="gsm8k:qwen_prm")
    parser.add_argument("--models", default="gemma,qwen,deepseek,mistral,llama")
    parser.add_argument("--run_name", default="full_repeated_split_table3")
    parser.add_argument("--per_domain_test_limit", type=int, default=0, help="Debug/smoke limit; 0 uses all test traces.")
    parser.add_argument("--repair_workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Prepare manifests and summaries of pending jobs without Ollama calls.")
    parser.add_argument("--reuse_errors", action="store_true")
    parser.add_argument("--no_import_existing", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    domains = {item.strip() for item in args.domains.split(",") if item.strip()}
    domain_scores = parse_domain_scores(args.domain_scores, args.repair_score)
    model_keys = [key.strip() for key in args.models.split(",") if key.strip()]
    configs = [BASE_BY_MODEL[key] for key in model_keys]
    out_dir = OUT_ROOT / re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_name)
    prompt_dir = out_dir / "prompts"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "generations.csv"

    cache_paths = [] if args.no_import_existing else default_existing_generation_paths()
    if raw_path.exists() and not args.force:
        cache_paths.append(raw_path)
    cache = load_generation_cache(cache_paths, reuse_errors=args.reuse_errors)

    rows: list[dict[str, Any]] = []
    done: set[tuple[int, str, str, str, str]] = set()
    if raw_path.exists() and not args.force:
        try:
            existing = pd.read_csv(raw_path)
        except EmptyDataError:
            existing = pd.DataFrame()
        rows = existing.to_dict("records")
        for _, row in existing.iterrows():
            done.add(
                (
                    int(row["split_seed"]),
                    str(row["model"]),
                    str(row["trace_id"]),
                    str(row["mode"]),
                    str(row["prompt_sha256"]),
                )
            )
    jobs_by_model: dict[str, list[dict[str, Any]]] = {}
    pending_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    threshold_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    reused = 0

    for split_index, seed in enumerate(seeds):
        print(f"Preparing split seed={seed}", flush=True)
        ctx_args = target_args()
        ctx_args.candidate_names = sorted({args.repair_score, *domain_scores.values()})
        ctx = _prepare_seed_context(ctx_args, "Target", TARGET_TEXT, TARGET_COMBINED, TARGET_QWEN, seed)
        crop_lengths, whole_accepts, cal_info = calibrate_domains(ctx, args.repair_score, domain_scores, domains)
        for info in cal_info.values():
            threshold_rows.append({"split_seed": seed, **info.__dict__})
        traces = selected_test_traces(
            ctx,
            domains,
            args.per_domain_test_limit if args.per_domain_test_limit > 0 else None,
            seed,
        )
        manifest_rows.extend(
            {
                "split_seed": seed,
                "trace_id": trace.trace_id,
                "domain": trace.domain,
                "is_test_trace": True,
            }
            for trace in traces
        )
        for cfg in configs:
            for trace in traces:
                info = cal_info[trace.domain]
                total_steps = len(trace.steps)
                mode_context = {
                    "question_only": (0, None),
                    "full_trace": (total_steps, True),
                    "whole_trace_abstention": (
                        total_steps if bool(whole_accepts[trace.trace_id]) else 0,
                        bool(whole_accepts[trace.trace_id]),
                    ),
                    "cpcc_prefix": (
                        int(crop_lengths[trace.trace_id]),
                        int(crop_lengths[trace.trace_id]) == total_steps,
                    ),
                }
                for mode, (context_len, accepted) in mode_context.items():
                    job = build_prompt_job(
                        split_seed=seed,
                        split_index=split_index,
                        trace=trace,
                        mode=mode,
                        context_len=int(context_len),
                        accepted=accepted,
                        cfg=cfg,
                        domain_info=info,
                        prompt_dir=prompt_dir,
                    )
                    done_key = (
                        int(seed),
                        str(cfg["model"]),
                        str(trace.trace_id),
                        str(mode),
                        str(job["prompt_sha256"]),
                    )
                    if done_key in done:
                        continue
                    cached = cache.get(cache_key(str(cfg["model"]), str(job["prompt_sha256"])))
                    if cached is not None:
                        rows.append(row_from_job_and_response(job, response_fields_from_row(cached)))
                        done.add(done_key)
                        reused += 1
                    else:
                        key = cache_key(str(cfg["model"]), str(job["prompt_sha256"]))
                        pending_groups.setdefault(key, []).append(job)
                        if len(pending_groups[key]) == 1:
                            jobs_by_model.setdefault(str(cfg["label"]), []).append(job)

    pd.DataFrame(threshold_rows).to_csv(out_dir / "thresholds_by_split_domain.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(out_dir / "test_trace_manifest.csv", index=False)
    pending_rows = [
        {
            "model": job["model"],
            "model_label": job["model_label"],
            "split_seed": job["split_seed"],
            "domain": job["domain"],
            "trace_id": job["trace_id"],
            "mode": job["mode"],
            "context_steps": job["context_steps"],
            "prompt_sha256": job["prompt_sha256"],
            "prompt_path": job["prompt_path"],
        }
        for jobs in jobs_by_model.values()
        for job in jobs
    ]
    pd.DataFrame(pending_rows).to_csv(out_dir / "pending_generations.csv", index=False)

    if args.dry_run:
        if rows:
            pd.DataFrame(rows).to_csv(raw_path, index=False)
        (out_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "seeds": seeds,
                    "domains": sorted(domains),
                    "repair_score": args.repair_score,
                    "domain_scores": domain_scores,
                    "models": model_keys,
                    "dry_run": True,
                    "cached_rows_reused": reused,
                    "pending_generations": len(pending_rows),
                    "split_protocol": "trace-level score-train/cal/test 60/20/20; no selection split; adapters refit on score-training traces each split; Direct PRM is frozen",
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(f"Dry run wrote {out_dir}; reused={reused}, pending={len(pending_rows)}", flush=True)
        return

    for cfg in configs:
        label = str(cfg["label"])
        jobs = jobs_by_model.get(label, [])
        if not jobs:
            continue
        print(f"Running {len(jobs)} missing repeated-split repair prompts for {label}", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, int(args.repair_workers))) as pool:
            futures = [pool.submit(run_generation, args, job, int(cfg["num_predict"])) for job in jobs]
            for future in as_completed(futures):
                row = future.result()
                key = cache_key(str(row["model"]), str(row["prompt_sha256"]))
                cache[key] = pd.Series(row)
                response = response_fields_from_row(pd.Series(row))
                group_jobs = pending_groups.get(key, [row])
                for grouped_job in group_jobs:
                    if isinstance(grouped_job, dict) and "prompt" in grouped_job:
                        rows.append(row_from_job_and_response(grouped_job, response))
                    else:
                        rows.append(row)
                print(
                    f"repair split={row['split_seed']} model={row['model_label']} "
                    f"domain={row['domain']} mode={row['mode']} correct={row['final_correct']} "
                    f"copies={len(group_jobs)} error={row['error']}",
                    flush=True,
                )
                pd.DataFrame(rows).to_csv(raw_path, index=False)

    raw = pd.DataFrame(rows)
    raw.to_csv(raw_path, index=False)
    split_modes, _split_deltas, inference = summarize_split_modes(raw, out_dir)
    write_compact_tables(split_modes, inference, out_dir)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "domains": sorted(domains),
                "repair_score": args.repair_score,
                "domain_scores": domain_scores,
                "models": model_keys,
                "cached_rows_reused": reused,
                "pending_generations_initial": len(pending_rows),
                "split_protocol": "trace-level score-train/cal/test 60/20/20; no selection split; adapters refit on score-training traces each split; Direct PRM is frozen",
                "alpha": ALPHA_MAIN,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Wrote repeated-split repair outputs to {out_dir}", flush=True)
    if not inference.empty:
        print(inference[inference["baseline"].eq("best_non_crop")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
