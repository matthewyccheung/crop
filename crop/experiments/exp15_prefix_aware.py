"""Prefix-aware CPCC score-learning experiments."""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from crop.data import TraceRecord, load_many_npz
from crop.experiments.common import ScoreBundle, build_score_bundle
from crop.experiments.exp09_process_repeated import (
    COE_SCORE_COLUMNS,
    _artifact_views,
    _evaluate_bundle,
    _fit_model_bundle,
    _split_like,
)
from crop.metrics import full_trace_accept_rate, prefix_contamination_rate, safe_aupr, safe_auroc
from crop.models import fit_verifier, make_model, scores_by_trace_from_model
from crop.prefix_aware import (
    append_trace_score_feature,
    augment_with_prefix_features,
    flatten_hazard_labels,
    flatten_prefix_labels,
    select_named_feature_columns,
    traces_with_hazard_targets,
    traces_with_prefix_targets,
)
from crop.splits import Split, split_traces
from crop.risk_control import prefix_lengths, prefix_losses_by_lambda, select_lambda_crc
from crop.utils import ensure_dir, write_json


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _summarize_prefix_aware(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = [
        col
        for col in ("dataset", "score", "score_family", "feature_view", "training_target", "alpha")
        if col in df.columns
    ]
    numeric = [col for col in df.columns if col not in set(group_cols) and pd.api.types.is_numeric_dtype(df[col])]
    grouped = df.groupby(group_cols, dropna=False)
    mean = grouped[numeric].mean(numeric_only=True)
    std = grouped[numeric].std(numeric_only=True).fillna(0.0)
    count = grouped[numeric].count()
    pieces = []
    for col in numeric:
        pieces.append(
            pd.DataFrame(
                {
                    f"{col}_mean": mean[col],
                    f"{col}_std": std[col],
                    f"{col}_n": count[col],
                    f"{col}_ci95": 1.96 * std[col] / np.sqrt(count[col].clip(lower=1)),
                }
            )
        )
    return pd.concat(pieces, axis=1).reset_index()


def _paired_split_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Paired split intervals for target-isolation comparisons."""

    if df.empty:
        return pd.DataFrame()
    pairs = [
        ("prefix_combined_logistic_l2", "combined_logistic_l2", "Prefix combined - Step combined"),
        ("hazard_combined_logistic_l2", "combined_logistic_l2", "Hazard combined - Step combined"),
        ("hazard_combined_logistic_l2", "prefix_combined_logistic_l2", "Hazard combined - Prefix combined"),
        ("step_qwen_combined_logistic_l2", "qwen_prm_error", "Step+Qwen - Qwen PRM"),
        ("prefix_qwen_combined_logistic_l2", "step_qwen_combined_logistic_l2", "Prefix+Qwen - Step+Qwen"),
        ("hazard_qwen_combined_logistic_l2", "step_qwen_combined_logistic_l2", "Hazard+Qwen - Step+Qwen"),
        ("hazard_qwen_combined_logistic_l2", "prefix_qwen_combined_logistic_l2", "Hazard+Qwen - Prefix+Qwen"),
        ("hazard_no_artifact_logistic_l2", "prefix_no_artifact_logistic_l2", "Hazard no-artifact - Prefix no-artifact"),
        ("hazard_no_artifact_logistic_l2", "artifact_token_formatting_logistic_l2", "Hazard no-artifact - Token/format"),
    ]
    metric_cols = [
        ("prefix_retained_fraction", "delta_prefix_kept"),
        ("prefix_full_trace_rate", "delta_full_accept"),
        ("prefix_contamination", "delta_prefix_risk"),
    ]
    rows = []
    for alpha, alpha_df in df.groupby("alpha", dropna=False):
        for score_a, score_b, comparison in pairs:
            a = alpha_df[alpha_df["score"] == score_a].set_index("seed")
            b = alpha_df[alpha_df["score"] == score_b].set_index("seed")
            seeds = sorted(set(a.index) & set(b.index))
            if not seeds:
                continue
            row = {
                "alpha": float(alpha),
                "score_a": score_a,
                "score_b": score_b,
                "comparison": comparison,
                "n_paired_splits": len(seeds),
            }
            for metric, out_name in metric_cols:
                diff = a.loc[seeds, metric].to_numpy(dtype=float) - b.loc[seeds, metric].to_numpy(dtype=float)
                diff = diff[np.isfinite(diff)]
                row[f"{out_name}_mean"] = float(np.mean(diff)) if len(diff) else float("nan")
                row[f"{out_name}_ci_low"] = float(np.percentile(diff, 2.5)) if len(diff) else float("nan")
                row[f"{out_name}_ci_high"] = float(np.percentile(diff, 97.5)) if len(diff) else float("nan")
                row[f"{out_name}_ci95_halfwidth"] = (
                    float(1.96 * np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
                )
            suffix_a = 1.0 - a.loc[seeds, "prefix_retained_fraction"].to_numpy(dtype=float)
            suffix_b = 1.0 - b.loc[seeds, "prefix_retained_fraction"].to_numpy(dtype=float)
            suffix_diff = suffix_a - suffix_b
            row["delta_suffix_routed_fraction_mean"] = float(np.mean(suffix_diff))
            row["delta_suffix_routed_fraction_ci_low"] = float(np.percentile(suffix_diff, 2.5))
            row["delta_suffix_routed_fraction_ci_high"] = float(np.percentile(suffix_diff, 97.5))
            rows.append(row)
    return pd.DataFrame(rows)


def _losses_from_prefix_lengths(traces: list[TraceRecord], lengths: np.ndarray) -> np.ndarray:
    losses = []
    for trace, length in zip(traces, np.asarray(lengths, dtype=int)):
        y = trace.y_errors
        losses.append(bool(length > 0 and np.any(y[:length] > 0)))
    return np.asarray(losses, dtype=int)


def _cascade_policy_rows(
    *,
    dataset: str,
    seed: int,
    split: Split,
    cheap_score: str,
    strong_score: str,
    cheap_bundle: ScoreBundle,
    strong_bundle: ScoreBundle,
    alphas: list[float],
    lambdas: np.ndarray,
    rhos: list[float],
) -> list[dict]:
    """Evaluate predeclared two-stage policies.

    Each component threshold is calibrated at alpha/2. The final bad event is a
    subset of cheap_bad union strong_bad, so this is the finite-family
    Bonferroni protocol from the hypothesis file rather than a validation-tuned
    cascade.
    """

    rows = []
    totals = np.asarray([len(trace.steps) for trace in split.test], dtype=float)
    cheap_cal_losses = prefix_losses_by_lambda(split.cal, cheap_bundle.cal_scores_by_trace, lambdas)
    strong_cal_losses = prefix_losses_by_lambda(split.cal, strong_bundle.cal_scores_by_trace, lambdas)
    for alpha in alphas:
        component_alpha = alpha / 2.0
        cheap_lambda, cheap_cal_risk = select_lambda_crc(
            cheap_cal_losses, lambdas, alpha=component_alpha, direction="increasing"
        )
        strong_lambda, strong_cal_risk = select_lambda_crc(
            strong_cal_losses, lambdas, alpha=component_alpha, direction="increasing"
        )
        cheap_lengths = prefix_lengths(cheap_bundle.test_scores_by_trace, cheap_lambda)
        strong_lengths = prefix_lengths(strong_bundle.test_scores_by_trace, strong_lambda)
        cheap_fracs = cheap_lengths / np.maximum(totals, 1.0)
        for rho in rhos:
            route = cheap_fracs < rho
            final_lengths = np.where(route, strong_lengths, cheap_lengths)
            final_losses = _losses_from_prefix_lengths(split.test, final_lengths)
            suffix = np.maximum(totals - final_lengths, 0.0)
            policy = f"cascade_{cheap_score}_to_{strong_score}_rho_{rho:.2f}".replace(".", "p")
            rows.append(
                {
                    "dataset": dataset,
                    "score": policy,
                    "score_family": "cascade",
                    "policy": f"Cascade: {cheap_score} -> {strong_score} if prefix < {rho:.2f}",
                    "cheap_score": cheap_score,
                    "strong_score": strong_score,
                    "route_rule": "cheap_prefix_fraction_below_rho",
                    "rho": rho,
                    "seed": seed,
                    "alpha": alpha,
                    "component_alpha": component_alpha,
                    "calibration_protocol": "bonferroni_component_cpcc",
                    "cheap_lambda": cheap_lambda,
                    "strong_lambda": strong_lambda,
                    "prefix_cal_corrected_risk": min(cheap_cal_risk + strong_cal_risk, 1.0),
                    "cheap_component_cal_risk": cheap_cal_risk,
                    "strong_component_cal_risk": strong_cal_risk,
                    "prefix_contamination": prefix_contamination_rate(final_losses),
                    "prefix_retained_steps": float(np.mean(final_lengths)) if len(final_lengths) else float("nan"),
                    "prefix_retained_fraction": float(np.mean(final_lengths / np.maximum(totals, 1.0)))
                    if len(final_lengths)
                    else float("nan"),
                    "prefix_full_trace_rate": full_trace_accept_rate(final_lengths, totals),
                    "qwen_call_rate": float(np.mean(route)) if len(route) else float("nan"),
                    "relative_scoring_cost": 0.05 + float(np.mean(route)) if len(route) else float("nan"),
                    "review_steps_routed": float(np.mean(suffix)) if len(suffix) else float("nan"),
                    "review_steps_routed_fraction": float(np.mean(suffix / np.maximum(totals, 1.0)))
                    if len(suffix)
                    else float("nan"),
                }
            )
    return rows


def _direct_policy_rows(process: pd.DataFrame) -> pd.DataFrame:
    if process.empty:
        return pd.DataFrame()
    policy_map = {
        "combined_logistic_l2": ("Cheap only: Step combined", 0.0, 0.05),
        "prefix_combined_logistic_l2": ("Cheap only: Prefix combined", 0.0, 0.05),
        "hazard_combined_logistic_l2": ("Cheap only: Hazard combined", 0.0, 0.05),
        "qwen_prm_error": ("Full Qwen PRM", 1.0, 1.0),
        "prefix_qwen_combined_logistic_l2": ("Full Prefix+Qwen", 1.0, 1.0),
        "hazard_qwen_combined_logistic_l2": ("Full Hazard+Qwen", 1.0, 1.0),
    }
    rows = []
    for row in process.itertuples(index=False):
        score = str(getattr(row, "score"))
        if score not in policy_map:
            continue
        policy, call_rate, cost = policy_map[score]
        rows.append(
            {
                "dataset": getattr(row, "dataset"),
                "score": f"direct_{score}",
                "score_family": "direct",
                "policy": policy,
                "cheap_score": score if call_rate == 0.0 else "",
                "strong_score": score if call_rate == 1.0 else "",
                "route_rule": "none",
                "rho": np.nan,
                "seed": getattr(row, "seed"),
                "alpha": getattr(row, "alpha"),
                "component_alpha": getattr(row, "alpha"),
                "calibration_protocol": "single_score_cpcc",
                "prefix_cal_corrected_risk": getattr(row, "prefix_cal_corrected_risk"),
                "prefix_contamination": getattr(row, "prefix_contamination"),
                "prefix_retained_steps": getattr(row, "prefix_retained_steps"),
                "prefix_retained_fraction": getattr(row, "prefix_retained_fraction"),
                "prefix_full_trace_rate": getattr(row, "prefix_full_trace_rate"),
                "qwen_call_rate": call_rate,
                "relative_scoring_cost": cost,
                "review_steps_routed_fraction": 1.0 - float(getattr(row, "prefix_retained_fraction")),
            }
        )
    return pd.DataFrame(rows)


def _paired_cascade_deltas(cascade: pd.DataFrame) -> pd.DataFrame:
    if cascade.empty:
        return pd.DataFrame()
    rows = []
    for row in cascade[cascade["score_family"] == "cascade"].itertuples(index=False):
        cascade_score = str(getattr(row, "score"))
        alpha = float(getattr(row, "alpha"))
        seed = int(getattr(row, "seed"))
        for baseline_kind, baseline_score in (
            ("cheap", f"direct_{getattr(row, 'cheap_score')}"),
            ("strong", f"direct_{getattr(row, 'strong_score')}"),
        ):
            base = cascade[
                (cascade["score"] == baseline_score)
                & (cascade["alpha"] == alpha)
                & (cascade["seed"] == seed)
            ]
            if base.empty:
                continue
            base_row = base.iloc[0]
            rows.append(
                {
                    "alpha": alpha,
                    "seed": seed,
                    "cascade_score": cascade_score,
                    "baseline_kind": baseline_kind,
                    "baseline_score": baseline_score,
                    "delta_prefix_kept": float(getattr(row, "prefix_retained_fraction"))
                    - float(base_row["prefix_retained_fraction"]),
                    "delta_full_accept": float(getattr(row, "prefix_full_trace_rate")) - float(base_row["prefix_full_trace_rate"]),
                    "delta_prefix_risk": float(getattr(row, "prefix_contamination")) - float(base_row["prefix_contamination"]),
                    "delta_qwen_call_rate": float(getattr(row, "qwen_call_rate")) - float(base_row["qwen_call_rate"]),
                }
            )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    summary_rows = []
    for keys, sub in raw.groupby(["alpha", "cascade_score", "baseline_kind", "baseline_score"], dropna=False):
        summary = dict(zip(["alpha", "cascade_score", "baseline_kind", "baseline_score"], keys))
        summary["n_paired_splits"] = int(len(sub))
        for col in ("delta_prefix_kept", "delta_full_accept", "delta_prefix_risk", "delta_qwen_call_rate"):
            vals = sub[col].to_numpy(dtype=float)
            summary[f"{col}_mean"] = float(np.mean(vals))
            summary[f"{col}_ci_low"] = float(np.percentile(vals, 2.5))
            summary[f"{col}_ci_high"] = float(np.percentile(vals, 97.5))
        summary_rows.append(summary)
    return pd.DataFrame(summary_rows)


def _read_qwen_scores(path: str | None, score_col: str) -> dict[str, dict[int, float]] | None:
    if not path:
        return None
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    required = {"trace_id", "step_id", score_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    out: dict[str, dict[int, float]] = {}
    for row in df.itertuples(index=False):
        trace_id = str(getattr(row, "trace_id"))
        step_id = int(getattr(row, "step_id"))
        score = float(getattr(row, score_col))
        out.setdefault(trace_id, {})[step_id] = score
    return out


def _scores_by_trace_from_map(
    traces: list[TraceRecord],
    scores_by_trace_id: dict[str, dict[int, float]],
    *,
    missing_value: float = 0.5,
) -> list[np.ndarray]:
    out = []
    for trace in traces:
        trace_scores = scores_by_trace_id.get(trace.trace_id, {})
        out.append(
            np.asarray(
                [float(trace_scores.get(step.step_number, missing_value)) for step in trace.steps],
                dtype=float,
            )
        )
    return out


def _score_map_coverage(traces: list[TraceRecord], scores_by_trace_id: dict[str, dict[int, float]]) -> float:
    total = 0
    found = 0
    for trace in traces:
        trace_scores = scores_by_trace_id.get(trace.trace_id, {})
        for step in trace.steps:
            total += 1
            found += int(step.step_number in trace_scores)
    return float(found / total) if total else float("nan")


def _qwen_bundle(split: Split, scores_by_trace_id: dict[str, dict[int, float]]) -> ScoreBundle:
    cal_by_trace = _scores_by_trace_from_map(split.cal, scores_by_trace_id)
    test_by_trace = _scores_by_trace_from_map(split.test, scores_by_trace_id)
    return ScoreBundle(
        name="qwen_prm_error",
        cal_scores_by_trace=cal_by_trace,
        test_scores_by_trace=test_by_trace,
        cal_step_scores=np.concatenate(cal_by_trace) if cal_by_trace else np.asarray([]),
        test_step_scores=np.concatenate(test_by_trace) if test_by_trace else np.asarray([]),
    )


def _fit_prefix_bundle(model_name: str, split: Split, seed: int, class_weight: str = "balanced") -> ScoreBundle:
    train = traces_with_prefix_targets(split.train)
    model = fit_verifier(make_model(model_name, seed=seed, class_weight=class_weight), train)
    cal_by_trace = scores_by_trace_from_model(model, split.cal)
    test_by_trace = scores_by_trace_from_model(model, split.test)
    return ScoreBundle(
        name=model_name,
        cal_scores_by_trace=cal_by_trace,
        test_scores_by_trace=test_by_trace,
        cal_step_scores=np.concatenate(cal_by_trace) if cal_by_trace else np.asarray([]),
        test_step_scores=np.concatenate(test_by_trace) if test_by_trace else np.asarray([]),
        model=model,
    )


def _fit_hazard_bundle(model_name: str, split: Split, seed: int, class_weight: str = "balanced") -> ScoreBundle:
    train = traces_with_hazard_targets(split.train)
    model = fit_verifier(make_model(model_name, seed=seed, class_weight=class_weight), train)
    cal_by_trace = scores_by_trace_from_model(model, split.cal)
    test_by_trace = scores_by_trace_from_model(model, split.test)
    return ScoreBundle(
        name=model_name,
        cal_scores_by_trace=cal_by_trace,
        test_scores_by_trace=test_by_trace,
        cal_step_scores=np.concatenate(cal_by_trace) if cal_by_trace else np.asarray([]),
        test_step_scores=np.concatenate(test_by_trace) if test_by_trace else np.asarray([]),
        model=model,
    )


def _add_prefix_auroc(rows: list[dict], split: Split, bundle: ScoreBundle, training_target: str, feature_view: str) -> list[dict]:
    prefix_y = flatten_prefix_labels(split.test)
    hazard_y = flatten_hazard_labels(split.test)
    prefix_auroc = safe_auroc(prefix_y, bundle.test_step_scores) if len(prefix_y) else float("nan")
    first_error_aupr = safe_aupr(hazard_y, bundle.test_step_scores) if len(hazard_y) else float("nan")
    first_error_auroc = safe_auroc(hazard_y, bundle.test_step_scores) if len(hazard_y) else float("nan")
    for row in rows:
        row["training_target"] = training_target
        row["feature_view"] = feature_view
        row["prefix_auroc"] = prefix_auroc
        row["first_error_aupr"] = first_error_aupr
        row["first_error_auroc"] = first_error_auroc
    return rows


def _build_views(
    combined: list[TraceRecord],
    text: list[TraceRecord],
    scores_by_trace_id: dict[str, dict[int, float]] | None,
) -> dict[str, list[TraceRecord]]:
    views: dict[str, list[TraceRecord]] = {
        "combined": combined,
        "text": text,
        **_artifact_views(combined),
        "prefix_combined": augment_with_prefix_features(combined, include_position_features=True),
        "prefix_text": augment_with_prefix_features(text, include_position_features=True),
    }
    try:
        no_artifact = select_named_feature_columns(combined, keep_names=set(COE_SCORE_COLUMNS))
        views["prefix_no_artifact"] = augment_with_prefix_features(no_artifact, include_position_features=False)
    except ValueError:
        pass
    if scores_by_trace_id is not None:
        qwen_combined = append_trace_score_feature(combined, scores_by_trace_id)
        views["prefix_qwen_combined"] = augment_with_prefix_features(qwen_combined, include_position_features=True)
    return views


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_text_features", required=True)
    parser.add_argument("--step_combined_features", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--qwen_scores_csv", default=None)
    parser.add_argument("--qwen_score_col", default="qwen_prm_error")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(2806, 2816)))
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.025, 0.05, 0.075, 0.10])
    parser.add_argument("--lambda_grid_size", type=int, default=101)
    parser.add_argument("--dataset_name", default="target")
    parser.add_argument("--class_weight", default="balanced")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--table_fixed_only",
        action="store_true",
        help="Run only the fixed-score rows needed for the manuscript prefix-utility table.",
    )
    args = parser.parse_args()

    if args.quick:
        args.seeds = args.seeds[:1]
        args.alphas = [0.05]
        args.lambda_grid_size = min(args.lambda_grid_size, 51)

    outdir = ensure_dir(args.output_dir)
    combined = load_many_npz([args.step_combined_features], ["mixed"], allow_nan=True)
    text = load_many_npz([args.step_text_features], ["mixed"], allow_nan=True)
    scores_by_trace_id = _read_qwen_scores(args.qwen_scores_csv, args.qwen_score_col)
    views = _build_views(combined, text, scores_by_trace_id)
    lambdas = np.linspace(0.0, 1.0, args.lambda_grid_size)

    specs: list[dict] = [
        {
            "score": "random",
            "family": "cheap_control",
            "view": "combined",
            "source": "random",
            "training_target": "random",
        },
        {
            "score": "artifact_token_formatting_logistic_l2",
            "family": "artifact_control",
            "view": "artifact_token_formatting",
            "source": "step",
            "training_target": "hand-designed/artifact",
        },
        {
            "score": "text_logistic_l2",
            "family": "learned_detector",
            "view": "text",
            "source": "step",
            "training_target": "Y_t",
        },
        {
            "score": "combined_logistic_l2",
            "family": "learned_detector",
            "view": "combined",
            "source": "step",
            "training_target": "Y_t",
        },
        {
            "score": "prefix_text_logistic_l2",
            "family": "prefix_aware",
            "view": "prefix_text",
            "source": "prefix",
            "training_target": "C_t",
        },
        {
            "score": "prefix_combined_logistic_l2",
            "family": "prefix_aware",
            "view": "prefix_combined",
            "source": "prefix",
            "training_target": "C_t",
        },
        {
            "score": "hazard_combined_logistic_l2",
            "family": "first_error_hazard",
            "view": "prefix_combined",
            "source": "hazard",
            "training_target": "H_t",
        },
    ]
    if "prefix_no_artifact" in views:
        specs.extend(
            [
                {
                    "score": "prefix_no_artifact_logistic_l2",
                    "family": "prefix_aware_ablation",
                    "view": "prefix_no_artifact",
                    "source": "prefix",
                    "training_target": "C_t",
                },
                {
                    "score": "hazard_no_artifact_logistic_l2",
                    "family": "hazard_ablation",
                    "view": "prefix_no_artifact",
                    "source": "hazard",
                    "training_target": "H_t",
                },
            ]
        )
    if scores_by_trace_id is not None:
        specs.extend(
            [
                {
                    "score": "qwen_prm_error",
                    "family": "external_prm",
                    "view": "combined",
                    "source": "qwen",
                    "training_target": "external PRM",
                },
                {
                    "score": "step_qwen_combined_logistic_l2",
                    "family": "qwen_ensemble",
                    "view": "prefix_qwen_combined",
                    "source": "step",
                    "training_target": "Y_t + Qwen feature",
                },
                {
                    "score": "prefix_qwen_combined_logistic_l2",
                    "family": "prefix_aware_ensemble",
                    "view": "prefix_qwen_combined",
                    "source": "prefix",
                    "training_target": "C_t + Qwen feature",
                },
                {
                    "score": "hazard_qwen_combined_logistic_l2",
                    "family": "hazard_ensemble",
                    "view": "prefix_qwen_combined",
                    "source": "hazard",
                    "training_target": "H_t + Qwen feature",
                },
            ]
        )
    if args.table_fixed_only:
        table_scores = {
            "random",
            "artifact_token_formatting_logistic_l2",
            "combined_logistic_l2",
            "qwen_prm_error",
            "step_qwen_combined_logistic_l2",
        }
        specs = [spec for spec in specs if spec["score"] in table_scores]

    process_rows = []
    cascade_rows = []
    runtime_rows = []
    coverage_rows = []
    for seed in args.seeds:
        reference = split_traces(combined, seed=seed)
        split_by_view = {name: _split_like(reference, traces) for name, traces in views.items()}
        bundles_for_seed: dict[str, tuple[Split, ScoreBundle]] = {}
        if scores_by_trace_id is not None:
            coverage_rows.append(
                {
                    "seed": seed,
                    "dataset": args.dataset_name,
                    "train_qwen_step_coverage": _score_map_coverage(reference.train, scores_by_trace_id),
                    "cal_qwen_step_coverage": _score_map_coverage(reference.cal, scores_by_trace_id),
                    "test_qwen_step_coverage": _score_map_coverage(reference.test, scores_by_trace_id),
                }
            )
        for spec in specs:
            split = split_by_view[spec["view"]]
            started = time.perf_counter()
            if spec["source"] == "random":
                bundle = build_score_bundle("random", split, seed=seed, class_weight=args.class_weight)
            elif spec["source"] == "qwen":
                if scores_by_trace_id is None:
                    continue
                bundle = _qwen_bundle(split, scores_by_trace_id)
            elif spec["source"] == "prefix":
                bundle = _fit_prefix_bundle("logistic_l2", split, seed=seed, class_weight=args.class_weight)
            elif spec["source"] == "hazard":
                bundle = _fit_hazard_bundle("logistic_l2", split, seed=seed, class_weight=args.class_weight)
            else:
                bundle = _fit_model_bundle("logistic_l2", split, seed=seed, class_weight=args.class_weight)
            elapsed = time.perf_counter() - started
            rows = _evaluate_bundle(
                score_name=spec["score"],
                score_family=spec["family"],
                split=split,
                bundle=bundle,
                seed=seed,
                alphas=args.alphas,
                lambdas=lambdas,
                runtime_seconds=elapsed,
            )
            process_rows.extend(_add_prefix_auroc(rows, split, bundle, spec["training_target"], spec["view"]))
            if spec["score"] in {
                "combined_logistic_l2",
                "prefix_combined_logistic_l2",
                "hazard_combined_logistic_l2",
                "qwen_prm_error",
                "prefix_qwen_combined_logistic_l2",
                "hazard_qwen_combined_logistic_l2",
            }:
                bundles_for_seed[spec["score"]] = (split, bundle)
            runtime_rows.append(
                {
                    "dataset": args.dataset_name,
                    "score": spec["score"],
                    "score_family": spec["family"],
                    "feature_view": spec["view"],
                    "training_target": spec["training_target"],
                    "seed": seed,
                    "runtime_seconds": elapsed,
                }
            )
        if scores_by_trace_id is not None and "hazard_combined_logistic_l2" in bundles_for_seed:
            cheap_split, cheap_bundle = bundles_for_seed["hazard_combined_logistic_l2"]
            for strong_score in ("qwen_prm_error", "hazard_qwen_combined_logistic_l2"):
                if strong_score not in bundles_for_seed:
                    continue
                _, strong_bundle = bundles_for_seed[strong_score]
                cascade_rows.extend(
                    _cascade_policy_rows(
                        dataset=args.dataset_name,
                        seed=seed,
                        split=cheap_split,
                        cheap_score="hazard_combined_logistic_l2",
                        strong_score=strong_score,
                        cheap_bundle=cheap_bundle,
                        strong_bundle=strong_bundle,
                        alphas=args.alphas,
                        lambdas=lambdas,
                        rhos=[0.25, 0.40, 0.60],
                    )
                )

    process = pd.DataFrame(process_rows)
    process.insert(0, "dataset", args.dataset_name)
    process.to_csv(outdir / "table_prefix_aware.csv", index=False)
    summary = _summarize_prefix_aware(process)
    summary.to_csv(outdir / "table_prefix_aware_summary.csv", index=False)
    paired = _paired_split_deltas(process)
    paired.to_csv(outdir / "table_paired_deltas.csv", index=False)
    cascade = pd.concat([_direct_policy_rows(process), pd.DataFrame(cascade_rows)], ignore_index=True)
    cascade.to_csv(outdir / "table_cascade.csv", index=False)
    _summarize_prefix_aware(cascade).to_csv(outdir / "table_cascade_summary.csv", index=False)
    _paired_cascade_deltas(cascade).to_csv(outdir / "table_cascade_paired_deltas.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(outdir / "table_runtime.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(outdir / "qwen_score_coverage.csv", index=False)
    write_json(outdir / "run_config.json", vars(args))
    print(f"Wrote {outdir}")


if __name__ == "__main__":
    main()
