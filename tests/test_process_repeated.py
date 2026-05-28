from __future__ import annotations

import numpy as np

from crop.data import make_toy_traces
from crop.experiments.exp09_process_repeated import (
    COE_SCORE_COLUMNS,
    _build_cache_tables,
    _conditional_upper_bound,
    _select_lambda_selective_risk,
)
from crop.scripts.import_external_process import _parse_math_shepherd_steps, export_external_process
from crop.metrics import first_error_diagnostics, prefix_diagnostics
from crop.sequence import candidate_first_error_set


def test_process_cache_alignment_preserves_trace_and_step_ids():
    traces = make_toy_traces(n_traces=6, min_steps=2, max_steps=3, n_features=4, seed=7)
    coe_traces = make_toy_traces(n_traces=6, min_steps=2, max_steps=3, n_features=len(COE_SCORE_COLUMNS), seed=7)
    step_cache, trace_cache = _build_cache_tables(traces, coe_traces)

    assert set(["trace_id", "step_id", "label_step_error", "first_error_step"]).issubset(step_cache.columns)
    assert len(step_cache) == sum(len(trace.steps) for trace in traces)
    assert len(trace_cache) == len(traces)
    first = step_cache.sort_values(["trace_id", "step_id"]).iloc[0]
    trace = next(t for t in traces if t.trace_id == first["trace_id"])
    assert int(first["label_step_error"]) == int(trace.steps[int(first["step_id"])].y_error)


def test_first_error_diagnostics_reports_error_only_metrics():
    traces = make_toy_traces(n_traces=4, min_steps=3, max_steps=3, n_features=2, seed=1)
    for trace in traces:
        for step in trace.steps:
            step.y_error = 0
    traces[0].steps[0].y_error = 1
    traces[1].steps[1].y_error = 1

    scores = [
        np.asarray([0.9, 0.1, 0.1]),
        np.asarray([0.1, 0.8, 0.1]),
        np.asarray([0.2, 0.1, 0.1]),
        np.asarray([0.9, 0.1, 0.1]),
    ]
    candidate_sets = [candidate_first_error_set(s, 0.5, include_no_error=True) for s in scores]
    metrics = first_error_diagnostics(candidate_sets, scores, traces)

    assert metrics["fe_coverage_error_only"] == 1.0
    assert metrics["fe_candidate_size_excluding_empty"] == 0.75
    assert metrics["false_localization_on_clean"] == 0.5
    assert metrics["fe_top1_accuracy_error_only"] == 1.0
    assert metrics["fe_within1_error_only"] == 1.0
    assert metrics["fe_within2_error_only"] == 1.0
    assert metrics["fe_mean_nearest_distance_error_only"] == 0.0
    assert metrics["fe_top1_mean_abs_distance_error_only"] == 0.0
    assert metrics["fe_top1_median_abs_distance_error_only"] == 0.0
    assert metrics["fe_candidate_before_first_error_rate"] == 0.0
    assert metrics["fe_candidate_after_first_error_rate"] == 0.0
    assert metrics["clean_trace_false_alarm_rate"] == 0.5


def test_prefix_diagnostics_identifies_overruns():
    traces = make_toy_traces(n_traces=3, min_steps=3, max_steps=3, n_features=2, seed=2)
    for trace in traces:
        for step in trace.steps:
            step.y_error = 0
    traces[0].steps[1].y_error = 1
    traces[1].steps[2].y_error = 1
    metrics = prefix_diagnostics(traces, np.asarray([1, 3, 2]))
    assert metrics["prefix_nonempty_rate"] == 1.0
    assert metrics["prefix_stops_at_or_before_first_error_rate"] == 0.5
    assert metrics["prefix_overruns_first_error_rate"] == 0.5


def test_selective_risk_bound_is_conservative_for_empty_acceptance():
    assert _conditional_upper_bound(0, 0, 0.05) == 1.0
    assert _conditional_upper_bound(0, 10, 0.05) < 0.4

    traces = make_toy_traces(n_traces=10, min_steps=1, max_steps=1, n_features=2, seed=3)
    for trace in traces:
        trace.steps[0].y_error = 0
    scores = [np.asarray([0.1]) for _ in traces]
    lam, upper, bad, total = _select_lambda_selective_risk(traces, scores, np.asarray([0.0, 0.2]), alpha=0.5)
    assert lam == 0.2
    assert bad == 0
    assert total == 10
    assert upper <= 0.5


def test_external_process_export_normalizes_fake_trace(tmp_path):
    steps, errors = _parse_math_shepherd_steps(
        "Question? Step 1: correct work ки Step 2: bad work",
        "Question? Step 1: correct work + Step 2: bad work *",
    )
    assert steps == ["correct work", "bad work"]
    assert errors == {1}

    from crop.cheap_baselines import CheapTrace

    trace = CheapTrace(
        trace_id="fake:0",
        domain="fake",
        source_file="fake",
        source_stem="fake",
        complexity=None,
        expression_id="0",
        original_expression="Question?",
        correct_value="",
        predicted_value="",
        steps=tuple(
            {"step_number": idx, "step_content": text, "step_label": idx not in errors}
            for idx, text in enumerate(steps)
        ),
    )
    paths = export_external_process([trace], tmp_path, "fake")
    assert paths["text_npz"].exists()
    assert paths["combined_npz"].exists()
    with np.load(paths["combined_npz"], allow_pickle=True) as data:
        assert data["features"].shape == (2, 63)
