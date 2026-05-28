from __future__ import annotations

import numpy as np
import pandas as pd

from crop.data import make_toy_traces
from crop.experiments.exp08_cheap_baselines import (
    _base_args,
    _mondrian_step_cp_rows,
    _summarize_with_ci,
)
from scripts.make_strengthening_artifacts import compute_routing_metrics


def test_repeated_split_ci_aggregation():
    df = pd.DataFrame(
        [
            {"dataset": "d", "score": "s", "alpha": 0.1, "seed": 1, "coverage": 0.9},
            {"dataset": "d", "score": "s", "alpha": 0.1, "seed": 2, "coverage": 1.0},
        ]
    )
    summary = _summarize_with_ci(df)
    assert summary.loc[0, "coverage_mean"] == 0.95
    assert summary.loc[0, "coverage_n"] == 2
    assert summary.loc[0, "coverage_ci95"] > 0


def test_mondrian_score_source_calibration_runs_on_column(tmp_path):
    traces = make_toy_traces(n_traces=80, n_features=2, seed=3, domain="toy")
    metadata = []
    features = []
    for trace in traces:
        for step in trace.steps:
            meta = dict(step.metadata)
            meta["trace_id"] = trace.trace_id
            meta["domain"] = "even" if int(trace.trace_id.rsplit("_", 1)[-1]) % 2 == 0 else "odd"
            metadata.append(meta)
            features.append([float(step.y_error) + 0.01 * step.step_number])
    path = tmp_path / "scores.npz"
    np.savez(
        path,
        features=np.asarray(features, dtype=float),
        metadata=np.asarray(metadata, dtype=object),
        feature_names=np.asarray(["score_error"], dtype=object),
    )
    args = _base_args(str(path), [0.1], 21)
    from crop.experiments.common import load_traces_from_args

    loaded = load_traces_from_args(args, seed=0)
    rows = _mondrian_step_cp_rows(args, loaded, seed=0, score_source="column:score_error", group_by="domain")
    assert rows[0]["coverage"] >= 0.0
    assert rows[0]["n_groups"] == 2


def test_compute_routing_metrics_counts_routed_first_errors():
    labels = [
        [0, 0, 1, 0],
        [0, 0, 0],
        [1, 0, 0],
    ]
    metrics = compute_routing_metrics([], [2, 3, 1], labels=labels)
    assert np.isclose(metrics["prefix_risk"], 1 / 3)
    assert np.isclose(metrics["prefix_kept"], (2 / 4 + 1 + 1 / 3) / 3)
    assert metrics["error_in_suffix_recall"] == 0.5
    assert np.isclose(metrics["full_accept"], 1 / 3)
    assert metrics["accepted_error"] == 0.0
