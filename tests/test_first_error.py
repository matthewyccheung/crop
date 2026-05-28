import numpy as np

from crop.data import make_toy_traces
from crop.risk_control import (
    calibrate_selective_acceptance,
    first_error_error_only_losses_by_lambda,
    first_error_losses_by_lambda,
    select_lambda_crc,
)
from crop.sequence import candidate_first_error_set, first_error_index


def test_first_error_index_and_candidate_set():
    y_errors = np.array([0, 0, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.4, 0.8])
    candidates = candidate_first_error_set(scores, 0.75)
    assert first_error_index(y_errors) == 2
    assert 2 in candidates


def test_first_error_crc_selects_grid_value():
    traces = make_toy_traces(n_traces=60, seed=8, error_rate=0.2)
    scores = [trace.y_errors.astype(float) for trace in traces]
    lambdas = np.linspace(0.0, 1.0, 51)
    losses = first_error_losses_by_lambda(traces, scores, lambdas)
    lambda_hat, risk = select_lambda_crc(losses, lambdas, alpha=0.1, direction="increasing")
    assert lambda_hat in set(lambdas)
    assert risk <= 0.1


def test_error_only_first_error_losses_skip_clean_traces():
    traces = make_toy_traces(n_traces=3, min_steps=3, max_steps=3, n_features=2, seed=4)
    for trace in traces:
        for step in trace.steps:
            step.y_error = 0
    traces[0].steps[1].y_error = 1
    traces[2].steps[2].y_error = 1
    scores = [np.asarray([0.1, 0.8, 0.2]), np.asarray([0.9, 0.1, 0.1]), np.asarray([0.2, 0.3, 0.7])]
    losses = first_error_error_only_losses_by_lambda(traces, scores, np.asarray([0.5]))
    assert losses.shape == (1, 2)
    assert losses[0].tolist() == [0, 0]


def test_selective_acceptance_rejects_all_when_no_threshold_is_feasible():
    traces = make_toy_traces(n_traces=4, min_steps=1, max_steps=1, n_features=2, seed=5)
    for trace in traces:
        trace.steps[0].y_error = 1
    scores = [np.asarray([0.1]) for _ in traces]
    selected = calibrate_selective_acceptance(traces, scores, np.asarray([0.1, 0.2]), beta_level=0.05, delta=0.05)
    assert selected["feasible"] is False
    assert selected["accepted"] == 0
