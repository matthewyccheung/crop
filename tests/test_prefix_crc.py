import numpy as np

from crop.data import make_toy_traces
from crop.risk_control import (
    corrected_risk,
    prefix_contamination_losses,
    prefix_length,
    prefix_losses_by_lambda,
    select_lambda_crc,
)


def test_prefix_length_examples():
    scores = np.array([0.1, 0.2, 0.8, 0.3])
    assert prefix_length(scores, 0.05) == 0
    assert prefix_length(scores, 0.10) == 1
    assert prefix_length(scores, 0.20) == 2
    assert prefix_length(scores, 0.79) == 2
    assert prefix_length(scores, 0.80) == 4


def test_prefix_contamination_loss_monotone():
    traces = make_toy_traces(n_traces=1, seed=6)
    traces[0].steps[0].y_error = 0
    traces[0].steps[1].y_error = 1
    scores = [np.array([0.1, 0.2, 0.8, 0.3])]
    lambdas = np.array([0.05, 0.10, 0.20, 0.80])
    losses = [prefix_contamination_losses(traces, scores, lam)[0] for lam in lambdas]
    assert losses == sorted(losses)


def test_crc_threshold_smoke():
    traces = make_toy_traces(n_traces=80, seed=7, error_rate=0.2)
    cal = traces[:40]
    scores_by_trace = [trace.y_errors.astype(float) for trace in cal]
    lambdas = np.linspace(0.0, 1.0, 101)
    losses = prefix_losses_by_lambda(cal, scores_by_trace, lambdas)
    lambda_hat, risk = select_lambda_crc(losses, lambdas, alpha=0.1, direction="increasing")
    assert lambda_hat in set(lambdas)
    idx = int(np.where(lambdas == lambda_hat)[0][0])
    assert corrected_risk(losses[idx]) <= 0.1
    assert risk <= 0.1
