import numpy as np
import pytest

from crop import fit_prefix_calibrator
from crop.data import make_toy_traces


def test_high_level_api_fits_and_predicts():
    traces = make_toy_traces(n_traces=80, seed=11, error_rate=0.2)
    scores = [trace.y_errors.astype(float) for trace in traces]
    calibrator = fit_prefix_calibrator(traces[:40], scores[:40], alpha=0.1)
    lengths = calibrator.predict(scores[40:])

    assert calibrator.feasible
    assert calibrator.corrected_calibration_risk <= 0.1
    assert lengths.shape == (40,)
    assert np.all(lengths >= 0)


def test_high_level_api_reports_infeasible_small_calibration_set():
    traces = make_toy_traces(n_traces=3, seed=12)
    scores = [np.zeros(len(trace.steps)) for trace in traces]
    calibrator = fit_prefix_calibrator(traces, scores, alpha=0.05)

    assert not calibrator.feasible
    assert np.isneginf(calibrator.threshold)
    assert np.all(calibrator.predict(scores) == 0)


def test_high_level_api_validates_alignment():
    traces = make_toy_traces(n_traces=2, seed=13)
    with pytest.raises(ValueError, match="steps"):
        fit_prefix_calibrator(traces, [[0.1], [0.2]], alpha=0.5)
