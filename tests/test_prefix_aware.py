from __future__ import annotations

import numpy as np

from crop.data import make_toy_traces
from crop.prefix_aware import (
    augment_with_prefix_features,
    first_error_hazard_labels,
    first_error_risk_set_mask,
    flatten_hazard_labels,
    flatten_prefix_labels,
    prefix_contamination_labels,
    traces_with_hazard_targets,
    traces_with_prefix_targets,
)


def test_prefix_contamination_labels_clean_trace():
    assert prefix_contamination_labels([0, 0, 0]).tolist() == [0, 0, 0]


def test_prefix_contamination_labels_first_error():
    assert prefix_contamination_labels([0, 1, 0, 0]).tolist() == [0, 1, 1, 1]


def test_prefix_contamination_labels_multiple_errors():
    assert prefix_contamination_labels([0, 1, 0, 1]).tolist() == [0, 1, 1, 1]


def test_traces_with_prefix_targets_does_not_mutate_original_labels():
    traces = make_toy_traces(n_traces=1, min_steps=4, max_steps=4, n_features=3, seed=4)
    for step in traces[0].steps:
        step.y_error = 0
    traces[0].steps[1].y_error = 1

    copied = traces_with_prefix_targets(traces)

    assert traces[0].y_errors.tolist() == [0, 1, 0, 0]
    assert copied[0].y_errors.tolist() == [0, 1, 1, 1]
    assert flatten_prefix_labels(traces).tolist() == [0, 1, 1, 1]


def test_first_error_hazard_labels_only_marks_boundary():
    assert first_error_hazard_labels([0, 0, 0]).tolist() == [0, 0, 0]
    assert first_error_hazard_labels([0, 1, 0, 1]).tolist() == [0, 1, 0, 0]
    assert first_error_risk_set_mask([0, 1, 0, 1]).tolist() == [True, True, False, False]


def test_traces_with_hazard_targets_censors_post_error_steps():
    traces = make_toy_traces(n_traces=1, min_steps=5, max_steps=5, n_features=3, seed=7)
    for step in traces[0].steps:
        step.y_error = 0
    traces[0].steps[2].y_error = 1
    traces[0].steps[4].y_error = 1

    copied = traces_with_hazard_targets(traces)

    assert traces[0].y_errors.tolist() == [0, 0, 1, 0, 1]
    assert copied[0].y_errors.tolist() == [0, 0, 1]
    assert flatten_hazard_labels(traces).tolist() == [0, 0, 1, 0, 0]


def test_prefix_feature_augmentation_preserves_trace_shape_and_adds_features():
    traces = make_toy_traces(n_traces=2, min_steps=3, max_steps=3, n_features=4, seed=5)
    augmented = augment_with_prefix_features(traces)

    assert len(augmented) == len(traces)
    assert len(augmented[0].steps) == len(traces[0].steps)
    assert augmented[0].steps[0].x.shape[0] == 4 * 4 + 3
    assert np.allclose(augmented[0].steps[0].x[:4], traces[0].steps[0].x)
