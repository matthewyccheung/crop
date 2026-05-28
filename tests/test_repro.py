import numpy as np

from crop.data import make_toy_traces, truncate_after_first_error
from crop.experiments.common import evaluate_base_model
from crop.metrics import fpr_at_recall_95
from crop.models import make_model
from crop.splits import split_traces


def test_fpr_at_recall_95_uses_observed_threshold():
    y = np.array([1, 1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.75, 0.15, 0.05])
    assert fpr_at_recall_95(y, scores) == 2 / 3


def test_zero_calibration_split_for_repro():
    traces = make_toy_traces(n_traces=50, seed=3)
    split = split_traces(traces, train_frac=0.8, cal_frac=0.0, test_frac=0.2, seed=3)
    assert len(split.train) > 0
    assert len(split.cal) == 0
    assert len(split.test) > 0


def test_class_weight_none_string_uses_sklearn_default():
    model = make_model("gradient_boosting", seed=0, class_weight="none")
    assert model.class_weight is None


def test_truncate_after_first_error_keeps_error_step_only():
    trace = make_toy_traces(n_traces=1, min_steps=5, max_steps=5, seed=0)[0]
    for idx, step in enumerate(trace.steps):
        step.y_error = int(idx in {2, 4})
        step.is_correct = step.y_error == 0
    truncated = truncate_after_first_error([trace])[0]
    assert [step.y_error for step in truncated.steps] == [0, 0, 1]
    assert [step.step_number for step in truncated.steps] == [0, 1, 2]


def test_base_model_step_split_reports_step_unit():
    class Args:
        split_unit = "step"
        train_frac = 0.8
        cal_frac = 0.0
        test_frac = 0.2
        model = "gradient_boosting"
        class_weight = "none"
        calibration = None

    traces = make_toy_traces(n_traces=40, seed=4)
    row = evaluate_base_model(Args(), traces, seed=0)
    assert row["split_unit"] == "step"
    assert row["n_cal_steps"] == 0
    assert row["n_train_steps"] > row["n_test_steps"] > 0
