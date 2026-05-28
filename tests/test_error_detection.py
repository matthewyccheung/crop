from crop.conformal import lower_conformal_quantile
from crop.data import make_toy_traces
from crop.metrics import error_detection_metrics
from crop.models import fit_verifier, make_model, predict_probs
from crop.splits import flatten_steps, split_traces


def test_error_detection_smoke_recall():
    traces = make_toy_traces(n_traces=400, seed=5, error_rate=0.2)
    split = split_traces(traces, seed=5)
    model = fit_verifier(make_model("logistic_l2", seed=5, class_weight="balanced"), split.train)
    cal_probs = predict_probs(model, split.cal)
    test_probs = predict_probs(model, split.test)
    _, cal_y, _, _, _ = flatten_steps(split.cal)
    _, test_y, _, _, _ = flatten_steps(split.test)
    threshold = lower_conformal_quantile(cal_probs[cal_y == 1, 1], alpha=0.1)
    metrics = error_detection_metrics(test_y, test_probs[:, 1], threshold)
    assert metrics["error_recall"] >= 0.75
