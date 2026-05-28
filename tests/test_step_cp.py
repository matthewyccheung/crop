from crop.conformal import fit_lac_threshold, predict_lac_sets
from crop.data import make_toy_traces
from crop.metrics import prediction_set_coverage
from crop.models import fit_verifier, make_model, predict_probs
from crop.splits import flatten_steps, split_traces


def test_lac_coverage_smoke():
    traces = make_toy_traces(n_traces=300, seed=4, error_rate=0.15)
    split = split_traces(traces, seed=4)
    model = fit_verifier(make_model("logistic_l2", seed=4, class_weight="balanced"), split.train)
    cal_probs = predict_probs(model, split.cal)
    test_probs = predict_probs(model, split.test)
    _, cal_y, _, _, _ = flatten_steps(split.cal)
    _, test_y, _, _, _ = flatten_steps(split.test)
    qhat = fit_lac_threshold(cal_probs, cal_y, alpha=0.1)
    sets = predict_lac_sets(test_probs, qhat)
    assert prediction_set_coverage(sets, test_y) >= 0.85
    assert all(len(s) > 0 for s in sets)
