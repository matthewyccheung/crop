import numpy as np

from crop.data import load_many_npz, make_toy_traces
from crop.splits import flatten_steps, split_summary, split_traces


def test_trace_split_has_no_leakage():
    traces = make_toy_traces(n_traces=80, seed=1)
    split = split_traces(traces, seed=2)
    train_ids = {t.trace_id for t in split.train}
    cal_ids = {t.trace_id for t in split.cal}
    test_ids = {t.trace_id for t in split.test}
    assert not train_ids & cal_ids
    assert not train_ids & test_ids
    assert not cal_ids & test_ids


def test_flatten_after_split_and_summary_counts():
    traces = make_toy_traces(n_traces=50, seed=3)
    split = split_traces(traces, seed=3)
    X, y, groups, trace_ids, step_numbers = flatten_steps(split.train)
    assert X.shape[0] == len(y) == len(groups) == len(trace_ids) == len(step_numbers)
    summary = split_summary(split)
    assert set(summary) == {"train", "cal", "test"}
    assert summary["train"]["traces"] == len(split.train)


def test_colliding_expr_ids_from_different_feature_sources_stay_separate(tmp_path):
    paths = []
    for dataset, value in (("arith_nt3", 3.0), ("arith_nt5", 5.0)):
        path = tmp_path / "repro_graph_shards" / dataset / f"{dataset}_0_1.npz"
        path.parent.mkdir(parents=True)
        np.savez(
            path,
            features=np.asarray([[value, 0.0]], dtype=float),
            metadata=np.asarray(
                [
                    {
                        "expr_id": 0,
                        "step_number": 0,
                        "before_after": "after",
                        "step_labels": {"step_label": True},
                    }
                ],
                dtype=object,
            ),
        )
        paths.append(path)

    traces = load_many_npz(paths, ["arithmetic", "arithmetic"])
    assert len(traces) == 2
    assert {trace.trace_id for trace in traces} == {"arithmetic:arith_nt3:0", "arithmetic:arith_nt5:0"}
    assert {trace.complexity for trace in traces} == {3, 5}
