from __future__ import annotations

from crop.data import make_toy_traces
from crop.experiments.exp16_adaptive_adapters import _adaptive_split_like, split_traces_four_way
from crop.prefix_aware import augment_with_prefix_features


def test_split_traces_four_way_is_trace_disjoint_and_exhaustive():
    traces = make_toy_traces(n_traces=48, min_steps=2, max_steps=4, seed=13)

    split = split_traces_four_way(traces, seed=7)

    parts = [split.train, split.select, split.cal, split.test]
    part_ids = [{trace.trace_id for trace in part} for part in parts]
    assert sum(len(part) for part in parts) == len(traces)
    assert set().union(*part_ids) == {trace.trace_id for trace in traces}
    for left_idx, left in enumerate(part_ids):
        for right in part_ids[left_idx + 1 :]:
            assert not (left & right)
    assert all(parts)


def test_adaptive_split_like_reuses_reference_trace_ids_on_feature_views():
    traces = make_toy_traces(n_traces=24, min_steps=3, max_steps=3, n_features=5, seed=3)
    augmented = augment_with_prefix_features(traces)
    reference = split_traces_four_way(traces, seed=11)

    view_split = _adaptive_split_like(reference, augmented)

    assert [trace.trace_id for trace in view_split.train] == [trace.trace_id for trace in reference.train]
    assert [trace.trace_id for trace in view_split.select] == [trace.trace_id for trace in reference.select]
    assert [trace.trace_id for trace in view_split.cal] == [trace.trace_id for trace in reference.cal]
    assert [trace.trace_id for trace in view_split.test] == [trace.trace_id for trace in reference.test]
    assert view_split.train[0].steps[0].x.shape[0] > reference.train[0].steps[0].x.shape[0]
