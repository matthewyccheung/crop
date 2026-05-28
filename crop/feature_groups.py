"""Feature-family masks for CROP advanced graph features."""

from __future__ import annotations

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


def infer_feature_groups(n_features: int, n_layers: int = 32) -> dict[str, np.ndarray]:
    all_idx = np.arange(n_features, dtype=int)
    expected = 11 + n_layers + 12
    if n_features == expected:
        global_idx = np.arange(0, 5, dtype=int)
        layer_hist = np.arange(11, 11 + n_layers, dtype=int)
        node_idx = np.concatenate([np.arange(5, 11, dtype=int), layer_hist])
        topo_idx = np.arange(11 + n_layers, expected, dtype=int)
    else:
        LOGGER.warning(
            "Expected %s CROP advanced features, got %s; using conservative inferred groups",
            expected,
            n_features,
        )
        global_idx = np.arange(0, min(5, n_features), dtype=int)
        logit = np.asarray([i for i in (3, 4) if i < n_features], dtype=int)
        topo_start = max(0, n_features - min(12, n_features))
        topo_idx = np.arange(topo_start, n_features, dtype=int)
        node_candidates = [i for i in range(n_features) if i not in set(global_idx) | set(topo_idx)]
        node_idx = np.asarray(node_candidates, dtype=int)

    logit_only = np.asarray([i for i in (3, 4) if i < n_features], dtype=int)
    graph_only_no_logits = np.asarray([i for i in all_idx if i not in set(logit_only)], dtype=int)

    return {
        "all": all_idx,
        "global": global_idx,
        "node": node_idx,
        "topological": topo_idx,
        "logit_only": logit_only,
        "graph_only_no_logits": graph_only_no_logits,
        "global_only": global_idx,
        "node_only": node_idx,
        "topological_only": topo_idx,
        "no_global": np.asarray([i for i in all_idx if i not in set(global_idx)], dtype=int),
        "no_node": np.asarray([i for i in all_idx if i not in set(node_idx)], dtype=int),
        "no_node_activation": np.asarray([i for i in all_idx if i not in set(node_idx)], dtype=int),
        "no_topological": np.asarray([i for i in all_idx if i not in set(topo_idx)], dtype=int),
    }


def select_feature_set(traces, feature_set: str, n_layers: int = 32):
    if not traces:
        return traces
    n_features = traces[0].steps[0].x.shape[0]
    groups = infer_feature_groups(n_features, n_layers=n_layers)
    if feature_set not in groups:
        raise ValueError(f"Unknown feature_set={feature_set!r}; choices={sorted(groups)}")
    idx = groups[feature_set]
    for trace in traces:
        for step in trace.steps:
            step.x = step.x[idx]
    return traces
