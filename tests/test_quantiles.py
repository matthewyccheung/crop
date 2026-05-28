import numpy as np

from crop.conformal import conformal_quantile, lower_conformal_quantile


def test_conformal_quantile_indexing():
    scores = np.array([1, 2, 3, 4, 5])
    assert conformal_quantile(scores, alpha=0.2) == 5


def test_conformal_quantile_returns_inf_when_index_exceeds_n():
    scores = np.array([1, 2, 3, 4, 5])
    assert np.isinf(conformal_quantile(scores, alpha=0.01))


def test_lower_tail_quantile_is_conservative():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert lower_conformal_quantile(scores, alpha=0.2) == 0.1
