import numpy as np

from crop.data import load_crop_npz
from crop.labels import normalize_step_label


def test_step_label_true_means_correct():
    is_correct, y_error = normalize_step_label(True)
    assert is_correct is True
    assert y_error == 0


def test_step_label_false_means_error():
    is_correct, y_error = normalize_step_label(False)
    assert is_correct is False
    assert y_error == 1


def test_npz_loader_normalizes_nested_step_labels(tmp_path):
    path = tmp_path / "features.npz"
    features = np.zeros((2, 3))
    metadata = np.asarray(
        [
            {"expr_id": "a", "step_labels": {"step_label": True, "step_number": 1}},
            {"expr_id": "b", "step_labels": {"step_label": False, "step_number": 1}},
        ],
        dtype=object,
    )
    np.savez(path, features=features, metadata=metadata)
    records = load_crop_npz(path, domain="toy")
    assert records[0].is_correct is True
    assert records[0].y_error == 0
    assert records[1].is_correct is False
    assert records[1].y_error == 1
