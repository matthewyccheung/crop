"""Label normalization for CROP step annotations.

CROP/HF metadata uses ``step_label=True`` for a correct step.  This
project uses ``y_error=1`` as the positive class, so the conversion must
be applied immediately at data-loading boundaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np


TRUE_STRINGS = {"true", "correct", "yes", "1", "valid"}
FALSE_STRINGS = {"false", "incorrect", "wrong", "no", "0", "invalid", "error"}


def coerce_step_label(step_label: Any) -> bool:
    """Return ``is_correct`` from a CROP ``step_label`` value."""

    if isinstance(step_label, (bool, np.bool_)):
        return bool(step_label)
    if isinstance(step_label, (int, np.integer)) and step_label in (0, 1):
        return bool(step_label)
    if isinstance(step_label, str):
        normalized = step_label.strip().lower()
        if normalized in TRUE_STRINGS:
            return True
        if normalized in FALSE_STRINGS:
            return False
    raise ValueError(f"Cannot interpret step_label={step_label!r} as correctness")


def normalize_step_label(step_label: Any) -> tuple[bool, int]:
    """Convert CROP correctness labels to ``(is_correct, y_error)``.

    ``y_error=1`` means the step is incorrect/error, which is the positive
    class used by all metrics and models in this package.
    """

    is_correct = coerce_step_label(step_label)
    return is_correct, int(not is_correct)


def metadata_get(metadata: dict[str, Any], key: str, default: Any = None) -> Any:
    """Fetch a key from top-level metadata or nested ``step_labels``."""

    if key in metadata:
        return metadata[key]
    nested = metadata.get("step_labels")
    if isinstance(nested, dict) and key in nested:
        return nested[key]
    return default


def extract_step_label(metadata: dict[str, Any]) -> Any:
    """Extract a correctness label from a CROP-like metadata dictionary."""

    candidates = [
        ("step_label", metadata_get(metadata, "step_label")),
        ("is_correct", metadata_get(metadata, "is_correct")),
    ]
    for _, value in candidates:
        if value is not None:
            return value

    y_error = metadata_get(metadata, "y_error")
    if y_error is not None:
        if int(y_error) not in (0, 1):
            raise ValueError(f"y_error must be 0/1, got {y_error!r}")
        return not bool(int(y_error))

    label = metadata_get(metadata, "label")
    if label is not None:
        return label

    raise KeyError("Could not find step_label/is_correct/y_error in metadata")
