"""Small reproducibility and file helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def environment_info(command: str | None = None) -> dict[str, Any]:
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        git_commit = None
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except Exception:
        sklearn_version = None
    return {
        "git_commit": git_commit,
        "python_version": sys.version,
        "platform": platform.platform(),
        "sklearn_version": sklearn_version,
        "numpy_version": np.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command if command is not None else " ".join(sys.argv),
        "cwd": os.getcwd(),
    }


def parse_optional_ints(values: list[str] | None, n: int) -> list[int | None]:
    if values is None:
        return [None] * n
    if len(values) != n:
        raise ValueError("complexities length must match features length")
    out = []
    for value in values:
        out.append(None if str(value).lower() in {"none", "null"} else int(value))
    return out
