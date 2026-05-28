"""Verifier model wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .splits import flatten_steps


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"", "none", "null"}:
        return None
    return value


@dataclass
class BaseVerifier:
    estimator: object
    name: str

    def fit(self, X_train, y_train):
        self.estimator.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        probs = self.estimator.predict_proba(X)
        return ensure_binary_proba(probs, getattr(self.estimator, "classes_", np.array([0, 1])))

    def score_error(self, X):
        return self.predict_proba(X)[:, 1]


class WeightedGradientBoosting(BaseVerifier):
    def __init__(self, estimator, name: str, class_weight: Optional[str] = None):
        super().__init__(estimator=estimator, name=name)
        self.class_weight = class_weight

    def fit(self, X_train, y_train):
        if self.class_weight == "balanced":
            weights = compute_sample_weight(class_weight="balanced", y=y_train)
            self.estimator.fit(X_train, y_train, clf__sample_weight=weights)
        else:
            self.estimator.fit(X_train, y_train)
        return self


def ensure_binary_proba(probs: np.ndarray, classes) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    classes = np.asarray(classes)
    out = np.zeros((len(probs), 2), dtype=float)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    for col, cls in enumerate(classes):
        if int(cls) in (0, 1):
            out[:, int(cls)] = probs[:, col]
    row_sums = out.sum(axis=1)
    missing = row_sums == 0
    if np.any(missing):
        out[missing, :] = 0.5
        row_sums = out.sum(axis=1)
    out = out / row_sums[:, None]
    return out


def _pipeline(clf, scaler: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


def make_model(
    name: str,
    seed: int,
    class_weight: str | None = None,
    calibration: str | None = None,
) -> BaseVerifier:
    """Create a scikit-learn verifier with ``y_error=1`` as positive."""

    class_weight = _normalize_optional(class_weight)
    calibration = _normalize_optional(calibration)

    if name == "dummy_prior":
        model: BaseVerifier = BaseVerifier(
            _pipeline(DummyClassifier(strategy="prior", random_state=seed), scaler=False),
            name=name,
        )
    elif name == "logistic_l2":
        clf = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=500,
            class_weight=class_weight,
            random_state=seed,
        )
        model = BaseVerifier(_pipeline(clf, scaler=True), name=name)
    elif name == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=seed)
        model = WeightedGradientBoosting(_pipeline(clf, scaler=False), name=name, class_weight=class_weight)
    elif name == "hist_gradient_boosting":
        clf = HistGradientBoostingClassifier(
            random_state=seed,
            class_weight=class_weight,
            max_iter=30,
            max_leaf_nodes=15,
        )
        model = BaseVerifier(_pipeline(clf, scaler=False), name=name)
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=10,
            max_depth=12,
            min_samples_leaf=2,
            class_weight=class_weight,
            n_jobs=2,
            random_state=seed,
        )
        model = BaseVerifier(_pipeline(clf, scaler=False), name=name)
    else:
        raise ValueError(f"Unknown model name: {name}")

    if calibration in (None, "none"):
        return model
    if calibration == "platt":
        calibrated = CalibratedClassifierCV(model.estimator, method="sigmoid", cv=3)
    elif calibration == "isotonic":
        calibrated = CalibratedClassifierCV(model.estimator, method="isotonic", cv=3)
    else:
        raise ValueError(f"Unknown calibration={calibration!r}")
    return BaseVerifier(calibrated, name=f"{name}_{calibration}")


def fit_verifier(model: BaseVerifier, train_traces) -> BaseVerifier:
    X_train, y_train, _, _, _ = flatten_steps(train_traces)
    if len(np.unique(y_train)) < 2:
        fallback = make_model("dummy_prior", seed=0)
        fallback.fit(X_train, y_train)
        return fallback
    return model.fit(X_train, y_train)


def predict_error_scores(model: BaseVerifier, steps_or_traces) -> np.ndarray:
    X, _, _, _, _ = flatten_steps(steps_or_traces)
    return model.score_error(X)


def predict_probs(model: BaseVerifier, traces) -> np.ndarray:
    X, _, _, _, _ = flatten_steps(traces)
    return model.predict_proba(X)


def scores_by_trace_from_model(model: BaseVerifier, traces) -> list[np.ndarray]:
    out = []
    for trace in traces:
        out.append(model.score_error(trace.X))
    return out
