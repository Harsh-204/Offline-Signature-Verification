"""Classifier training, persistence, and inference.

The saved model artifact is a single ``.pkl`` file containing:
    - ``scaler``: fitted ``StandardScaler``
    - ``reducer``: fitted PCA/LDA transform (or ``None``)
    - ``clf``: trained ``SVC`` with ``probability=True``
    - ``feature_config``: dict recording which feature groups and reducer
      settings were used, so that ``verify`` can reproduce the exact
      transform chain without refitting.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_pipeline() -> Pipeline:
    """Return an untrained StandardScaler → SVM pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", probability=True, class_weight="balanced")),
    ])


def train_model(
    X_pairs: np.ndarray,
    y_labels: np.ndarray,
    reducer=None,
    feature_config: dict | None = None,
) -> dict[str, Any]:
    """Fit scaler + SVM on pair-feature matrix.

    Parameters
    ----------
    X_pairs : (n_samples, n_pair_features) array
    y_labels : (n_samples,) binary labels (1=genuine, 0=forgery)
    reducer : a fitted PCA/LDA transformer (or ``None``)
    feature_config : metadata about features/reducer for reproducibility

    Returns
    -------
    artifact dict with keys ``scaler``, ``clf``, ``reducer``, ``feature_config``
    """
    pipe = build_pipeline()
    pipe.fit(X_pairs, y_labels)

    return {
        "scaler": pipe.named_steps["scaler"],
        "clf": pipe.named_steps["clf"],
        "reducer": reducer,
        "feature_config": feature_config or {},
    }


def save_model(artifact: dict, path: str) -> None:
    """Persist model artifact to *path*."""
    joblib.dump(artifact, path)


def load_model(path: str) -> dict:
    """Load a persisted model artifact."""
    return joblib.load(path)


def predict(
    artifact: dict,
    pair_features: np.ndarray,
) -> tuple[int, float]:
    """Run inference on a single pair-feature vector.

    Returns ``(label, confidence)`` where label is 1 (genuine) or 0 (forged)
    and confidence is the probability of the predicted class.
    """
    scaler = artifact["scaler"]
    clf = artifact["clf"]

    x = pair_features.reshape(1, -1)
    x = scaler.transform(x)
    label = int(clf.predict(x)[0])
    proba = clf.predict_proba(x)[0]
    genuine_idx = list(clf.classes_).index(label)
    confidence = float(proba[genuine_idx])
    return label, confidence


def predict_batch(
    artifact: dict,
    X_pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch inference.  Returns ``(labels, probabilities_of_genuine)``."""
    scaler = artifact["scaler"]
    clf = artifact["clf"]
    X_scaled = scaler.transform(X_pairs)
    proba = clf.predict_proba(X_scaled)
    labels = np.argmax(proba, axis=1)
    # probability of class 1 (genuine)
    genuine_idx = list(clf.classes_).index(1)
    scores = proba[:, genuine_idx]
    return labels, scores
