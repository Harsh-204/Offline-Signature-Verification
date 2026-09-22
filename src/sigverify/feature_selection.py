"""Dimensionality reduction (PCA / LDA) for feature vectors.

The reducer is **fit only on training data** and then reused at inference
time via :func:`transform`.  It is persisted as part of the model artifact
in :mod:`sigverify.classifier`.
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_reducer(
    X_train: np.ndarray,
    y_train: np.ndarray | None = None,
    method: str = "pca",
    n_components: int = 50,
):
    """Fit a dimensionality-reduction transform on training data.

    Parameters
    ----------
    method : ``"pca"`` (unsupervised) or ``"lda"`` (supervised, needs *y_train*).
    n_components : target dimensionality.  For LDA this is silently capped at
        ``min(n_components, n_classes - 1, n_features)``.

    Returns the fitted transformer (sklearn estimator).
    """
    if method == "pca":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        n_comp = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(X_scaled)
        reducer = Pipeline([("scaler", scaler), ("pca", pca)])
    elif method == "lda":
        if y_train is None:
            raise ValueError("LDA requires y_train labels")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        n_classes = len(np.unique(y_train))
        n_comp = min(n_components, n_classes - 1, X_scaled.shape[1])
        lda = LDA(n_components=n_comp)
        lda.fit(X_scaled, y_train)
        reducer = Pipeline([("scaler", scaler), ("lda", lda)])
    else:
        raise ValueError(f"Unknown method: {method!r}")
    return reducer


def transform(X: np.ndarray, reducer) -> np.ndarray:
    """Apply a fitted reducer to feature matrix *X*."""
    return reducer.transform(X)


def save_reducer(reducer, path: str) -> None:
    """Persist a fitted reducer to disk."""
    joblib.dump(reducer, path)


def load_reducer(path: str):
    """Load a persisted reducer."""
    return joblib.load(path)
