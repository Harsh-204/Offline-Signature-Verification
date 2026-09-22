"""Unit tests for feature_selection.py."""

import numpy as np
import pytest

from sigverify.feature_selection import (
    fit_reducer,
    load_reducer,
    save_reducer,
    transform,
)


@pytest.fixture
def dummy_features():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 100)
    y = np.array([0] * 15 + [1] * 15)
    return X, y


def test_pca_reduces_dimensions(dummy_features):
    X, _ = dummy_features
    pca = fit_reducer(X, method="pca", n_components=10)
    X_reduced = transform(X, pca)
    assert X_reduced.shape == (30, 10)


def test_lda_reduces_dimensions(dummy_features):
    X, y = dummy_features
    # Binary classification LDA produces max 1 component
    lda = fit_reducer(X, y, method="lda", n_components=10)
    X_reduced = transform(X, lda)
    assert X_reduced.shape == (30, 1)


def test_lda_missing_labels(dummy_features):
    X, _ = dummy_features
    with pytest.raises(ValueError):
        fit_reducer(X, y_train=None, method="lda")


def test_save_load_roundtrip(tmp_path, dummy_features):
    X, _ = dummy_features
    pca = fit_reducer(X, method="pca", n_components=5)
    save_path = str(tmp_path / "pca.pkl")
    
    save_reducer(pca, save_path)
    loaded_pca = load_reducer(save_path)
    
    t1 = transform(X, pca)
    t2 = transform(X, loaded_pca)
    assert np.allclose(t1, t2)
