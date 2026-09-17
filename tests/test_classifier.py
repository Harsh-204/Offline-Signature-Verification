"""Unit tests for classifier.py."""

import numpy as np
import pytest

from sigverify.classifier import (
    load_model,
    predict,
    predict_batch,
    save_model,
    train_model,
)


@pytest.fixture
def synthetic_pairs():
    rng = np.random.RandomState(42)
    # 20 pairs with 10 features each
    X = rng.randn(20, 10)
    # Trivial distinction: class 1 has positive mean, class 0 negative
    X[:10] += 2.0
    X[10:] -= 2.0
    y = np.array([1] * 10 + [0] * 10)
    return X, y


def test_train_returns_artifact(synthetic_pairs):
    X, y = synthetic_pairs
    artifact = train_model(X, y)
    assert "scaler" in artifact
    assert "clf" in artifact
    assert "reducer" in artifact
    assert "feature_config" in artifact


def test_predict_output_format(synthetic_pairs):
    X, y = synthetic_pairs
    artifact = train_model(X, y)
    
    label, conf = predict(artifact, X[0])
    assert label in (0, 1)
    assert 0.0 <= conf <= 1.0


def test_predict_batch(synthetic_pairs):
    X, y = synthetic_pairs
    artifact = train_model(X, y)

    labels, scores = predict_batch(artifact, X)
    assert labels.shape == (20,)
    assert scores.shape == (20,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_save_load_roundtrip(tmp_path, synthetic_pairs):
    X, y = synthetic_pairs
    artifact = train_model(X, y, feature_config={"test": True})
    save_path = str(tmp_path / "model.pkl")

    save_model(artifact, save_path)
    loaded_art = load_model(save_path)

    l1, c1 = predict(artifact, X[0])
    l2, c2 = predict(loaded_art, X[0])

    assert l1 == l2
    assert np.isclose(c1, c2)
    assert loaded_art["feature_config"] == {"test": True}


def test_overfits_trivial_data(synthetic_pairs):
    X, y = synthetic_pairs
    artifact = train_model(X, y)
    labels, _ = predict_batch(artifact, X)
    acc = np.mean(labels == y)
    assert acc == 1.0
