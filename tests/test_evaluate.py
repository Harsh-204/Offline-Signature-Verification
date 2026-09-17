"""Unit tests for evaluate.py."""

import json
import numpy as np
import pytest

from sigverify.evaluate import (
    compute_eer,
    compute_metrics,
    extract_pair_features_from_pairs,
    plot_confusion_matrix,
    plot_roc_curve,
    run_ablation,
    run_evaluation,
)
from sigverify.classifier import train_model, save_model
from sigverify.dataset import create_dummy_dataset, load_cedar_dataset, create_pairs


def test_compute_eer_perfect():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_scores = np.array([0.9, 0.85, 0.95, 0.1, 0.15, 0.05])
    eer = compute_eer(y_true, y_scores)
    assert np.isclose(eer, 0.0)


def test_compute_eer_random():
    rng = np.random.RandomState(42)
    y_true = np.array([1] * 50 + [0] * 50)
    y_scores = rng.rand(100)
    eer = compute_eer(y_true, y_scores)
    assert 0.3 <= eer <= 0.7


def test_metrics_keys():
    y_true = np.array([1, 1, 0, 0])
    y_scores = np.array([0.8, 0.7, 0.3, 0.2])
    y_pred = np.array([1, 1, 0, 0])
    
    m = compute_metrics(y_true, y_scores, y_pred)
    for key in ["accuracy", "precision", "recall", "f1_score", "auc", "eer"]:
        assert key in m
        assert 0.0 <= m[key] <= 1.0


def test_plot_roc_creates_file(tmp_path):
    y_true = np.array([1, 1, 0, 0])
    y_scores = np.array([0.8, 0.7, 0.3, 0.2])
    out_file = tmp_path / "roc.png"

    plot_roc_curve(y_true, y_scores, out_file)
    assert out_file.is_file()
    assert out_file.stat().st_size > 0


def test_plot_confusion_creates_file(tmp_path):
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    out_file = tmp_path / "cm.png"

    plot_confusion_matrix(y_true, y_pred, out_file)
    assert out_file.is_file()
    assert out_file.stat().st_size > 0


def test_run_evaluation_and_ablation_on_dummy_data(tmp_path):
    cedar_dir = create_dummy_dataset(tmp_path, n_writers=4, n_genuine=3, n_forged=3)
    
    # Train dummy model on actual pairs extracted from dummy dataset
    model_path = tmp_path / "model.pkl"
    writers = load_cedar_dataset(cedar_dir)
    train_pairs, _ = create_pairs(writers, protocol="writer-independent")
    
    X_tr, y_tr = extract_pair_features_from_pairs(train_pairs, groups=["hu", "grid"])
    artifact = train_model(X_tr, y_tr, feature_config={"groups": ["hu", "grid"]})
    save_model(artifact, str(model_path))

    results_dir = tmp_path / "results"
    metrics = run_evaluation(model_path, cedar_dir, protocol="writer-independent", out_dir=results_dir)

    assert (results_dir / "metrics.json").is_file()
    assert (results_dir / "roc_curve.png").is_file()
    assert (results_dir / "confusion_matrix.png").is_file()
    assert (results_dir / "per_writer_accuracy.csv").is_file()

    # Test ablation run
    df_ablation = run_ablation(cedar_dir, protocol="writer-independent", out_dir=results_dir)
    assert (results_dir / "ablation.csv").is_file()
    assert len(df_ablation) == 5
