"""Evaluation metrics, plotting, per-writer reporting, and ablation study.

Computes ROC curve, AUC, Equal Error Rate (EER), confusion matrix,
per-writer accuracy, and generates feature-ablation tables.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless CPU execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)

from sigverify.classifier import load_model, predict_batch, train_model
from sigverify.dataset import Pair, create_pairs, load_cedar_dataset
from sigverify.feature_selection import fit_reducer, transform
from sigverify.features import extract_features
from sigverify.pair_features import compute_pair_features
from sigverify.preprocessing import preprocess


# ---------------------------------------------------------------------------
# Metrics & EER
# ---------------------------------------------------------------------------

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Equal Error Rate (EER) where FAR == FRR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer


def compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute standard metrics: accuracy, precision, recall, f1, auc, eer."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = float(auc(fpr, tpr))
    eer = compute_eer(y_true, y_scores)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auc": roc_auc,
        "eer": eer,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, out_path: str | pathlib.Path) -> None:
    """Generate and save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (1 - FRR)")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: str | pathlib.Path) -> None:
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Forged (0)", "Genuine (1)"]

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Feature extraction helper for pair lists
# ---------------------------------------------------------------------------

def extract_pair_features_from_pairs(
    pairs: list[Pair],
    groups: list[str] | None = None,
    reducer=None,
    feature_cache: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Helper to load, preprocess, extract features, apply reducer, and pair."""
    if feature_cache is None:
        feature_cache = {}

    if not pairs:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int32)

    # Determine pair feature dimension from first pair
    p0 = pairs[0]
    for path in (p0.ref_path, p0.query_path):
        if path not in feature_cache:
            img = preprocess(path)
            feature_cache[path] = extract_features(img, groups=groups)
    f0_ref = feature_cache[p0.ref_path]
    f0_q = feature_cache[p0.query_path]
    if reducer is not None:
        f0_ref = transform(f0_ref.reshape(1, -1), reducer).flatten()
        f0_q = transform(f0_q.reshape(1, -1), reducer).flatten()
    feat_dim = len(compute_pair_features(f0_ref, f0_q))

    X_mat = np.empty((len(pairs), feat_dim), dtype=np.float32)
    y_vec = np.empty(len(pairs), dtype=np.int32)

    for i, pair in enumerate(pairs):
        if pair.ref_path not in feature_cache:
            img_ref = preprocess(pair.ref_path)
            feature_cache[pair.ref_path] = extract_features(img_ref, groups=groups)
        if pair.query_path not in feature_cache:
            img_q = preprocess(pair.query_path)
            feature_cache[pair.query_path] = extract_features(img_q, groups=groups)

        f_ref = feature_cache[pair.ref_path]
        f_q = feature_cache[pair.query_path]

        if reducer is not None:
            f_ref = transform(f_ref.reshape(1, -1), reducer).flatten()
            f_q = transform(f_q.reshape(1, -1), reducer).flatten()

        X_mat[i] = compute_pair_features(f_ref, f_q)
        y_vec[i] = pair.label

    return X_mat, y_vec


# ---------------------------------------------------------------------------
# Full Evaluation Orchestration
# ---------------------------------------------------------------------------

def run_evaluation(
    model_path: str | pathlib.Path,
    data_dir: str | pathlib.Path,
    protocol: str = "writer-independent",
    out_dir: str | pathlib.Path = "results",
) -> dict[str, float]:
    """Load model, create test evaluation set, compute metrics and save artifacts."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact = load_model(str(model_path))
    reducer = artifact.get("reducer")
    groups = artifact.get("feature_config", {}).get("groups")

    writers = load_cedar_dataset(data_dir)
    _, test_pairs = create_pairs(writers, protocol=protocol)

    X_test, y_test = extract_pair_features_from_pairs(test_pairs, groups=groups, reducer=reducer)
    y_pred, y_scores = predict_batch(artifact, X_test)

    metrics = compute_metrics(y_test, y_scores, y_pred)
    metrics["protocol"] = protocol

    # Save metrics JSON
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save plots
    plot_roc_curve(y_test, y_scores, out_dir / "roc_curve.png")
    plot_confusion_matrix(y_test, y_pred, out_dir / "confusion_matrix.png")

    # Save per-writer accuracy breakdown CSV
    writer_ids = [p.writer_id for p in test_pairs]
    df_writer = pd.DataFrame({"writer_id": writer_ids, "correct": (y_test == y_pred).astype(int)})
    writer_acc = df_writer.groupby("writer_id")["correct"].mean().reset_index(name="accuracy")
    writer_acc.to_csv(out_dir / "per_writer_accuracy.csv", index=False)

    return metrics


# ---------------------------------------------------------------------------
# Feature Ablation Study
# ---------------------------------------------------------------------------

def run_ablation(
    data_dir: str | pathlib.Path,
    protocol: str = "writer-independent",
    out_dir: str | pathlib.Path = "results",
) -> pd.DataFrame:
    """Run ablation evaluation across Hu-only, HOG-only, LBP-only, Grid-only, and Combined."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    writers = load_cedar_dataset(data_dir)
    train_pairs, test_pairs = create_pairs(writers, protocol=protocol)

    ablation_sets = {
        "Hu-only": ["hu"],
        "HOG-only": ["hog"],
        "LBP-only": ["lbp"],
        "Grid-only": ["grid"],
        "Combined (All)": ["hu", "hog", "lbp", "grid"],
    }

    results = []

    for name, groups in ablation_sets.items():
        cache: dict[str, np.ndarray] = {}
        # Pre-reducer individual features
        X_tr_raw, y_tr = extract_pair_features_from_pairs(train_pairs, groups=groups, feature_cache=cache)
        X_te_raw, y_te = extract_pair_features_from_pairs(test_pairs, groups=groups, feature_cache=cache)

        # Train model
        model_art = train_model(X_tr_raw, y_tr, reducer=None, feature_config={"groups": groups})
        y_pred, y_scores = predict_batch(model_art, X_te_raw)

        m = compute_metrics(y_te, y_scores, y_pred)
        results.append({
            "Feature Group": name,
            "Accuracy": round(m["accuracy"], 4),
            "EER": round(m["eer"], 4),
            "AUC": round(m["auc"], 4),
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1-Score": round(m["f1_score"], 4),
        })

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "ablation.csv", index=False)
    return df
