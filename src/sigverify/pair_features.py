"""Pair-wise feature computation for signature verification.

Verification is a *pair-comparison* task — the classifier should see
a representation of the **relationship** between a reference and a query
feature vector, not raw individual vectors.

The pair feature vector concatenates:
    1. Element-wise absolute difference  ``|ref - query|``
    2. Element-wise product              ``ref * query``
    3. Cosine similarity (scalar)
    4. Euclidean distance  (scalar)
"""

from __future__ import annotations

import numpy as np


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (clamped to [-1, 1])."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def compute_pair_features(
    feat_ref: np.ndarray,
    feat_query: np.ndarray,
) -> np.ndarray:
    """Compute a pair-wise feature vector from two individual feature vectors.

    Returns a 1-D array of length ``2 * len(feat_ref) + 2``.
    """
    abs_diff = np.abs(feat_ref - feat_query)
    product = feat_ref * feat_query
    cos_sim = _cosine_similarity(feat_ref, feat_query)
    euc_dist = float(np.linalg.norm(feat_ref - feat_query))
    return np.concatenate([abs_diff, product, [cos_sim, euc_dist]])
