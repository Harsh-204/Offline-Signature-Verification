"""Unit tests for pair_features.py."""

import numpy as np
import pytest

from sigverify.pair_features import compute_pair_features


def test_identical_pair():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    pair_feat = compute_pair_features(v, v)
    
    dim = len(v)
    abs_diff = pair_feat[:dim]
    product = pair_feat[dim:2*dim]
    cos_sim = pair_feat[-2]
    euc_dist = pair_feat[-1]

    assert np.allclose(abs_diff, 0.0)
    assert np.allclose(product, v ** 2)
    assert np.isclose(cos_sim, 1.0)
    assert np.isclose(euc_dist, 0.0)


def test_different_pair():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    pair_feat = compute_pair_features(v1, v2)

    cos_sim = pair_feat[-2]
    euc_dist = pair_feat[-1]

    assert np.isclose(cos_sim, 0.0)
    assert np.isclose(euc_dist, np.sqrt(2.0))


def test_output_length():
    dim = 50
    v1 = np.ones(dim)
    v2 = np.ones(dim) * 2.0
    pair_feat = compute_pair_features(v1, v2)

    assert pair_feat.shape == (2 * dim + 2,)


def test_symmetry():
    v1 = np.array([1.0, 2.5, 3.1])
    v2 = np.array([4.2, 0.5, 1.1])

    pair1 = compute_pair_features(v1, v2)
    pair2 = compute_pair_features(v2, v1)

    # Cosine similarity and Euclidean distance scalars should be symmetric
    assert np.isclose(pair1[-2], pair2[-2])
    assert np.isclose(pair1[-1], pair2[-1])
