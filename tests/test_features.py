"""Unit tests for features.py."""

import numpy as np
import pytest

from sigverify.features import (
    extract_features,
    hog_features,
    hu_moments,
    lbp_histogram,
    pixel_density_grid,
)


@pytest.fixture
def sample_norm_image():
    # 150x220 canvas normalized to [0, 1]
    img = np.full((150, 220), 1.0, dtype=np.float64)
    # Add fake ink strokes (0.0)
    img[50:100, 60:160] = 0.0
    return img


def test_hu_moments_shape(sample_norm_image):
    hu = hu_moments(sample_norm_image)
    assert isinstance(hu, np.ndarray)
    assert hu.shape == (7,)
    assert np.all(np.isfinite(hu))


def test_hog_features_deterministic(sample_norm_image):
    hog1 = hog_features(sample_norm_image)
    hog2 = hog_features(sample_norm_image)
    assert isinstance(hog1, np.ndarray)
    assert hog1.ndim == 1
    assert len(hog1) > 0
    assert np.array_equal(hog1, hog2)


def test_lbp_histogram_shape(sample_norm_image):
    n_bins = 26
    lbp = lbp_histogram(sample_norm_image, n_bins=n_bins)
    assert isinstance(lbp, np.ndarray)
    assert lbp.shape == (n_bins,)
    assert np.isclose(np.sum(lbp), 1.0, atol=1e-2)


def test_pixel_density_grid(sample_norm_image):
    grid = (5, 5)
    densities = pixel_density_grid(sample_norm_image, grid=grid)
    assert isinstance(densities, np.ndarray)
    assert densities.shape == (25,)
    assert np.all((densities >= 0.0) & (densities <= 1.0))
    # Cell inside the ink rectangle should have higher density than empty cell
    assert densities.max() > densities.min()


def test_extract_features_combined(sample_norm_image):
    feat = extract_features(sample_norm_image)
    hu = hu_moments(sample_norm_image)
    hog = hog_features(sample_norm_image)
    lbp = lbp_histogram(sample_norm_image)
    grid = pixel_density_grid(sample_norm_image)
    
    expected_len = len(hu) + len(hog) + len(lbp) + len(grid)
    assert len(feat) == expected_len


def test_feature_groups_selectable(sample_norm_image):
    feat_hu_only = extract_features(sample_norm_image, groups=["hu"])
    assert len(feat_hu_only) == 7

    feat_subset = extract_features(sample_norm_image, groups=["hu", "grid"])
    assert len(feat_subset) == 7 + 25
