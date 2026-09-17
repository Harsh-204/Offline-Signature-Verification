"""Feature extraction for signature images.

Extracts four complementary feature groups from a preprocessed (float64,
normalized) signature image and concatenates them into a single vector:

1. **Hu moments** (7-d) — rotation/scale-invariant shape descriptors
2. **HOG descriptor** — gradient orientation histograms (texture + edge)
3. **LBP histogram** — local binary pattern texture descriptor
4. **Pixel-density grid** — ink distribution across spatial cells
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from skimage.feature import hog as _hog
from skimage.feature import local_binary_pattern as _lbp


# ---------------------------------------------------------------------------
# Individual feature extractors
# ---------------------------------------------------------------------------

def hu_moments(img: np.ndarray) -> np.ndarray:
    """Return 7 log-transformed Hu moments from a [0,1] float image."""
    img_u8 = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
    moments = cv2.moments(img_u8)
    hu = cv2.HuMoments(moments).flatten()  # shape (7,)
    hu_log = np.zeros_like(hu)
    for i in range(len(hu)):
        abs_h = abs(hu[i])
        if abs_h > 1e-12:
            hu_log[i] = -np.sign(hu[i]) * np.log10(abs_h)
    return hu_log


def hog_features(
    img: np.ndarray,
    pixels_per_cell: tuple[int, int] = (16, 16),
    cells_per_block: tuple[int, int] = (2, 2),
    orientations: int = 9,
) -> np.ndarray:
    """Histogram of Oriented Gradients descriptor."""
    # skimage hog expects float in [0,1] — our preprocessed images already are
    descriptor = _hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return descriptor


def lbp_histogram(
    img: np.ndarray,
    radius: int = 3,
    n_points: int = 24,
    n_bins: int = 26,
) -> np.ndarray:
    """Uniform Local Binary Pattern histogram (normalized)."""
    img_u8 = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
    lbp_map = _lbp(img_u8, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp_map, bins=n_bins, range=(0, n_points + 2), density=True)
    return hist


def pixel_density_grid(img: np.ndarray, grid: tuple[int, int] = (5, 5)) -> np.ndarray:
    """Divide image into *grid* cells and compute ink-pixel fraction per cell.

    Ink is defined as pixels < 0.5 (dark on white background, [0,1] float).
    """
    rows, cols = grid
    h, w = img.shape[:2]
    cell_h, cell_w = h // rows, w // cols

    densities = []
    for r in range(rows):
        for c in range(cols):
            cell = img[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            ink_fraction = np.mean(cell < 0.5)
            densities.append(ink_fraction)
    return np.array(densities, dtype=np.float64)


# ---------------------------------------------------------------------------
# Registry of feature groups (used for ablation)
# ---------------------------------------------------------------------------

_FEATURE_GROUPS: dict[str, callable] = {
    "hu": hu_moments,
    "hog": hog_features,
    "lbp": lbp_histogram,
    "grid": pixel_density_grid,
}


def extract_features(
    img: np.ndarray,
    groups: Sequence[str] | None = None,
) -> np.ndarray:
    """Concatenate selected feature groups into a single 1-D vector.

    Parameters
    ----------
    img : preprocessed float64 image in [0, 1]
    groups : subset of ``{"hu", "hog", "lbp", "grid"}`` or ``None`` for all
    """
    if groups is None:
        groups = list(_FEATURE_GROUPS.keys())

    parts = []
    for name in groups:
        fn = _FEATURE_GROUPS[name]
        parts.append(fn(img))
    return np.concatenate(parts)
