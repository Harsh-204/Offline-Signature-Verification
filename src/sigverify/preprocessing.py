"""Preprocessing pipeline for signature images.

Steps: load → grayscale → binarize → denoise → crop → resize/normalize.
Optionally: skeletonize for structural features.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize as _skeletonize


# ---------------------------------------------------------------------------
# Individual stages
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Read an image from *path* and return it as a grayscale uint8 array."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def binarize(img: np.ndarray) -> np.ndarray:
    """Otsu binarization.  Ink → 0, background → 255."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure ink is dark (0).  If the image has more white than black already
    # Otsu usually gets this right, but guard against inverted scans.
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    return binary


def remove_noise(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Morphological opening to remove small speckle noise."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def crop_to_ink(img: np.ndarray, pad: int = 10) -> np.ndarray:
    """Tight-crop around ink pixels (value 0) with *pad* pixel margin."""
    # Ink is 0 in our convention
    ink_mask = img < 128
    if not ink_mask.any():
        return img  # blank image — return as-is

    rows = np.where(ink_mask.any(axis=1))[0]
    cols = np.where(ink_mask.any(axis=0))[0]

    r_min = max(rows[0] - pad, 0)
    r_max = min(rows[-1] + pad + 1, img.shape[0])
    c_min = max(cols[0] - pad, 0)
    c_max = min(cols[-1] + pad + 1, img.shape[1])

    return img[r_min:r_max, c_min:c_max]


def resize_normalize(
    img: np.ndarray,
    size: tuple[int, int] = (150, 220),
) -> np.ndarray:
    """Resize preserving aspect ratio, pad to *size*, normalize to [0, 1].

    Parameters
    ----------
    size : (height, width)
    """
    target_h, target_w = size
    h, w = img.shape[:2]

    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to target canvas (centre the signature)
    canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

    return canvas.astype(np.float64) / 255.0


def skeletonize_img(img: np.ndarray) -> np.ndarray:
    """Return the morphological skeleton of a binarized image.

    Expects ink = 0, bg = 255 (our convention).  Returns a boolean array
    where ``True`` marks skeleton pixels.
    """
    # skimage expects True = foreground
    foreground = img < 128
    return _skeletonize(foreground)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def preprocess(path: str, size: tuple[int, int] = (150, 220)) -> np.ndarray:
    """Full preprocessing pipeline: load → binarize → denoise → crop → resize.

    Returns a float64 array of shape *size* with values in [0, 1].
    """
    img = load_image(path)
    img = binarize(img)
    img = remove_noise(img)
    img = crop_to_ink(img)
    img = resize_normalize(img, size=size)
    return img
