"""Unit tests for preprocessing.py."""

import cv2
import numpy as np
import pytest

from sigverify.preprocessing import (
    binarize,
    crop_to_ink,
    load_image,
    preprocess,
    remove_noise,
    resize_normalize,
    skeletonize_img,
)


def test_load_image(tmp_path):
    img_path = tmp_path / "test.png"
    arr = np.zeros((100, 100), dtype=np.uint8)
    cv2.imwrite(str(img_path), arr)

    loaded = load_image(str(img_path))
    assert isinstance(loaded, np.ndarray)
    assert loaded.dtype == np.uint8
    assert loaded.shape == (100, 100)


def test_load_image_nonexistent():
    with pytest.raises(FileNotFoundError):
        load_image("nonexistent_image_xyz_123.png")


def test_binarize(dummy_signature_img):
    binary = binarize(dummy_signature_img)
    assert isinstance(binary, np.ndarray)
    unique_vals = set(np.unique(binary))
    assert unique_vals.issubset({0, 255})


def test_remove_noise(dummy_signature_img):
    denoised = remove_noise(dummy_signature_img)
    assert denoised.shape == dummy_signature_img.shape


def test_crop_to_ink(dummy_signature_img):
    cropped = crop_to_ink(dummy_signature_img, pad=5)
    assert cropped.shape[0] <= dummy_signature_img.shape[0]
    assert cropped.shape[1] <= dummy_signature_img.shape[1]


def test_resize_normalize(dummy_signature_img):
    target_size = (150, 220)
    norm = resize_normalize(dummy_signature_img, size=target_size)
    assert norm.shape == target_size
    assert norm.dtype == np.float64
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0


def test_skeletonize_img(dummy_signature_img):
    binary = binarize(dummy_signature_img)
    skel = skeletonize_img(binary)
    assert skel.dtype == bool
    assert skel.shape == binary.shape
    # Skeleton pixels should be significantly fewer than total ink pixels
    assert np.sum(skel) < np.sum(binary == 0)


def test_preprocess_e2e(tmp_path, dummy_signature_img):
    img_path = tmp_path / "sig.png"
    cv2.imwrite(str(img_path), dummy_signature_img)

    result = preprocess(str(img_path), size=(150, 220))
    assert result.shape == (150, 220)
    assert result.dtype == np.float64
    assert 0.0 <= result.min() <= result.max() <= 1.0
