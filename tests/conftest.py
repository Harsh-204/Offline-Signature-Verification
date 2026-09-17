"""Shared pytest fixtures for sigverify tests.

Provides:
- dummy_signature_img: a synthetic ink-on-white image
- dummy_forged_img: a visually different synthetic signature
- tmp_data_dir: a temporary CEDAR-structured folder tree with N writers
"""

import os
import pathlib

import cv2
import numpy as np
import pytest


def _draw_random_signature(height: int = 300, width: int = 500, seed: int = 0) -> np.ndarray:
    """Generate a synthetic signature image (black ink strokes on white)."""
    rng = np.random.RandomState(seed)
    canvas = np.full((height, width), 255, dtype=np.uint8)

    # Draw 4-8 random polylines to simulate pen strokes
    n_strokes = rng.randint(4, 9)
    for _ in range(n_strokes):
        n_points = rng.randint(3, 8)
        pts = np.column_stack([
            rng.randint(int(width * 0.1), int(width * 0.9), size=n_points),
            rng.randint(int(height * 0.15), int(height * 0.85), size=n_points),
        ]).astype(np.int32)
        thickness = rng.randint(1, 4)
        cv2.polylines(canvas, [pts], isClosed=False, color=0, thickness=thickness)

    return canvas


@pytest.fixture
def dummy_signature_img() -> np.ndarray:
    """A synthetic genuine-like signature (deterministic, seed=42)."""
    return _draw_random_signature(seed=42)


@pytest.fixture
def dummy_forged_img() -> np.ndarray:
    """A visually different synthetic signature (deterministic, seed=99)."""
    return _draw_random_signature(seed=99)


@pytest.fixture
def tmp_data_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a mini CEDAR-structured dataset with 5 writers, 4 genuine + 4 forged each.

    Returns the path to the ``cedar/`` directory.
    """
    cedar_dir = tmp_path / "cedar"
    org_dir = cedar_dir / "full_org"
    forg_dir = cedar_dir / "full_forg"
    org_dir.mkdir(parents=True)
    forg_dir.mkdir(parents=True)

    n_writers = 5
    n_genuine = 4
    n_forged = 4

    for writer in range(1, n_writers + 1):
        for sample in range(1, n_genuine + 1):
            img = _draw_random_signature(seed=writer * 100 + sample)
            fname = f"original_{writer}_{sample}.png"
            cv2.imwrite(str(org_dir / fname), img)

        for sample in range(1, n_forged + 1):
            img = _draw_random_signature(seed=writer * 200 + sample)
            fname = f"forgeries_{writer}_{sample}.png"
            cv2.imwrite(str(forg_dir / fname), img)

    return cedar_dir
