"""Dataset loading, pair generation, and dummy-data creation.

Handles the CEDAR signature dataset layout and provides train/test pair
generation under both *writer-dependent* and *writer-independent* protocols.
"""

from __future__ import annotations

import os
import pathlib
import re
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class WriterData:
    """All image paths for a single writer."""
    writer_id: int
    genuine_paths: list[str] = field(default_factory=list)
    forged_paths: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

_ORG_RE = re.compile(r"original_(\d+)_(\d+)\.png$")
_FORG_RE = re.compile(r"forgeries_(\d+)_(\d+)\.png$")


def load_cedar_dataset(data_dir: str | pathlib.Path) -> list[WriterData]:
    """Parse a CEDAR-structured directory into a list of :class:`WriterData`.

    Expects::

        data_dir/
            full_org/   original_{writer}_{sample}.png
            full_forg/  forgeries_{writer}_{sample}.png
    """
    data_dir = pathlib.Path(data_dir)
    org_dir = data_dir / "full_org"
    forg_dir = data_dir / "full_forg"

    if not org_dir.is_dir() or not forg_dir.is_dir():
        raise FileNotFoundError(
            f"Expected full_org/ and full_forg/ inside {data_dir}"
        )

    writers: dict[int, WriterData] = {}

    for fname in sorted(os.listdir(org_dir)):
        m = _ORG_RE.match(fname)
        if not m:
            continue
        wid = int(m.group(1))
        writers.setdefault(wid, WriterData(writer_id=wid))
        writers[wid].genuine_paths.append(str(org_dir / fname))

    for fname in sorted(os.listdir(forg_dir)):
        m = _FORG_RE.match(fname)
        if not m:
            continue
        wid = int(m.group(1))
        writers.setdefault(wid, WriterData(writer_id=wid))
        writers[wid].forged_paths.append(str(org_dir.parent / "full_forg" / fname))

    return sorted(writers.values(), key=lambda w: w.writer_id)


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

@dataclass
class Pair:
    """A labelled pair of image paths."""
    ref_path: str
    query_path: str
    label: int  # 1 = genuine match, 0 = forgery
    writer_id: int


def _generate_pairs_for_writer(
    wd: WriterData,
    genuine_indices: Sequence[int],
    forged_indices: Sequence[int],
    ref_indices: Sequence[int],
) -> list[Pair]:
    """Generate genuine-genuine and genuine-forged pairs for one writer.

    ``ref_indices`` — indices into ``genuine_paths`` used as references.
    ``genuine_indices`` — indices into ``genuine_paths`` used as query (genuine match).
    ``forged_indices`` — indices into ``forged_paths`` used as query (forgery).
    """
    pairs: list[Pair] = []
    for ri in ref_indices:
        ref = wd.genuine_paths[ri]
        for qi in genuine_indices:
            if qi == ri:
                continue  # don't pair an image with itself
            pairs.append(Pair(ref, wd.genuine_paths[qi], label=1, writer_id=wd.writer_id))
        for fi in forged_indices:
            pairs.append(Pair(ref, wd.forged_paths[fi], label=0, writer_id=wd.writer_id))
    return pairs


def create_pairs(
    writers: list[WriterData],
    protocol: str = "writer-independent",
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[Pair], list[Pair]]:
    """Create train/test pair sets under the specified *protocol*.

    Parameters
    ----------
    protocol : ``"writer-dependent"`` or ``"writer-independent"``
    test_ratio : fraction of writers (writer-independent) or samples
        (writer-dependent) held out for testing.
    seed : random seed for reproducibility.
    """
    rng = np.random.RandomState(seed)

    if protocol == "writer-independent":
        n_test = max(1, int(len(writers) * test_ratio))
        indices = rng.permutation(len(writers))
        test_idx = set(indices[:n_test])

        train_writers = [w for i, w in enumerate(writers) if i not in test_idx]
        test_writers = [w for i, w in enumerate(writers) if i in test_idx]

        train_pairs: list[Pair] = []
        for wd in train_writers:
            n_g = len(wd.genuine_paths)
            n_f = len(wd.forged_paths)
            all_g = list(range(n_g))
            all_f = list(range(n_f))
            # Use first half as references, second half as queries
            mid_g = max(1, n_g // 2)
            refs = all_g[:mid_g]
            queries_g = all_g[mid_g:]
            train_pairs.extend(_generate_pairs_for_writer(wd, queries_g, all_f, refs))

        test_pairs: list[Pair] = []
        for wd in test_writers:
            n_g = len(wd.genuine_paths)
            n_f = len(wd.forged_paths)
            all_g = list(range(n_g))
            all_f = list(range(n_f))
            mid_g = max(1, n_g // 2)
            refs = all_g[:mid_g]
            queries_g = all_g[mid_g:]
            test_pairs.extend(_generate_pairs_for_writer(wd, queries_g, all_f, refs))

        return train_pairs, test_pairs

    elif protocol == "writer-dependent":
        train_pairs = []
        test_pairs = []
        for wd in writers:
            n_g = len(wd.genuine_paths)
            n_f = len(wd.forged_paths)
            g_idx = rng.permutation(n_g)
            f_idx = rng.permutation(n_f)

            split_g = max(1, int(n_g * (1 - test_ratio)))
            split_f = max(1, int(n_f * (1 - test_ratio)))

            train_g = g_idx[:split_g].tolist()
            test_g = g_idx[split_g:].tolist()
            train_f = f_idx[:split_f].tolist()
            test_f = f_idx[split_f:].tolist()

            # References always come from train genuine
            refs = train_g[:max(1, len(train_g) // 2)]
            train_pairs.extend(
                _generate_pairs_for_writer(wd, train_g, train_f, refs)
            )
            test_pairs.extend(
                _generate_pairs_for_writer(wd, test_g, test_f, refs)
            )

        return train_pairs, test_pairs

    else:
        raise ValueError(f"Unknown protocol: {protocol!r}")


# ---------------------------------------------------------------------------
# Dummy / synthetic dataset generation
# ---------------------------------------------------------------------------

def _draw_random_signature(
    height: int = 300,
    width: int = 500,
    seed: int = 0,
) -> np.ndarray:
    """Generate a synthetic signature image (black ink strokes on white)."""
    rng = np.random.RandomState(seed)
    canvas = np.full((height, width), 255, dtype=np.uint8)
    n_strokes = rng.randint(4, 9)
    for _ in range(n_strokes):
        n_pts = rng.randint(3, 8)
        pts = np.column_stack([
            rng.randint(int(width * 0.1), int(width * 0.9), size=n_pts),
            rng.randint(int(height * 0.15), int(height * 0.85), size=n_pts),
        ]).astype(np.int32)
        thickness = rng.randint(1, 4)
        cv2.polylines(canvas, [pts], isClosed=False, color=0, thickness=thickness)
    return canvas


def create_dummy_dataset(
    base_dir: str | pathlib.Path,
    n_writers: int = 5,
    n_genuine: int = 4,
    n_forged: int = 4,
) -> pathlib.Path:
    """Create a synthetic CEDAR-structured dataset for testing.

    Returns the path to the ``cedar/`` directory.
    """
    base_dir = pathlib.Path(base_dir)
    cedar_dir = base_dir / "cedar"
    org_dir = cedar_dir / "full_org"
    forg_dir = cedar_dir / "full_forg"
    org_dir.mkdir(parents=True, exist_ok=True)
    forg_dir.mkdir(parents=True, exist_ok=True)

    for writer in range(1, n_writers + 1):
        for sample in range(1, n_genuine + 1):
            img = _draw_random_signature(seed=writer * 100 + sample)
            cv2.imwrite(str(org_dir / f"original_{writer}_{sample}.png"), img)
        for sample in range(1, n_forged + 1):
            img = _draw_random_signature(seed=writer * 200 + sample)
            cv2.imwrite(str(forg_dir / f"forgeries_{writer}_{sample}.png"), img)

    return cedar_dir
