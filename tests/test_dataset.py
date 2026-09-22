"""Unit tests for dataset.py."""

import pathlib
import pytest

from sigverify.dataset import (
    create_dummy_dataset,
    create_pairs,
    load_cedar_dataset,
)


def test_create_dummy_dataset(tmp_path):
    cedar_dir = create_dummy_dataset(tmp_path, n_writers=3, n_genuine=2, n_forged=2)
    assert (cedar_dir / "full_org").is_dir()
    assert (cedar_dir / "full_forg").is_dir()

    org_files = list((cedar_dir / "full_org").glob("original_*.png"))
    forg_files = list((cedar_dir / "full_forg").glob("forgeries_*.png"))

    assert len(org_files) == 3 * 2
    assert len(forg_files) == 3 * 2


def test_load_cedar_dataset(tmp_data_dir):
    writers = load_cedar_dataset(tmp_data_dir)
    assert len(writers) == 5
    for wd in writers:
        assert len(wd.genuine_paths) == 4
        assert len(wd.forged_paths) == 4


def test_writer_independent_split(tmp_data_dir):
    writers = load_cedar_dataset(tmp_data_dir)
    train_pairs, test_pairs = create_pairs(writers, protocol="writer-independent", test_ratio=0.4)

    assert len(train_pairs) > 0
    assert len(test_pairs) > 0

    train_writers = set(p.writer_id for p in train_pairs)
    test_writers = set(p.writer_id for p in test_pairs)

    # Disjoint writer sets
    assert train_writers.isdisjoint(test_writers)


def test_writer_dependent_split(tmp_data_dir):
    writers = load_cedar_dataset(tmp_data_dir)
    train_pairs, test_pairs = create_pairs(writers, protocol="writer-dependent", test_ratio=0.5)

    assert len(train_pairs) > 0
    assert len(test_pairs) > 0

    train_writers = set(p.writer_id for p in train_pairs)
    test_writers = set(p.writer_id for p in test_pairs)

    # In writer-dependent split, writers overlap
    assert len(train_writers.intersection(test_writers)) > 0


def test_pair_labels_balanced(tmp_data_dir):
    writers = load_cedar_dataset(tmp_data_dir)
    train_pairs, _ = create_pairs(writers, protocol="writer-independent")

    genuine_count = sum(1 for p in train_pairs if p.label == 1)
    forged_count = sum(1 for p in train_pairs if p.label == 0)

    assert genuine_count > 0
    assert forged_count > 0
