"""
Basic Python tests for radix_mlp.

Most algorithmic correctness is tested in Rust. These tests focus on:
- Python API validation
- Input type/shape checking
- Basic functionality smoke test
"""

import numpy as np
import pytest

from radix_mlp import compute_fold_and_scatter


def test_basic_import():
    """Test that the module can be imported."""
    assert compute_fold_and_scatter is not None


def test_basic_functionality():
    """Test basic functionality with a simple example."""
    input_ids = np.array([1, 2, 3, 1, 2, 4], dtype=np.uint32)
    position_ids = np.array([0, 1, 2, 0, 1, 2], dtype=np.uint32)
    cu_seq_lengths = np.array([0, 3, 6], dtype=np.uint32)

    compact_ids, compact_pos, scatter, fold = compute_fold_and_scatter(
        input_ids, position_ids, cu_seq_lengths
    )

    assert isinstance(compact_ids, np.ndarray)
    assert isinstance(compact_pos, np.ndarray)
    assert isinstance(scatter, np.ndarray)
    assert isinstance(fold, np.ndarray)


def test_input_validation_wrong_dtype():
    """Test that wrong dtype raises TypeError."""
    input_ids = np.array([1, 2, 3], dtype=np.int32)  # Wrong dtype
    position_ids = np.array([0, 1, 2], dtype=np.uint32)
    cu_seq_lengths = np.array([0, 3], dtype=np.uint32)

    with pytest.raises(TypeError, match="input_ids must be uint32"):
        compute_fold_and_scatter(input_ids, position_ids, cu_seq_lengths)


def test_input_validation_wrong_shape():
    """Test that wrong shape raises ValueError."""
    input_ids = np.array([[1, 2], [3, 4]], dtype=np.uint32)  # 2D array
    position_ids = np.array([0, 1, 2, 3], dtype=np.uint32)
    cu_seq_lengths = np.array([0, 4], dtype=np.uint32)

    with pytest.raises(ValueError, match="input_ids must be 1-dimensional"):
        compute_fold_and_scatter(input_ids, position_ids, cu_seq_lengths)


def test_input_validation_length_mismatch():
    """Test that length mismatch raises ValueError."""
    input_ids = np.array([1, 2, 3], dtype=np.uint32)
    position_ids = np.array([0, 1], dtype=np.uint32)  # Wrong length
    cu_seq_lengths = np.array([0, 3], dtype=np.uint32)

    with pytest.raises(ValueError, match="input_ids and position_ids must have same length"):
        compute_fold_and_scatter(input_ids, position_ids, cu_seq_lengths)


def test_empty_input():
    """Test empty input handling."""
    input_ids = np.array([], dtype=np.uint32)
    position_ids = np.array([], dtype=np.uint32)
    cu_seq_lengths = np.array([], dtype=np.uint32)

    compact_ids, compact_pos, scatter, fold = compute_fold_and_scatter(
        input_ids, position_ids, cu_seq_lengths
    )

    assert len(compact_ids) == 0
    assert len(compact_pos) == 0
    assert len(scatter) == 0
    assert len(fold) == 0


def test_single_sequence():
    """Test single sequence (identity case)."""
    input_ids = np.array([1, 2, 3], dtype=np.uint32)
    position_ids = np.array([0, 1, 2], dtype=np.uint32)
    cu_seq_lengths = np.array([0, 3], dtype=np.uint32)

    compact_ids, compact_pos, scatter, fold = compute_fold_and_scatter(
        input_ids, position_ids, cu_seq_lengths
    )

    np.testing.assert_array_equal(compact_ids, input_ids)
    np.testing.assert_array_equal(compact_pos, position_ids)
    np.testing.assert_array_equal(scatter, np.array([0, 1, 2], dtype=np.uint32))
    np.testing.assert_array_equal(fold, np.array([0, 1, 2], dtype=np.uint32))


def test_identical_sequences():
    """Test two identical sequences (should deduplicate)."""
    input_ids = np.array([1, 2, 3, 1, 2, 3], dtype=np.uint32)
    position_ids = np.array([0, 1, 2, 0, 1, 2], dtype=np.uint32)
    cu_seq_lengths = np.array([0, 3, 6], dtype=np.uint32)

    compact_ids, compact_pos, scatter, fold = compute_fold_and_scatter(
        input_ids, position_ids, cu_seq_lengths
    )

    assert len(compact_ids) == 3  # Should deduplicate to single sequence
    np.testing.assert_array_equal(compact_ids, np.array([1, 2, 3], dtype=np.uint32))
    np.testing.assert_array_equal(scatter, np.array([0, 1, 2, 0, 1, 2], dtype=np.uint32))


def test_version_attribute():
    """Test that __version__ is accessible."""
    from radix_mlp import __version__

    assert __version__.count(".") == 2  # Simple check for semantic versioning format
