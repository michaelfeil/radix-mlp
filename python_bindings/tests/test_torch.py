"""
Tests for PyTorch interface of radix_mlp.

These tests focus on PyTorch-specific functionality:
- Device handling (CPU/GPU)
- Dtype conversion (int32/int64)
- Tensor conversion
- Basic functionality
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from radix_mlp.torch import compute_fold_and_scatter_torch


def test_torch_basic_import():
    """Test that the torch module can be imported."""
    assert compute_fold_and_scatter_torch is not None


def test_torch_basic_functionality():
    """Test basic functionality with torch tensors."""
    input_ids = torch.tensor([1, 2, 3, 1, 2, 4], dtype=torch.int32)
    position_ids = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32)
    cu_seq_lengths = torch.tensor([0, 3, 6], dtype=torch.int32)

    compact_ids, compact_pos, scatter, fold = compute_fold_and_scatter_torch(
        input_ids, position_ids, cu_seq_lengths
    )

    assert isinstance(compact_ids, torch.Tensor)
    assert isinstance(compact_pos, torch.Tensor)
    assert isinstance(scatter, torch.Tensor)
    assert isinstance(fold, torch.Tensor)


def test_torch_int32_dtype():
    """Test that int32 tensors are accepted."""
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    position_ids = torch.tensor([0, 1, 2], dtype=torch.int32)
    cu_seq_lengths = torch.tensor([0, 3], dtype=torch.int32)

    compact_ids, _, _, _ = compute_fold_and_scatter_torch(input_ids, position_ids, cu_seq_lengths)

    assert compact_ids.dtype == torch.int32


def test_torch_int64_dtype():
    """Test that int64 tensors are accepted and converted."""
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int64)
    position_ids = torch.tensor([0, 1, 2], dtype=torch.int64)
    cu_seq_lengths = torch.tensor([0, 3], dtype=torch.int64)

    compact_ids, _, _, _ = compute_fold_and_scatter_torch(input_ids, position_ids, cu_seq_lengths)

    assert compact_ids.dtype == torch.int32


def test_torch_cpu_device():
    """Test that CPU tensors work correctly."""
    input_ids = torch.tensor([1, 2, 3, 1, 2, 4], dtype=torch.int32)
    position_ids = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32)
    cu_seq_lengths = torch.tensor([0, 3, 6], dtype=torch.int32)

    compact_ids, compact_pos, scatter, fold = compute_fold_and_scatter_torch(
        input_ids, position_ids, cu_seq_lengths
    )

    assert compact_ids.device == torch.device("cpu")
    assert compact_pos.device == torch.device("cpu")
    assert scatter.device == torch.device("cpu")
    assert fold.device == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_torch_gpu_device():
    """Test that GPU tensors work correctly."""
    device = torch.device("cuda")

    input_ids = torch.tensor([1, 2, 3, 1, 2, 4], dtype=torch.int32, device=device)
    position_ids = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32, device=device)
    cu_seq_lengths = torch.tensor([0, 3, 6], dtype=torch.int32, device=device)

    compact_ids, compact_pos, scatter, fold = compute_fold_and_scatter_torch(
        input_ids, position_ids, cu_seq_lengths
    )

    assert compact_ids.device.type == device.type
    assert compact_pos.device.type == device.type
    assert scatter.device.type == device.type
    assert fold.device.type == device.type


def test_torch_device_mismatch_error():
    """Test that device mismatch raises ValueError."""
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    position_ids = torch.tensor([0, 1, 2], dtype=torch.int32, device="cpu")
    cu_seq_lengths = torch.tensor([0, 3], dtype=torch.int32, device="cpu")

    # This should work (all on CPU)
    compute_fold_and_scatter_torch(input_ids, position_ids, cu_seq_lengths)

    # This should fail (different devices)
    if torch.cuda.is_available():
        input_ids_gpu = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")
        position_ids_cpu = torch.tensor([0, 1, 2], dtype=torch.int32, device="cpu")
        cu_seq_lengths_cpu = torch.tensor([0, 3], dtype=torch.int32, device="cpu")

        with pytest.raises(ValueError, match="same device"):
            compute_fold_and_scatter_torch(input_ids_gpu, position_ids_cpu, cu_seq_lengths_cpu)


def test_torch_wrong_type_error():
    """Test that non-tensor inputs raise TypeError."""
    input_ids = np.array([1, 2, 3], dtype=np.uint32)  # numpy array, not tensor
    position_ids = torch.tensor([0, 1, 2], dtype=torch.int32)
    cu_seq_lengths = torch.tensor([0, 3], dtype=torch.int32)

    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        compute_fold_and_scatter_torch(input_ids, position_ids, cu_seq_lengths)


def test_torch_wrong_shape_error():
    """Test that wrong shape raises ValueError."""
    input_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)  # 2D tensor
    position_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    cu_seq_lengths = torch.tensor([0, 4], dtype=torch.int32)

    with pytest.raises(ValueError, match="must be 1-dimensional"):
        compute_fold_and_scatter_torch(input_ids, position_ids, cu_seq_lengths)


def test_torch_length_mismatch_error():
    """Test that length mismatch raises ValueError."""
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    position_ids = torch.tensor([0, 1], dtype=torch.int32)  # Wrong length
    cu_seq_lengths = torch.tensor([0, 3], dtype=torch.int32)

    with pytest.raises(ValueError, match="must have same length"):
        compute_fold_and_scatter_torch(input_ids, position_ids, cu_seq_lengths)


def test_torch_empty_input():
    """Test empty input handling."""
    input_ids = torch.tensor([], dtype=torch.int32)
    position_ids = torch.tensor([], dtype=torch.int32)
    cu_seq_lengths = torch.tensor([], dtype=torch.int32)

    compact_ids, compact_pos, scatter, fold = compute_fold_and_scatter_torch(
        input_ids, position_ids, cu_seq_lengths
    )

    assert len(compact_ids) == 0
    assert len(compact_pos) == 0
    assert len(scatter) == 0
    assert len(fold) == 0


def test_torch_single_sequence():
    """Test single sequence (identity case)."""
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    position_ids = torch.tensor([0, 1, 2], dtype=torch.int32)
    cu_seq_lengths = torch.tensor([0, 3], dtype=torch.int32)

    compact_ids, compact_pos, scatter, fold = compute_fold_and_scatter_torch(
        input_ids, position_ids, cu_seq_lengths
    )

    torch.testing.assert_close(compact_ids, input_ids)
    torch.testing.assert_close(compact_pos, position_ids)
    torch.testing.assert_close(scatter, torch.tensor([0, 1, 2], dtype=torch.int32))
    torch.testing.assert_close(fold, torch.tensor([0, 1, 2], dtype=torch.int32))


def test_torch_identical_sequences():
    """Test two identical sequences (should deduplicate)."""
    input_ids = torch.tensor([1, 2, 3, 1, 2, 3], dtype=torch.int32)
    position_ids = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32)
    cu_seq_lengths = torch.tensor([0, 3, 6], dtype=torch.int32)

    compact_ids, compact_pos, scatter, fold = compute_fold_and_scatter_torch(
        input_ids, position_ids, cu_seq_lengths
    )

    assert len(compact_ids) == 3  # Should deduplicate to single sequence
    torch.testing.assert_close(compact_ids, torch.tensor([1, 2, 3], dtype=torch.int32))
    torch.testing.assert_close(scatter, torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32))


def test_torch_matches_numpy():
    """Test that torch interface produces same results as numpy interface."""
    from radix_mlp import compute_fold_and_scatter

    input_ids_np = np.array([1, 2, 3, 1, 2, 4], dtype=np.uint32)
    position_ids_np = np.array([0, 1, 2, 0, 1, 2], dtype=np.uint32)
    cu_seq_lengths_np = np.array([0, 3, 6], dtype=np.uint32)

    input_ids_torch = torch.tensor([1, 2, 3, 1, 2, 4], dtype=torch.int32)
    position_ids_torch = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32)
    cu_seq_lengths_torch = torch.tensor([0, 3, 6], dtype=torch.int32)

    # Numpy version
    (
        compact_ids_np,
        compact_pos_np,
        scatter_np,
        fold_np,
    ) = compute_fold_and_scatter(input_ids_np, position_ids_np, cu_seq_lengths_np)

    # Torch version
    (
        compact_ids_torch,
        compact_pos_torch,
        scatter_torch,
        fold_torch,
    ) = compute_fold_and_scatter_torch(input_ids_torch, position_ids_torch, cu_seq_lengths_torch)

    # Compare results (convert numpy to int32 for comparison)
    torch.testing.assert_close(compact_ids_torch, torch.from_numpy(compact_ids_np).to(torch.int32))
    torch.testing.assert_close(compact_pos_torch, torch.from_numpy(compact_pos_np).to(torch.int32))
    torch.testing.assert_close(scatter_torch, torch.from_numpy(scatter_np).to(torch.int32))
    torch.testing.assert_close(fold_torch, torch.from_numpy(fold_np).to(torch.int32))


def test_torch_with_padding():
    """Test with padding enabled."""
    input_ids = torch.tensor([1, 2, 3, 1, 2, 4], dtype=torch.int32)
    position_ids = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32)
    cu_seq_lengths = torch.tensor([0, 3, 6], dtype=torch.int32)

    compact_ids, _, _, _ = compute_fold_and_scatter_torch(
        input_ids, position_ids, cu_seq_lengths, pad_multiple_of=True
    )

    # Should be padded to multiple of 8
    assert len(compact_ids) % 8 == 0 or len(compact_ids) == 0
