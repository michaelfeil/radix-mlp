"""
PyTorch interface for RadixMLP.

This module provides a convenient PyTorch wrapper around the numpy-based
RadixMLP implementation, handling device and dtype conversions automatically.
"""

from __future__ import annotations

from typing import Tuple

try:
    import torch
    import numpy as np
except ImportError as e:
    raise ImportError(
        "PyTorch+Numpy is required for the torch interface. Install it with: pip install radix-mlp torch numpy"
    ) from e

from radix_mlp import compute_fold_and_scatter as _compute_fold_and_scatter_numpy


def compute_fold_and_scatter_torch(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    cu_seq_lengths: torch.Tensor,
    pad_multiple_of: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute indices for RadixMLP-style folding and scattering using PyTorch tensors.

    This function identifies shared prefixes among sequences in a batch and produces
    a compact representation containing only unique subsequences. It handles device
    and dtype conversions automatically.

    Args:
        input_ids: Flattened tensor of token IDs for all sequences.
            Can be int32 or int64, on any device.
        position_ids: Flattened tensor of position IDs corresponding to each token.
            Can be int32 or int64, on any device.
        cu_seq_lengths: Cumulative sequence lengths, e.g., [0, len_seq1, len_seq1+len_seq2, ...].
            Can be int32 or int64, on any device.
        pad_multiple_of: If True, pad output to multiple of 8 (small) or 64 (large) for performance.

    Returns:
        A tuple of four PyTorch tensors on the same device as the inputs:
        - compact_input_ids: Unique token IDs (compacted representation)
        - compact_position_ids: Corresponding position IDs
        - scatter_indices: Index map to unfold from compact to original space
        - fold_gather: Index map to gather from original to compact space

    Note:
        - All input tensors must be on the same device.
        - Uses `tensor.numpy(force=True)` for conversion, which handles:
          - Detaching from computation graph
          - Moving to CPU if needed
          - Resolving conjugate/negative bits
        - For CPU tensors without grad, conversion is zero-copy (shares storage).
        - For GPU tensors or tensors with grad, a copy is made.
        - Input tensors are converted to uint32 internally.

    Example:
        >>> import torch
        >>> from radix_mlp.torch import compute_fold_and_scatter_torch
        >>>
        >>> # Two sequences with shared prefix [1, 2]
        >>> input_ids = torch.tensor([1, 2, 3, 1, 2, 4], dtype=torch.int32)
        >>> position_ids = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32)
        >>> cu_seq_lengths = torch.tensor([0, 3, 6], dtype=torch.int32)
        >>>
        >>> compact_ids, compact_pos, scatter, fold = compute_fold_and_scatter_torch(
        ...     input_ids, position_ids, cu_seq_lengths
        ... )
        >>> print(f"Original: {len(input_ids)} -> Compact: {len(compact_ids)}")
    """
    # Validate inputs are torch tensors
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError(f"input_ids must be a torch.Tensor, got {type(input_ids)}")
    if not isinstance(position_ids, torch.Tensor):
        raise TypeError(f"position_ids must be a torch.Tensor, got {type(position_ids)}")
    if not isinstance(cu_seq_lengths, torch.Tensor):
        raise TypeError(f"cu_seq_lengths must be a torch.Tensor, got {type(cu_seq_lengths)}")

    # Validate all tensors are on the same device
    device = input_ids.device
    if position_ids.device != device:
        raise ValueError(
            f"position_ids must be on the same device as input_ids. "
            f"Got {position_ids.device} vs {device}"
        )
    if cu_seq_lengths.device != device:
        raise ValueError(
            f"cu_seq_lengths must be on the same device as input_ids. "
            f"Got {cu_seq_lengths.device} vs {device}"
        )

    # Validate input shapes
    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1-dimensional, got {input_ids.ndim}D")
    if position_ids.ndim != 1:
        raise ValueError(f"position_ids must be 1-dimensional, got {position_ids.ndim}D")
    if cu_seq_lengths.ndim != 1:
        raise ValueError(f"cu_seq_lengths must be 1-dimensional, got {cu_seq_lengths.ndim}D")

    if len(input_ids) != len(position_ids):
        raise ValueError(
            f"input_ids and position_ids must have same length: "
            f"{len(input_ids)} != {len(position_ids)}"
        )


    input_ids_np = input_ids.numpy(force=True).astype(np.uint32)
    position_ids_np = position_ids.numpy(force=True).astype(np.uint32)
    cu_seq_lengths_np = cu_seq_lengths.numpy(force=True).astype(np.uint32)

    # Call numpy implementation
    (
        compact_input_ids_np,
        compact_position_ids_np,
        scatter_indices_np,
        fold_gather_np,
    ) = _compute_fold_and_scatter_numpy(
        input_ids_np, position_ids_np, cu_seq_lengths_np, pad_multiple_of
    )

    # Convert back to torch tensors (as int32 for better compatibility)
    compact_input_ids = torch.from_numpy(compact_input_ids_np).to(torch.int32)
    compact_position_ids = torch.from_numpy(compact_position_ids_np).to(torch.int32)
    scatter_indices = torch.from_numpy(scatter_indices_np).to(torch.int32)
    fold_gather = torch.from_numpy(fold_gather_np).to(torch.int32)

    # Move to original device if needed
    if device.type != "cpu":
        compact_input_ids = compact_input_ids.to(device)
        compact_position_ids = compact_position_ids.to(device)
        scatter_indices = scatter_indices.to(device)
        fold_gather = fold_gather.to(device)

    return compact_input_ids, compact_position_ids, scatter_indices, fold_gather


__all__ = ["compute_fold_and_scatter_torch"]
