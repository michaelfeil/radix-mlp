"""
RadixMLP: Prefix-based computation sharing for transformer models.

This module provides Python bindings for the RadixMLP algorithm, which enables
efficient computation sharing across sequences with shared prefixes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple

# Import the Rust extension
try:
    from radix_mlp._core import compute_fold_and_scatter as _compute_fold_and_scatter
    from radix_mlp._core import __version__
except ImportError as e:
    raise ImportError(
        "Failed to import RadixMLP Rust extension. Please ensure the package is built with maturin."
    ) from e


def compute_fold_and_scatter(
    input_ids: NDArray[np.uint32],
    position_ids: NDArray[np.uint32],
    cu_seq_lengths: NDArray[np.uint32],
    pad_multiple_of: bool = False,
) -> Tuple[NDArray[np.uint32], NDArray[np.uint32], NDArray[np.uint32], NDArray[np.uint32]]:
    """
    Compute indices for RadixMLP-style folding and scattering.

    This function identifies shared prefixes among sequences in a batch and produces
    a compact representation containing only unique subsequences.

    Args:
        input_ids: Flattened array of token IDs for all sequences.
        position_ids: Flattened array of position IDs corresponding to each token.
        cu_seq_lengths: Cumulative sequence lengths, e.g., [0, len_seq1, len_seq1+len_seq2, ...].
        pad_multiple_of: If True, pad output to multiple of 8 (small) or 64 (large) for performance.

    Returns:
        A tuple of four numpy arrays:
        - compact_input_ids: Unique token IDs (compacted representation)
        - compact_position_ids: Corresponding position IDs
        - scatter_indices: Index map to unfold from compact to original space
        - fold_gather: Index map to gather from original to compact space

    Example:
        >>> import numpy as np
        >>> from radix_mlp import compute_fold_and_scatter
        >>>
        >>> # Two sequences with shared prefix [1, 2]
        >>> input_ids = np.array([1, 2, 3, 1, 2, 4], dtype=np.uint32)
        >>> position_ids = np.array([0, 1, 2, 0, 1, 2], dtype=np.uint32)
        >>> cu_seq_lengths = np.array([0, 3, 6], dtype=np.uint32)
        >>>
        >>> compact_ids, compact_pos, scatter, fold = compute_fold_and_scatter(
        ...     input_ids, position_ids, cu_seq_lengths
        ... )
        >>> print(f"Original: {len(input_ids)} -> Compact: {len(compact_ids)}")
    """
    # Validate input dtypes
    if input_ids.dtype != np.uint32:
        raise TypeError(f"input_ids must be uint32, got {input_ids.dtype}")
    if position_ids.dtype != np.uint32:
        raise TypeError(f"position_ids must be uint32, got {position_ids.dtype}")
    if cu_seq_lengths.dtype != np.uint32:
        raise TypeError(f"cu_seq_lengths must be uint32, got {cu_seq_lengths.dtype}")

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

    # Call Rust implementation
    return _compute_fold_and_scatter(input_ids, position_ids, cu_seq_lengths, pad_multiple_of)


__all__ = ["compute_fold_and_scatter"]

# Export torch interface if available
try:
    from radix_mlp.torch import compute_fold_and_scatter_torch

    __all__.append("compute_fold_and_scatter_torch")
except ImportError:
    pass
