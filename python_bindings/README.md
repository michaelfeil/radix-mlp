# RadixMLP Python Bindings

Python bindings for the RadixMLP algorithm, enabling prefix-based computation
sharing for transformer models.

## Installation

### Development Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Using pip
pip install -e .
```

### Release Installation

```bash
pip install radix-mlp
```

## Usage

```python
import numpy as np
from radix_mlp import compute_fold_and_scatter

# Example: Two sequences with shared prefix
input_ids = np.array([1, 2, 3, 1, 2, 4], dtype=np.uint32)
position_ids = np.array([0, 1, 2, 0, 1, 2], dtype=np.uint32)
cu_seq_lengths = np.array([0, 3, 6], dtype=np.uint32)

compact_ids, compact_pos, scatter, fold = compute_fold_and_scatter(
    input_ids, position_ids, cu_seq_lengths
)

print(f"Original tokens: {len(input_ids)}")
print(f"Compact tokens: {len(compact_ids)}")
print(f"Compression ratio: {len(compact_ids) / len(input_ids):.2%}")
```

## API Reference

### `compute_fold_and_scatter`

Computes indices for RadixMLP-style folding and scattering.

**Parameters:**
- `input_ids` (np.ndarray[np.uint32]): Flattened token IDs
- `position_ids` (np.ndarray[np.uint32]): Flattened position IDs
- `cu_seq_lengths` (np.ndarray[np.uint32]): Cumulative sequence lengths
- `pad_multiple_of` (bool): Pad output for performance (default: False)

**Returns:**
- `compact_input_ids`: Unique token IDs
- `compact_position_ids`: Corresponding position IDs
- `scatter_indices`: Unfold indices (compact -> original)
- `fold_gather`: Gather indices (original -> compact)

## Development

### Building

```bash
# Development build
maturin develop

# Release build
maturin build --release
```

## License

MIT License - Copyright (c) 2025 michaelfeil