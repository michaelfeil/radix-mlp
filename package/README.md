# RadixMLP

Pure Rust library for prefix-based computation sharing in transformer models.

## Overview

RadixMLP identifies shared prefixes among sequences in a batch and produces a compact
representation containing only unique subsequences. This enables efficient computation
sharing across sequences with shared prefixes.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
radix_mlp = "0.1.0"
```

## Usage

```rust
use radix_mlp::compute_fold_and_scatter;

let input_ids = vec![1, 2, 3, 1, 2, 4];
let position_ids = vec![0, 1, 2, 0, 1, 2];
let cu_seq_lengths = vec![0, 3, 6];

let (compact_input_ids, compact_position_ids, scatter_indices, fold_gather) =
    compute_fold_and_scatter(&input_ids, &position_ids, &cu_seq_lengths, false);

println!("Original: {} -> Compact: {}", input_ids.len(), compact_input_ids.len());
```

## API

### `compute_fold_and_scatter`

Computes indices for RadixMLP-style folding and scattering.

**Parameters:**
- `input_ids`: Flattened vector of token IDs
- `position_ids`: Flattened vector of position IDs
- `cu_seq_lengths`: Cumulative sequence lengths
- `pad_multiple_of`: Pad output for performance

**Returns:**
- `compact_input_ids`: Unique token IDs
- `compact_position_ids`: Corresponding position IDs
- `scatter_indices`: Unfold indices (compact -> original)
- `fold_gather`: Gather indices (original -> compact)

## Testing

Run tests with:

```bash
cargo test
```

## License

MIT License - Copyright (c) 2025 michaelfeil