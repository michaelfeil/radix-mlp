# RadixMLP

RadixMLP enables prefix-based computation sharing for transformer models, eliminating redundant MLP activations when processing batches with shared prefixes. Achieves up to 5× speedup for embedding workloads.

## Key Features

- Prefix deduplication: Automatically identifies and compacts shared subsequences across batched sequences
- Stateless operation: Single forward pass optimization, no cache management required
- Weight compatible: Drop-in replacement for standard transformer models
- compatible with autograd
- Production ready: Integrated into [text-embeddings-inference upstream](https://github.com/huggingface/text-embeddings-inference/pull/761)

## Project Structure

```
radix-mlp/
├── package/           # Rust core library - `cargo add radix_mlp`
├── python_bindings/   # Python interface (PyTorch + NumPy) `pip install radix_mlp`
├── train/             # Training scripts & experimental results for autograd
├── benchmark/         # Performance benchmarks via rest api
├── kernels            # cuda kernels for fast index-select 
└── Readme.md          # This file
```

## Implementations

### Rust Core Library (`package/`)
- High-performance `compute_fold_and_scatter()` algorithm
- Comprehensive test suite and benchmarks
- MIT License, `pip install radix-mlp`

### Python Bindings (`python_bindings/`)
- NumPy interface: `compute_fold_and_scatter()`
- PyTorch interface: `compute_fold_and_scatter_torch()`
- Device support (CPU/GPU) with automatic conversion

### Training & Proofs (`train/`)
- PyTorch: `radix_torch_varlen.py` - Complete Qwen3 implementation
- Rust: `radix_mlp_qwen3_modeling_varlen.rs` - Ground truth implementation
- Mathematical Proofs:
  - `proof_radix_forward_backward.py` - Forward/backward equivalence
  - `proof_radix_identical_inference.py` - Inference equivalence
  - `test_huggingface_comparison.py` - Weight compatibility

## Benchmarks

### Synthetic Benchmarks
- Up to 5× speedup on Qwen3 models (0.6B-8B)
- Performance gains scale with model size and prefix length
- Detailed results in package/benches/

### Real-World Benchmarks (`benchmark/`)
- MSMARCO v1.1 query-passage embedding
- End-to-end TEI integration testing
- 1.4-1.6× latency reduction in production workloads

## Usage

### Python
```python
from radix_mlp import compute_fold_and_scatter_torch
import torch

# Two sequences with shared prefix
input_ids = torch.tensor([1, 2, 3, 1, 2, 4])
position_ids = torch.tensor([0, 1, 2, 0, 1, 2])
cu_seq_lengths = torch.tensor([0, 3, 6])

compact_ids, _, _, _ = compute_fold_and_scatter_torch(
    input_ids, position_ids, cu_seq_lengths
)
print(f"Compression: {len(input_ids)} → {len(compact_ids)} tokens")
```

### Rust
```rust
use radix_mlp::compute_fold_and_scatter;

let input_ids = vec![1, 2, 3, 1, 2, 4];
let position_ids = vec![0, 1, 2, 0, 1, 2];
let cu_seq_lengths = vec![0, 3, 6];

let (compact_ids, _, _, _) = compute_fold_and_scatter(
    &input_ids, &position_ids, &cu_seq_lengths, None
);
```

## Production Integration

### Text-Embeddings-Inference
Upstream PR: [huggingface/text-embeddings-inference#761](https://github.com/huggingface/text-embeddings-inference/pull/761)

- Zero-configuration enablement
- Automatic thresholding
- Production-tested with MSMARCO workloads

## Documentation

- Rust Library: See `package/README.md`
- Python Bindings: See `python_bindings/README.md`  
- Training & Proofs: See `train/README.md`
- Benchmarks: See `benchmark/README.md`

## Verification Status

- Forward pass: Numerically identical to baseline
- Backward pass: Gradients identical to baseline
- Weight compatibility: 100% compatible with transformers
- Production: TEI upstream integration complete

## License

MIT License - Copyright (c) 2025 michaelfeil

## Performance Summary

| Model | Synthetic Speedup | End-to-End Speedup |
|-------|-------------------|-------------------|
| 0.6B  | 2.7×              | 1.44×             |
| 4B    | 4.1×              | 1.56×             |
| 8B    | 5.0×              | 1.59×             |

*Results from paper - see benchmarks for details*
