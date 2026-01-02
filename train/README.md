# RadixMLP

RadixMLP is a prefix-based computation sharing optimization for transformer models that enables significant speedup when processing multiple sequences with shared prefixes. This implementation provides both Rust (ground truth) and PyTorch versions with full weight compatibility.

## Key Features

- **Prefix-based Computation Sharing**: Automatically identifies and deduplicates shared subsequences across batched sequences
- **Weight Compatible**: Drop-in replacement for standard transformer models - same weights, same results
- **Variable Length Support**: Efficient handling of sequences with different lengths using flash attention
- **Multi-framework**: Rust (Candle) and PyTorch implementations
- **Proven Correctness**: Multiple mathematical proofs verifying numerical equivalence, for forward pass (in Rust/Torch) as well as in torch for backward. 

## Project Structure

```
radix-mlp/
├── train/                          # Training and proof scripts
│   ├── radix_torch_varlen.py       # PyTorch RadixMLP implementation of qwen3
│   ├── radix_mlp_qwen3_modeling_varlen.rs  # Rust ground truth implementation (reference only)
│   ├── proof_radix_forward_backward.py    # Forward/backward pass equivalence 
│   ├── proof_radix_identical_inference.py      # Inference equivalence proof
│   └── test_huggingface_comparison.py          # Weight loading and compatibility test