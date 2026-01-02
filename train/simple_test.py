"""
Simple test script to validate the RadixMLP implementation.

This script tests the radix-enabled model without requiring the vanilla model imports.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import radix implementation
from radix_torch_varlen import RadixMLPQwen3ForCausalLM, RadixMLPQwen3Config


def create_test_sequences() -> List[Tuple[List[int], List[int]]]:
    """
    Create test sequences with shared prefixes.

    Returns:
        List of (sequence, position_ids) tuples
    """
    test_cases = []

    # Test case 1: Simple shared prefix
    # Sequence 1: [1, 2, 3, 4, 5]
    # Sequence 2: [1, 2, 3, 6, 7]
    # Shared prefix: [1, 2, 3]
    seq1 = [1, 2, 3, 4, 5]
    seq2 = [1, 2, 3, 6, 7]
    pos1 = [0, 1, 2, 3, 4]
    pos2 = [0, 1, 2, 3, 4]
    test_cases.append((seq1, pos1))
    test_cases.append((seq2, pos2))

    # Test case 2: Identical sequences (maximum sharing)
    seq3 = [100, 101, 102, 103]
    seq4 = [100, 101, 102, 103]
    pos3 = [0, 1, 2, 3]
    pos4 = [0, 1, 2, 3]
    test_cases.append((seq3, pos3))
    test_cases.append((seq4, pos4))

    return test_cases


def prepare_batchless_inputs(
    sequences: List[Tuple[List[int], List[int]]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Prepare batchless inputs for radix model.

    Args:
        sequences: List of (token_ids, position_ids) tuples

    Returns:
        Tuple of (input_ids, position_ids, cu_seq_lengths, max_seq_len)
    """
    all_input_ids = []
    all_position_ids = []
    seq_lengths = []

    for token_ids, position_ids in sequences:
        all_input_ids.extend(token_ids)
        all_position_ids.extend(position_ids)
        seq_lengths.append(len(token_ids))

    # Create cumulative sequence lengths
    cu_seq_lengths = [0] + np.cumsum(seq_lengths).tolist()
    max_seq_len = max(seq_lengths)

    input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    position_ids = torch.tensor(all_position_ids, dtype=torch.long)
    cu_seq_lengths = torch.tensor(cu_seq_lengths, dtype=torch.long)

    return input_ids, position_ids, cu_seq_lengths, max_seq_len


def test_radix_model():
    """Test the radix model implementation."""
    print("🧪 Testing RadixMLP Qwen3 Model")
    print("=" * 50)

    # Create config
    config = RadixMLPQwen3Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,  # Small for testing
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        use_radix_mlp=True,
        use_flash_attn_varlen=True,
        radix_pad_multiple_of=8,
    )

    # Create model
    model = RadixMLPQwen3ForCausalLM(config)
    model.eval()

    print(f"Model created with {config.num_hidden_layers} layers")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Using RadixMLP: {config.use_radix_mlp}")
    print(f"Using Flash Attention Varlen: {config.use_flash_attn_varlen}")

    # Create test sequences
    test_cases = create_test_sequences()

    print(f"\nTesting with {len(test_cases)} sequences:")
    for i, (seq, pos) in enumerate(test_cases):
        print(f"  Sequence {i + 1}: {seq}")

    # Prepare inputs
    input_ids, position_ids, cu_seq_lengths, max_seq_len = prepare_batchless_inputs(test_cases)

    print(f"\nInput shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  position_ids: {position_ids.shape}")
    print(f"  cu_seq_lengths: {cu_seq_lengths.shape}")
    print(f"  max_seq_len: {max_seq_len}")

    print(f"\nCumulative sequence lengths: {cu_seq_lengths.tolist()}")

    # Test forward pass
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seq_lengths=cu_seq_lengths,
                max_seq_len=max_seq_len,
            )

        print(f"\n✅ Forward pass successful!")
        print(f"Logits shape: {outputs.logits.shape}")
        print(f"Loss: {outputs.loss}")

        # Check output properties
        expected_vocab_size = config.vocab_size
        actual_vocab_size = outputs.logits.shape[-1]

        if actual_vocab_size == expected_vocab_size:
            print(f"✅ Output vocab size correct: {actual_vocab_size}")
        else:
            print(
                f"❌ Output vocab size incorrect: {actual_vocab_size} (expected {expected_vocab_size})"
            )

        # Test with labels
        print(f"\n🧪 Testing with labels...")
        labels = torch.randint(0, config.vocab_size, (input_ids.shape[0],))

        with torch.no_grad():
            outputs_with_loss = model(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seq_lengths=cu_seq_lengths,
                max_seq_len=max_seq_len,
                labels=labels,
            )

        print(f"✅ Forward pass with labels successful!")
        print(f"Loss value: {outputs_with_loss.loss.item():.4f}")

        return True

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_radix_vs_no_radix():
    """Test radix model vs non-radix model."""
    print("\n🧪 Testing RadixMLP vs Standard MLP")
    print("=" * 50)

    # Create radix config
    radix_config = RadixMLPQwen3Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        use_radix_mlp=True,
        use_flash_attn_varlen=True,
        radix_pad_multiple_of=8,
    )

    # Create standard config
    standard_config = RadixMLPQwen3Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        use_radix_mlp=False,  # Disable radix
        use_flash_attn_varlen=True,
        radix_pad_multiple_of=8,
    )

    # Create models
    radix_model = RadixMLPQwen3ForCausalLM(radix_config)
    standard_model = RadixMLPQwen3ForCausalLM(standard_config)

    # Copy weights from standard to radix for fair comparison
    radix_model.model.embed_tokens.weight.data = (
        standard_model.model.embed_tokens.weight.data.clone()
    )

    for i in range(len(radix_model.model.layers)):
        # Copy attention weights
        radix_model.model.layers[i].self_attn.q_proj.weight.data = standard_model.model.layers[
            i
        ].self_attn.q_proj.weight.data.clone()
        radix_model.model.layers[i].self_attn.k_proj.weight.data = standard_model.model.layers[
            i
        ].self_attn.k_proj.weight.data.clone()
        radix_model.model.layers[i].self_attn.v_proj.weight.data = standard_model.model.layers[
            i
        ].self_attn.v_proj.weight.data.clone()
        radix_model.model.layers[i].self_attn.o_proj.weight.data = standard_model.model.layers[
            i
        ].self_attn.o_proj.weight.data.clone()

        # Copy normalization weights
        radix_model.model.layers[i].self_attn.q_norm.weight.data = standard_model.model.layers[
            i
        ].self_attn.q_norm.weight.data.clone()
        radix_model.model.layers[i].self_attn.k_norm.weight.data = standard_model.model.layers[
            i
        ].self_attn.k_norm.weight.data.clone()
        radix_model.model.layers[i].input_layernorm.weight.data = standard_model.model.layers[
            i
        ].input_layernorm.weight.data.clone()
        radix_model.model.layers[
            i
        ].post_attention_layernorm.weight.data = standard_model.model.layers[
            i
        ].post_attention_layernorm.weight.data.clone()

        # Copy MLP weights
        radix_model.model.layers[i].mlp.gate_proj.weight.data = standard_model.model.layers[
            i
        ].mlp.gate_proj.weight.data.clone()
        radix_model.model.layers[i].mlp.up_proj.weight.data = standard_model.model.layers[
            i
        ].mlp.up_proj.weight.data.clone()
        radix_model.model.layers[i].mlp.down_proj.weight.data = standard_model.model.layers[
            i
        ].mlp.down_proj.weight.data.clone()

    # Copy final weights
    radix_model.model.norm.weight.data = standard_model.model.norm.weight.data.clone()
    radix_model.lm_head.weight.data = standard_model.lm_head.weight.data.clone()

    # Set to eval mode
    radix_model.eval()
    standard_model.eval()

    # Create test sequences with shared prefixes
    test_cases = create_test_sequences()

    # Prepare inputs
    input_ids, position_ids, cu_seq_lengths, max_seq_len = prepare_batchless_inputs(test_cases)

    print(f"Testing with sequences that have shared prefixes")
    print(f"Total tokens: {input_ids.shape[0]}")

    try:
        # Forward pass through standard model
        with torch.no_grad():
            standard_outputs = standard_model(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seq_lengths=cu_seq_lengths,
                max_seq_len=max_seq_len,
            )

        # Forward pass through radix model
        with torch.no_grad():
            radix_outputs = radix_model(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seq_lengths=cu_seq_lengths,
                max_seq_len=max_seq_len,
            )

        # Compare outputs
        diff = torch.abs(standard_outputs.logits - radix_outputs.logits)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"\n📊 Comparison Results:")
        print(f"Standard logits shape: {standard_outputs.logits.shape}")
        print(f"Radix logits shape: {radix_outputs.logits.shape}")
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")

        # Check if they are close
        are_close = torch.allclose(
            standard_outputs.logits, radix_outputs.logits, rtol=1e-3, atol=1e-3
        )

        if are_close:
            print(f"✅ PASS: Radix and standard models produce identical outputs!")
            return True
        else:
            print(f"❌ FAIL: Radix and standard models differ!")
            return False

    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function to run all tests."""
    print("🚀 Starting RadixMLP Implementation Tests")
    print("=" * 60)

    results = []

    # Test 1: Basic functionality
    result1 = test_radix_model()
    results.append(("Basic Functionality", result1))

    # Test 2: Radix vs Standard comparison
    result2 = test_radix_vs_no_radix()
    results.append(("Radix vs Standard", result2))

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        if result:
            passed += 1
        print(f"{test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! RadixMLP implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

    return results


if __name__ == "__main__":
    results = main()
