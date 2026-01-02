"""
Proof script to demonstrate identical forward passes between vanilla and radix-enabled Qwen3 models.

This script creates test cases with shared prefixes and validates that both implementations
produce numerically identical outputs.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import both implementations
from vanilla_modeling_qwen3 import Qwen3ForCausalLM as VanillaQwen3ForCausalLM
from vanilla_modeling_qwen3 import Qwen3Config as VanillaQwen3Config
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

    # Test case 2: Longer shared prefix
    # Sequence 3: [10, 11, 12, 13, 14, 15, 16]
    # Sequence 4: [10, 11, 12, 13, 20, 21, 22]
    # Shared prefix: [10, 11, 12, 13]
    seq3 = [10, 11, 12, 13, 14, 15, 16]
    seq4 = [10, 11, 12, 13, 20, 21, 22]
    pos3 = [0, 1, 2, 3, 4, 5, 6]
    pos4 = [0, 1, 2, 3, 4, 5, 6]
    test_cases.append((seq3, pos3))
    test_cases.append((seq4, pos4))

    # Test case 3: Identical sequences (maximum sharing)
    seq5 = [100, 101, 102, 103]
    seq6 = [100, 101, 102, 103]
    pos5 = [0, 1, 2, 3]
    pos6 = [0, 1, 2, 3]
    test_cases.append((seq5, pos5))
    test_cases.append((seq6, pos6))

    # Test case 4: No shared prefix
    seq7 = [200, 201, 202]
    seq8 = [300, 301, 302]
    pos7 = [0, 1, 2]
    pos8 = [0, 1, 2]
    test_cases.append((seq7, pos7))
    test_cases.append((seq8, pos8))

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


def prepare_vanilla_inputs(
    sequences: List[Tuple[List[int], List[int]]],
) -> Dict[str, torch.Tensor]:
    """
    Prepare inputs for vanilla model (batched).

    Args:
        sequences: List of (token_ids, position_ids) tuples

    Returns:
        Dictionary of inputs for vanilla model
    """
    max_len = max(len(seq) for seq, _ in sequences)

    # Pad sequences to max length
    batch_input_ids = []
    batch_position_ids = []
    batch_attention_mask = []

    for token_ids, position_ids in sequences:
        # Pad with pad token (0)
        padded_ids = token_ids + [0] * (max_len - len(token_ids))
        padded_pos = position_ids + [0] * (max_len - len(position_ids))

        batch_input_ids.append(padded_ids)
        batch_position_ids.append(padded_pos)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(token_ids) + [0] * (max_len - len(token_ids))
        batch_attention_mask.append(attention_mask)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "position_ids": torch.tensor(batch_position_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
    }


def create_models() -> Tuple[Any, Any]:
    """
    Create vanilla and radix models with shared weights.

    Returns:
        Tuple of (vanilla_model, radix_model)
    """
    # Create config
    config = RadixMLPQwen3Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        use_radix_mlp=True,
        use_flash_attn_varlen=True,
        radix_pad_multiple_of=8,
    )

    # Create models
    vanilla_config = VanillaQwen3Config(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        hidden_act=config.hidden_act,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
    )

    vanilla_model = VanillaQwen3ForCausalLM(vanilla_config)
    radix_model = RadixMLPQwen3ForCausalLM(config)

    # Set both models to eval mode
    vanilla_model.eval()
    radix_model.eval()

    return vanilla_model, radix_model


def copy_weights(vanilla_model: Any, radix_model: Any) -> None:
    """
    Copy weights from vanilla model to radix model.

    Args:
        vanilla_model: Source vanilla model
        radix_model: Target radix model
    """
    # Copy embedding weights
    radix_model.model.embed_tokens.weight.data = (
        vanilla_model.model.embed_tokens.weight.data.clone()
    )

    # Copy layer weights
    for i in range(len(vanilla_model.model.layers)):
        # Attention weights
        radix_model.model.layers[i].self_attn.q_proj.weight.data = vanilla_model.model.layers[
            i
        ].self_attn.q_proj.weight.data.clone()
        radix_model.model.layers[i].self_attn.k_proj.weight.data = vanilla_model.model.layers[
            i
        ].self_attn.k_proj.weight.data.clone()
        radix_model.model.layers[i].self_attn.v_proj.weight.data = vanilla_model.model.layers[
            i
        ].self_attn.v_proj.weight.data.clone()
        radix_model.model.layers[i].self_attn.o_proj.weight.data = vanilla_model.model.layers[
            i
        ].self_attn.o_proj.weight.data.clone()

        # Normalization weights
        radix_model.model.layers[i].self_attn.q_norm.weight.data = vanilla_model.model.layers[
            i
        ].self_attn.q_norm.weight.data.clone()
        radix_model.model.layers[i].self_attn.k_norm.weight.data = vanilla_model.model.layers[
            i
        ].self_attn.k_norm.weight.data.clone()
        radix_model.model.layers[i].input_layernorm.weight.data = vanilla_model.model.layers[
            i
        ].input_layernorm.weight.data.clone()
        radix_model.model.layers[
            i
        ].post_attention_layernorm.weight.data = vanilla_model.model.layers[
            i
        ].post_attention_layernorm.weight.data.clone()

        # MLP weights
        radix_model.model.layers[i].mlp.gate_proj.weight.data = vanilla_model.model.layers[
            i
        ].mlp.gate_proj.weight.data.clone()
        radix_model.model.layers[i].mlp.up_proj.weight.data = vanilla_model.model.layers[
            i
        ].mlp.up_proj.weight.data.clone()
        radix_model.model.layers[i].mlp.down_proj.weight.data = vanilla_model.model.layers[
            i
        ].mlp.down_proj.weight.data.clone()

    # Copy final norm and lm_head weights
    radix_model.model.norm.weight.data = vanilla_model.model.norm.weight.data.clone()
    radix_model.lm_head.weight.data = vanilla_model.lm_head.weight.data.clone()


def test_forward_pass_equality(
    vanilla_model: Any,
    radix_model: Any,
    sequences: List[Tuple[List[int], List[int]]],
    test_name: str,
) -> Dict[str, Any]:
    """
    Test forward pass equality between vanilla and radix models.

    Args:
        vanilla_model: Vanilla Qwen3 model
        radix_model: RadixMLP Qwen3 model
        sequences: Test sequences
        test_name: Name of the test

    Returns:
        Dictionary with test results
    """
    print(f"\n=== {test_name} ===")
    print(f"Testing {len(sequences)} sequences")

    # Prepare inputs
    vanilla_inputs = prepare_vanilla_inputs(sequences)
    radix_input_ids, radix_position_ids, cu_seq_lengths, max_seq_len = prepare_batchless_inputs(
        sequences
    )

    print(f"Vanilla input shape: {vanilla_inputs['input_ids'].shape}")
    print(f"Radix input shape: {radix_input_ids.shape}")
    print(f"Cumulative seq lengths: {cu_seq_lengths.tolist()}")
    print(f"Max seq length: {max_seq_len}")

    # Forward pass through vanilla model
    with torch.no_grad():
        vanilla_outputs = vanilla_model(**vanilla_inputs)
        vanilla_logits = vanilla_outputs.logits  # [batch_size, seq_len, vocab_size]

    # Forward pass through radix model
    with torch.no_grad():
        radix_outputs = radix_model(
            input_ids=radix_input_ids,
            position_ids=radix_position_ids,
            cu_seq_lengths=cu_seq_lengths,
            max_seq_len=max_seq_len,
        )
        radix_logits = radix_outputs.logits  # [total_tokens, vocab_size]

    print(f"Vanilla logits shape: {vanilla_logits.shape}")
    print(f"Radix logits shape: {radix_logits.shape}")

    # Compare outputs
    # We need to map radix outputs back to batch format for comparison
    results = {
        "test_name": test_name,
        "vanilla_logits_shape": vanilla_logits.shape,
        "radix_logits_shape": radix_logits.shape,
        "max_diff": None,
        "mean_diff": None,
        "are_close": False,
        "error": None,
    }

    try:
        # Map radix logits back to batch format
        batch_size, seq_len, vocab_size = vanilla_logits.shape
        radix_logits_reshaped = torch.zeros_like(vanilla_logits)

        token_idx = 0
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                if vanilla_inputs["attention_mask"][batch_idx, seq_idx] == 1:  # Real token
                    radix_logits_reshaped[batch_idx, seq_idx] = radix_logits[token_idx]
                    token_idx += 1

        # Compare logits
        diff = torch.abs(vanilla_logits - radix_logits_reshaped)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Check if they are close (using a relatively loose tolerance)
        are_close = torch.allclose(vanilla_logits, radix_logits_reshaped, rtol=1e-3, atol=1e-3)

        results.update(
            {
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "are_close": are_close,
            }
        )

        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        print(f"Are close (rtol=1e-3, atol=1e-3): {are_close}")

        if are_close:
            print("✅ PASS: Forward passes are numerically identical!")
        else:
            print("❌ FAIL: Forward passes differ significantly!")

    except Exception as e:
        results["error"] = str(e)
        print(f"❌ ERROR: {e}")

    return results


def main():
    """Main function to run all tests."""
    print("🧪 Starting RadixMLP vs Vanilla Qwen3 Forward Pass Comparison")
    print("=" * 70)

    # Create models
    print("Creating models...")
    vanilla_model, radix_model = create_models()

    # Copy weights to ensure identical initialization
    print("Copying weights...")
    copy_weights(vanilla_model, radix_model)

    # Create test cases
    test_cases = create_test_sequences()

    # Run tests
    all_results = []

    # Test 1: All sequences together
    result1 = test_forward_pass_equality(
        vanilla_model, radix_model, test_cases, "All Test Sequences"
    )
    all_results.append(result1)

    # Test 2: Only sequences with shared prefixes
    shared_prefix_sequences = test_cases[:4]  # First 4 sequences have shared prefixes
    result2 = test_forward_pass_equality(
        vanilla_model, radix_model, shared_prefix_sequences, "Shared Prefix Sequences"
    )
    all_results.append(result2)

    # Test 3: Only identical sequences (maximum radix benefit)
    identical_sequences = test_cases[4:6]  # Sequences 5 and 6 are identical
    result3 = test_forward_pass_equality(
        vanilla_model, radix_model, identical_sequences, "Identical Sequences"
    )
    all_results.append(result3)

    # Test 4: Only sequences with no shared prefixes
    no_shared_sequences = test_cases[6:8]  # Sequences 7 and 8 have no shared prefixes
    result4 = test_forward_pass_equality(
        vanilla_model, radix_model, no_shared_sequences, "No Shared Prefix Sequences"
    )
    all_results.append(result4)

    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)

    passed = 0
    total = len(all_results)

    for result in all_results:
        status = "✅ PASS" if result["are_close"] else "❌ FAIL"
        if result["error"]:
            status = "❌ ERROR"
        else:
            passed += 1 if result["are_close"] else 0

        print(f"{result['test_name']}: {status}")
        if result["max_diff"] is not None:
            print(f"  Max diff: {result['max_diff']:.6f}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! RadixMLP implementation is correct.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

    return all_results


if __name__ == "__main__":
    results = main()
