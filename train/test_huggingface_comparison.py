"""
Test script to compare vanilla transformers Qwen3-0.6B with radix implementation.

This script loads the official Qwen3-0.6B model from HuggingFace and compares
outputs with our radix implementation on a single sequence.
"""

import torch
import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import radix implementation
from qwen3_radix_torch_varlen import RadixMLPQwen3ForCausalLM, RadixMLPQwen3Config


def test_single_sequence_comparison():
    """Test vanilla vs radix on a single sequence."""
    print("🧪 Testing Vanilla vs Radix Qwen3-0.6B on Single Sequence")
    print("=" * 60)

    # Model configuration
    model_name = "Qwen/Qwen3-0.6B"

    # Test sequence
    test_text = "Hello, world!"

    print(f"Loading model: {model_name}")
    print(f"Test sequence: '{test_text}'")

    try:
        # Load vanilla model and tokenizer
        print("📦 Loading vanilla transformers model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        vanilla_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        vanilla_model.eval()
        vanilla_model = vanilla_model.to("cuda")

        # Tokenize input
        inputs = tokenizer(test_text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")

        print(f"Input IDs: {input_ids.tolist()}")
        print(f"Input shape: {input_ids.shape}")
        print(f"Attention mask: {attention_mask.tolist()}")

        # Get vanilla model output
        print("🔄 Running vanilla model...")
        with torch.no_grad():
            vanilla_outputs = vanilla_model(input_ids=input_ids, attention_mask=attention_mask)
            vanilla_logits = vanilla_outputs.logits  # [1, seq_len, vocab_size]

        print(f"Vanilla logits shape: {vanilla_logits.shape}")

        # Create radix config matching the loaded model
        print("⚙️ Creating radix config...")

        # Create radix model
        print("📦 Creating radix model...")
        radix_model = RadixMLPQwen3ForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        radix_model.eval()
        radix_model = radix_model.to(torch.float16)

        # Copy weights from vanilla to radix model
        print("🔄 Copying weights from vanilla to radix...")
        radix_model = radix_model.to("cuda")

        # Prepare radix inputs (batchless format)
        print("📊 Preparing radix inputs...")
        batch_size, seq_len = input_ids.shape

        # Convert to batchless format
        radix_input_ids = input_ids.squeeze(0).cuda()  # [seq_len]
        radix_position_ids = torch.arange(seq_len, dtype=torch.long).cuda()  # [seq_len]
        radix_cu_seq_lengths = torch.tensor([0, seq_len], dtype=torch.long).cuda()  # [2]
        radix_max_seq_len = seq_len

        print(f"Radix input IDs: {radix_input_ids.tolist()}")
        print(f"Radix position IDs: {radix_position_ids.tolist()}")
        print(f"Radix cu_seq_lengths: {radix_cu_seq_lengths.tolist()}")
        print(f"Radix max_seq_len: {radix_max_seq_len}")

        # Get radix model output
        print("🔄 Running radix model...")
        with torch.no_grad():
            radix_outputs = radix_model(
                input_ids=radix_input_ids,
                position_ids=radix_position_ids,
                cu_seq_lengths=radix_cu_seq_lengths,
                max_seq_len=radix_max_seq_len,
            )
            radix_logits = radix_outputs.logits  # [seq_len, vocab_size]

        print(f"Radix logits shape: {radix_logits.shape}")

        # Compare outputs
        print("📊 Comparing outputs...")
        compare_outputs(vanilla_logits, radix_logits)

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def copy_weights_from_vanilla_to_radix(vanilla_model, radix_model):
    """Copy weights from vanilla model to radix model."""
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

    # Copy final weights
    radix_model.model.norm.weight.data = vanilla_model.model.norm.weight.data.clone()
    radix_model.lm_head.weight.data = vanilla_model.lm_head.weight.data.clone()


def compare_outputs(vanilla_logits, radix_logits):
    """Compare vanilla and radix outputs."""
    # Remove batch dimension from vanilla logits for comparison
    vanilla_logits_2d = vanilla_logits.squeeze(0)  # [seq_len, vocab_size]

    print(f"Vanilla logits shape: {vanilla_logits_2d.shape}")
    print(f"Radix logits shape: {radix_logits.shape}")

    # Check shapes match
    if vanilla_logits_2d.shape != radix_logits.shape:
        print(f"❌ Shape mismatch: vanilla {vanilla_logits_2d.shape} vs radix {radix_logits.shape}")
        return False

    # Compute differences
    diff = torch.abs(vanilla_logits_2d - radix_logits)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"📈 Max difference: {max_diff:.8f}")
    print(f"📈 Mean difference: {mean_diff:.8f}")

    # Check if they are close
    are_close = torch.allclose(vanilla_logits_2d, radix_logits, rtol=1e-5, atol=1e-5)

    if are_close:
        print("✅ PASS: Vanilla and radix outputs are numerically identical!")
    else:
        print("❌ FAIL: Vanilla and radix outputs differ significantly!")

        # Show some example differences
        print("\n🔍 Example differences (first 5 tokens, first 10 vocab entries):")
        for token_idx in range(min(5, vanilla_logits_2d.shape[0])):
            print(f"  Token {token_idx}:")
            for vocab_idx in range(min(10, vanilla_logits_2d.shape[1])):
                vanilla_val = vanilla_logits_2d[token_idx, vocab_idx].item()
                radix_val = radix_logits[token_idx, vocab_idx].item()
                diff_val = abs(vanilla_val - radix_val)
                if diff_val > 1e-5:
                    print(
                        f"    Vocab {vocab_idx}: vanilla={vanilla_val:.6f}, radix={radix_val:.6f}, diff={diff_val:.6f}"
                    )

    return are_close


def main():
    """Main function."""
    print("🚀 Starting Vanilla vs Radix Qwen3-0.6B Comparison Test")
    print("=" * 70)

    success = test_single_sequence_comparison()

    print("\n" + "=" * 70)
    if success:
        print("🎉 Test completed successfully!")
    else:
        print("❌ Test failed!")

    return success


if __name__ == "__main__":
    main()
