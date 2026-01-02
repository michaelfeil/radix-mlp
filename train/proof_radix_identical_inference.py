"""
Proof script to demonstrate identical inference between radix and non-radix modes.

This script creates comprehensive test cases to verify that use_radix_mlp=True and
use_radix_mlp=False produce identical outputs in the radix_torch_varlen implementation.

Key insights:
- Tests single sequences (baseline)
- Tests identical sequences (maximum radix optimization)
- Tests shared prefixes (partial radix optimization)
- Tests no sharing (fallback behavior)
- Investigates MLP layer behavior with radix indices
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import radix implementation
from qwen3_radix_torch_varlen import RadixMLPQwen3ForCausalLM, RadixMLPQwen3Config


class TestSequenceGenerator:
    """Generate test sequences for radix proof."""

    @staticmethod
    def create_test_cases() -> Dict[str, List[List[int]]]:
        """Create comprehensive test cases."""
        test_cases = {}

        # Test case 1: Single sequence (baseline - no radix possible)
        test_cases["single_sequence"] = [
            [1, 2, 3, 4, 5]  # "Hello world"
        ]

        # Test case 2: Identical sequences (maximum radix optimization)
        test_cases["identical_sequences"] = [
            [1, 2, 3, 4, 5],  # "Hello"
            [1, 2, 3, 4, 5],  # "Hello" (identical)
        ]

        # Test case 3: Shared prefix (partial radix optimization)
        test_cases["shared_prefix"] = [
            [1, 2, 3, 4, 5],  # "Hello world"
            [1, 2, 3, 6, 7],  # "Hello there" (shared prefix: [1, 2, 3])
        ]

        # Test case 4: No sharing (fallback behavior)
        test_cases["no_sharing"] = [
            [1, 2, 3],  # "Hello"
            [4, 5, 6],  # "Goodbye" (no shared prefix)
        ]

        # Test case 5: Mixed lengths (variable length support)
        test_cases["mixed_lengths"] = [
            [1, 2],  # "Hi"
            [1, 2, 3, 4, 5],  # "Hello world"
            [1, 2, 3],  # "Hey"
        ]

        # Test case 6: Complex sharing pattern
        test_cases["complex_sharing"] = [
            [1, 2, 3, 4, 5],  # Sequence 1
            [1, 2, 3, 6, 7],  # Sequence 2 (shares prefix with 1)
            [1, 2, 8, 9, 10],  # Sequence 3 (shares shorter prefix)
            [11, 12, 13, 14, 15],  # Sequence 4 (no sharing)
        ]

        return test_cases


class RadixModelComparator:
    """Compare radix vs non-radix model outputs."""

    def __init__(self, config: RadixMLPQwen3Config):
        self.config = config
        self.model = self._create_model()

    def _create_model(self) -> RadixMLPQwen3ForCausalLM:
        """Create test model with specified config."""
        model = RadixMLPQwen3ForCausalLM(self.config)
        model = model.to("cuda").to(torch.float32)
        return model

    def prepare_batchless_inputs(
        self, sequences: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Prepare batchless inputs for radix model.

        Args:
            sequences: List of token sequences

        Returns:
            Tuple of (input_ids, position_ids, cu_seq_lengths, max_seq_len)
        """
        all_input_ids = []
        all_position_ids = []
        seq_lengths = []

        for seq in sequences:
            all_input_ids.extend(seq)
            all_position_ids.extend(range(len(seq)))
            seq_lengths.append(len(seq))

        # Create cumulative sequence lengths
        cu_seq_lengths = [0] + np.cumsum(seq_lengths).tolist()
        max_seq_len = max(seq_lengths)

        input_ids = torch.tensor(all_input_ids, dtype=torch.long).to("cuda")
        position_ids = torch.tensor(all_position_ids, dtype=torch.long).to("cuda")
        cu_seq_lengths = torch.tensor(cu_seq_lengths, dtype=torch.long).to("cuda")

        return input_ids, position_ids, cu_seq_lengths, max_seq_len

    def compare_radix_vs_nonradix(
        self, sequences: List[List[int]], test_name: str
    ) -> Dict[str, Any]:
        """
        Compare radix vs non-radix outputs for given sequences.

        Args:
            sequences: Test sequences
            test_name: Name of the test

        Returns:
            Dictionary with comparison results
        """
        print(f"\n=== {test_name} ===")
        print(f"Sequences: {sequences}")

        # Prepare inputs
        input_ids, position_ids, cu_seq_lengths, max_seq_len = self.prepare_batchless_inputs(
            sequences
        )

        print(f"Input shape: {input_ids.shape}")
        print(f"Cumulative seq lengths: {cu_seq_lengths.tolist()}")
        print(f"Max seq length: {max_seq_len}")

        with torch.no_grad():
            # Run with radix enabled
            radix_output = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seq_lengths=cu_seq_lengths,
                max_seq_len=max_seq_len,
                use_radix_mlp=True,
            ).logits.cpu()

            # Run with radix disabled
            nonradix_output = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seq_lengths=cu_seq_lengths,
                max_seq_len=max_seq_len,
                use_radix_mlp=False,
            ).logits.cpu()

        # Compare outputs
        radix_logits = radix_output
        nonradix_logits = nonradix_output

        print(f"Radix logits shape: {radix_logits.shape}")
        print(f"Non-radix logits shape: {nonradix_logits.shape}")

        # Compute differences
        diff = torch.abs(radix_logits - nonradix_logits)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Check numerical equality
        are_close = torch.allclose(radix_logits, nonradix_logits, rtol=1e-5, atol=1e-5)

        results = {
            "test_name": test_name,
            "sequences": sequences,
            "radix_logits_shape": radix_logits.shape,
            "nonradix_logits_shape": nonradix_logits.shape,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "are_close": are_close,
            "radix_logits": radix_logits,
            "nonradix_logits": nonradix_logits,
        }

        print(f"Max difference: {max_diff:.8f}")
        print(f"Mean difference: {mean_diff:.8f}")
        print(f"Numerically identical (rtol=1e-5, atol=1e-5): {are_close}")

        if are_close:
            print("✅ PASS: Radix and non-radix outputs are identical!")
        else:
            print("❌ FAIL: Radix and non-radix outputs differ!")

            # Show some example differences
            print("\n🔍 Example differences (first 3 tokens, first 5 vocab entries):")
            for token_idx in range(min(3, radix_logits.shape[0])):
                print(f"  Token {token_idx}:")
                for vocab_idx in range(min(5, radix_logits.shape[1])):
                    radix_val = radix_logits[token_idx, vocab_idx].item()
                    nonradix_val = nonradix_logits[token_idx, vocab_idx].item()
                    diff_val = abs(radix_val - nonradix_val)
                    if diff_val > 1e-5:
                        print(
                            f"    Vocab {vocab_idx}: radix={radix_val:.6f}, nonradix={nonradix_val:.6f}, diff={diff_val:.6f}"
                        )

        return results

    def compare_radix_vs_nonradix_backward(
        self, sequences: List[List[int]], test_name: str
    ) -> Dict[str, Any]:
        """
        Compare radix vs non-radix gradients during inference mode.

        Args:
            sequences: Test sequences
            test_name: Name of the test

        Returns:
            Dictionary with gradient comparison results
        """
        print(f"\n=== {test_name} (Backward Pass) ===")
        print(f"Sequences: {sequences}")

        # Prepare inputs
        input_ids, position_ids, cu_seq_lengths, max_seq_len = self.prepare_batchless_inputs(
            sequences
        )

        # Move to GPU and enable gradients
        input_ids = input_ids.to("cuda")
        position_ids = position_ids.to("cuda")
        cu_seq_lengths = cu_seq_lengths.to("cuda")

        print(f"Input shape: {input_ids.shape}")
        print(f"Cumulative seq lengths: {cu_seq_lengths.tolist()}")
        print(f"Max seq length: {max_seq_len}")

        # Store gradients for comparison
        radix_grads = {}
        nonradix_grads = {}

        try:
            # Run with radix enabled (gradient mode)
            self.model.zero_grad()
            radix_output = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seq_lengths=cu_seq_lengths,
                max_seq_len=max_seq_len,
                use_radix_mlp=True,
            ).logits

            # Create a simple loss (sum of all logits)
            radix_loss = radix_output.sum()
            radix_loss.backward(retain_graph=True)

            # Store input gradients
            if input_ids.grad is not None:
                radix_grads["input_grad"] = input_ids.grad.detach().clone()

            # Store parameter gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    radix_grads[name] = param.grad.detach().clone()

            # Run with radix disabled (gradient mode)
            self.model.zero_grad()
            input_ids.grad = None  # Clear input gradients
            nonradix_output = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seq_lengths=cu_seq_lengths,
                max_seq_len=max_seq_len,
                use_radix_mlp=False,
            ).logits

            # Create a simple loss (sum of all logits)
            nonradix_loss = nonradix_output.sum()
            nonradix_loss.backward(retain_graph=True)

            # Store input gradients
            if input_ids.grad is not None:
                nonradix_grads["input_grad"] = input_ids.grad.detach().clone()

            # Store parameter gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    nonradix_grads[name] = param.grad.detach().clone()

            print(f"Radix loss: {radix_loss.item():.6f}")
            print(f"Non-radix loss: {nonradix_loss.item():.6f}")
            print(f"Loss difference: {abs(radix_loss.item() - nonradix_loss.item()):.8f}")

            # Compare gradients
            grad_diffs = []
            for name in nonradix_grads:
                if name in radix_grads:
                    diff = torch.abs(nonradix_grads[name] - radix_grads[name])
                    max_diff = diff.max().item()
                    mean_diff = diff.mean().item()
                    grad_diffs.append((name, max_diff, mean_diff))

            # Also compare input gradients
            if "input_grad" in nonradix_grads and "input_grad" in radix_grads:
                input_grad_diff = torch.abs(
                    nonradix_grads["input_grad"] - radix_grads["input_grad"]
                )
                input_max_diff = input_grad_diff.max().item()
                input_mean_diff = input_grad_diff.mean().item()
                grad_diffs.append(("input_grad", input_max_diff, input_mean_diff))
                print(f"Input gradient max diff: {input_max_diff:.8f}")

            if grad_diffs:
                max_grad_diff = max(d[1] for d in grad_diffs)
                mean_grad_diff = np.mean([d[2] for d in grad_diffs])

                gradients_close = all(d[1] < 1e-5 for d in grad_diffs)

                print(f"Max gradient difference: {max_grad_diff:.8f}")
                print(f"Mean gradient difference: {mean_grad_diff:.8f}")
                print(f"Gradients close: {gradients_close}")

                if gradients_close:
                    print("✅ PASS: Backward gradients are identical!")
                else:
                    print("❌ FAIL: Backward gradients differ!")

                    # Show parameters with largest differences
                    print("\n🔍 Parameters with largest gradient differences:")
                    grad_diffs.sort(key=lambda x: x[1], reverse=True)
                    for name, max_diff, mean_diff in grad_diffs[:5]:
                        print(f"  {name}: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")

        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback

            traceback.print_exc()
            return {"test_name": test_name, "error": str(e)}

        results = {
            "test_name": test_name,
            "sequences": sequences,
            "are_close": gradients_close if grad_diffs else False,
            "max_diff": max_grad_diff if grad_diffs else None,
            "max_grad_diff": max_grad_diff if grad_diffs else 0.0,
            "mean_grad_diff": mean_grad_diff if grad_diffs else 0.0,
            "gradients_close": gradients_close if grad_diffs else False,
            "grad_diffs": grad_diffs,
        }

        return results


class RadixIdenticalInferenceProof:
    """Main proof runner for radix identical inference."""

    def __init__(self):
        self.config = self._create_test_config()
        self.comparator = RadixModelComparator(self.config)
        self.test_generator = TestSequenceGenerator()

    def _create_test_config(self) -> RadixMLPQwen3Config:
        """Create test configuration."""
        return RadixMLPQwen3Config(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,  # Small for fast testing
            num_attention_heads=8,
            num_key_value_heads=2,
            hidden_act="silu",
            max_position_embeddings=128,
            rms_norm_eps=1e-5,
            use_radix_mlp=True,
            use_flash_attn_varlen=True,
        )

    def run_all_proofs(self) -> Dict[str, Any]:
        """Run all proof tests."""
        print("🧪 Starting RadixMLP Identical Inference Proof")
        print("=" * 70)

        test_cases = self.test_generator.create_test_cases()
        results = {}

        # Run all comparison tests
        print("\n📊 RUNNING COMPARISON TESTS")
        print("=" * 50)

        for test_name, sequences in test_cases.items():
            result = self.comparator.compare_radix_vs_nonradix(sequences, test_name)
            results[test_name] = result
            # Run backward comparison ALL LAYERS
            backward_result = self.comparator.compare_radix_vs_nonradix_backward(
                sequences, test_name + " (backward)"
            )
            results[test_name + "_backward"] = backward_result
        
        # Generate summary
        self._generate_summary(results)

        return results

    def _generate_summary(self, results: Dict[str, Any]):
        """Generate proof summary."""
        print("\n" + "=" * 70)
        print("📊 PROOF SUMMARY")
        print("=" * 70)

        # Count passed/failed tests
        comparison_tests = [k for k in results.keys() if k != "mlp_investigation"]
        passed_tests = 0
        total_tests = len(comparison_tests)

        for test_name in comparison_tests:
            if results[test_name]["are_close"]:
                passed_tests += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"

            max_diff = results[test_name]["max_diff"]
            print(f"{test_name}: {status} (max_diff: {max_diff:.8f})")

        # Overall conclusion
        print(f"\nOverall Results: {passed_tests}/{total_tests} comparison tests passed")

        if passed_tests == total_tests:
            print("🎉 PROOF COMPLETE: RadixMLP produces identical inference!")
        else:
            print("❌ PROOF FAILED: Numerical differences detected!")

        print("=" * 70)


def main():
    """Main function to run the proof."""
    proof = RadixIdenticalInferenceProof()
    results = proof.run_all_proofs()
    return results


if __name__ == "__main__":
    results = main()
