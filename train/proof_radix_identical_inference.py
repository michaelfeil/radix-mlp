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
import functools
from typing import List, Tuple, Dict, Any
import sys
import os
import argparse

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Enable CUDA determinism for better precision
def setup_cuda_determinism():
    """Setup CUDA determinism and precision settings."""
    # Set CuBLAS workspace config for deterministic behavior
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    if torch.cuda.is_available():
        # Enable deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set deterministic behavior for CUDA (warn_only to avoid runtime errors)
        torch.use_deterministic_algorithms(True)

        # Set precision for cuDNN
        torch.backends.cudnn.allow_tf32 = False  # Disable TF32 for better precision

        # Set matmul precision
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        print("✅ CUDA determinism enabled for maximum precision")


# Setup determinism at import
setup_cuda_determinism()
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

    def __init__(self, config: RadixMLPQwen3Config, use_dummy_attn: bool = False):
        self.config = config
        self.use_dummy_attn = use_dummy_attn
        self.model = self._create_model()

    @functools.lru_cache(maxsize=1)
    def _create_model(self) -> RadixMLPQwen3ForCausalLM:
        """Create and cache the test model."""
        return self.__create_model(self.config)

    @staticmethod
    def __create_model(config: RadixMLPQwen3Config) -> RadixMLPQwen3ForCausalLM:
        """Create test model with specified config."""
        model = RadixMLPQwen3ForCausalLM(config)
        model = model.to("cuda").to(
            torch.float32
        )  # Use double precision for maximum accuracy
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
        input_ids, position_ids, cu_seq_lengths, max_seq_len = (
            self.prepare_batchless_inputs(sequences)
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
                use_dummy_attn=self.use_dummy_attn,
            ).logits.cpu()

            # Run with radix disabled
            nonradix_output = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seq_lengths=cu_seq_lengths,
                max_seq_len=max_seq_len,
                use_radix_mlp=False,
                use_dummy_attn=self.use_dummy_attn,
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
        are_close = torch.allclose(radix_logits, nonradix_logits, rtol=1e-4, atol=1e-4)

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
        print(f"Numerically identical (rtol=1e-4, atol=1e-4): {are_close}")

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

    def _run_backward_pass(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seq_lengths: torch.Tensor,
        max_seq_len: int,
        use_radix_mlp: bool,
        use_dummy_attn: bool,
        attn_implementation: str = "flash_attention_2",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run a single backward pass configuration and collect gradients.

        Args:
            input_ids: Input token IDs
            position_ids: Position IDs
            cu_seq_lengths: Cumulative sequence lengths
            max_seq_len: Maximum sequence length
            use_radix_mlp: Whether to use radix MLP
            use_dummy_attn: Whether to use dummy attention
            attn_implementation: Attention implementation to use

        Returns:
            Tuple of (loss, gradients_dict)
        """
        self.model.zero_grad()
        if input_ids.grad is not None:
            input_ids.grad = None

        output = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seq_lengths=cu_seq_lengths,
            max_seq_len=max_seq_len,
            use_radix_mlp=use_radix_mlp,
            use_dummy_attn=use_dummy_attn,
            attn_implementation=attn_implementation,
        ).logits

        loss = output.sum()
        loss.backward()

        # Collect gradients
        grads = {}
        if input_ids.grad is not None:
            grads["input_grad"] = input_ids.grad.detach().clone()

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().clone()

        return loss, grads

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
        input_ids, position_ids, cu_seq_lengths, max_seq_len = (
            self.prepare_batchless_inputs(sequences)
        )

        # Move to GPU and enable gradients
        input_ids = input_ids.to("cuda")
        position_ids = position_ids.to("cuda")
        cu_seq_lengths = cu_seq_lengths.to("cuda")

        print(f"Input shape: {input_ids.shape}")
        print(f"Cumulative seq lengths: {cu_seq_lengths.tolist()}")
        print(f"Max seq length: {max_seq_len}")

        # Test configurations
        test_configs = [
            (False, False, "flash_attention_2", "no_radix+flash"),
            (True, False, "flash_attention_2", "radix+flash"),
        ]
        if not self.use_dummy_attn:
            test_configs += [
                (False, False, "sdpa", "no_radix+sdpa"),
                (True, False, "sdpa", "radix+sdpa"),
            ]

        # Store results for all configurations
        all_results = {}
        grad_diffs = []
        gradients_close = False
        max_grad_diff = 0.0
        mean_grad_diff = 0.0

        try:
            # Run all configurations
            for use_radix, use_dummy, attn_impl, config_name in test_configs:
                print(f"\n  📊 Running: {config_name}")

                loss, grads = self._run_backward_pass(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    cu_seq_lengths=cu_seq_lengths,
                    max_seq_len=max_seq_len,
                    use_radix_mlp=use_radix,
                    use_dummy_attn=use_dummy,
                    attn_implementation=attn_impl,
                )

                print(f"    Loss: {loss.item():.6f}")
                all_results[config_name] = {"loss": loss.item(), "grads": grads}

            # Compare all configurations
            print(f"\n  🔍 Comparing gradients across configurations:")

            config_names = [cfg[3] for cfg in test_configs]
            for i, config1 in enumerate(config_names):
                for j, config2 in enumerate(config_names):
                    if i >= j:
                        continue

                    grads1 = all_results[config1]["grads"]
                    grads2 = all_results[config2]["grads"]

                    # Compare gradients
                    for name in grads1:
                        if name in grads2:
                            diff = torch.abs(grads1[name] - grads2[name])
                            max_diff = diff.max().item()
                            mean_diff = diff.mean().item()
                            grad_diffs.append(
                                (f"{config1}_vs_{config2}", name, max_diff, mean_diff)
                            )

            # Find maximum gradient difference
            if grad_diffs:
                max_grad_diff = max(d[2] for d in grad_diffs)
                mean_grad_diff = np.mean([d[3] for d in grad_diffs])
                gradients_close = all(d[2] < 1e-4 for d in grad_diffs)

                print(f"\n  Max gradient difference: {max_grad_diff:.8f}")
                print(f"  Mean gradient difference: {mean_grad_diff:.8f}")
                print(f"  Gradients close: {gradients_close}")

                if gradients_close:
                    print("  ✅ PASS: Backward gradients are identical!")
                else:
                    print("  ❌ FAIL: Backward gradients differ!")

                    # Show parameters with largest differences
                    print("\n  🔍 Parameters with largest gradient differences:")
                    grad_diffs.sort(key=lambda x: x[2], reverse=True)
                    for comparison, name, max_diff, mean_diff in grad_diffs[:5]:
                        print(
                            f"    {comparison} ({name}): max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}"
                        )

        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback

            traceback.print_exc()
            return {"test_name": test_name, "error": str(e)}

        results = {
            "test_name": test_name,
            "sequences": sequences,
            "are_close": gradients_close,
            "max_diff": max_grad_diff,
            "max_grad_diff": max_grad_diff,
            "mean_grad_diff": mean_grad_diff,
            "gradients_close": gradients_close,
            "grad_diffs": grad_diffs,
            "all_results": all_results,
        }

        return results


class RadixIdenticalInferenceProof:
    """Main proof runner for radix identical inference."""

    def __init__(self):
        self.config = self._create_test_config()
        self.comparator = RadixModelComparator(self.config)
        self.comparator_dummy = RadixModelComparator(self.config, use_dummy_attn=True)
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
            # Run backward comparison without dummy attention
            backward_result = self.comparator.compare_radix_vs_nonradix_backward(
                sequences, test_name + "_backward"
            )
            results[test_name + "_backward"] = backward_result
            # Run backward comparison with dummy attention
            backward_result_dummy = (
                self.comparator_dummy.compare_radix_vs_nonradix_backward(
                    sequences, test_name + "_backward_dummy_attn"
                )
            )
            results[test_name + "_backward_dummy_attn"] = backward_result_dummy

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

        # Group tests by base name for clearer comparison
        base_tests = set()
        for test_name in comparison_tests:
            if "_backward" in test_name:
                base_name = test_name.replace("_backward", "").replace(
                    "_dummy_attn", ""
                )
                base_tests.add(base_name)
            elif "_backward_dummy_attn" not in test_name:
                base_tests.add(test_name)

        for base_name in sorted(base_tests):
            # Forward pass
            if base_name in results:
                result = results[base_name]
                status = "✅ PASS" if result["are_close"] else "❌ FAIL"
                max_diff = result["max_diff"]
                print(f"{base_name}: {status} (max_diff: {max_diff:.8f})")

            # Backward pass without dummy attention
            backward_key = base_name + "_backward"
            if backward_key in results:
                result = results[backward_key]
                status = "✅ PASS" if result["are_close"] else "❌ FAIL"
                max_diff = result["max_diff"]
                print(f"{backward_key}: {status} (max_diff: {max_diff:.8f})")

            # Backward pass with dummy attention
            backward_dummy_key = base_name + "_backward_dummy_attn"
            if backward_dummy_key in results:
                result = results[backward_dummy_key]
                status = "✅ PASS" if result["are_close"] else "❌ FAIL"
                max_diff = result["max_diff"]
                print(f"{backward_dummy_key}: {status} (max_diff: {max_diff:.8f})")

        # Overall conclusion
        print(
            f"\nOverall Results: {passed_tests}/{total_tests} comparison tests passed"
        )

        if passed_tests == total_tests:
            print("🎉 PROOF COMPLETE: RadixMLP produces identical inference!")
        else:
            print("❌ PROOF FAILED: Numerical differences detected!")

        print("=" * 70)


def main():
    """Main function to run the proof."""
    # Run original radix identical inference proof
    print("🧪 Running Radix Identical Inference Proof")
    print("=" * 70)
    proof = RadixIdenticalInferenceProof()
    radix_results = proof.run_all_proofs()

    # Run attention implementation comparison

    # Combined results
    combined_results = {
        "radix_proof": radix_results,
    }

    return combined_results


if __name__ == "__main__":
    results = main()
