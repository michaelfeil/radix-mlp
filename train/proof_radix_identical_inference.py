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

    def __init__(self, config: RadixMLPQwen3Config):
        self.config = config
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
                attn_implementation="sdpa",
            ).logits.cpu()

            # Run with radix disabled
            nonradix_output = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seq_lengths=cu_seq_lengths,
                max_seq_len=max_seq_len,
                use_radix_mlp=False,
                attn_implementation="sdpa",
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

    def _run_forward_pass(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seq_lengths: torch.Tensor,
        max_seq_len: int,
        use_radix_mlp: bool,
        attn_implementation: str = "flash_attention_2",
    ) -> torch.Tensor:
        """
        Run a single forward pass configuration.

        Args:
            input_ids: Input token IDs
            position_ids: Position IDs
            cu_seq_lengths: Cumulative sequence lengths
            max_seq_len: Maximum sequence length
            use_radix_mlp: Whether to use radix MLP
            attn_implementation: Attention implementation to use

        Returns:
            Output logits
        """
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seq_lengths=cu_seq_lengths,
                max_seq_len=max_seq_len,
                use_radix_mlp=use_radix_mlp,
                attn_implementation=attn_implementation,
            ).logits

        return output

    def _run_backward_pass(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seq_lengths: torch.Tensor,
        max_seq_len: int,
        use_radix_mlp: bool,
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
            (False, "flash_attention_2", "no_radix+flash"),
            (True, "flash_attention_2", "radix+flash"),
            (False, "sdpa", "no_radix+sdpa"),
            (True, "sdpa", "radix+sdpa"),
        ]

        # Store results for all configurations
        all_results = {}
        grad_diffs = []
        gradients_close = False
        max_grad_diff = 0.0
        mean_grad_diff = 0.0

        try:
            # Run all configurations
            for use_radix, attn_impl, config_name in test_configs:

                loss, grads = self._run_backward_pass(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    cu_seq_lengths=cu_seq_lengths,
                    max_seq_len=max_seq_len,
                    use_radix_mlp=use_radix,
                    attn_implementation=attn_impl,
                )

                all_results[config_name] = {"loss": loss.item(), "grads": grads}

            # Compare only SDPA configurations
            print(f"\n  🔍 Comparing SDPA gradients:")

            # Only compare SDPA configurations
            sdpa_configs = [cfg for cfg in test_configs if cfg[1] == "sdpa"]
            if len(sdpa_configs) == 2:
                # Get the two SDPA configurations (no_radix and radix)
                no_radix_config = next(cfg for cfg in sdpa_configs if not cfg[0])
                radix_config = next(cfg for cfg in sdpa_configs if cfg[0])

                no_radix_name = no_radix_config[2]
                radix_name = radix_config[2]

                grads1 = all_results[no_radix_name]["grads"]
                grads2 = all_results[radix_name]["grads"]

                # Compare gradients
                sdpa_grad_diffs = []
                for name in grads1:
                    if name in grads2:
                        diff = torch.abs(grads1[name] - grads2[name])
                        max_diff = diff.max().item()
                        mean_diff = diff.mean().item()
                        sdpa_grad_diffs.append(
                            (
                                f"{no_radix_name}_vs_{radix_name}",
                                name,
                                max_diff,
                                mean_diff,
                            )
                        )

                # Calculate statistics for SDPA comparison
                if sdpa_grad_diffs:
                    max_grad_diff = max(d[2] for d in sdpa_grad_diffs)
                    mean_grad_diff = np.mean([d[3] for d in sdpa_grad_diffs])
                    gradients_close = all(d[2] < 1e-4 for d in sdpa_grad_diffs)

                    print(f"\n  SDPA vs SDPA+radix gradient comparison:")
                    print(f"  Max gradient difference: {max_grad_diff:.8f}")
                    print(f"  Mean gradient difference: {mean_grad_diff:.8f}")
                    print(f"  Gradients close: {gradients_close}")

                    if gradients_close:
                        print("  ✅ PASS: SDPA and SDPA+radix gradients are identical!")
                    else:
                        print("  ❌ FAIL: SDPA and SDPA+radix gradients differ!")

                        # Show parameters with largest differences for SDPA comparison
                        print(
                            "\n  🔍 SDPA parameters with largest gradient differences:"
                        )
                        sdpa_grad_diffs.sort(key=lambda x: x[2], reverse=True)
                        for comparison, name, max_diff, mean_diff in sdpa_grad_diffs[
                            :5
                        ]:
                            print(
                                f"    {name}: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}"
                            )
                else:
                    print("  ⚠️  No gradients found for SDPA comparison")
            else:
                print(
                    "  ⚠️  Expected 2 SDPA configurations but found", len(sdpa_configs)
                )

        except Exception as e:
            print(f"❌ ERROR: {e}")
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
            # Run backward comparison
            backward_result = self.comparator.compare_radix_vs_nonradix_backward(
                sequences, test_name + "_backward"
            )
            results[test_name + "_backward"] = backward_result

        # Generate summary
        self._generate_summary(results)

        return results

    def compare_configurations_for_test_case(
        self, sequences: List[List[int]], test_name: str
    ) -> Dict[str, Any]:
        """
        Compare all 4 configurations (radix/no-radix × flash/sdpa) for a single test case.
        Runs both forward and backward passes.

        Args:
            sequences: Test sequences
            test_name: Name of the test

        Returns:
            Dictionary with comparison results for all configurations
        """
        print(f"\n{'=' * 70}")
        print(f"📊 ATTENTION IMPLEMENTATION COMPARISON")
        print(f"Test Case: {test_name}")
        print(f"Baseline: flash_attention_2 + no_radix")
        print(f"{'=' * 70}")

        # Prepare inputs
        input_ids, position_ids, cu_seq_lengths, max_seq_len = (
            self.comparator.prepare_batchless_inputs(sequences)
        )

        # Move to GPU and enable gradients
        input_ids = input_ids.to("cuda")
        position_ids = position_ids.to("cuda")
        cu_seq_lengths = cu_seq_lengths.to("cuda")

        print(f"Sequences: {sequences}")
        print(f"Total tokens: {input_ids.shape[0]}")
        print(f"Cumulative seq lengths: {cu_seq_lengths.tolist()}")
        print(f"Max seq length: {max_seq_len}")

        # Test configurations
        test_configs = [
            (False, "sdpa", "sdpa + no_radix", "BASELINE"),
            (True, "sdpa", "sdpa + radix", "EXPECTED"),
            # comparing to flash-attn package.
            (False, "flash_attention_2", "flash + no_radix", "EXPECTED"),
            (True, "flash_attention_2", "flash + radix", "EXPECTED"),
        ]

        # Store results for all configurations
        all_results = {}

        try:
            # Run forward passes
            print(f"\n  Running forward passes...")
            with torch.no_grad():
                for (
                    use_radix,
                    attn_impl,
                    config_name,
                    expected_status,
                ) in test_configs:
                    output = self.comparator.model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        cu_seq_lengths=cu_seq_lengths,
                        max_seq_len=max_seq_len,
                        use_radix_mlp=use_radix,
                        attn_implementation=attn_impl,
                    ).logits.cpu()

                    all_results[config_name] = {
                        "forward_logits": output,
                        "use_radix": use_radix,
                        "attn_impl": attn_impl,
                        "expected_status": expected_status,
                    }

            # Run backward passes
            print(f"\n  Running backward passes...")
            for (
                use_radix,
                attn_impl,
                config_name,
                expected_status,
            ) in test_configs:
                loss, grads = self.comparator._run_backward_pass(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    cu_seq_lengths=cu_seq_lengths,
                    max_seq_len=max_seq_len,
                    use_radix_mlp=use_radix,
                    attn_implementation=attn_impl,
                )

                all_results[config_name].update(
                    {
                        "loss": loss.item(),
                        "grads": grads,
                    }
                )

            # Print comparison tables
            self._print_configuration_comparison_table(test_name, all_results)

        except Exception as e:
            print(f"❌ ERROR: {e}")
            return {"test_name": test_name, "error": str(e)}

        return {"test_name": test_name, "all_results": all_results}

    def _print_configuration_comparison_table(
        self,
        test_name: str,
        all_results: Dict[str, Any],
    ):
        """Print a formatted table comparing all configurations."""
        baseline_output = all_results.get("sdpa + no_radix", {}).get("forward_logits")
        baseline_grads = all_results.get("sdpa + no_radix", {}).get("grads")

        # Forward pass table
        print(f"\n📊 FORWARD PASS COMPARISON")
        print(f"┌{'─' * 21}┬{'─' * 14}┬{'─' * 14}┬{'─' * 14}┐")
        print(
            f"│ {'Configuration':^19} │ {'Max Diff':^12} │ {'Mean Diff':^12} │ {'Status':^12} │"
        )
        print(f"├{'─' * 21}┼{'─' * 14}┼{'─' * 14}┼{'─' * 14}┤")

        for config_name in [
            "flash + no_radix",
            "flash + radix",
            "sdpa + no_radix",
            "sdpa + radix",
        ]:
            if config_name not in all_results:
                continue

            result = all_results[config_name]
            output = result["forward_logits"]

            if config_name == "sdpa + no_radix":
                max_diff = 0.0
                mean_diff = 0.0
                status = "BASELINE"
            else:
                # Compare to baseline
                diff = torch.abs(output - baseline_output)
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()

                # Determine status
                if max_diff < 1e-4:
                    status = "✅ PASS"
                elif max_diff < 1e-2:
                    status = "⚠️  EXPECTED"
                else:
                    status = "❌ FAIL"

            print(
                f"│ {config_name:^19} │ {max_diff:^12.8f} │ {mean_diff:^12.8f} │ {status:^12} │"
            )

        print(f"└{'─' * 21}┴{'─' * 14}┴{'─' * 14}┴{'─' * 14}┘")

        # Backward pass table
        print(f"\n📊 BACKWARD PASS COMPARISON")
        print(f"┌{'─' * 21}┬{'─' * 14}┬{'─' * 14}┬{'─' * 14}┐")
        print(
            f"│ {'Configuration':^19} │ {'Max Diff':^12} │ {'Mean Diff':^12} │ {'Status':^12} │"
        )
        print(f"├{'─' * 21}┼{'─' * 14}┼{'─' * 14}┼{'─' * 14}┤")

        for config_name in [
            "flash + no_radix",
            "flash + radix",
            "sdpa + no_radix",
            "sdpa + radix",
        ]:
            if config_name not in all_results:
                continue

            result = all_results[config_name]
            grads = result["grads"]

            if config_name == "sdpa + no_radix":
                max_diff = 0.0
                mean_diff = 0.0
                status = "BASELINE"
            else:
                # Compare to baseline
                max_diffs = []
                mean_diffs = []
                for name in grads:
                    if baseline_grads and name in baseline_grads:
                        diff = torch.abs(grads[name] - baseline_grads[name])
                        max_diffs.append(diff.max().item())
                        mean_diffs.append(diff.mean().item())

                max_diff = max(max_diffs) if max_diffs else 0.0
                mean_diff = np.mean(mean_diffs) if mean_diffs else 0.0

                # Determine status
                if max_diff < 1e-4:
                    status = "✅ PASS"
                elif max_diff < 1e-2:
                    status = "⚠️  EXPECTED"
                else:
                    status = "❌ FAIL"

            print(
                f"│ {config_name:^19} │ {max_diff:^12.8f} │ {mean_diff:^12.8f} │ {status:^12} │"
            )

        print(f"└{'─' * 21}┴{'─' * 14}┴{'─' * 14}┴{'─' * 14}┘")

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
                base_name = test_name.replace("_backward", "")
                base_tests.add(base_name)
            else:
                base_tests.add(test_name)

        for base_name in sorted(base_tests):
            # Forward pass
            if base_name in results:
                result = results[base_name]
                status = "✅ PASS" if result["are_close"] else "❌ FAIL"
                max_diff = result["max_diff"]
                print(f"{base_name}: {status} (max_diff: {max_diff:.8f})")

            # Backward pass
            backward_key = base_name + "_backward"
            if backward_key in results:
                result = results[backward_key]
                status = "✅ PASS" if result["are_close"] else "❌ FAIL"
                max_diff = result["max_diff"]
                print(f"{backward_key}: {status} (max_diff: {max_diff:.8f})")

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

    test_cases = proof.test_generator.create_test_cases()
    attention_results = {}

    for test_name, sequences in test_cases.items():
        result = proof.compare_configurations_for_test_case(sequences, test_name)
        attention_results[test_name] = result

    # Combined results
    combined_results = {
        "radix_proof": radix_results,
        "attention_comparison": attention_results,
    }

    return combined_results


if __name__ == "__main__":
    results = main()
