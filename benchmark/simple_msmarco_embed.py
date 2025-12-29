#!/usr/bin/env python3
"""
Simple MSMARCO Passage Embedding Script

Loads MSMARCO passages and prepares them for embedding with Qwen3 chat template.
"""

import os
import json
import argparse
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

from datasets import load_dataset
from baseten_performance_client import (
    PerformanceClient,
    RequestProcessingPreference,
)


@dataclass
class ConfigurationResult:
    """Store results for a single configuration test."""

    name: str
    base_url: str
    server_info: Dict[str, Any]
    times: List[float]
    total_time: float
    success: bool
    error_message: Optional[str] = None


def load_msmarco_passages(max_samples: Optional[int] = None) -> List[Dict[str, str]]:
    """Load MSMARCO query-passage pairs from validation set."""
    print("Loading MSMARCO query-passage pairs...")

    # Load the document corpus using IRDS format
    dataset = load_dataset("microsoft/ms_marco", "v1.1")
    validation = dataset["validation"]

    rerank_documents = []
    for entry in validation:
        query = entry[0] if isinstance(entry, (list, tuple)) else entry["query"]
        passages = entry[1] if isinstance(entry, (list, tuple)) else entry["passages"]
        passage_texts = (
            passages["passage_text"] if isinstance(passages, dict) else passages
        )
        for i, doc in enumerate(passage_texts):
            rerank_documents.append({"text": doc, "query": query})

            # Apply max_samples limit if specified
            if max_samples and len(rerank_documents) >= max_samples:
                break
        if max_samples and len(rerank_documents) >= max_samples:
            break

    print(
        f"Loaded {len(rerank_documents)} query-passage pairs from MSMARCO validation set"
    )
    return rerank_documents


def apply_qwen3_template(
    passage_data: List[Dict[str, str]],
    instruction: str | None = None,
) -> List[str]:
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def format_instruction(instruction, query, doc):
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        output = f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}{suffix}"
        return output

    formatted_passages = []
    for passage in passage_data:
        formatted = format_instruction(None, passage["query"], passage["text"])
        formatted_passages.append(formatted)

    return formatted_passages


def get_server_info(base_url: str) -> Dict[str, Any]:
    """Get server info from /info endpoint."""
    try:
        response = requests.get(f"{base_url}/info")
        response.raise_for_status()
        info = response.json()
        radix_mlp_threshold = info.get("radix_mlp_threshold", 0.0)

        return {
            "model_name": info.get("model_id", "Unknown"),
            "model_dtype": info.get("model_dtype", "Unknown"),
            "version": info.get("version", "Unknown"),
            "max_concurrent_requests": info.get("max_concurrent_requests", "Unknown"),
            "max_input_length": info.get("max_input_length", "Unknown"),
            "max_batch_tokens": info.get("max_batch_tokens", "Unknown"),
            "radix_mlp_threshold": radix_mlp_threshold,
            "radix_mlp_enabled": radix_mlp_threshold > 0,
        }
    except Exception as e:
        print(f"Error fetching server info: {e}")
        return {
            "model_name": "Unknown",
            "model_dtype": "Unknown",
            "version": "Unknown",
            "max_concurrent_requests": "Unknown",
            "max_input_length": "Unknown",
            "max_batch_tokens": "Unknown",
            "radix_mlp_threshold": "Unknown",
        }


def run_single_configuration(
    config_name: str,
    base_url: str,
    passages: List[str],
    model_name: str,
    batch_size: int,
    api_key: str = "",
) -> ConfigurationResult:
    """Run embedding test for a single configuration."""
    print(f"\n=== Testing {config_name} ({base_url}) ===")

    # Get server info
    server_info = get_server_info(base_url)
    if server_info["model_name"] == "Unknown":
        return ConfigurationResult(
            name=config_name,
            base_url=base_url,
            server_info=server_info,
            times=[],
            total_time=0.0,
            success=False,
            error_message="Failed to fetch server info",
        )

    # Run embedding
    try:
        response = embed_passages(
            passages=passages,
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            batch_size=batch_size,
        )

        if response is None:
            return ConfigurationResult(
                name=config_name,
                base_url=base_url,
                server_info=server_info,
                times=[],
                total_time=0.0,
                success=False,
                error_message="Embedding failed",
            )

        return ConfigurationResult(
            name=config_name,
            base_url=base_url,
            server_info=server_info,
            times=response.individual_request_times or [],
            total_time=response.total_time or 0.0,
            success=True,
        )

    except Exception as e:
        return ConfigurationResult(
            name=config_name,
            base_url=base_url,
            server_info=server_info,
            times=[],
            total_time=0.0,
            success=False,
            error_message=str(e),
        )


def run_all_configurations_parallel(
    ports: List[int],
    passages: List[str],
    model_name: str,
    batch_size: int,
    api_key: str = "",
) -> List[ConfigurationResult]:
    """Run tests for all configurations in parallel."""
    base_url = "http://localhost"

    # Create configuration names
    configs = []
    for port in ports:
        url = f"{base_url}:{port}"
        # Get server info to determine configuration name
        server_info = get_server_info(url)
        threshold = server_info.get("radix_mlp_threshold", 0.0)
        max_batch_tokens = server_info.get("max_batch_tokens", "Unknown")

        if threshold > 0:
            config_name = f"RadixMLP Enabled ({threshold}, mbt={max_batch_tokens})"
        else:
            config_name = f"RadixMLP Disabled (mbt={max_batch_tokens})"

        configs.append((config_name, url))

    # Run configurations in parallel
    results = []
    with ThreadPoolExecutor(max_workers=len(ports)) as executor:
        # Submit all tasks
        future_to_config = {
            executor.submit(
                run_single_configuration,
                config_name,
                url,
                passages,
                model_name,
                batch_size,
                api_key,
            ): (config_name, url)
            for config_name, url in configs
        }

        # Collect results as they complete
        for future in as_completed(future_to_config):
            config_name, url = future_to_config[future]
            try:
                result = future.result()
                results.append(result)
                status = "✓" if result.success else "✗"
                print(
                    f"{status} {config_name}: {len(result.times)} requests in {result.total_time:.2f}s"
                )
            except Exception as e:
                print(f"✗ {config_name}: Exception - {e}")
                results.append(
                    ConfigurationResult(
                        name=config_name,
                        base_url=url,
                        server_info={},
                        times=[],
                        total_time=0.0,
                        success=False,
                        error_message=str(e),
                    )
                )

    return results


def plot_comparison_violin(
    results: List[ConfigurationResult], output_file: str = "radix_mlp_comparison.png"
):
    """Create violin plot comparison of all configurations."""
    # Filter successful results
    successful_results = [r for r in results if r.success and r.times]

    if not successful_results:
        print("No successful configurations to plot")
        return

    # Set seaborn style
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 8))

    # Prepare data for violin plot
    plot_data = []
    for result in successful_results:
        for time in result.times:
            plot_data.append({"Configuration": result.name, "Time (seconds)": time})

    df = pd.DataFrame(plot_data)

    # Create violin plot
    configs = df["Configuration"].unique()
    colors = []
    for config in configs:
        if "Disabled" in config:
            colors.append("#1f77b4")
        else:
            colors.append("#ff7f0e")
    ax = sns.violinplot(
        data=df,
        x="Configuration",
        y="Time (seconds)",
        hue="Configuration",
        palette=colors,
        inner="box",
        cut=0,
        legend=False,
    )

    # Add statistical annotations
    for i, result in enumerate(successful_results):
        if result.times:
            # Calculate statistics
            p50 = (
                statistics.quantiles(result.times, n=100)[49]
                if len(result.times) > 1
                else result.times[0]
            )
            p90 = (
                statistics.quantiles(result.times, n=100)[89]
                if len(result.times) > 1
                else result.times[0]
            )
            mean = statistics.mean(result.times)

            # Add text annotations
            stats_text = f"P50: {p50:.2f}s\nP90: {p90:.2f}s\nMean: {mean:.2f}s\nTotal: {result.total_time:.2f}s"
            ax.text(
                i,
                max(result.times) * 0.9,
                stats_text,
                ha="center",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=9,
            )

    # Customize plot
    model_name = successful_results[0].server_info.get("model_name", "Unknown Model")
    plt.title(
        f"Mean latency {model_name}\n",
        fontsize=14,
        pad=20,
    )
    plt.xlabel("Configuration", fontsize=12)
    plt.ylabel("Request Time (seconds)", fontsize=12)
    plt.xticks(rotation=15, ha="right")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#1f77b4", label="RadixMLP Disabled"),
        Patch(facecolor="#ff7f0e", label="RadixMLP Enabled"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nComparison plot saved to {output_file}")
    plt.close()

    # Print summary
    print(f"\n=== Configuration Summary ===")
    for result in successful_results:
        if result.times:
            p50 = (
                statistics.quantiles(result.times, n=100)[49]
                if len(result.times) > 1
                else result.times[0]
            )
            p90 = (
                statistics.quantiles(result.times, n=100)[89]
                if len(result.times) > 1
                else result.times[0]
            )
            mean = statistics.mean(result.times)
            threshold = result.server_info.get("radix_mlp_threshold", 0.0)

            print(f"\n{result.name}:")
            print(f"  Port: {result.base_url.split(':')[-1]}")
            print(f"  Radix MLP threshold: {threshold}")
            print(
                f"  Max batch tokens: {result.server_info.get('max_batch_tokens', 'Unknown')}"
            )
            print(f"  Requests: {len(result.times)}")
            print(f"  P50: {p50:.3f}s, P90: {p90:.3f}s, Mean: {mean:.3f}s")
            print(f"  Total duration: {result.total_time:.2f}s")


def save_timing_data(
    results: List[ConfigurationResult], output_file: str = "timing_data.csv"
):
    """Save timing data to CSV file."""
    all_data = []

    for result in results:
        if result.success and result.times:
            for i, time in enumerate(result.times):
                all_data.append(
                    {
                        "Configuration": result.name,
                        "Base_URL": result.base_url,
                        "Request_Index": i,
                        "Time_seconds": time,
                        "Radix_MLP_Threshold": result.server_info.get(
                            "radix_mlp_threshold", ""
                        ),
                        "Max_Batch_Tokens": result.server_info.get(
                            "max_batch_tokens", ""
                        ),
                        "Model_Name": result.server_info.get("model_name", ""),
                    }
                )

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False)
        print(f"Timing data saved to {output_file}")
    else:
        print("No timing data to save")


def plot_latency_percentiles(times: List[float], server_info: Dict[str, Any]):
    """Plot p50 and p90 latencies using seaborn."""
    if not times:
        print("No timing data available for plotting")
        return

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Calculate percentiles
    sorted_times = sorted(times)
    p50 = sorted_times[int(len(sorted_times) * 0.5)]
    p90 = sorted_times[int(len(sorted_times) * 0.9)]

    print(f"\n=== Latency Statistics ===")
    print(f"P50 latency: {p50:.3f} seconds")
    print(f"P90 latency: {p90:.3f} seconds")
    print(f"Average latency: {sum(times) / len(times):.3f} seconds")
    print(f"Min latency: {min(times):.3f} seconds")
    print(f"Max latency: {max(times):.3f} seconds")
    print(f"\n=== Server Information ===")
    print(f"Model: {server_info['model_name']}")
    print(f"Data type: {server_info['model_dtype']}")
    print(f"Version: {server_info['version']}")
    print(f"Max concurrent requests: {server_info['max_concurrent_requests']}")
    print(f"Max input length: {server_info['max_input_length']}")
    print(f"Max batch tokens: {server_info['max_batch_tokens']}")
    print(f"Radix MLP enabled: {server_info.get('radix_mlp_enabled', False)}")
    if server_info.get("radix_mlp_enabled", False):
        print(f"Radix MLP threshold: {server_info.get('radix_mlp_threshold', 0.0)}")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Histogram with KDE
    sns.histplot(data=times, bins=30, kde=True, ax=axes[0], alpha=0.7)
    axes[0].axvline(
        p50, color="red", linestyle="--", linewidth=2, label=f"P50: {p50:.3f}s"
    )
    axes[0].axvline(
        p90, color="orange", linestyle="--", linewidth=2, label=f"P90: {p90:.3f}s"
    )
    axes[0].set_xlabel("Request Time (seconds)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title(
        f"Latency Distribution with KDE - {server_info['model_name']}", fontsize=14
    )
    axes[0].legend(fontsize=11)

    # Plot 2: Box plot with additional info
    sns.boxplot(data=times, ax=axes[1], orient="h", width=0.3)
    axes[1].set_xlabel("Request Time (seconds)", fontsize=12)
    axes[1].set_title("Latency Box Plot", fontsize=14)

    # Add percentile annotations to box plot
    axes[1].text(
        p50,
        0.7,
        f"P50: {p50:.3f}s",
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3),
    )
    axes[1].text(
        p90,
        0.7,
        f"P90: {p90:.3f}s",
        ha="center",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.3),
    )

    # Add server info as figure text
    info_text = f"Model: {server_info['model_name']}\n"
    info_text += f"Data type: {server_info['model_dtype']}\n"
    info_text += f"Version: {server_info['version']}\n"
    info_text += f"Max input length: {server_info['max_input_length']}"
    info_text += f"\nMax batch tokens: {server_info['max_batch_tokens']}\n"
    if server_info.get("radix_mlp_enabled", False):
        info_text += f"Radix MLP enabled: with threshold {server_info.get('radix_mlp_threshold', 0.0)}\n"
    else:
        info_text += f"Radix MLP enabled: False\n"

    fig.text(
        0.02,
        0.98,
        info_text,
        transform=fig.transFigure,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()

    # Save plot
    plot_filename = "latency_distribution.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"\nLatency plot saved to {plot_filename}")
    plt.close()


def embed_passages(
    passages: List[str],
    base_url: str = "http://localhost:8000",
    api_key: str = "",
    model_name: str = "qwen3",
    batch_size: int = 64,
):
    """Embed passages using baseten_performance_client."""
    print(f"Embedding {len(passages)} passages...")

    # Create performance client
    client = PerformanceClient(base_url=base_url, api_key=api_key, http_version=1)

    # Prepare request preference
    preference = RequestProcessingPreference(
        batch_size=batch_size,
        max_concurrent_requests=16,
        timeout_s=600.0,
    )
    try:
        # Use batch_post for embedding
        response = client.embed(
            model=model_name,
            input=passages,
            preference=preference,
        )

        print(
            f"Successfully embedded {len(passages)} passages in {response.total_time:.2f} seconds"
        )
        return response

    except Exception as e:
        print(f"Error embedding passages: {e}")
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Embed MSMARCO passages and compare configurations"
    )
    parser.add_argument(
        "--max-samples", type=int, help="Maximum number of passages to load"
    )
    parser.add_argument(
        "--ports",
        default="3000,3001,3002,3003",
        help="Comma-separated list of ports to test (default: 3000,3001,3002,3003)",
    )
    parser.add_argument(
        "--single-port",
        type=int,
        help="Test single port only (disables comparison mode)",
    )
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--model", default="qwen3", help="Model name")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for embedding"
    )
    parser.add_argument(
        "--output-plot", default="radix_mlp_comparison.png", help="Output plot file"
    )
    parser.add_argument(
        "--save-data", default="timing_data.csv", help="Save timing data to CSV file"
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Custom prefix",
    )
    args = parser.parse_args()

    # Load query-passage pairs
    passage_data = load_msmarco_passages(args.max_samples)
    if not passage_data:
        print("No passage data loaded. Exiting.")
        return

    # Apply Qwen3 template
    formatted_passages = apply_qwen3_template(passage_data, args.prefix)
    print(f"Applied Qwen3 template to {len(formatted_passages)} passages")

    # Show example
    print(f"\nExample formatted passage:")
    print(formatted_passages[0][:200] + "...")

    # Determine mode: single port or comparison
    if args.single_port:
        # Single port mode (backward compatibility)
        base_url = f"http://localhost:{args.single_port}"
        print(f"\n=== Single Port Mode: {base_url} ===")

        server_info = get_server_info(base_url)
        embeddings = embed_passages(
            formatted_passages,
            base_url,
            args.api_key or "",
            args.model,
            args.batch_size,
        )

        if embeddings is None:
            print("Failed to embed passages")
            return

        print("Performance stats:")
        print(f"Total time: {embeddings.total_time:.2f} seconds")
        times = embeddings.individual_request_times

        if times:
            plot_latency_percentiles(times, server_info)
        else:
            print("No timing data available")

    else:
        # Comparison mode (default)
        ports = [int(p.strip()) for p in args.ports.split(",")]
        print(f"\n=== Comparison Mode: Testing ports {ports} ===")

        # Run all configurations in parallel
        results = run_all_configurations_parallel(
            ports=ports,
            passages=formatted_passages,
            model_name=args.model,
            batch_size=args.batch_size,
            api_key=args.api_key or "",
        )

        # Generate comparison plot
        plot_comparison_violin(results, args.output_plot)

        # Save timing data
        save_timing_data(results, args.save_data)

        # Print failed configurations
        failed_results = [r for r in results if not r.success]
        if failed_results:
            print(f"\n=== Failed Configurations ===")
            for result in failed_results:
                print(f"✗ {result.name}: {result.error_message}")


if __name__ == "__main__":
    main()
