#!/usr/bin/env python3
"""
Simple MSMARCO Passage Embedding Script

Loads MSMARCO passages and prepares them for embedding with Qwen3 chat template.
"""

import os
import json
import argparse
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from typing import List, Dict, Any

from datasets import load_dataset
from baseten_performance_client import (
    PerformanceClient,
    RequestProcessingPreference,
)


def load_msmarco_passages(max_samples: int = None) -> List[Dict[str, str]]:
    """Load MSMARCO query-passage pairs from validation set."""
    print("Loading MSMARCO query-passage pairs...")

    # Load the document corpus using IRDS format
    dataset = load_dataset("microsoft/ms_marco", "v1.1")
    validation = dataset["validation"]

    rerank_documents = []
    for entry in validation:
        query = entry["query"]
        for i, doc in enumerate(entry["passages"]["passage_text"]):
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
        }
    except Exception as e:
        print(f"Error fetching server info: {e}")
        return {"model_name": "Unknown", "radix_mlp_threshold": "Unknown"}


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
    parser = argparse.ArgumentParser(description="Embed MSMARCO passages")
    parser.add_argument(
        "--max-samples", type=int, help="Maximum number of passages to load"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:3000",
        help="Base URL for embedding service",
    )
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--model", default="qwen3", help="Model name")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for embedding"
    )
    parser.add_argument(
        "--output", default="msmarco_embeddings.json", help="Output file"
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

    # Get server info first
    server_info = get_server_info(args.base_url)

    # Embed passages
    embeddings = embed_passages(
        formatted_passages,
        args.base_url,
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
        # Plot latency percentiles instead of just average
        plot_latency_percentiles(times, server_info)
    else:
        print("No timing data available")


if __name__ == "__main__":
    main()
