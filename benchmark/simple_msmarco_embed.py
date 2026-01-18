#!/usr/bin/env python3
"""
Simple MSMARCO Passage Embedding Script

Loads MSMARCO passages and prepares them for embedding with Qwen3 chat template.
"""

import os
import json
import argparse
import re
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


def create_output_directory(model_name: str, base_dir: str = "results") -> str:
    """Create organized output directory structure for model results."""
    # Sanitize model name for directory
    safe_model_name = model_name.lower().replace("-", "_").replace(" ", "_")
    model_dir = os.path.join(base_dir, safe_model_name)

    # Create subdirectories
    subdirs = ["plots", "data", "latex"]
    for subdir in subdirs:
        os.makedirs(os.path.join(model_dir, subdir), exist_ok=True)

    print(f"Created output directory: {model_dir}")
    return model_dir


def save_plots_multiple_formats(
    results: List["ConfigurationResult"],
    base_path: str,
    base_name: str,
    plot_formats: Optional[List[str]] = None,
    plot_dpi: int = 600,
):
    """Save plots in multiple formats (SVG, PNG, PDF)."""
    if plot_formats is None:
        plot_formats = ["svg", "png", "pdf"]

    # Filter successful results
    successful_results = [r for r in results if r.success and r.times]

    if not successful_results:
        print("No successful configurations to plot")
        return

    # Generate the plot first
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 8))

    # Prepare data for violin plot
    plot_data = []
    for result in successful_results:
        for time in result.times:
            plot_data.append({"Configuration": result.name, "Time (seconds)": time})

    df = pd.DataFrame(plot_data)

    # Create violin plot
    colors = [
        "#1f77b4" if "Disabled" in config else "#ff7f0e"
        for config in df["Configuration"].unique()
    ]
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
        f"Latency Distribution Comparison - {model_name}\n(Violin Plot: Full Distribution Shape)",
        fontsize=14,
        pad=20,
    )
    plt.xlabel("Configuration", fontsize=12)
    plt.ylabel("Request Time (seconds)", fontsize=12)
    plt.xticks(rotation=0, ha="right")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#1f77b4", label="Radix MLP Disabled"),
        Patch(facecolor="#ff7f0e", label="Radix MLP Enabled"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    # Save in multiple formats
    plots_dir = os.path.join(base_path, "plots")
    for fmt in plot_formats:
        filename = os.path.join(plots_dir, f"{base_name}.{fmt}")

        if fmt == "png":
            plt.savefig(filename, format=fmt, dpi=plot_dpi, bbox_inches="tight")
        else:
            plt.savefig(filename, format=fmt, bbox_inches="tight")

        print(f"Saved plot as {filename}")

    plt.close()


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    # Replace LaTeX special characters with escaped versions
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("$", r"\$")
    text = text.replace("#", r"\#")
    text = text.replace("_", r"\_")
    text = text.replace("^", r"\^{}")
    text = text.replace("{", r"\{")
    text = text.replace("}", r"\}")
    text = text.replace("~", r"\textasciitilde{}")
    text = text.replace("\\", r"\textbackslash{}")
    return text


def export_latex_table(
    results: List["ConfigurationResult"],
    output_path: str,
    filename: str = "results_table.tex",
):
    """Export results as LaTeX table using booktabs style."""
    # Filter successful results
    successful_results = [r for r in results if r.success and r.times]

    if not successful_results:
        print("No successful configurations for LaTeX table")
        return

    # Generate LaTeX table content
    latex_content = []
    latex_content.append("% LaTeX table generated by simple_msmarco_embed.py")
    latex_content.append("% Use with: \\input{results_table.tex}")
    latex_content.append("")
    latex_content.append("\\begin{tabular}{lcccccc}")
    latex_content.append("\\toprule")
    latex_content.append(
        "Configuration & Max Batch Tokens & Threshold & Requests & P50 (s) & P90 (s) & Mean (s) \\\\"
    )
    latex_content.append("\\midrule")

    for result in successful_results:
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

            # Get configuration info
            config_name = escape_latex(result.name)
            max_batch_tokens = result.server_info.get("max_batch_tokens", "Unknown")
            threshold = result.server_info.get("radix_mlp_threshold", 0.0)
            num_requests = len(result.times)

            # Format table row
            row = f"{config_name} & {max_batch_tokens} & {threshold} & {num_requests} & {p50:.2f} & {p90:.2f} & {mean:.2f} \\\\"
            latex_content.append(row)

    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")

    # Write to file
    output_file = os.path.join(output_path, filename)
    with open(output_file, "w") as f:
        f.write("\n".join(latex_content))

    print(f"LaTeX table saved to {output_file}")


def export_latex_summary(
    results: List["ConfigurationResult"],
    plot_filename: str,
    output_path: str,
    filename: str = "results_summary.tex",
):
    """Export complete LaTeX document with results summary."""
    # Filter successful results
    successful_results = [r for r in results if r.success and r.times]

    if not successful_results:
        print("No successful configurations for LaTeX summary")
        return

    # Get model info
    model_name = successful_results[0].server_info.get("model_name", "Unknown Model")
    model_dtype = successful_results[0].server_info.get("model_dtype", "Unknown")

    # Generate LaTeX document
    latex_content = []
    latex_content.append("% LaTeX document generated by simple_msmarco_embed.py")
    latex_content.append("")
    latex_content.append("\\documentclass[11pt,a4paper]{article}")
    latex_content.append("\\usepackage[utf8]{inputenc}")
    latex_content.append("\\usepackage{graphicx}")
    latex_content.append("\\usepackage{booktabs}")
    latex_content.append("\\usepackage{amsmath}")
    latex_content.append("\\usepackage{svg}")
    latex_content.append("")
    latex_content.append(
        "\\title{Performance Benchmark: " + escape_latex(model_name) + "}"
    )
    latex_content.append("\\author{Radix MLP Benchmark}")
    latex_content.append("\\date{\\today}")
    latex_content.append("")
    latex_content.append("\\begin{document}")
    latex_content.append("")
    latex_content.append("\\maketitle")
    latex_content.append("")
    latex_content.append("\\section{Methodology}")
    latex_content.append(
        "This benchmark evaluates the performance of "
        + escape_latex(model_name)
        + " ("
        + escape_latex(model_dtype)
        + ") with and without Radix MLP optimization. "
    )
    latex_content.append(
        "The test uses MSMARCO validation query-passage pairs formatted with the Qwen3 chat template. "
    )
    latex_content.append(
        "Latency measurements are collected for individual embedding requests and analyzed using percentile statistics."
    )
    latex_content.append("")
    latex_content.append("\\section{Results}")
    latex_content.append("")
    latex_content.append("\\subsection{Performance Comparison}")
    latex_content.append("")
    latex_content.append("\\begin{table}[h]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Performance comparison across configurations}")
    latex_content.append("\\input{results_table.tex}")
    latex_content.append("\\end{table}")
    latex_content.append("")
    latex_content.append("\\subsection{Latency Distribution}")
    latex_content.append("")
    latex_content.append("\\begin{figure}[h]")
    latex_content.append("\\centering")
    latex_content.append("\\includesvg{" + plot_filename + "}")
    latex_content.append(
        "\\caption{Latency distribution comparison across configurations}"
    )
    latex_content.append("\\label{fig:latency_comparison}")
    latex_content.append("\\end{figure}")
    latex_content.append("")
    latex_content.append("\\section{Analysis}")
    latex_content.append("")

    # Add analysis based on results
    enabled_configs = [r for r in successful_results if "Enabled" in r.name]
    disabled_configs = [r for r in successful_results if "Disabled" in r.name]

    if enabled_configs and disabled_configs:
        enabled_mean = statistics.mean(
            [statistics.mean(r.times) for r in enabled_configs]
        )
        disabled_mean = statistics.mean(
            [statistics.mean(r.times) for r in disabled_configs]
        )
        improvement = ((disabled_mean - enabled_mean) / disabled_mean) * 100

        latex_content.append(
            f"The Radix MLP optimization shows an average performance improvement of {improvement:.1f}\\% "
        )
        latex_content.append(
            f"compared to the baseline configuration (mean latency: {enabled_mean:.2f}s vs {disabled_mean:.2f}s)."
        )
    else:
        latex_content.append(
            "Performance analysis requires both enabled and disabled configurations for comparison."
        )

    latex_content.append("")
    latex_content.append("\\end{document}")

    # Write to file
    output_file = os.path.join(output_path, filename)
    with open(output_file, "w") as f:
        f.write("\n".join(latex_content))

    print(f"LaTeX summary saved to {output_file}")


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


def load_msmarco_passages(split: str, max_samples: Optional[int] = None) -> List[Dict[str, str]]:
    """Load MSMARCO query-passage pairs from validation set."""
    print("Loading MSMARCO query-passage pairs...")

    # Load the document corpus using IRDS format
    dataset = load_dataset("microsoft/ms_marco", "v1.1")
    validation = dataset[split]

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
    swap_docs_queries: bool = False,
) -> List[str]:
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def format_instruction(instruction, query, doc):
        if swap_docs_queries:
            # augmentation technique, swap query and document for different kind of benchmark.
            query, doc = doc, query 
        if instruction is None:
            instruction = "<Instruct>: Given a web search query, retrieve relevant passages that answer the query"
        output = f"{prefix}{instruction}\n<Query>: {query}\n<Document>: {doc}{suffix}"
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
            "model_name": "Qwen/Qwen-3-Embedding",
            "model_dtype": "fp16",
            "version": "vllm",
            "max_concurrent_requests": "NA",
            "max_input_length": "Unknown",
            "max_batch_tokens": "16384,vllm=True",
            "radix_mlp_threshold": 0.0,
            "radix_mlp_enabled": False,
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
    results: List["ConfigurationResult"],
    output_file: str = "radix_mlp_comparison.png",
    output_dir: Optional[str] = None,
    plot_formats: Optional[List[str]] = None,
    plot_dpi: int = 300,
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
            name_pretty = result.name.replace(" (", "\n(")
            plot_data.append({"Configuration": result.name, "Time (seconds)": time})

    df = pd.DataFrame(plot_data)

    # Create violin plot
    colors = [
        "#1f77b4" if "Disabled" in config else "#ff7f0e"
        for config in df["Configuration"].unique()
    ]
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
        f"Latency Distribution Comparison - {model_name}\n",
        fontsize=14,
        pad=20,
    )
    plt.xlabel("Configuration", fontsize=12)
    plt.ylabel("Request Time (seconds)", fontsize=12)
    plt.xticks(rotation=0, ha="center")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#1f77b4", label="Radix MLP Disabled"),
        Patch(facecolor="#ff7f0e", label="Radix MLP Enabled"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    # Determine output path and formats
    if output_dir and plot_formats:
        # Save in multiple formats to specified directory
        base_name = os.path.splitext(os.path.basename(output_file))[0]
        for fmt in plot_formats:
            filename = os.path.join(output_dir, f"{base_name}.{fmt}")
            if fmt == "png":
                plt.savefig(filename, format=fmt, dpi=plot_dpi, bbox_inches="tight")
            else:
                plt.savefig(filename, format=fmt, bbox_inches="tight")
            print(f"Comparison plot saved as {filename}")
    else:
        # Save single format to original file
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
            print(f"  P50: {p50:.2f}s, P90: {p90:.2f}s, Mean: {mean:.2f}s")
            print(f"  Total duration: {result.total_time:.2f}s")


def save_timing_data(
    results: List["ConfigurationResult"],
    output_file: str = "timing_data.csv",
    output_dir: Optional[str] = None,
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

        # Determine output path
        if output_dir:
            output_path = os.path.join(output_dir, output_file)
        else:
            output_path = output_file

        df.to_csv(output_path, index=False)
        print(f"Timing data saved to {output_path}")
    else:
        print("No timing data to save")


def plot_latency_percentiles(
    times: List[float],
    server_info: Dict[str, Any],
    output_dir: Optional[str] = None,
    plot_formats: Optional[List[str]] = None,
    plot_dpi: int = 300,
):
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
    print(f"P50 latency: {p50:.2f} seconds")
    print(f"P90 latency: {p90:.2f} seconds")
    print(f"Average latency: {sum(times) / len(times):.2f} seconds")
    print(f"Min latency: {min(times):.2f} seconds")
    print(f"Max latency: {max(times):.2f} seconds")
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
        p50, color="red", linestyle="--", linewidth=2, label=f"P50: {p50:.2f}s"
    )
    axes[0].axvline(
        p90, color="orange", linestyle="--", linewidth=2, label=f"P90: {p90:.2f}s"
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
        f"P50: {p50:.2f}s",
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3),
    )
    axes[1].text(
        p90,
        0.7,
        f"P90: {p90:.2f}s",
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

    # Determine output path and formats
    if output_dir and plot_formats:
        # Save in multiple formats to specified directory
        base_name = "latency_distribution"
        for fmt in plot_formats:
            filename = os.path.join(output_dir, f"{base_name}.{fmt}")
            if fmt == "png":
                plt.savefig(filename, format=fmt, dpi=plot_dpi, bbox_inches="tight")
            else:
                plt.savefig(filename, format=fmt, bbox_inches="tight")
            print(f"Latency plot saved as {filename}")
    else:
        # Save single format to original location
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
    parser.add_argument(
        "--export-latex",
        action="store_true",
        help="Export LaTeX table and summary document",
    )
    parser.add_argument(
        "--plot-formats",
        default="svg,png,pdf",
        help="Comma-separated plot formats (default: svg,png,pdf)",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=600,
        help="DPI for PNG plots (default: 600)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Base output directory (default: results)",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="MSMARCO split to load (default: validation), e.g., 'train', 'validation', 'test'",
    )
    parser.add_argument(
        "--swap-docs-queries",
        action="store_true",
        help="Swap documents and queries in the MSMARCO dataset",
    )
    args = parser.parse_args()

    # Parse plot formats
    plot_formats = [fmt.strip() for fmt in args.plot_formats.split(",")]

    # Create output directory structure
    model_dir = create_output_directory(args.model, args.output_dir)
    plots_dir = os.path.join(model_dir, "plots")
    data_dir = os.path.join(model_dir, "data")
    latex_dir = os.path.join(model_dir, "latex")

    # Load query-passage pairs
    passage_data = load_msmarco_passages(args.split, args.max_samples)
    if not passage_data:
        print("No passage data loaded. Exiting.")
        return

    # Apply Qwen3 template
    formatted_passages = apply_qwen3_template(passage_data, args.prefix, swap_docs_queries=args.swap_docs_queries)
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
            # Use new organized output structure
            plot_latency_percentiles(
                times,
                server_info,
                output_dir=plots_dir,
                plot_formats=plot_formats,
                plot_dpi=args.plot_dpi,
            )

            # Export LaTeX if requested
            if args.export_latex:
                # Create a single result for LaTeX export
                from dataclasses import dataclass

            # Create a single result for LaTeX export
            single_result = ConfigurationResult(
                name=f"Single Port ({args.single_port})",
                base_url=base_url,
                server_info=server_info,
                times=times,
                total_time=embeddings.total_time or 0.0,
                success=True,
            )
            export_latex_table([single_result], latex_dir)
            export_latex_summary([single_result], "latency_distribution", latex_dir)
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

        # Generate comparison plot with new organized structure
        plot_comparison_violin(
            results,
            args.output_plot,
            output_dir=plots_dir,
            plot_formats=plot_formats,
            plot_dpi=args.plot_dpi,
        )

        # Save timing data to organized location
        save_timing_data(results, args.save_data, output_dir=data_dir)

        # Export LaTeX if requested
        if args.export_latex:
            export_latex_table(results, latex_dir)
            export_latex_summary(results, "radix_mlp_comparison", latex_dir)

        # Print failed configurations
        failed_results = [r for r in results if not r.success]
        if failed_results:
            print(f"\n=== Failed Configurations ===")
            for result in failed_results:
                print(f"✗ {result.name}: {result.error_message}")


if __name__ == "__main__":
    main()
