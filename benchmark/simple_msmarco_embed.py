#!/usr/bin/env python3
"""
Simple MSMARCO Passage Embedding Script

Loads MSMARCO passages and prepares them for embedding with Qwen3 chat template.
"""

import os
import json
import argparse
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


def embed_passages(
    passages: List[str],
    base_url: str = "http://localhost:8000",
    api_key: str = "",
    model_name: str = "qwen3",
    batch_size: int = 64,
) -> List[Dict[str, Any]]:
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
        return []


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

    # Embed passages
    embeddings = embed_passages(
        formatted_passages,
        args.base_url,
        args.api_key or "",
        args.model,
        args.batch_size,
    )
    print("Performance stats:")
    print(embeddings.total_time)
    times = embeddings.individual_request_times
    print(f"Average batch request time: {sum(times) / len(times):.2f} seconds")


if __name__ == "__main__":
    main()
