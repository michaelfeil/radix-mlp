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
    """Load MSMARCO passages from IRDS."""
    print("Loading MSMARCO passages...")

    # Load the document corpus using IRDS format
    dataset = load_dataset("microsoft/ms_marco", "v1.1")
    validation = dataset["validation"]

    rerank_documents = []
    for entry in validation:
        query = entry["query"]
        for i, doc in enumerate(entry["passages"]["passage_text"]):
            rerank_documents.append({"text": doc, "query": query})

    print(f"Loaded {len(rerank_documents)} passages from MSMARCO validation set")
    return rerank_documents
    


def apply_qwen3_template(
    passages: List[str],
    prefix: str = "embed the following sentences that is part of bing query dataset, with target to help find relevant web documents.",
) -> List[str]:
    """Apply Qwen3 chat template to passages."""
    template = """<|im_start|>system
You are a helpful assistant specialized in embedding text for retrieval tasks.<|im_end|>
<|im_start|>user
{prefix} {passage}<|im_end|>
<|im_start|>assistant
"""

    formatted_passages = []
    for passage in passages:
        formatted = template.format(prefix=passage["query"], passage=passage["text"])
        formatted_passages.append(formatted)

    return formatted_passages


def embed_passages(
    passages: List[str],
    base_url: str = "http://localhost:8000",
    api_key: str = "",
    model_name: str = "qwen3",
    batch_size: int = 128,
) -> List[Dict[str, Any]]:
    """Embed passages using baseten_performance_client."""
    print(f"Embedding {len(passages)} passages...")

    # Create performance client
    client = PerformanceClient(base_url=base_url, api_key=api_key, http_version=1)

    # Prepare request preference
    preference = RequestProcessingPreference(
        batch_size=batch_size,
        max_concurrent_requests=256,
        timeout_s=600.0,
    )
    try:
        # Use batch_post for embedding
        response = client.embed(
            model=model_name,
            input=passages,
            preference=preference,
        )

        print(f"Successfully embedded {len(response.data)} passages")
        return response.data

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
        "--batch-size", type=int, default=32, help="Batch size for embedding"
    )
    parser.add_argument(
        "--output", default="msmarco_embeddings.json", help="Output file"
    )
    parser.add_argument(
        "--prefix",
        default="embed the following sentences that is part of bing query dataset, with target to help find relevant web documents.",
        help="Custom prefix",
    )
    parser.add_argument(
        "--load-queries",
        action="store_true",
        help="Load dev queries instead of passages",
    )

    args = parser.parse_args()

    # Load data
    if args.load_queries:
        # Load dev queries
        queries = load_dev_queries()
        if not queries:
            print("No queries loaded. Exiting.")
            return

        data = queries
        data_type = "queries"
        print(f"Loaded {len(queries)} dev queries")
    else:
        # Load passages
        passages = load_msmarco_passages(args.max_samples)
        if not passages:
            print("No passages loaded. Exiting.")
            return

        data = passages
        data_type = "passages"

    # Apply Qwen3 template
    formatted_data = apply_qwen3_template(data, args.prefix)
    print(f"Applied Qwen3 template to {len(formatted_data)} {data_type}")

    # Show example
    print(f"\nExample formatted {data_type[:-1]}:")
    print(formatted_data[0][:200] + "...")

    # Embed data
    embeddings = embed_passages(
        formatted_data,
        args.base_url,
        args.api_key or "",
        args.model,
        args.batch_size,
    )

    if embeddings:
        # Save results
        results = {
            data_type: data,
            f"formatted_{data_type}": formatted_data,
            "embeddings": embeddings,
            "config": {
                "max_samples": args.max_samples,
                "base_url": args.base_url,
                "model": args.model,
                "batch_size": args.batch_size,
                "prefix": args.prefix,
                "load_queries": args.load_queries,
            },
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {args.output}")
        print(f"Processed {len(data)} {data_type}")
    else:
        print("Embedding failed. No results to save.")


if __name__ == "__main__":
    main()
