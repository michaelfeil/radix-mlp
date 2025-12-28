# MSMARCO Query-Passage Embedding

Simple script to load MSMARCO query-passage pairs and embed them using Qwen3 chat template with baseten_performance_client.

## Files

- `simple_msmarco_embed.py` - Main script for loading and embedding MSMARCO query-passage pairs
- `requirements.txt` - Dependencies

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Embed all MSMARCO query-passage pairs from validation set
python simple_msmarco_embed.py

# Embed limited number of pairs
python simple_msmarco_embed.py --max-samples 1000

# Custom server and model
python simple_msmarco_embed.py --base-url http://localhost:8000 --model qwen3-7b

# Custom batch size
python simple_msmarco_embed.py --batch-size 64
```

### Command Line Arguments

- `--max-samples` - Maximum number of query-passage pairs to load (default: all)
- `--base-url` - Base URL for embedding service (default: http://localhost:3000)
- `--api-key` - API key for authentication
- `--model` - Model name (default: qwen3)
- `--batch-size` - Batch size for embedding (default: 128)
- `--output` - Output JSON file (default: msmarco_embeddings.json)
- `--prefix` - Custom prefix for embedding (default: Bing query dataset prefix)

### Environment Variables

```bash
export BASE_URL="http://localhost:3000"
export API_KEY="your-api-key"
export MODEL_NAME="qwen3"
python simple_msmarco_embed.py
```

## Qwen3 Chat Template

The script automatically applies the Qwen3 chat template to each query-passage pair:

```
<|im_start|>system
You are a helpful assistant specialized in embedding text for retrieval tasks.<|im_end|>
<|im_start|>user
{query} embed the following sentences that is part of bing query dataset, with target to help find relevant web documents. {passage}<|im_end|>
<|im_start|>assistant
```

## Data Source

The script loads MSMARCO validation data from:
- **Dataset**: `microsoft/ms_marco` (v1.1)
- **Split**: `validation`
- **Format**: Query-passage pairs for reranking

Each validation entry contains a query and multiple passages. The script extracts all query-passage pairs for embedding.

## Output

The script generates a JSON file with:
- `passage_data` - Original query-passage pairs with `query` and `text` fields
- `formatted_passages` - Query-passage pairs with Qwen3 template applied
- `embeddings` - Embedding results from the service
- `config` - Configuration used

## Example

```bash
python simple_msmarco_embed.py --max-samples 100 --batch-size 64
```

Output:
```
Loading MSMARCO query-passage pairs...
Loaded 100 query-passage pairs from MSMARCO validation set
Applied Qwen3 template to 100 passages
Embedding 100 passages...
Successfully embedded 100 passages
Results saved to msmarco_embeddings.json
Processed 100 query-passage pairs
```