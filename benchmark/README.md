# MSMARCO Passage Embedding

Simple script to load MSMARCO passages and embed them using Qwen3 chat template with baseten_performance_client.

## Files

- `simple_msmarco_embed.py` - Main script for loading and embedding MSMARCO passages
- `requirements.txt` - Dependencies

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Embed all MSMARCO passages
python simple_msmarco_embed.py

# Embed limited number of passages
python simple_msmarco_embed.py --max-samples 1000

# Load and embed dev queries
python simple_msmarco_embed.py --load-queries

# Custom server and model
python simple_msmarco_embed.py --base-url http://localhost:8000 --model qwen3-7b

# Custom batch size
python simple_msmarco_embed.py --batch-size 64
```

### Command Line Arguments

- `--max-samples` - Maximum number of passages to load (default: all)
- `--load-queries` - Load dev queries instead of passages
- `--base-url` - Base URL for embedding service (default: http://localhost:8000)
- `--api-key` - API key for authentication
- `--model` - Model name (default: qwen3)
- `--batch-size` - Batch size for embedding (default: 32)
- `--output` - Output JSON file (default: msmarco_embeddings.json)
- `--prefix` - Custom prefix for embedding (default: Bing query dataset prefix)

### Environment Variables

```bash
export BASE_URL="http://localhost:8000"
export API_KEY="your-api-key"
export MODEL_NAME="qwen3"
python simple_msmarco_embed.py
```

## Qwen3 Chat Template

The script automatically applies the Qwen3 chat template:

```
<|im_start|>system
You are a helpful assistant specialized in embedding text for retrieval tasks.<|im_end|>
<|im_start|>user
embed the following sentences that is part of bing query dataset, with target to help find relevant web documents. {passage}<|im_end|>
<|im_start|>assistant
```

## Output

The script generates a JSON file with:
- `passages` - Original MSMARCO passages
- `formatted_passages` - Passages with Qwen3 template applied
- `embeddings` - Embedding results from the service
- `config` - Configuration used

## Example

```bash
# Embed passages
python simple_msmarco_embed.py --max-samples 100 --batch-size 16

# Embed dev queries
python simple_msmarco_embed.py --load-queries
```

Output:
```
Loading MSMARCO passages...
Loaded 8841827 passages
Limited to 100 passages
Applied Qwen3 template to 100 passages
Embedding 100 passages...
Successfully embedded 100 passages
Results saved to msmarco_embeddings.json
Processed 100 passages
```

## Data Sources

The script uses IRDS format to load MSMARCO data:

- **Passages**: `irds/msmarco-passage` (docs) - Full document corpus
- **Dev Queries**: `irds/msmarco-passage_dev` (queries) - Development set queries