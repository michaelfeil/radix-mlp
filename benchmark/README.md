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

# Enhanced MSMARCO Query-Passage Embedding

## 🆕 New Features

### 📁 Organized Output Structure
Results are now organized by model name in a clean directory structure:
```
results/
├── qwen3/
│   ├── plots/          # SVG, PNG, PDF plots
│   ├── data/           # CSV timing data
│   └── latex/          # LaTeX tables and documents
├── qwen3_7b/
└── [other_models]/
```

### 📊 Multiple Plot Formats
- **SVG**: Vector format with unlimited scalability (recommended)
- **PNG**: High-resolution raster format (configurable DPI, default 600)
- **PDF**: LaTeX-friendly vector format

### 📝 LaTeX Export
- **Professional tables** using booktabs package
- **Complete documents** with methodology and analysis
- **Automatic escaping** of LaTeX special characters
- **Ready for publication** in academic papers

### 🎛️ Enhanced Configuration Support
- **max_batch_tokens** information included in all outputs
- **Detailed server info** extracted and displayed
- **Better configuration naming** with all relevant parameters

## 🚀 Usage

### Basic Usage (Organized Output)
```bash
# Run with organized output by model name
python simple_msmarco_embed.py --model qwen3-7b
```

### Advanced Usage (All Features)
```bash
python simple_msmarco_embed.py \
    --model qwen3-7b \
    --export-latex \
    --plot-formats svg,png,pdf \
    --plot-dpi 1200 \
    --output-dir results \
    --max-samples 1000
```

### 📋 New Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--export-latex` | Export LaTeX table and summary document | False |
| `--plot-formats` | Comma-separated plot formats | svg,png,pdf |
| `--plot-dpi` | DPI for PNG plots | 600 |
| `--output-dir` | Base output directory | results |

## 📁 Output Structure

### Directory Organization
```
results/qwen3_7b/
├── plots/
│   ├── radix_mlp_comparison.svg     # Main comparison plot
│   ├── radix_mlp_comparison.png     # High-res PNG (600 DPI)
│   ├── radix_mlp_comparison.pdf     # LaTeX-friendly PDF
│   ├── latency_distribution.svg     # Single port latency plot
│   ├── latency_distribution.png
│   └── latency_distribution.pdf
├── data/
│   └── timing_data.csv              # Raw timing data
└── latex/
    ├── results_table.tex            # Booktabs-style table
    └── results_summary.tex          # Complete document
```

### LaTeX Integration

#### Table Only
```latex
\documentclass{article}
\usepackage{booktabs}
\begin{document}
\input{results/qwen3_7b/latex/results_table.tex}
\end{document}
```

#### Complete Document
```bash
# Compile the complete summary
pdflatex results/qwen3_7b/latex/results_summary.tex
```

#### With SVG Plots
```latex
\documentclass{article}
\usepackage{svg}
\begin{document}
\input{results/qwen3_7b/latex/results_summary.tex}
\end{document}
```

## 📊 Enhanced Metrics

### Configuration Naming
Configurations now include all relevant parameters:
- `Radix MLP Enabled (threshold=0.5, max_batch_tokens=8192)`
- `Radix MLP Disabled (max_batch_tokens=4096)`

### LaTeX Table Format
```latex
\begin{tabular}{lcccccc}
\toprule
Configuration & Max Batch Tokens & Threshold & Requests & P50 (s) & P90 (s) & Mean (s) \\
\midrule
Radix MLP Enabled (threshold=0.5, max_batch_tokens=8192) & 8192 & 0.5 & 1000 & 0.123 & 0.456 & 0.234 \\
...
\bottomrule
\end{tabular}
```

### Enhanced CSV Output
Additional columns in timing_data.csv:
- `Max_Batch_Tokens`
- `Radix_MLP_Threshold`
- `Model_Name`

## 🎯 Quality Settings

### Plot Quality
- **SVG**: Unlimited resolution, vector format
- **PNG**: Configurable DPI (default 600, recommended 1200 for print)
- **PDF**: Vector format, ideal for LaTeX

### LaTeX Quality
- **Booktabs**: Professional table formatting
- **Automatic escaping**: Handles special characters
- **Complete documents**: Ready for publication

## 📖 Examples

### Example 1: Quick Benchmark
```bash
python simple_msmarco_embed.py --model qwen3 --max-samples 100
```
Creates: `results/qwen3/` with basic plots and data

### Example 2: Publication-Ready Results
```bash
python simple_msmarco_embed.py \
    --model qwen3-7b \
    --export-latex \
    --plot-formats svg,png \
    --plot-dpi 1200
```
Creates: Publication-ready LaTeX document + high-res plots

### Example 3: Custom Organization
```bash
python simple_msmarco_embed.py \
    --model custom-model \
    --output-dir my_results \
    --export-latex
```
Creates: `my_results/custom_model/` with LaTeX export

## 🔧 Technical Details

### Plot Formats Comparison
| Format | Advantages | Use Case |
|--------|------------|----------|
| SVG | Unlimited scalability, small file size, crisp text | Web, presentations, modern LaTeX |
| PNG | Universal compatibility, high resolution | Print, compatibility |
| PDF | LaTeX integration, vector format | Academic papers |

### LaTeX Features
- **Booktabs**: Professional table styling
- **SVG support**: Modern vector graphics
- **Automatic analysis**: Performance improvement calculations
- **Special character handling**: Underscores, symbols, etc.

### Directory Structure Benefits
- **Model isolation**: Each model gets its own directory
- **Format separation**: Plots, data, LaTeX in separate subdirs
- **Scalability**: Easy to add new models and formats
- **Clean workspace**: No clutter in main directory

## 📝 Migration from Previous Version

### Breaking Changes
- Output files now organized in model-specific directories
- Plot filenames may have different extensions based on formats

### Backward Compatibility
- All original command line arguments still work
- Single port mode still supported
- Original CSV format maintained (with additional columns)

### Recommended Migration
```bash
# Old command (still works)
python simple_msmarco_embed.py --ports 3000,3001

# New equivalent command
python simple_msmarco_embed.py --model qwen3 --export-latex
```
