# blockrank-rag

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2510.05396-b31b1b.svg)](https://arxiv.org/abs/2510.05396)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Model-yellow)](https://huggingface.co/quicktensor/blockrank-msmarco-mistral-7b)

**Efficient, scalable In-Context Ranking (ICR) for RAG using the BlockRank technique.**

This is a clean, production-oriented implementation and toolkit inspired by the paper:

> **Scalable In-context Ranking with Generative Models** (Gupta et al., arXiv:2510.05396, 2025)

Original research repo: https://github.com/nilesh2797/BlockRank (we keep a reference clone for study).

## Quick Links

- [Paper](https://arxiv.org/abs/2510.05396) · [Original Code](https://github.com/nilesh2797/BlockRank)
- [HF Model](https://huggingface.co/quicktensor/blockrank-msmarco-mistral-7b)
- [BEIR Evaluation Script](./scripts/eval_beir.py)
- [Training Collate Example](./examples/train_block_collate_example.py)
- [Chunking + Aggregation](#long-documents-chunking--aggregation)

## Table of Contents

- [Quick Links](#quick-links)
- [Why BlockRank?](#why-blockrank)
- [Quick Start](#quick-start-inference-with-published-model)
- [Usage](#usage)
- [Architecture](#architecture)
- [Features](#features)
- [Project Layout](#project-layout)
- [Citation](#citation)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Why BlockRank?

Standard listwise LLM reranking puts many candidate documents in one prompt and asks the model to output IDs. It works well but is **slow**:
- Quadratic attention over long contexts.
- Expensive autoregressive decoding to emit the ranked list.

BlockRank exploits two empirical structures in fine-tuned ICR LLMs:
1. **Block sparsity** — Documents mostly attend to themselves + the shared instruction (not every doc to every other doc).
2. **Latent relevance signals** — In middle layers, attention from certain "query" tokens to document blocks is highly predictive of true relevance.

**Result**:
- Architecturally enforce **structured sparse attention** → near-linear cost.
- Train with an **auxiliary contrastive loss** on attention scores.
- **Rank directly from attention** during prefill (no token generation needed) → often **2-4× faster** with competitive accuracy on BEIR.

See the paper for full analysis and results (Mistral-7B variant reaches ~54.9 avg BEIR using only 10% MS MARCO data).

## Quick Start (Inference with Published Model)

```bash
# 1. Install (rank-bm25 is in the [rag] extra and highly recommended)
cd ~/Developer/projects/blockrank-rag
pip install -e ".[rag,ui]"
# or
# pip install -r requirements-rag.txt

# 2. Fast ranking demo (uses the official HF BlockRank-Mistral + real BM25 first stage)
python examples/quick_inference.py

# Or try the end-to-end RAG demo
python examples/simple_rag_demo.py
```

`quick_inference.py` will:
- Load `quicktensor/blockrank-msmarco-mistral-7b` (or a base model + patch)
- Format a query + candidate docs
- Return ranked results using attention scores (fast path)

## Usage

All examples below assume you have installed the package with the recommended extras.

### Basic Reranking with Real Model
```python
from blockrank_rag import BlockRanker, BlockRankerConfig

ranker = BlockRanker(BlockRankerConfig(mock=False))  # loads published model
documents = ["Paris is the capital of France.", "Berlin is the capital of Germany."]
results = ranker.rank("capital of France?", documents)
```

### Full RAG Pipeline
```python
from blockrank_rag import RAGPipeline, BlockRanker, BlockRankerConfig

my_corpus = [
    "Paris is the capital of France and home to the Eiffel Tower.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain.",
]

pipeline = RAGPipeline(
    ranker=BlockRanker(BlockRankerConfig(mock=False)),
    first_stage_k=50,
    final_k=10
)
result = pipeline.answer("What is the capital of France?", my_corpus)
print(result["answer"])
print(result["sources"])
```

### Long Documents: Chunking + Aggregation
```python
long_documents = [
    "Paris is the capital of France. " * 50,   # artificially long doc
    "Berlin is the capital of Germany.",
]

cfg = BlockRankerConfig(
    max_chunk_tokens=128,     # split docs longer than this
    chunk_aggregate="max"     # "max", "mean", or "sum"
)
ranker = BlockRanker(cfg)
results = ranker.rank("capital of France?", long_documents)
```

### BEIR-style Evaluation
```bash
# Loads directly from the official ICR-formatted BEIR data on HF
python scripts/eval_beir.py \
    --hf_dataset quicktensor/icr-beir-evals \
    --subset trec_covid \
    --max_examples 100 \
    --mock

# With official qrels for nDCG/MRR
python scripts/eval_beir.py \
    --hf_dataset quicktensor/icr-beir-evals \
    --subset trec_covid \
    --qrels_path path/to/trec_covid.tsv
```

### Training Data & Block Collation
```bash
# Generate synthetic data in the exact format expected by the original research code
python examples/train_synthetic.py

# Full example of block tokenization + collate + model forward
python examples/train_block_collate_example.py
```

See `scripts/` and `examples/` for more utilities.

## Architecture

```
Query + Candidates
        │
        ▼
First-stage retriever (rank-bm25 by default)
        │
        ▼
BlockRankRanker
  ├─ Build prompt (with BLOCK_SEPARATOR between docs)
  ├─ tokenize_with_block_boundaries → exact token spans per doc
  ├─ (optional) Chunk long documents + prepare_chunked_inputs
  ├─ Forward + output_attentions at chosen layer
  ├─ Aggregate attention scores (per doc or per chunk, using max/mean/sum)
  └─ Return ranked results
        │
        ▼
RAGPipeline (optional generator with citations)
```

The production path uses `prepare_block_inputs` / adapted `block_icr_collate_fn` (based on the original research code) to get precise document token boundaries instead of rough heuristics. Chunking is supported on top of the exact boundaries.

## Features

- Real first-stage retrieval with `rank-bm25` (strongly recommended) + lexical fallback
- Exact block tokenization via `tokenize_with_block_boundaries` and `prepare_block_inputs`
- Adapted `block_icr_collate_fn` from the original research code for proper per-block structure
- Long-document support with chunking + per-document score aggregation (`max` / `mean` / `sum`)
- Accurate mid-layer attention scoring using real token boundaries (no more rough splits)
- Rich evaluation support:
  - `scripts/eval_beir.py` – loads BEIR subsets directly from HF (`quicktensor/icr-beir-evals`)
  - `calculate_accuracy` + qrels support (nDCG, MRR) ported from the original
- Training & data tooling:
  - Synthetic data generator matching the original ICR format
  - Complete examples showing block collation + model forward passes
- Full RAG pipeline, Gradio demo, and FastAPI server
- Latency comparison tools (attention-based vs full decode)
- Works out of the box with the published model `quicktensor/blockrank-msmarco-mistral-7b`

## Project Layout

```
blockrank-rag/
├── pyproject.toml
├── requirements-rag.txt
├── README.md
├── src/blockrank_rag/
│   ├── __init__.py
│   ├── ranker.py                 # BlockRanker (with chunking support)
│   ├── pipeline.py               # RAGPipeline + SimpleFirstStage (BM25)
│   ├── collate.py                # prepare_block_inputs, block_icr_collate_fn
│   ├── utils.py                  # calculate_accuracy, load_qrels
│   └── core/
│       ├── formatting.py         # block prompts, tokenize_with_block_boundaries, chunking
│       └── attention.py
├── scripts/
│   ├── eval_beir.py              # Dedicated BEIR evaluation (HF + qrels)
│   ├── eval_synthetic.py
│   └── bench_latency.py          # attn vs decode comparison
├── examples/
│   ├── quick_inference.py
│   ├── simple_rag_demo.py
│   ├── train_block_collate_example.py   # Full block collation + forward example
│   └── train_synthetic.py
├── data/                         # Generated synthetic data & configs
├── docs/
│   └── research-notes.md
└── blockrank-original/           # Reference clone of the research repo (read-only)
```

## Citation

If you use the ideas or this implementation, please cite the original paper:

```bibtex
@article{gupta2025blockrank,
  title={Scalable In-context Ranking with Generative Models},
  author={Gupta, Nilesh and You, Chong and Bhojanapalli, Srinadh and Kumar, Sanjiv and Dhillon, Inderjit and Yu, Felix},
  journal={arXiv preprint arXiv:2510.05396},
  year={2025}
}
```

## Development

```bash
# Install with all extras (recommended for full development)
pip install -e ".[dev,all]"

# Run linting and tests
ruff check .
pytest -q

# Run specific examples
python examples/quick_inference.py
python examples/simple_rag_demo.py
python examples/train_block_collate_example.py

# Run evaluation and benchmarking tools
python scripts/eval_beir.py --mock --max_examples 20
python scripts/bench_latency.py --n-docs 10 30 50 --mock
python scripts/eval_synthetic.py --n 30
```

### Working with the Original Reference
A full copy of the research implementation lives in the sibling `blockrank-original/` directory. You can explore the original attention kernels, training scripts, and data formats there:

```bash
cd blockrank-original
pip install -e .
python scripts/eval_attn.py --help
```

See `docs/research-notes.md` for internal implementation decisions and how the code relates to the original research repo.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the full guide.

Quick summary:
- Contributions are welcome (bug reports, docs, new model support, better chunking, BEIR evals, etc.).
- Run `ruff check .` and `pytest` before PRs.
- When touching block logic, review `blockrank-original/` for compatibility.

## License

MIT (same spirit as the original research release).
