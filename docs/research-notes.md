# Research Notes & Implementation Decisions (BlockRank)

## Core Ideas (from paper + code exploration 2026-06)

- In-context Ranking (ICR): single prompt containing instruction + N docs + query, model outputs relevant doc IDs.
- Two key structures observed in attention:
  1. Strong intra-block (within doc) + instruction attention, weak cross-document.
  2. Mid-layer query tokens (esp. around the final "answer" trigger) concentrate attention on relevant docs.
- BlockRank fixes:
  - Enforce the sparsity in the attention mask/forward (block causal + full instr access).
  - Aux InfoNCE loss on aggregated attention scores to the docs from the bracket token.
  - Inference: read the attention, rank, done. No generation.
- Published model works well with the specific prompt format used during its training.

## Code Study Highlights

- `blockrank-original/src/blockrank/`:
  - `blockrank_std_attention.py`: detailed eager implementation + mask creation for 5D block masks. Uses HF AttentionInterface.
  - `losses.py`: precise logic to locate bracket token, isolate doc tokens, compute multi-positive InfoNCE.
  - `dataset.py` + `utils.py`: block tokenization using special separator, remapping, prompt builders for Mistral/Qwen.
  - `trainer.py`: extends TRL SFTTrainer, special forward with `layers_to_return_scores`, grad checkpointing fix.
- `scripts/`: train + two evals (attn fast path vs decode).

## Design Decisions for This Project

- **New project vs fork**: Clean slate for senior engineering practices (typing, tests, modularity, packaging). Reference clone kept read-only.
- **Attention extraction**: Start with portable hooks + `output_attentions=True`. Support registered custom impl later.
- **Block boundaries**: Initially approximate for quick demo. Later use exact per-block lengths from a block-aware collate.
- **Formatting**: Jinja + template names so users can easily add Llama3 / Gemma variants.
- **MVP first**: Get published model ranking + tiny RAG demo running, then add training / serving.
- **Mac friendly**: CPU/MPS path always works for demos (even if slow). Document CUDA for real speed.
- **No blind copy**: Port math and ideas; rewrite for clarity and testability.

## Next Experiments (after MVP)

- Exact block tokenization matching original collate.
- Try flex_attention for cleaner custom block mask.
- Layer sensitivity study across more models.
- Distill the attention signal into a small cross-encoder head.
- Integrate as drop-in reranker in LangChain / LlamaIndex.

See the top-level plan.md for the full execution roadmap.
