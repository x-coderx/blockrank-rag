"""
Portable attention score extraction for BlockRank-style ranking.

Primary path: forward hooks to capture raw attentions at a target layer.
Secondary (future): registered BlockRank sparse attention (from original kernels).

This gives us document relevance scores without full generation.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from contextlib import contextmanager


@contextmanager
def capture_attentions(
    model: nn.Module,
    target_layer_idx: int,
    num_last_queries: int = 32,
):
    """
    Context manager that captures attention weights from a specific layer.

    Yields:
        list of captured attention tensors (or None if not available)
    """
    captured: List[torch.Tensor] = []
    handles = []

    # Find the layers container (works for most HF CausalLM)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        layers = None

    if layers is None or target_layer_idx >= len(layers):
        yield captured
        return

    layer = layers[target_layer_idx]

    def hook_fn(module, input, output):
        # output for attention is usually (attn_output, attn_weights) or just output
        # In HF with output_attentions=True we get attentions separately in model outputs.
        # This hook is a fallback. We prefer using model(..., output_attentions=True)
        pass

    # We rely primarily on output_attentions=True returned in forward.
    # This context is for future richer hooks (e.g. patching specific attn modules).
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


def aggregate_doc_scores_from_attentions(
    attentions: torch.Tensor,          # (B, num_heads, seq, seq) or from model.attentions list
    block_boundaries: List[Tuple[int, int]],  # [(start, end), ...] token indices for each doc block (excl instr/query)
    query_positions: List[int],        # positions of the "signal" query tokens (e.g. last 8-32)
    agg: str = "sum",
) -> torch.Tensor:
    """
    Aggregate attention mass from query positions to each document block.

    Simplified portable version. For full fidelity with published models
    you should also respect how the original loss computed "bracket" token and
    normalized over doc tokens only.

    Returns:
        (B, num_docs) relevance scores (higher = more relevant)
    """
    if attentions.dim() == 4:  # (B, H, S, S)
        # average over heads
        attn = attentions.mean(dim=1)  # (B, S, S)
    else:
        attn = attentions

    B, S, _ = attn.shape
    scores = []

    for b in range(B):
        q_pos = [p for p in query_positions if p < S]
        if not q_pos:
            q_pos = [S-1]

        doc_scores = []
        for start, end in block_boundaries:
            if start >= S:
                doc_scores.append(0.0)
                continue
            end = min(end, S)

            # Sum attention from query positions into this doc's tokens
            mass = attn[b, q_pos, start:end].sum().item()
            doc_scores.append(mass)

        scores.append(doc_scores)

    return torch.tensor(scores, dtype=torch.float32)


def extract_block_boundaries_from_lengths(
    block_lengths: List[int],
    num_instr_blocks: int = 1,
    num_query_blocks: int = 1,
) -> List[Tuple[int, int]]:
    """
    Convert per-block token lengths into absolute (start, end) spans.
    Assumes concatenated layout: [instr, doc0, doc1, ..., query]
    """
    boundaries: List[Tuple[int, int]] = []
    cursor = 0
    for i, blen in enumerate(block_lengths):
        if num_instr_blocks <= i < len(block_lengths) - num_query_blocks:
            boundaries.append((cursor, cursor + blen))
        cursor += blen
    return boundaries
