"""
Block-aware collation adapted from blockrank-original/src/blockrank/dataset.py

The original `block_icr_collate_fn` reshapes inputs into per-block (B, M, H) for the
custom sparse attention.

For production RAG (inference-focused), the most valuable part is:
- exact `doc_boundaries` (token spans) so attention scoring is accurate.

This module provides:
- `prepare_block_inputs`: easy function to get flat_ids + exact doc spans (used by ranker)
- `block_icr_collate_fn`: full adapted collate (for training compatibility or advanced use)
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np

from .core.formatting import BLOCK_SEPARATOR, tokenize_with_block_boundaries


def prepare_block_inputs(
    tokenizer,
    query: str,
    documents: List[str],
    template: str = "mistral",
    max_block_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    High-level helper for inference / reranking.

    Returns everything needed for accurate BlockRank attention scoring:
      - input_ids (flat)
      - doc_boundaries: exact (start, end) token positions for each document
      - plus padded block view if requested
    """
    tok_result = tokenize_with_block_boundaries(
        tokenizer, query, documents, template=template
    )

    input_ids = tok_result["input_ids"]
    doc_boundaries = tok_result.get("doc_boundaries", [])

    # Optional per-block padding (useful if you want to feed the (B, M, H) style to custom attn)
    if max_block_length is not None:
        # simple per-block pad/trunc for demonstration
        block_view = []
        for start, end in doc_boundaries:  # or all blocks
            block = input_ids[start:end]
            if len(block) > max_block_length:
                block = block[:max_block_length]
            else:
                block = block + [tokenizer.pad_token_id] * (max_block_length - len(block))
            block_view.append(block)
        return {
            "input_ids": input_ids,
            "doc_boundaries": doc_boundaries,
            "block_view": block_view,
        }

    return {
        "input_ids": input_ids,
        "doc_boundaries": doc_boundaries,
        "num_blocks": tok_result.get("num_blocks", len(doc_boundaries) + 2),
    }


def block_icr_collate_fn(
    batch: List[Dict],
    tok,
    pad_to_multiple_of: int = 16,
    max_block_length: Optional[int] = None,
    always_max_len: bool = False,
    permutation_invariant_pos: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Adapted version of the original block_icr_collate_fn from BlockRank research code.

    Expects each item in batch to have come from a block-tokenized dataset, i.e.:
        item = {
            "input_ids": List[int],           # flat tokens
            "block_lengths": List[int],       # lengths of each block (instr + docs + query)
            "answer_ids": List[int],          # for aux loss / labels (optional for inference)
            ...
        }

    Returns tensors in the shape expected by BlockRank attention (flattened + metadata).
    """
    pad_token_id = tok.pad_token_id or tok.eos_token_id
    padding_side = getattr(tok, "padding_side", "right")
    B = len(batch)

    if B == 0:
        return {}

    # Copy to avoid mutating caller data
    batch = [dict(item) for item in batch]

    # Determine number of blocks (instruction + docs + query/completion)
    M = len(batch[0]["block_lengths"]) - 1 if "block_lengths" in batch[0] else 2

    # Merge last two blocks (query + completion) like the original
    for item in batch:
        bl = item.get("block_lengths", [])
        if len(bl) >= 2:
            item["last_prompt_block_lengths"] = int(bl[-2])
            item["completion_lengths"] = int(bl[-1])
            item["block_lengths"] = bl[:-1]
            item["block_lengths"][-1] += item["completion_lengths"]
        else:
            item.setdefault("block_lengths", [len(item.get("input_ids", []))])

    # Compute max block length
    all_block_lens = []
    for item in batch:
        all_block_lens.extend(item.get("block_lengths", []))

    if always_max_len:
        max_block_length = max_block_length or (max(all_block_lens) if all_block_lens else 128)
    else:
        max_block_length = min(
            max(all_block_lens) if all_block_lens else 128,
            max_block_length or int(1e9)
        )

    if pad_to_multiple_of and max_block_length:
        max_block_length = ((max_block_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    # Build per-block padded sequences (simplified from original)
    all_block_input_ids: List[List[int]] = []
    for item in batch:
        lengths = item.get("block_lengths", [])
        ids = item.get("input_ids", [])
        if not lengths:
            lengths = [len(ids)]

        indptr = list(np.cumsum([0] + lengths))
        blocks = []
        for s, e in zip(indptr, indptr[1:]):
            blk = ids[s:e][:max_block_length]
            blk = blk + [pad_token_id] * (max_block_length - len(blk))
            blocks.append(blk)
        all_block_input_ids.extend(blocks)

    # Pad sequence of blocks
    padding_block = [pad_token_id] * max_block_length
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(b, dtype=torch.long) for b in all_block_input_ids] + [torch.tensor(padding_block)],
        batch_first=True,
        padding_value=pad_token_id,
    )[:-1]

    BM, H = input_ids.shape
    input_ids = input_ids.view(B, M, H)
    attention_mask = (input_ids != pad_token_id).long()
    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100
    labels[:, :-1, :] = -100  # only train on last block

    # Position ids (permutation invariant like original)
    if permutation_invariant_pos:
        position_ids = attention_mask.cumsum(-1)
        if M > 1:
            position_ids[:, 1:-1] += position_ids[:, 0].max(dim=-1).values[:, None, None]
        position_ids[:, -1] += 16384
        position_ids = torch.clamp_min(position_ids - 1, 0)
        position_ids[~attention_mask.bool()] = 0
    else:
        position_ids = attention_mask.view(B, -1).cumsum(-1) * attention_mask.view(B, -1)
        position_ids = torch.clamp_min(position_ids - 1, 0)

    # answer_ids for aux loss (padded)
    answer_ids_list = [item.get("answer_ids", []) for item in batch]
    answer_ids_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(a, dtype=torch.long) if a else torch.tensor([], dtype=torch.long) for a in answer_ids_list],
        batch_first=True,
        padding_value=-1,
    )

    return {
        "input_ids": input_ids.view(B, M * H),
        "position_ids": position_ids.view(B, M * H),
        "attention_mask": attention_mask.view(B, M * H),
        "labels": labels.view(B, M * H),
        "num_blocks": torch.tensor(M, dtype=torch.long),
        "answer_ids": answer_ids_padded,
        "doc_boundaries": [item.get("doc_boundaries", []) for item in batch],  # pass through if present
    }
