#!/usr/bin/env python
"""
More complete example of using block_icr_collate_fn with training data.

This demonstrates the full flow adapted from the original BlockRank:
1. Generate ICR data in the expected format.
2. Preprocess with block tokenization (using the separator).
3. Use block_icr_collate_fn to get (B, M, H) style inputs + answer_ids.
4. Forward through a model (with output_attentions for aux loss).
5. (Optional) compute aux loss + LM loss.

Requires: transformers, datasets, torch, (trl for full trainer)

Run:
    python examples/train_block_collate_example.py
"""

import json
from pathlib import Path
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

from blockrank_rag.collate import block_icr_collate_fn
from blockrank_rag.core.formatting import BLOCK_SEPARATOR
from blockrank_rag.utils import remap_documents


def generate_icr_data(n: int = 4, docs_per: int = 6) -> List[Dict]:
    """Generate data compatible with BlockRank (same as train_synthetic)."""
    examples = []
    for i in range(n):
        q = f"capital of country number {i}?"
        raw_docs = {str(j): f"Information about country {j}. Capital facts here." for j in range(docs_per)}
        ans = [str(i % docs_per)]
        rem_docs, rem_ids, rem_ans = remap_documents(raw_docs, ans, docs_per)
        examples.append({
            "query": q,
            "query_id": f"q{i}",
            "documents": [{"doc_id": rid, "text": d} for rid, d in zip(rem_ids, rem_docs)],
            "answer_ids": rem_ans,
        })
    return examples


def simple_block_tokenize(tokenizer, ex: Dict, sep: str = BLOCK_SEPARATOR) -> Dict:
    """Minimal version of the reference _block_tokenize_batch."""
    # Build prompt string with separator (like create_prompt... + sep)
    # For simplicity we build a flat string with SEP between doc sections
    prompt_parts = [f"Query: {ex['query']}"]
    for d in ex["documents"]:
        prompt_parts.append(f"ID: {d['doc_id']} CONTENT: {d['text']} END")
    prompt_parts.append("Answer with relevant IDs.")
    full_text = sep.join(prompt_parts)

    # Split and tokenize per block
    segments = full_text.split(sep)
    block_ids = []
    for seg in segments:
        ids = tokenizer(seg, add_special_tokens=False)["input_ids"]
        block_ids.append(ids)

    # Flat input_ids and block_lengths (list of lens)
    flat = []
    lengths = []
    for b in block_ids:
        flat.extend(b)
        lengths.append(len(b))

    return {
        "input_ids": flat,
        "block_lengths": lengths,
        "answer_ids": ex["answer_ids"],
    }


def main():
    print("Generating synthetic ICR data...")
    raw_data = generate_icr_data(3, 5)
    ds = Dataset.from_list(raw_data)

    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"  # tiny for demo
    print(f"Loading tokenizer + tiny model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()

    print("Preprocessing with block tokenization (inserting SEP)...")
    processed = []
    for ex in ds:
        proc = simple_block_tokenize(tok, ex)
        processed.append(proc)

    print("Collating with block_icr_collate_fn...")
    batch = block_icr_collate_fn(
        processed,
        tok,
        pad_to_multiple_of=8,
        max_block_length=32,
    )

    print("Collated batch keys:", list(batch.keys()))
    print("input_ids shape:", batch["input_ids"].shape)
    print("num_blocks:", batch["num_blocks"])
    print("answer_ids shape:", batch["answer_ids"].shape)

    # Forward example (for aux loss style)
    print("\nForward pass (requesting attentions for aux loss)...")
    with torch.no_grad():
        # Note: for real BlockRank you set attn_implementation and use layers_to_return_scores
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_attentions=True,
        )
    print("logits shape:", out.logits.shape)
    print("num attention layers returned:", len(out.attentions) if out.attentions else 0)

    print("\n=== Training collate usage example complete ===")
    print("For real training: adapt load_icr_dataset_hf + block_icr_collate_fn + BlockRankAuxLossTrainer from the reference.")


if __name__ == "__main__":
    main()
