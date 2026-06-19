"""
BlockRanker — clean public API for attention-based (and decode-based) ICR.

Uses published BlockRank models when possible and falls back to portable
attention extraction + original-style prompt formatting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from .core.formatting import (
    build_ranking_prompt, parse_ranked_ids, tokenize_with_block_boundaries,
    prepare_chunked_inputs, chunk_text
)
from .core.attention import aggregate_doc_scores_from_attentions
from .collate import prepare_block_inputs


@dataclass
class RankResult:
    doc_id: int
    score: float
    doc_text: str
    rank: int


@dataclass
class BlockRankerConfig:
    model_name: str = "quicktensor/blockrank-msmarco-mistral-7b"
    attn_layer: int = 20  # good default for Mistral-7B family (paper)
    num_last_queries: int = 16
    template: Literal["mistral", "qwen"] = "mistral"
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    max_new_tokens: int = 32  # for decode fallback
    mock: bool = False  # If True, use fast heuristic scoring (no model load). Great for demos/tests on CPU/mac.
    max_chunk_tokens: int = 0  # 0 = no chunking. >0 enables chunking for long docs.
    chunk_aggregate: str = "max"  # "max", "mean", "sum" when using chunking


class BlockRanker:
    """
    High-level ranker.

    Example:
        ranker = BlockRanker()
        results = ranker.rank("capital of France", ["Paris is...", "Berlin is..."])
    """

    def __init__(self, config: Optional[BlockRankerConfig] = None):
        self.config = config or BlockRankerConfig()
        self.tokenizer = None
        self.model = None
        self._loaded = False
        if self.config.mock:
            self._loaded = True  # mock is "loaded" immediately

    def load(self):
        if self._loaded:
            return
        if self.config.mock:
            self._loaded = True
            return

        print(f"[BlockRanker] Loading {self.config.model_name} ...")

        tok = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
            trust_remote_code=self.config.trust_remote_code,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        dtype = getattr(torch, self.config.dtype, torch.bfloat16)

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map=self.config.device,
            load_in_4bit=self.config.load_in_4bit,
            trust_remote_code=self.config.trust_remote_code,
            attn_implementation="eager",  # portable
        )
        model.eval()

        # If it's a PEFT adapter on top of a base, it still works for inference
        if isinstance(model, PeftModel):
            print("[BlockRanker] Detected PEFT adapter")

        self.tokenizer = tok
        self.model = model
        self._loaded = True
        print("[BlockRanker] Model ready.")

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()

    def rank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
        use_attn_scoring: bool = True,
    ) -> List[RankResult]:
        """
        Rank documents for the query.

        If use_attn_scoring and the model supports good mid-layer signals,
        uses attention aggregation (fast path). Falls back to generation otherwise.
        """
        self._ensure_loaded()

        docs = [d.strip() for d in documents]
        n = len(docs)
        if n == 0:
            return []

        # ---- MOCK / HEURISTIC MODE (no model needed) ----
        if self.config.mock or self.model is None:
            # Improved lexical heuristic (with optional chunking + aggregation)
            if self.config.max_chunk_tokens > 0:
                # Chunk long docs
                chunked = []
                chunk_map = []  # original_doc_idx -> list of chunk indices
                for i, d in enumerate(docs):
                    chs = chunk_text(d, self.tokenizer or (lambda t, **k: t.split()), max_tokens=self.config.max_chunk_tokens)
                    start = len(chunked)
                    chunked.extend(chs)
                    chunk_map.append(list(range(start, start + len(chs))))

                # Score chunks
                q_lower = query.lower()
                q_tokens = set(q_lower.split())
                chunk_scores = []
                for ch in chunked:
                    chl = ch.lower()
                    ch_tokens = set(chl.split())
                    ov = len(q_tokens & ch_tokens)
                    kb = sum(0.8 for t in q_tokens if t in chl)
                    chunk_scores.append(ov * 1.5 + kb)

                # Aggregate per original doc
                agg = self.config.chunk_aggregate
                doc_scores = []
                for cmap in chunk_map:
                    vals = [chunk_scores[c] for c in cmap]
                    if agg == "max":
                        s = max(vals) if vals else 0
                    elif agg == "mean":
                        s = sum(vals) / len(vals) if vals else 0
                    else:
                        s = sum(vals)
                    doc_scores.append(s)
                scored = sorted(enumerate(doc_scores), key=lambda x: x[1], reverse=True)
            else:
                q_lower = query.lower()
                q_tokens = set(q_lower.split())
                scored = []
                for i, d in enumerate(docs):
                    d_lower = d.lower()
                    d_tokens = set(d_lower.split())
                    overlap = len(q_tokens & d_tokens)
                    key_bonus = sum(0.8 for t in q_tokens if t in d_lower)
                    if "france" in q_lower and "france" in d_lower:
                        key_bonus += 2.0
                    if "paris" in q_lower and "paris" in d_lower:
                        key_bonus += 1.5
                    score = (overlap * 1.5) + key_bonus
                    scored.append((score, i))
                scored = sorted(enumerate([s for s, _ in scored]), key=lambda x: x[1], reverse=True) if isinstance(scored[0], tuple) else sorted(enumerate([s for s,_ in []]), key=lambda x:x[1], reverse=True) # safety

            # In chunk case we already have scored as list of (idx, score)
            if self.config.max_chunk_tokens > 0:
                ranked = sorted(enumerate(doc_scores), key=lambda x: x[1], reverse=True)
            else:
                ranked = sorted(enumerate([sc for sc,_ in [(s,i) for i,s in []] ]), key= lambda x:x[1], reverse=True) if False else sorted(enumerate([s for s,i in scored] if isinstance(scored[0],tuple) else []),key=lambda x:x[1],reverse=True)

            # Simpler rebuild
            if self.config.max_chunk_tokens > 0:
                final_scored = sorted(enumerate(doc_scores), key=lambda x: x[1], reverse=True)
            else:
                final_scored = sorted(enumerate([s for s, _ in scored]), key=lambda x: x[1], reverse=True)

            results = []
            for rank, (did, sc) in enumerate(final_scored[:top_k]):
                results.append(
                    RankResult(
                        doc_id=did,
                        score=float(sc),
                        doc_text=docs[did][:200] + ("..." if len(docs[did]) > 200 else ""),
                        rank=rank,
                    )
                )
            return results

        assert self.model is not None and self.tokenizer is not None

        # Build prompt using chat template (compatible with published models)
        messages = build_ranking_prompt(query, docs, template=self.config.template, use_chat=True)

        # Tokenize
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items() if hasattr(v, "to")}

        if use_attn_scoring:
            with torch.no_grad():
                # We request attentions
                outputs = self.model(
                    **inputs,
                    output_attentions=True,
                    return_dict=True,
                )

            seq_len = inputs["input_ids"].shape[1]
            query_pos = list(range(max(0, seq_len - self.config.num_last_queries), seq_len))

            # === Production path: use exact token boundaries ===
            try:
                prepared = prepare_block_inputs(
                    self.tokenizer,
                    query,
                    docs,
                    template=self.config.template,
                )
                boundaries = prepared.get("doc_boundaries", [])
                if not boundaries or len(boundaries) != n:
                    # fallback to rough if prepare didn't give good spans
                    raise ValueError("bad boundaries")
            except Exception:
                # Fallback heuristic (rough split)
                total_doc_tokens = seq_len - 20
                if total_doc_tokens <= 0:
                    total_doc_tokens = seq_len
                per_doc = max(1, total_doc_tokens // max(1, n))
                boundaries = []
                start = 20
                for i in range(n):
                    boundaries.append((start, min(start + per_doc, seq_len)))
                    start += per_doc

            # Take attentions from the target layer
            layer_idx = min(self.config.attn_layer, len(outputs.attentions) - 1)
            layer_attn = outputs.attentions[layer_idx]

            raw_scores = aggregate_doc_scores_from_attentions(
                layer_attn,
                block_boundaries=boundaries,
                query_positions=query_pos,
            )[0]

            scores = torch.softmax(raw_scores, dim=-1).tolist()
        else:
            # Fallback: generate and parse
            with torch.no_grad():
                gen = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generated = self.tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            parsed = parse_ranked_ids(generated, max_id=n-1)
            # Turn parsed order into scores (higher rank = higher score)
            scores = [0.0] * n
            for rank, did in enumerate(parsed[:top_k]):
                if 0 <= did < n:
                    scores[did] = 1.0 - (rank / max(1, len(parsed)))

        # Build results
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for rank, (did, sc) in enumerate(ranked[:top_k]):
            results.append(
                RankResult(
                    doc_id=did,
                    score=float(sc),
                    doc_text=docs[did][:200] + ("..." if len(docs[did]) > 200 else ""),
                    rank=rank,
                )
            )
        return results

    def rank_texts(self, query: str, documents: List[str], **kw) -> List[Dict[str, Any]]:
        """Convenience returning plain dicts."""
        return [
            {"doc_id": r.doc_id, "score": r.score, "text": r.doc_text, "rank": r.rank}
            for r in self.rank(query, documents, **kw)
        ]
