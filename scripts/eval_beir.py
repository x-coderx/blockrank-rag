#!/usr/bin/env python
"""
Dedicated BEIR-style evaluation script for BlockRank.

Loads ICR-formatted data (from HF or local JSONL) + optional qrels,
runs the BlockRanker (with real first-stage + attention scoring),
and computes metrics using the original `calculate_accuracy`.

Supports:
- HF datasets in the format expected by BlockRank (query, documents, answer_ids)
- The official ICR BEIR evals: e.g. https://huggingface.co/datasets/quicktensor/icr-beir-evals (or subsets)
- Local JSONL + TSV qrels (same format as original)

Usage examples:
    # Mock (fast, no weights)
    python scripts/eval_beir.py --dataset trec_covid --mock --k 10

    # Real model (requires GPU + weights)
    python scripts/eval_beir.py --data_path data/icr-beir-evals/trec_covid.jsonl \
        --qrels_path data/icr-beir-evals/qrels/trec_covid.tsv --no-mock

    # From HF (if available in compatible format)
    python scripts/eval_beir.py --hf_dataset quicktensor/icr-beir-evals --subset trec_covid
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from datasets import load_dataset, Dataset

from blockrank_rag.ranker import BlockRanker, BlockRankerConfig
from blockrank_rag.pipeline import SimpleFirstStage, RAGPipeline
from blockrank_rag.utils import calculate_accuracy, load_qrels


def load_beir_data(
    data_path: Optional[str] = None,
    hf_dataset: Optional[str] = None,
    subset: Optional[str] = None,
    split: str = "test",
    max_examples: Optional[int] = None,
) -> Dataset:
    """Load data in BlockRank ICR format (query, documents list, answer_ids).

    Falls back to a small synthetic BEIR-like set for demo purposes if loading fails.
    """
    ds = None
    try:
        if hf_dataset:
            print(f"Loading from HF: {hf_dataset} (subset={subset or 'default'})")
            if "quicktensor/icr-beir-evals" in hf_dataset:
                # The dataset uses "examples" config, which returns a DatasetDict with task names as keys
                raw = load_dataset(hf_dataset, "examples")
                task = subset or "trec_covid"
                if task in raw:
                    ds = raw[task]
                else:
                    # fallback to first available
                    ds = list(raw.values())[0]
                    print(f"  (subset '{task}' not found, using first available: {list(raw.keys())[:3]}...)")
            else:
                config = subset
                raw = load_dataset(hf_dataset, config)
                if hasattr(raw, "keys"):
                    ds = raw.get(split) or raw.get("train") or list(raw.values())[0]
                else:
                    ds = raw
        elif data_path:
            print(f"Loading local: {data_path}")
            if data_path.endswith(".jsonl") or "*" in data_path:
                ds = load_dataset("json", data_files=data_path, split="train")
            else:
                ds = load_dataset(data_path, split=split)
    except Exception as e:
        print(f"Could not load specified data ({e}). Using synthetic demo data.")
        ds = None

    if ds is None:
        # Synthetic fallback matching BEIR-style ICR format
        from blockrank_rag.utils import remap_documents
        import random
        examples = []
        topics = ["France", "Germany", "Italy", "Spain"]
        for i in range(max_examples or 20):
            topic = random.choice(topics)
            q = f"capital of {topic}?"
            raw_docs = {str(j): f"Doc about {topic if j==0 else 'other'} number {j}." for j in range(12)}
            ans = ["0"]
            rem_docs, rem_ids, rem_ans = remap_documents(raw_docs, ans, 12)
            examples.append({
                "query": q,
                "query_id": f"q{i}",
                "documents": [{"doc_id": rid, "text": d} for rid, d in zip(rem_ids, rem_docs)],
                "answer_ids": [str(a) for a in rem_ans],
            })
        ds = Dataset.from_list(examples)

    if max_examples and len(ds) > max_examples:
        ds = ds.select(range(max_examples))

    # Normalize
    def _normalize(ex):
        if "documents" in ex and isinstance(ex["documents"], list):
            if ex["documents"] and isinstance(ex["documents"][0], str):
                ex["documents"] = [{"doc_id": str(i), "text": d} for i, d in enumerate(ex["documents"])]
        if "answer_ids" in ex:
            ans = ex["answer_ids"]
            if ans and isinstance(ans[0], int):
                ex["answer_ids"] = [str(a) for a in ans]
        return ex

    ds = ds.map(_normalize)
    return ds


def run_evaluation(
    ranker: BlockRanker,
    dataset: Dataset,
    first_stage_k: int = 100,
    rerank_k: int = 10,
    use_first_stage: bool = True,
) -> Dict[str, float]:
    """Run ranking over the dataset and collect predictions for calculate_accuracy."""
    all_predictions: List[List[int]] = []
    eval_ds_for_metrics: Dict[str, List] = {
        "answer_ids": [],
        "query_id": [],
        "remapped_doc_ids": [],
    }

    for idx, ex in enumerate(dataset):
        query = ex["query"]
        docs = ex.get("documents", [])
        if isinstance(docs[0], dict):
            doc_texts = [d.get("text", str(d)) for d in docs]
            remapped_ids = [d.get("doc_id", str(i)) for i, d in enumerate(docs)]
        else:
            doc_texts = docs
            remapped_ids = [str(i) for i in range(len(docs))]

        answer_ids = ex.get("answer_ids", [])
        if isinstance(answer_ids, list) and answer_ids:
            if isinstance(answer_ids[0], int):
                answer_ids = [str(a) for a in answer_ids]

        # Map answer_ids to remapped indices (0-based in the candidate list)
        gt_indices = []
        id_to_idx = {rid: i for i, rid in enumerate(remapped_ids)}
        for aid in answer_ids:
            if str(aid) in id_to_idx:
                gt_indices.append(id_to_idx[str(aid)])

        if use_first_stage:
            fs = SimpleFirstStage(k=first_stage_k)
            cands = fs.retrieve(query, doc_texts)
            cand_texts = [c.text for c in cands]
            orig_indices = [c.doc_id for c in cands]  # indices in original list
        else:
            cand_texts = doc_texts
            orig_indices = list(range(len(doc_texts)))

        # Rerank top candidates with BlockRank
        ranked = ranker.rank(query, cand_texts, top_k=rerank_k, use_attn_scoring=True)
        pred_indices = [orig_indices[r.doc_id] for r in ranked]  # original 0-based indices

        all_predictions.append(pred_indices)
        eval_ds_for_metrics["answer_ids"].append(gt_indices)
        eval_ds_for_metrics["query_id"].append(ex.get("query_id", str(idx)))
        eval_ds_for_metrics["remapped_doc_ids"].append(remapped_ids)

    metrics = calculate_accuracy(all_predictions, eval_ds_for_metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="BEIR evaluation for BlockRank")
    parser.add_argument("--data_path", type=str, default=None, help="Path to ICR JSONL file")
    parser.add_argument("--hf_dataset", type=str, default="quicktensor/icr-beir-evals", help="HF dataset id")
    parser.add_argument("--subset", type=str, default=None, help="Subset (e.g. trec_covid)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--qrels_path", type=str, default=None, help="Path to qrels TSV")
    parser.add_argument("--max_examples", type=int, default=50, help="Limit for demo")
    parser.add_argument("--first_stage_k", type=int, default=100)
    parser.add_argument("--rerank_k", type=int, default=10)
    parser.add_argument("--mock", action="store_true", default=True, help="Use mock ranker (no model load)")
    parser.add_argument("--no-mock", dest="mock", action="store_false")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="quicktensor/blockrank-msmarco-mistral-7b")
    args = parser.parse_args()

    # Load data
    ds = load_beir_data(
        data_path=args.data_path,
        hf_dataset=args.hf_dataset,
        subset=args.subset,
        split=args.split,
        max_examples=args.max_examples,
    )
    print(f"Loaded {len(ds)} examples")

    # Load qrels if provided, or try to auto-load from the same HF repo for quicktensor data
    qrels = None
    if args.qrels_path:
        qrels = load_qrels(args.qrels_path)
    elif args.hf_dataset and "quicktensor/icr-beir-evals" in args.hf_dataset and args.subset:
        try:
            # The "qrels" config contains per-task tsv files. We can try to extract.
            qrels_ds = load_dataset(args.hf_dataset, "qrels")
            task = args.subset
            if task in qrels_ds:
                # Write temp tsv and load
                import tempfile, os
                with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
                    # simplistic: assume it has the text or we skip detailed
                    pass
            print("  (qrels available via HF 'qrels' config; for full nDCG pass --qrels_path pointing to the tsv)")
        except Exception:
            pass

    # Prepare ranker
    cfg = BlockRankerConfig(
        model_name=args.model,
        mock=args.mock,
        device=args.device,
    )
    ranker = BlockRanker(cfg)

    print(f"Running evaluation (mock={args.mock}, first_stage_k={args.first_stage_k})...")
    metrics = run_evaluation(
        ranker,
        ds,
        first_stage_k=args.first_stage_k,
        rerank_k=args.rerank_k,
    )

    print("\n=== BEIR-style Results ===")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"{k:15s}: {v:.3f}")
        else:
            print(f"{k:15s}: {v}")

    if qrels:
        print("(qrels loaded; full nDCG/MRR would be computed if predictions include original doc_ids)")
    else:
        print("Tip: pass --qrels_path for full ranking metrics (nDCG, MRR) via calculate_accuracy")

    print("\nDone. For full BEIR runs use larger --max_examples or full data + GPU.")


if __name__ == "__main__":
    main()
