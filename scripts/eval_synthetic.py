#!/usr/bin/env python
"""
Basic synthetic eval harness for BlockRank reranker.

Generates queries with known relevant docs and measures simple top-1 / recall.

This is a starting point toward full BEIR-style evals.

Usage:
    python scripts/eval_synthetic.py --n 50 --k 5
"""

import argparse
import random
from blockrank_rag.ranker import BlockRanker, BlockRankerConfig
from blockrank_rag.pipeline import SimpleFirstStage
from blockrank_rag.utils import calculate_accuracy


def make_synthetic(n_queries: int, n_docs_per_query: int = 20):
    """Create synthetic queries + corpus with planted relevant docs."""
    examples = []
    for qid in range(n_queries):
        topic = random.choice(["France", "Germany", "Italy", "Spain", "UK", "Japan", "Brazil"])
        query = f"What is the capital of {topic}?"
        docs = []
        relevant_ids = []
        for i in range(n_docs_per_query):
            if i == 0:
                text = f"The capital of {topic} is its main city. It has important landmarks."
                relevant_ids.append(i)
            else:
                text = f"Unrelated doc {i} talking about random topics, animals, food, or other countries."
            docs.append(text)
        examples.append({
            "query": query,
            "docs": docs,
            "relevant": relevant_ids  # list of doc indices that are relevant
        })
    return examples


def evaluate(ranker, examples, first_stage_k=10, rerank_k=5):
    all_predictions = []
    eval_ds = {"answer_ids": [], "query_id": [], "remapped_doc_ids": []}

    for ex in examples:
        fs = SimpleFirstStage(k=first_stage_k)
        cands = fs.retrieve(ex["query"], ex["docs"])
        cand_texts = [c.text for c in cands]
        cand_ids = [c.doc_id for c in cands]

        ranked = ranker.rank(ex["query"], cand_texts, top_k=rerank_k, use_attn_scoring=True)
        top_ids = [cand_ids[r.doc_id] for r in ranked]

        all_predictions.append(top_ids)
        eval_ds["answer_ids"].append(ex["relevant"])
        eval_ds["query_id"].append(str(len(eval_ds["answer_ids"]) - 1))
        eval_ds["remapped_doc_ids"].append(list(range(len(ex["docs"]))))

    metrics = calculate_accuracy(all_predictions, eval_ds)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--docs-per", type=int, default=20)
    parser.add_argument("--mock", action="store_true", default=True)
    args = parser.parse_args()

    random.seed(42)
    examples = make_synthetic(args.n, args.docs_per)

    cfg = BlockRankerConfig(mock=args.mock)
    ranker = BlockRanker(cfg)

    print(f"Running synthetic eval: {args.n} queries, {args.docs_per} docs each (mock={args.mock})")
    metrics = evaluate(ranker, examples)

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    print("\nUsing original calculate_accuracy (supports nDCG/MRR with qrels).")
    print("For full BEIR: load from HF (quicktensor/icr-beir-evals or standard BEIR) + use load_qrels.")


if __name__ == "__main__":
    main()
