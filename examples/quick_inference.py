#!/usr/bin/env python
"""
Quick inference demo using BlockRank-style ranking.

Uses the published model by default and demonstrates attention-based scoring.
"""

from blockrank_rag.ranker import BlockRanker, BlockRankerConfig

def main():
    cfg = BlockRankerConfig(
        model_name="quicktensor/blockrank-msmarco-mistral-7b",
        attn_layer=20,
        num_last_queries=16,
        template="mistral",
        device="auto",
        mock=True,  # Set False to load real model (needs GPU + lots of RAM for 7B)
    )
    ranker = BlockRanker(cfg)
    # load() is a no-op for mock=True, but harmless
    ranker.load()

    query = "What is the capital of France?"
    docs = [
        "Berlin is the capital and largest city of Germany.",
        "Paris is the capital and most populous city of France. It is in Europe.",
        "Madrid is the capital of Spain and the largest city in the country.",
        "Rome is the capital city of Italy.",
        "London is the capital and largest city of England and the United Kingdom.",
    ]

    print("Query:", query)
    print("Ranking documents with attention scores...\n")

    results = ranker.rank(query, docs, top_k=5, use_attn_scoring=True)

    for r in results:
        print(f"#{r.rank+1} (doc {r.doc_id}, score={r.score:.4f}): {r.doc_text}")

    print("\nDone. (Expected: doc 1 'Paris' should rank very high.)")


if __name__ == "__main__":
    main()
