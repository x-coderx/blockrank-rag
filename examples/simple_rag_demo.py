#!/usr/bin/env python
"""
Minimal end-to-end RAG demo using BlockRank reranker.

Uses a tiny synthetic corpus so it runs quickly even on CPU.
"""

from blockrank_rag.pipeline import RAGPipeline
from blockrank_rag.ranker import BlockRanker, BlockRankerConfig

CORPUS = [
    "The capital of France is Paris. It is known for the Eiffel Tower and Louvre museum.",
    "Berlin is the capital of Germany and has a rich history including the Berlin Wall.",
    "Madrid is the capital of Spain. It is famous for the Prado museum and tapas.",
    "Rome, the capital of Italy, is home to the Colosseum and Vatican City.",
    "London is the capital of the United Kingdom. Big Ben and the Thames are landmarks.",
    "Python is a popular programming language created by Guido van Rossum.",
    "Machine learning models can be trained using PyTorch or TensorFlow frameworks.",
    "The Eiffel Tower is located in Paris and was completed in 1889.",
]

def main():
    print("Loading BlockRank reranker (mock mode by default for fast CPU demo)...")
    cfg = BlockRankerConfig(device="cpu", mock=True)  # Set mock=False + device="auto"/"cuda" for the real published model
    ranker = BlockRanker(cfg)

    pipeline = RAGPipeline(ranker=ranker, first_stage_k=8, final_k=3)

    query = "What is the capital of France and what famous landmark is there?"
    print("\nQuery:", query)
    print("Corpus size:", len(CORPUS))

    result = pipeline.answer(query, CORPUS)

    print("\n--- Generated Answer ---")
    print(result["answer"])

    print("\n--- Top Sources (BlockRank reranked) ---")
    for s in result["sources"]:
        print(f"  [{s['rank']}] score={s['score']:.4f} : {s['text'][:120]}...")

    print("\nDemo complete. Try editing CORPUS or the query!")


if __name__ == "__main__":
    main()
