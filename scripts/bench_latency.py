#!/usr/bin/env python
"""
Latency + quality comparison between BlockRank attention scoring and full decode.

Compares:
- use_attn_scoring=True  (the fast BlockRank path)
- use_attn_scoring=False (standard generate + parse, the "naive" listwise way)

Usage (mock for quick CPU run):
    python scripts/bench_latency.py --n-docs 10 30 50 --mock

For real model (slower, needs the HF weights):
    python scripts/bench_latency.py --n-docs 10 30 50 --no-mock --device auto
"""

import argparse
import time
from typing import List, Tuple
from blockrank_rag.ranker import BlockRanker, BlockRankerConfig


def timed_rank(ranker, query: str, docs: List[str], use_attn: bool, top_k: int = 5) -> Tuple[float, list]:
    t0 = time.perf_counter()
    results = ranker.rank(query, docs, top_k=top_k, use_attn_scoring=use_attn)
    dt = time.perf_counter() - t0
    return dt, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-docs", nargs="+", type=int, default=[10, 30, 50])
    parser.add_argument("--query", default="What is the capital of France?")
    parser.add_argument("--mock", action="store_true", default=True, help="Use fast mock mode (no model weights)")
    parser.add_argument("--no-mock", dest="mock", action="store_false")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3, help="Average over N runs")
    args = parser.parse_args()

    max_n = max(args.n_docs)
    corpus = [
        f"Document {i}: This passage discusses cities, countries, landmarks, and capitals around the world. "
        f"It contains information relevant to geography queries including France, Germany, Italy, Spain, UK etc. "
        f"Extra padding text to simulate longer documents. {i}" 
        for i in range(max_n)
    ]

    # Make one obviously relevant for quality check
    corpus[1] = "Paris is the capital and most populous city of France. Famous for the Eiffel Tower and Louvre."

    cfg = BlockRankerConfig(mock=args.mock, device=args.device)
    ranker = BlockRanker(cfg)

    print(f"Benchmarking BlockRank attention vs decode (mock={args.mock}, device={args.device})")
    print(f"Corpus size up to {max_n} docs. Repeats={args.repeats}\n")

    for nd in sorted(args.n_docs):
        subdocs = corpus[:nd]
        attn_times = []
        decode_times = []
        for _ in range(args.repeats):
            dt_attn, _ = timed_rank(ranker, args.query, subdocs, use_attn=True, top_k=args.top_k)
            dt_decode, _ = timed_rank(ranker, args.query, subdocs, use_attn=False, top_k=args.top_k)
            attn_times.append(dt_attn)
            decode_times.append(dt_decode)

        avg_attn = sum(attn_times) / len(attn_times)
        avg_decode = sum(decode_times) / len(decode_times)
        speedup = avg_decode / avg_attn if avg_attn > 0 else float('inf')

        print(f"N={nd:3d} | attn={avg_attn*1000:6.1f}ms | decode={avg_decode*1000:6.1f}ms | speedup={speedup:5.1f}x")

    # Quick quality sanity on the last N
    print("\n--- Quality sanity check (last N) ---")
    nd = sorted(args.n_docs)[-1]
    subdocs = corpus[:nd]
    res_attn = ranker.rank(args.query, subdocs, top_k=3, use_attn_scoring=True)
    res_decode = ranker.rank(args.query, subdocs, top_k=3, use_attn_scoring=False)

    print("Attn top:", [ (r.doc_id, round(r.score,3)) for r in res_attn ])
    print("Decode top:", [ (r.doc_id, round(r.score,3)) for r in res_decode ])

    print("\nDone. Real gains appear with actual LLM forward passes on GPU.")


if __name__ == "__main__":
    main()
