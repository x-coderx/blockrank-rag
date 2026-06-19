"""Simple CLI for blockrank-rag."""

import argparse
import json
from .ranker import BlockRanker, BlockRankerConfig


def main():
    p = argparse.ArgumentParser(description="BlockRank RAG toolkit CLI")
    p.add_argument("query", help="Search query")
    p.add_argument("docs", nargs="+", help="Candidate documents (or use --file)")
    p.add_argument("--model", default="quicktensor/blockrank-msmarco-mistral-7b")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--no-attn", action="store_true", help="Use decode instead of attention scoring")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    cfg = BlockRankerConfig(model_name=args.model)
    ranker = BlockRanker(cfg)
    results = ranker.rank(args.query, args.docs, top_k=args.top_k, use_attn_scoring=not args.no_attn)

    out = [{"rank": r.rank, "doc_id": r.doc_id, "score": r.score, "text": r.doc_text} for r in results]
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for item in out:
            print(f"#{item['rank']+1} [{item['doc_id']}] {item['score']:.4f} | {item['text'][:100]}")


if __name__ == "__main__":
    main()
