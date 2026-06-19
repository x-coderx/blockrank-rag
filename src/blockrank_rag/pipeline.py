"""
Simple RAG pipeline using BlockRank for the reranking stage.

First stage: naive lexical (or plug in BM25 / embeddings).
Rerank: BlockRanker
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from .ranker import BlockRanker, BlockRankerConfig, RankResult

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDoc:
    doc_id: int
    text: str
    score: float


class SimpleFirstStage:
    """Real first-stage retriever.

    Prefers rank-bm25 (BM25Okapi) when installed for proper keyword-based retrieval.
    Falls back to simple lexical overlap otherwise.
    """

    def __init__(self, k: int = 20):
        self.k = k
        self._bm25 = None
        self._corpus = None
        try:
            from rank_bm25 import BM25Okapi
            self._bm25_class = BM25Okapi
            logger.debug("rank-bm25 available - will use BM25 for first stage")
        except ImportError:
            self._bm25_class = None
            logger.debug("rank-bm25 not installed - using lexical fallback for first stage")

    def retrieve(self, query: str, corpus: List[str]) -> List[RetrievedDoc]:
        if not corpus:
            return []

        if self._bm25_class is not None:
            # Real BM25
            tokenized_corpus = [doc.lower().split() for doc in corpus]
            if self._bm25 is None or self._corpus != corpus:
                self._bm25 = self._bm25_class(tokenized_corpus)
                self._corpus = corpus
            tokenized_query = query.lower().split()
            scores = self._bm25.get_scores(tokenized_query)
            scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:self.k]
            return [
                RetrievedDoc(doc_id=i, text=corpus[i], score=float(s))
                for i, s in scored
            ]
        else:
            # Fallback lexical
            qtokens = set(query.lower().split())
            scored = []
            for i, doc in enumerate(corpus):
                dtokens = set(doc.lower().split())
                overlap = len(qtokens & dtokens)
                scored.append((overlap, i, doc))
            scored.sort(reverse=True)
            top = scored[: self.k]
            return [RetrievedDoc(doc_id=i, text=doc, score=float(sc)) for sc, i, doc in top]


class RAGPipeline:
    def __init__(
        self,
        ranker: Optional[BlockRanker] = None,
        first_stage_k: int = 20,
        final_k: int = 5,
    ):
        self.ranker = ranker or BlockRanker()
        self.first_stage = SimpleFirstStage(k=first_stage_k)
        self.final_k = final_k

    def answer(self, query: str, corpus: List[str]) -> Dict[str, Any]:
        # 1. Retrieve
        candidates = self.first_stage.retrieve(query, corpus)
        cand_texts = [c.text for c in candidates]
        if not cand_texts:
            return {"query": query, "answer": "No candidates.", "sources": []}

        # 2. Rerank with BlockRank
        ranked = self.ranker.rank(query, cand_texts, top_k=self.final_k)

        # 3. Naive generator (echo + citations). Replace with real LLM call.
        top_texts = [candidates[r.doc_id].text for r in ranked]
        answer = (
            f"Based on the top documents, here is a synthesized answer for: {query}\n\n"
            + " ".join(t[:300] for t in top_texts[:2])
            + "..."
        )

        sources = [
            {"rank": r.rank, "score": r.score, "text": candidates[r.doc_id].text[:200]}
            for r in ranked
        ]
        return {"query": query, "answer": answer, "sources": sources}
