"""
Minimal FastAPI server exposing /rank .

uvicorn blockrank_rag.serving.app:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from ...ranker import BlockRanker, BlockRankerConfig  # type: ignore

app = FastAPI(title="BlockRank RAG API")
_ranker = None


class RankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: int = 5


@app.on_event("startup")
def startup():
    global _ranker
    _ranker = BlockRanker(BlockRankerConfig())


@app.post("/rank")
def rank(req: RankRequest):
    global _ranker
    if _ranker is None:
        _ranker = BlockRanker()
    res = _ranker.rank(req.query, req.documents, top_k=req.top_k)
    return [{"doc_id": r.doc_id, "score": r.score, "rank": r.rank, "text": r.doc_text} for r in res]


@app.get("/health")
def health():
    return {"status": "ok"}
