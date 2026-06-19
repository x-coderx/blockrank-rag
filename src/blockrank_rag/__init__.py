"""blockrank-rag: Efficient In-Context Ranking toolkit for RAG.

Based on ideas from "Scalable In-context Ranking with Generative Models" (arXiv:2510.05396).
"""

__version__ = "0.1.0"

from .ranker import BlockRanker, BlockRankerConfig
from .pipeline import RAGPipeline, SimpleFirstStage
from .collate import prepare_block_inputs, block_icr_collate_fn
from .core.formatting import (
    BLOCK_SEPARATOR, build_block_formatted_prompt, tokenize_with_block_boundaries,
    chunk_text, prepare_chunked_inputs
)
from .utils import calculate_accuracy, load_qrels, remap_documents

__all__ = [
    "BlockRanker",
    "BlockRankerConfig",
    "RAGPipeline",
    "SimpleFirstStage",
    "prepare_block_inputs",
    "block_icr_collate_fn",
    "BLOCK_SEPARATOR",
    "build_block_formatted_prompt",
    "tokenize_with_block_boundaries",
    "chunk_text",
    "prepare_chunked_inputs",
    "calculate_accuracy",
    "load_qrels",
    "remap_documents",
]
