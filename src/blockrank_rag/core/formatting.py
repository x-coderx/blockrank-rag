"""
Prompt formatting for In-Context Ranking (ICR).

Generalized and configurable version inspired by the original BlockRank
utils (format_ranking_prompt_mistral / qwen + conversation wrappers).
Supports Jinja2 templates for easy extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import jinja2

# Default templates (very close to the paper/original for compatibility with published models)
DEFAULT_MISTRAL_INSTRUCT = """You will be given a query and a list of documents. Each document will be formatted as ID: <id> | CONTENT: <content> | END ID: <id>. You need to read carefully and understand all of them and your goal is to find all document(s) from the list that can help answer the query.

Query: {{ query }}

Documents:
{% for doc in documents -%}
ID: {{ loop.index0 }} | CONTENT: {{ doc }} | END ID: {{ loop.index0 }}
{% endfor %}

====== Now let's start! ======
Which document is most relevant to answer the query? Print out the ID of the document.
Query: {{ query }}
The following document(s) can help answer the query:
"""

DEFAULT_QWEN = """<Instruct>: You will be given a query and a list of documents. Each document will be formatted as <Document>: ID: <id> | CONTENT: <content> | END ID: <id>. You need to read and understand carefully the content of each document and your goal is to give the IDs of the document from the list that can help answer the query.

<Query>: {{ query }}
{% for doc in documents -%}
<Document>: ID: {{ loop.index0 }} | CONTENT: {{ doc }} | END ID: {{ loop.index0 }}
{% endfor %}
"""

# Simple Llama-3 / Llama-2 style instruction
DEFAULT_LLAMA = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant that ranks documents for a query.
<|eot_id|><|start_header_id|>user<|end_header_id|>
You will be given a query and a list of documents. Format: ID: <id> | CONTENT: <content> | END ID: <id>.
Find document(s) that help answer the query.

Query: {{ query }}

Documents:
{% for doc in documents -%}
ID: {{ loop.index0 }} | CONTENT: {{ doc }} | END ID: {{ loop.index0 }}
{% endfor %}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# Completion templates (for training data generation or few-shot)
MISTRAL_COMPLETION = "Final Answer: [{{ ids | join(', ') }}]"
QWEN_COMPLETION = "<think>I need to give the IDs of the most relevant documents in the list for the query: {{ query }} formatted as list [...].</think>\n<Answer>: [{{ ids | join(', ') }}]"


@dataclass
class RankingPromptTemplate:
    name: str
    instruction: str
    doc_format: str = "ID: {id} | CONTENT: {content} | END ID: {id}"
    separator: str = "\n"
    final_section: str = (
        "\n====== Now let's start! ======\n"
        "Which document is most relevant to answer the query? Print out the ID of the document.\n"
        "Query: {query}\n"
        "The following document(s) can help answer the query:"
    )
    # For chat models we often split user / assistant


def format_documents(documents: List[str]) -> List[str]:
    """Ensure list of clean document strings."""
    return [d.strip() for d in documents]


def build_ranking_prompt(
    query: str,
    documents: List[str],
    template: str = "mistral",
    use_chat: bool = True,
    use_blocks: bool = False,
) -> List[Dict[str, str]] | str:
    """
    Build prompt (or chat messages) for ICR.

    If use_blocks=True, inserts the original BlockRank separator between documents
    so that downstream tokenization can split into exact blocks for attention scoring.

    Returns:
        If use_chat=True: list of {"role": "...", "content": "..."} suitable for apply_chat_template
        Else: raw string prompt
    """
    documents = format_documents(documents)
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    sep = BLOCK_SEPARATOR if use_blocks else "\n"

    if template == "mistral":
        tmpl = env.from_string(DEFAULT_MISTRAL_INSTRUCT)
        user_content = tmpl.render(query=query, documents=documents)
        if use_blocks:
            # Insert separator after each document line in the rendered content
            # A simple approach: replace the natural doc separators
            user_content = user_content.replace("\nID: ", f"{sep}ID: ")
        if use_chat:
            return [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": MISTRAL_COMPLETION.format(ids=[])}
            ]
        return user_content

    elif template == "qwen":
        tmpl = env.from_string(DEFAULT_QWEN)
        user_content = tmpl.render(query=query, documents=documents)
        if use_blocks:
            user_content = user_content.replace("\n<Document>: ", f"{sep}<Document>: ")
        if use_chat:
            return [
                {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided."},
                {"role": "user", "content": user_content},
            ]
        return user_content

    elif template in ("llama", "llama3", "meta-llama"):
        tmpl = env.from_string(DEFAULT_LLAMA)
        user_content = tmpl.render(query=query, documents=documents)
        if use_blocks:
            user_content = user_content.replace("\nID: ", f"{sep}ID: ")
        if use_chat:
            return [{"role": "user", "content": user_content}]
        return user_content

    else:
        raise ValueError(f"Unknown template: {template}")


def build_completion(answer_ids: List[int | str]) -> str:
    """Simple completion string (for training data synthesis)."""
    ids_str = ", ".join(str(x) for x in answer_ids)
    return f"Final Answer: [{ids_str}]"


def parse_ranked_ids(text: str, max_id: int | None = None) -> List[int]:
    """
    Robust-ish parser for model output containing document IDs.
    Looks for bracketed lists or standalone numbers.
    """
    import re
    text = text.strip()

    # Prefer content inside [...]
    bracket = re.search(r"\[([^\]]+)\]", text)
    if bracket:
        candidates = re.findall(r"\d+", bracket.group(1))
    else:
        candidates = re.findall(r"\b\d+\b", text)

    ids = []
    seen = set()
    for c in candidates:
        iid = int(c)
        if max_id is not None and iid > max_id:
            continue
        if iid not in seen:
            seen.add(iid)
            ids.append(iid)
    return ids


# Original BlockRank style separator for block-aware processing
BLOCK_SEPARATOR = "<<end_of_block_prompt_segment>>"


def build_block_formatted_prompt(
    query: str,
    documents: List[str],
    template: str = "mistral",
) -> Tuple[str, List[int]]:
    """
    Build prompt string using the original BlockRank separator (for character-level preview).
    """
    base = build_ranking_prompt(query, documents, template=template, use_chat=False, use_blocks=True)
    if isinstance(base, list):
        base = base[0].get("content", "") if base else ""
    full_text = base  # already has separators if use_blocks was respected
    block_lengths = [len(base)] + [len(d) for d in documents]
    return full_text, block_lengths


def tokenize_with_block_boundaries(
    tokenizer,
    query: str,
    documents: List[str],
    template: str = "mistral",
    add_special_tokens: bool = True,
) -> Dict[str, Any]:
    """
    Tokenize the full ICR prompt while preserving exact token boundaries for each document block.

    This is the key production utility that wires build_block_formatted_prompt style
    + exact token boundaries (adapted from reference blockrank-original/dataset.py).

    Returns:
        {
            "input_ids": List[int] (flattened),
            "doc_boundaries": List[Tuple[int, int]],  # (start, end) token indices for each doc
            "num_blocks": int,
            "attention_mask": Optional[List[int]],
        }
    """
    documents = format_documents(documents)

    # Build the full string representation using block separators
    # We render the user content with separators between docs
    prompt_str = build_ranking_prompt(
        query, documents, template=template, use_chat=False, use_blocks=True
    )
    if isinstance(prompt_str, list):
        # Take the main user message
        prompt_str = prompt_str[0]["content"] if prompt_str else ""

    # Split into logical blocks using the separator
    # The rendered prompt has instruction + docs separated + final query part
    segments = prompt_str.split(BLOCK_SEPARATOR)

    # Tokenize each segment independently (like the reference _block_tokenize_batch)
    # This ensures tokenizer boundaries are respected per block
    all_block_input_ids = []
    for seg in segments:
        # clean a bit
        seg = seg.strip()
        if not seg:
            continue
        ids = tokenizer(seg, add_special_tokens=False)["input_ids"]
        all_block_input_ids.append(ids)

    if not all_block_input_ids:
        # fallback
        all_ids = tokenizer(prompt_str, add_special_tokens=add_special_tokens)["input_ids"]
        return {
            "input_ids": all_ids,
            "doc_boundaries": [(0, len(all_ids))],
            "num_blocks": 1,
        }

    # Reconstruct flat ids and compute boundaries
    # Convention from original: block 0 = instruction, middle = docs, last = query/completion
    flat_ids: List[int] = []
    boundaries: List[Tuple[int, int]] = []
    current_pos = 0

    for i, block_ids in enumerate(all_block_input_ids):
        start = current_pos
        flat_ids.extend(block_ids)
        end = current_pos + len(block_ids)
        current_pos = end

        # Document blocks are the middle ones (skip first instruction, last query)
        if 0 < i < len(all_block_input_ids) - 1:
            boundaries.append((start, end))

    # If we didn't get enough middle blocks (edge case), provide reasonable ones
    if not boundaries and len(all_block_input_ids) > 2:
        # fallback approximate
        for i in range(1, len(all_block_input_ids) - 1):
            # recompute
            pass

    result = {
        "input_ids": flat_ids,
        "doc_boundaries": boundaries or [(0, len(flat_ids))],
        "num_blocks": len(all_block_input_ids),
    }
    return result


def chunk_text(text: str, tokenizer, max_tokens: int = 128, overlap: int = 0) -> List[str]:
    """Simple sliding window chunker for long documents."""
    if hasattr(tokenizer, "__call__") and not hasattr(tokenizer, "encode"):
        # simple splitter fallback
        words = text.split()
        if len(words) <= max_tokens:
            return [text]
        return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(tokens) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
        start += max_tokens - overlap
        if start >= len(tokens):
            break
    return chunks


def prepare_chunked_inputs(
    tokenizer,
    query: str,
    documents: List[str],
    template: str = "mistral",
    max_chunk_tokens: int = 128,
    aggregate: str = "max",  # "max", "mean", "sum"
) -> Dict[str, Any]:
    """
    Prepare inputs with document chunking + mapping back to original docs.

    Returns doc_boundaries as before, but internally handles chunks.
    The caller (ranker) can compute per-chunk scores then aggregate per doc.
    """
    chunked_docs = []
    doc_chunk_map: List[List[int]] = []  # for each original doc, list of chunk indices in chunked_docs
    current_chunk_idx = 0

    for doc in documents:
        chunks = chunk_text(doc, tokenizer, max_tokens=max_chunk_tokens)
        start_idx = current_chunk_idx
        chunked_docs.extend(chunks)
        end_idx = current_chunk_idx + len(chunks)
        doc_chunk_map.append(list(range(start_idx, end_idx)))
        current_chunk_idx = end_idx

    # Now treat chunks as "documents" for block tokenization
    prepared = tokenize_with_block_boundaries(
        tokenizer, query, chunked_docs, template=template
    )

    return {
        "input_ids": prepared["input_ids"],
        "doc_boundaries": prepared["doc_boundaries"],  # boundaries for chunks
        "chunk_to_doc": doc_chunk_map,                 # map chunk idx -> original doc idx
        "num_original_docs": len(documents),
        "aggregate": aggregate,
    }

