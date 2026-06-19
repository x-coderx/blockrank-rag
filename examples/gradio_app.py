"""
Gradio demo for BlockRank ranking (requires gradio + model).

Run:
    python -m gradio examples/gradio_app.py
"""

import gradio as gr
from blockrank_rag.ranker import BlockRanker, BlockRankerConfig

ranker = None


def load_model():
    global ranker
    if ranker is None:
        ranker = BlockRanker(BlockRankerConfig(device="cpu"))
        ranker.load()
    return "Model loaded (or was cached)."


def rank_docs(query, docs_text, top_k):
    if not ranker:
        load_model()
    docs = [d.strip() for d in docs_text.strip().split("\n") if d.strip()]
    if not docs:
        return "Provide at least one document."
    results = ranker.rank(query, docs, top_k=int(top_k))
    out = []
    for r in results:
        out.append(f"**#{r.rank+1}** (id={r.doc_id}, score={r.score:.4f})\n{r.doc_text}\n")
    return "\n---\n".join(out)


with gr.Blocks(title="BlockRank Demo") as demo:
    gr.Markdown("# BlockRank RAG Demo\nEfficient in-context ranking using attention signals.")
    with gr.Row():
        load_btn = gr.Button("Load / Reload Model")
        status = gr.Textbox(label="Status", interactive=False)
    load_btn.click(load_model, outputs=status)

    query = gr.Textbox(label="Query", value="capital of france?")
    docs = gr.Textbox(label="Documents (one per line)", lines=8, value="""Paris is the capital of France.
Berlin is the capital of Germany.
London is the capital of the UK.""")
    k = gr.Slider(1, 10, value=3, step=1, label="Top K")
    run = gr.Button("Rank")
    out = gr.Markdown()

    run.click(rank_docs, inputs=[query, docs, k], outputs=out)

if __name__ == "__main__":
    demo.launch()
