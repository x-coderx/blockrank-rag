"""
Synthetic data + minimal training smoke test.

Generates data in the exact format expected by the original BlockRank repo
(JSONL with query, documents, answer_ids).

For a real smoke, after generating data you can:
    pip install -e ../blockrank-original   # or the reference
    python ../blockrank-original/scripts/train.py --config configs/tiny.yaml

This script also demonstrates how one would start a tiny LoRA run
using TRL (requires the full dependencies).
"""

import json
from pathlib import Path
import random


def generate_synthetic_icr(n: int = 50, docs_per: int = 12) -> list:
    """Generate ICR training data compatible with BlockRank format."""
    topics = ["France", "Germany", "Italy", "Spain", "UK", "Japan", "Brazil", "Australia"]
    examples = []
    for i in range(n):
        topic = random.choice(topics)
        q = f"What is the capital of {topic}?"
        docs = []
        ans = []
        for j in range(docs_per):
            if j == 0:
                text = f"The capital city of {topic} is well known and contains important government buildings and landmarks."
                ans.append(str(j))
            else:
                text = f"This is an unrelated document about topic {j} containing random facts, history, or other information."
            docs.append({"doc_id": str(j), "text": text})
        examples.append({
            "query": q,
            "query_id": f"q_{i}",
            "documents": docs,
            "answer_ids": ans,
        })
    return examples


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    data = generate_synthetic_icr(50)
    out = data_dir / "synthetic_icr.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(data)} ICR training examples to {out}")

    # Also write a tiny config example for reference
    cfg = {
        "model": {
            "model_name_or_path": "facebook/opt-125m",   # tiny for smoke
            "use_blockrank": True,
            "attn_implementation": "default_blockrank",
            "use_lora": True,
        },
        "data": {
            "data_path": str(out),
            "num_documents": 8,
            "max_block_length": 128,
        },
        "training": {
            "output_dir": "outputs/blockrank-smoke",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "use_aux_loss": True,
            "aux_layer_idx": 4,   # for very small model
            "aux_loss_weight": 0.1,
            "max_steps": 20,      # smoke only
        }
    }
    cfg_path = data_dir / "smoke_train_config.yaml"
    import yaml
    with cfg_path.open("w") as f:
        yaml.safe_dump(cfg, f)
    print(f"Wrote example training config to {cfg_path}")

    print("\nNext steps for real training smoke:")
    print("  1. pip install -e ../blockrank-original   (or git clone + editable)")
    print("  2. (Optional) pip install -e .[rag]      # for better data tools")
    print("  3. Use the reference scripts/train.py with a tiny config + --max-steps 20")
    print("  See docs/research-notes.md and blockrank-original/docs/TRAINING.md")


if __name__ == "__main__":
    main()
