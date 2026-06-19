"""
Utilities adapted from the original BlockRank research code (blockrank-original/src/blockrank/utils.py).

Includes:
- calculate_accuracy (with qrels support for nDCG/MRR)
- load_qrels
- remap_documents (light version)
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Tuple
import json

try:
    import pytrec_eval
except ImportError:
    pytrec_eval = None


def load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """Load qrels (BEIR or TREC format)."""
    qrels: Dict[str, Dict[str, int]] = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or (i == 0 and line.startswith("query-id")):
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) == 4:  # TREC
                qid, _, did, rel = parts
            elif len(parts) == 3:  # BEIR
                qid, did, rel = parts
            else:
                continue
            rel = int(rel)
            qrels.setdefault(str(qid), {})[str(did)] = rel
    return qrels


def calculate_accuracy(
    predictions: List[int | List[int]],
    eval_ds: Dict | Any,
    qrels: Optional[Dict[str, Dict[str, int]]] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Calculate accuracy and ranking metrics (adapted from original BlockRank).

    predictions: list of predicted doc indices (or top-k lists)
    eval_ds: must support ['answer_ids'], optionally ['query_id'], ['remapped_doc_ids']
    """
    n = len(predictions)
    if isinstance(getattr(eval_ds, "get", lambda k: None)("answer_ids"), list):
        ground_truth = eval_ds["answer_ids"][:n]
    else:
        ground_truth = list(eval_ds["answer_ids"])[:n] if hasattr(eval_ds, "__getitem__") else []

    normalized_preds: List[List[int]] = []
    for p in predictions:
        if isinstance(p, list):
            normalized_preds.append([int(x) for x in p])
        else:
            normalized_preds.append([int(p)])

    ground_truth = [
        [int(x) for x in (g if isinstance(g, (list, tuple)) else [g])]
        for g in ground_truth
    ]

    correct = 0
    invalid = 0
    for pred, gt in zip(normalized_preds, ground_truth):
        try:
            top1 = [pred[0]] if pred else []
            if set(top1).issubset(set(gt)):
                correct += 1
        except Exception:
            invalid += 1
        if verbose:
            print(f"Pred: {pred[:3]} | GT: {gt}")

    metrics: Dict[str, float] = {
        "accuracy": 100 * correct / n if n > 0 else 0.0,
        "exact_match": correct,
        "total": n,
        "invalid_predictions": invalid,
        "invalid_rate": 100 * invalid / n if n > 0 else 0.0,
    }

    if qrels is not None and pytrec_eval is not None:
        run: Dict[str, Dict[str, float]] = {}
        query_ids = eval_ds.get("query_id", [str(i) for i in range(n)])[:n]
        remapped = eval_ds.get("remapped_doc_ids", [list(range(20)) for _ in range(n)])[:n]

        for i, pred_ranking in enumerate(normalized_preds):
            qid = str(query_ids[i])
            if qid not in qrels:
                continue
            run[qid] = {}
            for rank, d_idx in enumerate(pred_ranking):
                if d_idx < len(remapped[i]):
                    did = str(remapped[i][d_idx])
                    run[qid][did] = float(len(pred_ranking) - rank)
            for d_idx, did in enumerate(remapped[i]):
                if d_idx not in pred_ranking:
                    run[qid][str(did)] = 0.0

        measures = {"ndcg_cut_1", "ndcg_cut_3", "ndcg_cut_5", "ndcg_cut_10", "recip_rank"}
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
        results = evaluator.evaluate(run)

        ndcg = {k: [] for k in [1, 3, 5, 10]}
        mrr = []
        for res in results.values():
            for k in ndcg:
                ndcg[k].append(res.get(f"ndcg_cut_{k}", 0.0))
            mrr.append(res.get("recip_rank", 0.0))

        for k in ndcg:
            if ndcg[k]:
                metrics[f"ndcg@{k}"] = 100 * sum(ndcg[k]) / len(ndcg[k])
            if mrr:
                metrics[f"mrr@{k}"] = 100 * sum(mrr) / len(mrr)

    return metrics


def remap_documents(
    documents: Dict[Any, str],
    answer_ids: Optional[List[str]],
    num_samples: int,
    seed: Optional[int] = None,
    add_padding_docs: bool = True,
) -> Tuple[List[str], List[str], List[int]]:
    """Lightweight version of the original remap (for data prep)."""
    import random
    if seed is not None:
        random.seed(seed)
    answer_ids = answer_ids or []
    all_ids = list(documents.keys())
    if num_samples > 0 and num_samples < len(all_ids):
        negs = [x for x in all_ids if x not in answer_ids]
        sampled = answer_ids + random.sample(negs, min(len(negs), num_samples - len(answer_ids)))
    else:
        sampled = all_ids[:num_samples] if num_samples > 0 else all_ids

    if add_padding_docs:
        while len(sampled) < num_samples:
            sampled.append(f"dummy_{len(sampled)}")
            documents[sampled[-1]] = "Padding document."

    random.shuffle(sampled)
    remapped = [documents[s] for s in sampled]
    remapped_ans = [sampled.index(a) for a in answer_ids if a in sampled]
    return remapped, sampled, remapped_ans
