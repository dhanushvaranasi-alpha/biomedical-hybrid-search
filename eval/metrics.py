"""Retrieval metrics with binary relevance. ranked: list of pmids best first; gold: set of relevant pmids."""
import math


def recall_at_k(ranked, gold, k):
    return len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0


def mrr_at_k(ranked, gold, k=10):
    for i, p in enumerate(ranked[:k], start=1):
        if p in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, gold, k=10):
    dcg = sum(1.0 / math.log2(i + 1) for i, p in enumerate(ranked[:k], start=1) if p in gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0


def all_metrics(ranked, gold):
    g = set(gold)
    return {"recall@5": recall_at_k(ranked, g, 5), "recall@10": recall_at_k(ranked, g, 10),
            "mrr@10": mrr_at_k(ranked, g, 10), "ndcg@10": ndcg_at_k(ranked, g, 10)}
