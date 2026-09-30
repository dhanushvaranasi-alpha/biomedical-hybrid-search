"""Rank fusion across retrievers and across (original + expanded) queries."""
from collections import defaultdict


def rrf(lists, k_rrf: int = 60):
    """lists: iterable of (weight, ranked [(pmid, score)]). Returns [(pmid, fused_score)] best first."""
    acc = defaultdict(float)
    for w, ranked in lists:
        for rank, (pmid, _) in enumerate(ranked, start=1):
            acc[pmid] += w / (k_rrf + rank)
    return sorted(acc.items(), key=lambda x: (-x[1], x[0]))


def _minmax(ranked):
    if not ranked:
        return {}
    vals = [s for _, s in ranked]
    lo, hi = min(vals), max(vals)
    return {p: ((s - lo) / (hi - lo) if hi > lo else 1.0) for p, s in ranked}


def weighted(groups, alpha: float = 0.5):
    """groups: iterable of (query_weight, lexical_ranked, dense_ranked). Min-max per list, alpha on dense."""
    acc = defaultdict(float)
    for w, lex, den in groups:
        for p, s in _minmax(lex).items():
            acc[p] += w * (1 - alpha) * s
        for p, s in _minmax(den).items():
            acc[p] += w * alpha * s
    return sorted(acc.items(), key=lambda x: (-x[1], x[0]))
