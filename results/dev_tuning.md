# Dev-set tuning (200 queries disjoint from the 100 eval queries; no LLM used)

Date: 2026-09-30. Index: BAAI/bge-small-en-v1.5 over 48,605 chunks / 27,977 passages; BM25 k1=1.2 b=0.75.
With n=200, differences of about 0.02 are within noise; these runs only guide the choice, they are not the final result.

| Setup | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| Lexical (BM25) | 0.552 | 0.708 | 0.855 | 0.749 |
| Dense (bge-small) | 0.500 | 0.641 | 0.783 | 0.677 |
| Hybrid, RRF k=60, equal weights | 0.558 | 0.733 | 0.823 | 0.744 |
| Hybrid, weighted min-max, alpha=0.4 | 0.582 | 0.746 | 0.858 | 0.775 |
| Hybrid RRF, lexical weight 2, k=30 | 0.585 | 0.752 | 0.855 | 0.774 |
| + cross-encoder rerank (bge-reranker-base, top-30), pure | 0.533 | 0.697 | 0.777 | 0.693 |
| + rerank blended (0.3 rerank / 0.7 fused z-scores) | 0.586 | 0.762 | 0.863 | 0.790 |

Findings:
- BM25 beats dense on every metric here; equal-weight RRF is no better than BM25 on nDCG. Down-weighting dense (alpha 0.3-0.4) helps.
- Reranking alone hurts; a blend gains about +0.015 nDCG, within noise, at roughly 9 s/query on 2 CPU cores versus about 25 ms without it. Not selected.
- Provisional strongest variant: `configs/best.yaml` (weighted hybrid, alpha=0.4), pending the query-expansion test.

Reproduce: `python -m eval.sweep`, `python -m eval.sweep_rerank BAAI/bge-reranker-base 30`.
