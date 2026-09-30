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
- Query expansion (below) gives a consistent gain, so the strongest variant is `configs/best.yaml`: weighted hybrid (alpha 0.4) + query expansion.

## Query expansion on the same 200 dev queries (model openai/gpt-6-luna, 3 alternates, reasoning disabled)

LLM cost for all 200 expansions: $0.0096 (OpenRouter usage accounting). Paired difference in nDCG@10 vs. the first row, mean (standard error):

| Setup | R@5 | R@10 | MRR@10 | nDCG@10 | paired dnDCG |
|---|---|---|---|---|---|
| Hybrid weighted a=0.4, no QE | 0.582 | 0.746 | 0.858 | 0.775 | - |
| + QE, alternate weight 0.3 | 0.607 | 0.772 | 0.873 | 0.800 | +0.025 (0.008) |
| + QE, alternate weight 0.5 | 0.610 | 0.773 | 0.870 | 0.800 | +0.025 (0.009) |
| + QE, alternate weight 1.0 | 0.613 | 0.778 | 0.857 | 0.799 | +0.024 (0.010) |
| Hybrid RRF equal weights, no QE | 0.558 | 0.733 | 0.823 | 0.744 | -0.031 (0.008) |
| + QE, RRF equal weights, alt 0.5 | 0.596 | 0.758 | 0.857 | 0.785 | +0.009 (0.011) |

Reproduce: `python -m eval.sweep_qe` (needs the OpenRouter key; repeat runs hit the cache and cost nothing).

Reproduce: `python -m eval.sweep`, `python -m eval.sweep_rerank BAAI/bge-reranker-base 30`.
