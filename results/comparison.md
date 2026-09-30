# Comparison table (fixed 100 queries, identical IDs for every run)

| Config | R@5 | R@10 | MRR@10 | nDCG@10 | Answer correctness | Groundedness | Context relevance | Insufficient % | Latency p50 / p95 (ms) | LLM cost / 100 q |
|---|---|---|---|---|---|---|---|---|---|---|
| Lexical (BM25) | 0.616 | 0.759 | 0.860 | 0.790 | 0.677 | 0.975 (n=85) | 0.920 | 15 | 1689 / 3813 | $0.68 |
| Dense (bge-small) | 0.500 | 0.632 | 0.838 | 0.692 | 0.647 | 0.979 (n=83) | 0.875 | 17 | 1497 / 2391 | $0.69 |
| Hybrid (RRF) | 0.606 | 0.762 | 0.868 | 0.785 | 0.703 | 0.993 (n=87) | 0.912 | 13 | 1439 / 2196 | $0.68 |
| Hybrid + query expansion | 0.613 | 0.771 | 0.895 | 0.807 | 0.710 | 0.982 (n=91) | 0.932 | 9 | 3151 / 4406 | $0.71 |
| Best: weighted hybrid + QE | 0.630 | 0.792 | 0.902 | 0.829 | 0.703 | 0.984 (n=89) | 0.925 | 11 | 3092 / 4425 | $0.70 |

Latency = retrieval + query-expansion LLM + answer LLM (LLM stage times are the original call latencies). Cost = expansion + answers + judge for the 100 queries.
Groundedness is only scored for answered questions (n shown). Insufficient-evidence answers score 0 on correctness. Judge: openrouter/google/gemini-3.8-flash, judge_v1.

## Paired differences (mean, 95% bootstrap CI over the 100 queries)

| Comparison | dnDCG@10 | dRecall@10 | dCorrectness |
|---|---|---|---|
| Best: weighted hybrid + QE vs Lexical (BM25) | +0.038 [+0.010, +0.069] * | +0.033 [+0.007, +0.062] * | +0.025 [-0.025, +0.077] |
| Best: weighted hybrid + QE vs Hybrid (RRF) | +0.044 [+0.020, +0.071] * | +0.030 [+0.010, +0.053] * | +0.000 [-0.052, +0.055] |
| Best: weighted hybrid + QE vs Hybrid + query expansion | +0.022 [+0.008, +0.037] * | +0.022 [-0.001, +0.045] | -0.007 [-0.037, +0.018] |
| Hybrid (RRF) vs Lexical (BM25) | -0.006 [-0.032, +0.020] | +0.003 [-0.025, +0.031] | +0.025 [-0.028, +0.075] |
| Hybrid + query expansion vs Hybrid (RRF) | +0.022 [-0.003, +0.049] | +0.008 [-0.022, +0.041] | +0.007 [-0.040, +0.055] |
| Hybrid (RRF) vs Dense (bge-small) | +0.093 [+0.062, +0.126] * | +0.130 [+0.087, +0.179] * | +0.055 [+0.000, +0.113] |

`*` = the 95% interval excludes 0.

## nDCG@10 by question type

| Type (n) | Lexical (BM25) | Dense (bge-small) | Hybrid (RRF) | Hybrid + query expansion | Best: weighted hybrid + QE |
|---|---|---|---|---|---|
| definition (17) | 0.841 | 0.657 | 0.816 | 0.849 | 0.876 |
| effect (16) | 0.767 | 0.706 | 0.766 | 0.779 | 0.801 |
| list (17) | 0.801 | 0.699 | 0.776 | 0.766 | 0.792 |
| mechanism (16) | 0.801 | 0.664 | 0.796 | 0.857 | 0.854 |
| treatment (17) | 0.792 | 0.727 | 0.801 | 0.832 | 0.862 |
| yesno (17) | 0.740 | 0.698 | 0.751 | 0.759 | 0.790 |

## Citation validity audit (code-enforced)

| Config | Answered | Insufficient | Answers with an invalid citation removed | Cited PMIDs outside context (after validation) |
|---|---|---|---|---|
| Lexical (BM25) | 85 | 15 | 0 | 0 |
| Dense (bge-small) | 83 | 17 | 0 | 0 |
| Hybrid (RRF) | 87 | 13 | 0 | 0 |
| Hybrid + query expansion | 91 | 9 | 0 | 0 |
| Best: weighted hybrid + QE | 89 | 11 | 0 | 0 |
