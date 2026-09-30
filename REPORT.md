# Report: Biomedical Hybrid Search on rag-mini-bioasq

**Selected best configuration:** `configs/best.yaml`: weighted hybrid retrieval (BM25 + dense `bge-small-en-v1.5`, min-max fusion, alpha = 0.4 on dense)
over the original query plus 3 LLM query expansions (weight 0.5 each), top-5 passages to the answer model.

## Setup in one paragraph

Corpus: 27,977 non-empty PubMed passages (12,244 empty placeholders removed). BM25 (`bm25s`) over whole passages; dense retrieval with ChromaDB over
48,605 chunks. Answers: `openai/gpt-6-luna`; judge: RAGAS 0.4.3 with `google/gemini-3.8-flash` (fixed prompts, temperature 0, low reasoning effort).
The same search and answer code serves the API and this evaluation. The 100 queries (`eval/queries_100.jsonl`, seed 42) are stratified over
yes/no, definition, list, treatment, mechanism and effect questions (17/17/17/17/16/16) and every configuration ran on exactly those IDs.
All fusion and expansion settings were tuned on a disjoint 200-query dev set (`results/dev_tuning.md`), never on these 100.

## Comparison table

| Config | R@5 | R@10 | MRR@10 | nDCG@10 | Answer correctness | Groundedness | Context relevance | Insufficient % | Latency p50 / p95 (ms) | LLM cost / 100 q |
|---|---|---|---|---|---|---|---|---|---|---|
| Lexical (BM25) | 0.616 | 0.759 | 0.860 | 0.790 | 0.677 | 0.975 (n=85) | 0.920 | 15 | 1689 / 3813 | $0.68 |
| Dense (bge-small) | 0.500 | 0.632 | 0.838 | 0.692 | 0.647 | 0.979 (n=83) | 0.875 | 17 | 1497 / 2391 | $0.69 |
| Hybrid (RRF) | 0.606 | 0.762 | 0.868 | 0.785 | 0.703 | 0.993 (n=87) | 0.912 | 13 | 1439 / 2196 | $0.68 |
| Hybrid + query expansion | 0.613 | 0.771 | 0.895 | 0.807 | 0.710 | 0.982 (n=91) | 0.932 | 9 | 3151 / 4406 | $0.71 |
| Best: weighted hybrid + QE | 0.630 | 0.792 | 0.902 | 0.829 | 0.703 | 0.984 (n=89) | 0.925 | 11 | 3092 / 4425 | $0.70 |

Latency = retrieval + query-expansion LLM call + answer LLM call (p50 / p95, ms; LLM times are the original call latencies, measured from this
environment over the internet, so they will vary). Cost is for the whole 100-query run including the judge; serving cost without the judge is
about $0.025 per 100 queries for plain hybrid and $0.029 for the best configuration. Groundedness is scored only for answered questions
(n shown); insufficient-evidence answers count as 0 for correctness. Full details, paired differences and per-type numbers: `results/comparison.md`.

## What won, and why

**The best configuration (weighted hybrid + query expansion) won on every retrieval metric.** nDCG@10 is 0.829 versus 0.790 for BM25, 0.785 for plain RRF hybrid
and 0.807 for hybrid + expansion. Paired over the 100 queries (95% bootstrap intervals): +0.038 nDCG@10 vs BM25 [+0.010, +0.069], +0.044 vs plain hybrid
[+0.020, +0.071], +0.022 vs hybrid + expansion [+0.008, +0.037]; Recall@10 is +0.033 vs BM25 [+0.007, +0.062]. These intervals exclude zero.

Why it won, based on the dev-set sweeps and the per-query results:

- **BM25 is strong on this corpus and dense is weak.** Dense-only is the worst retriever (nDCG@10 0.692). Plain equal-weight RRF hybrid therefore did **not** beat BM25 on nDCG (-0.006, interval [-0.032, +0.020]);
  its gain over dense is large (+0.093). Down-weighting the dense signal (alpha 0.4) fixed this on the dev set.
- **Query expansion adds the words the abstracts actually use.** For "What is the route of administration of vaxchora?" the expansions add "cholera vaccine ... oral";
  nDCG@10 goes from 0.39 (plain hybrid) to 1.00, and the answer is "orally, single dose" with a valid citation.
- Expansion is not free of harm: for "What is the indication for KYMRIAH?" the expansions diluted the query and nDCG@10 fell from 0.63 to 0.43.

**Answer quality did not separate reliably.** Answer correctness is 0.703 for the best configuration versus 0.677 (BM25), 0.703 (hybrid), 0.710 (hybrid + expansion) and 0.647 (dense).
The paired intervals for these differences include zero, so we cannot claim the retrieval gain shows up as better answers with 100 queries. The answer stage is limited by other factors
(next sections). Groundedness is high everywhere (0.975 to 0.993) and context relevance is 0.875 to 0.932. All citations were valid: 0 answers contained a citation outside the retrieved context, in every configuration.

## Trade-off: latency and cost

Query expansion adds one LLM call before retrieval: median end-to-end latency rises from 1.44 s (plain hybrid) to 3.09 s (best), p95 from 2.2 s to 4.4 s.
Retrieval itself is cheap (57 ms p50 with expansion, 29 ms without). Serving cost stays tiny: expansion adds about $0.004 per 100 queries. If latency matters more than the last 0.04 nDCG,
plain hybrid with the weighted fusion and no expansion is the cheaper choice; the cross-encoder reranker we tested on the dev set was rejected (about 9 s per query on 2 CPU cores
for a gain within noise).

## Successful searches

1. **"Does ziconotide bind to N-type calcium channels?"** (nDCG@10 1.00): "Yes. Ziconotide selectively binds to and blocks neuronal N-type calcium channels" citing PMID 16845440 and 22428804, both in the retrieved context (screenshot: `docs/screenshots/ui-answered.png`).
2. **"What is the route of administration of vaxchora?"**: query expansion turned a brand name into "cholera vaccine ... oral" terms (nDCG@10 0.39 to 1.00); answer: "administered orally as a single dose [PMID:28622736]".
3. **"What is known as the cause of subacute thyroiditis?"** (nDCG@10 1.00, Recall@10 1.00): cause unclear, viral infection frequently implicated (Coxsackie, cytomegalovirus, ...), with citations. The same question's answer in the BM25 run was wrongly rejected by our first citation checker (see Limitations) and is accepted by the fixed one.

## Failure cases

1. **Retrieval failure, correct refusal.** "Is Bcl-2-like protein 1 an pro apoptotic protein?" (nDCG@10 0.22): the model reports that the retrieved passages do not identify the protein, and the system shows an insufficient-evidence state instead of guessing
   (screenshot: `docs/screenshots/ui-insufficient.png`). The dataset answer is "No, it is an anti-apoptotic protein".
2. **Good retrieval, over-cautious refusal.** "List the four most important interferonopathies" (Recall@10 0.88): the passages mention Aicardi-Goutieres syndrome, USP18 deficiency, familial chilblain lupus and others, but no passage says which four are "most important",
   so the model refuses. Same pattern for "How does parathyroid hormone affect circulating levels of periostin?" (nDCG@10 1.00): the passages describe periostin mRNA in osteoblasts, not circulating levels. The strict "answer only from the passages" rule trades recall of answers for safety.
3. **Plausible answer scored 0 against an idiosyncratic gold answer.** "What is a J pouch?": the model gives an ileal pouch after ileal pouch-anal anastomosis (cited); the dataset answer describes a colonic J-pouch after low anterior resection for rectal carcinoma.
   "What is the mechanism of action of decitabine?": the model gives DNMT1 depletion and demethylation; the dataset answer is about p21WAF1 reactivation in AML cell lines. Dataset answers are often excerpts from one abstract, which makes the correctness metric noisy.
4. **Query expansion hurt:** the KYMRIAH example above.

**Manual review** (`results/review_best.csv`, verdict and notes for every row): for the best configuration I read the 12 cases where retrieval metrics were good but the judged correctness was low, comparing the model answer, the dataset answer and the cited passage text.
Seven were "insufficient evidence" outcomes and five were answered. Of the seven refusals, four were justified (three where the retrieved text really lacks the answer, one borderline about "circulating" levels), two were over-cautious (a partial list was supportable: sarcopenia trials, interferonopathies), and one was a false rejection by the citation validator (the Disambiguate answer was accurate and supported, but a single citation at the end of several sentences was counted as "mostly uncited"; this is a remaining limitation of the validator).
All five answered cases were correct and grounded; the judge scored them low because the dataset answer is a narrower or differently focused excerpt. Two of the gold passage sets contain unrelated passages (dataset noise), so their recall overstates retrieval.
Net: the low correctness scores in this group are mostly judge and gold-answer mismatch or strict abstention, not wrong answers. No invalid citations were found. This review was done by me reading the passages, not by an independent annotator.

## Limitations and honest notes

- **A checker bug was found and fixed after seeing eval output.** After the first configuration finished, 32% of its answers were marked insufficient evidence. Inspection showed my citation validator treated "Claim. [PMID:1]" as a claim with no citation plus a citation-only fragment,
  and counted "Yes." as an uncited claim; 17 of 100 answers were wrongly rejected. I fixed the validator (added tests), restarted, and all five configurations reported here use the fixed version. This changes how answers are graded, not the retrieval ranking.
  Invalid or missing citations are still rejected (unit-tested).
- **The corpus is easy.** The mini corpus is the union of every question's relevant passages, so absolute scores are higher than on an open corpus. Also 30% of passages are empty placeholders and were excluded (333 questions with no usable relevant passage were dropped).
- **100 queries is small.** Differences under about 0.03 are not reliable; per-type numbers (16 to 17 queries each) are only indicative.
- **Untried:** a biomedical embedding model (e.g. MedCPT), tuned chunking, and a second judge model. `bge-small` was chosen for CPU speed, not by comparison.
- **Judge:** one judge model at low reasoning effort; its scores were not calibrated against human labels beyond the manual review above.
- **Cost of the whole project's LLM use:** $3.30 of OpenRouter credit (development tests, dev sweeps and the five evaluation runs).

## Reproduce

`make test` (25 unit tests), then `make eval` (needs `OPENROUTER_API_KEY`; cached responses make re-runs free). Raw per-query results for every configuration are in `results/runs/`.
