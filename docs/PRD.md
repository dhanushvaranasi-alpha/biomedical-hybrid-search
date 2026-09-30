# PRD — Biomedical Hybrid Search

| | |
|---|---|
| **Status** | v0.2, implemented; outcomes in [REPORT.md](../REPORT.md) |
| **Owner** | Dhanush Varanasi |
| **Date** | 2026-09-30 |
| **Repo** | `dhanushvaranasi-alpha/biomedical-hybrid-search` |
| **Related** | [ARCHITECTURE.md](./ARCHITECTURE.md) · [SYSTEM_DESIGN.md](./SYSTEM_DESIGN.md) |

---

## 1. Summary

Build a small biomedical search and question-answering (RAG) application over the Hugging Face dataset **`rag-datasets/rag-mini-bioasq`**. The app retrieves relevant PubMed passages using lexical, dense and hybrid retrieval (with LLM query expansion), shows the evidence to the user, and uses an LLM to write a final answer that cites only the retrieved passage IDs. An offline evaluation over a fixed 100-query set is used to pick the best retrieval configuration.

## 2. Problem & goals

Biomedical questions need answers that are **traceable to evidence**. A plain LLM answer can't be trusted; a plain keyword search misses synonyms and paraphrases; a plain vector search misses exact gene/drug names.

**Goals**

1. Implement three retrieval modes over the same indexed passages: **lexical (BM25)**, **dense (vector)**, **hybrid (fusion)**.
2. Add **query expansion** that produces a small set of alternate queries while always keeping the original.
3. Provide a **UI** that shows query + expansions, ranked evidence, and a grounded answer with inline citations — or an explicit **"insufficient evidence"** state.
4. Run a **reproducible offline evaluation** (retrieval metrics + LLM-judge + manual review) and select the **best hybrid configuration** from data.
5. Document the system with a **system-design diagram** and a README explaining the main choices.

**Non-goals**

- Clinical-grade accuracy or medical advice. The UI carries a "research demo, not medical advice" disclaimer.
- Live PubMed ingestion or corpora beyond `rag-mini-bioasq`.
- User accounts, auth, multi-tenant hosting.
- Fine-tuning embedding or LLM models.

## 3. Users

| User | Need |
|---|---|
| **Evaluator / grader** | Run the app and the eval script end-to-end, see the comparison table, verify the chosen config is backed by results. |
| **Biomedical researcher (persona)** | Ask a question, scan evidence quickly, trust the answer because every claim links to a passage. |
| **Developer (me)** | Swap configurations (retriever, fusion, k, expansion, reranker, LLM) without code changes and compare runs. |

## 4. Dataset (verified by downloading and inspecting the actual files, 2026-09-30)

| File | Rows | Columns |
|---|---|---|
| `data/test.parquet` (question-answer-passages) | **4,719** | `question`, `answer`, `relevant_passage_ids` (string like `"[20598273, 6650562, …]"`), `id` (parquet index) |
| `data/passages.parquet` (text-corpus) | **40,221** | `passage`, `id` (parquet index; PubMed ID) |

The Hugging Face dataset card lists 4,720 and 40,200 rows; the downloaded files contain the counts above. License: CC-BY-2.5 (derived from BioASQ Task 11b).

**Data-quality findings (measured):**

- **12,244 of 40,221 passages (30%) are empty placeholders** (text `"nan"` ×12,220 or `"1."` ×24). They cannot be retrieved by content. They are excluded from all indexes → **27,977 indexed passages**, split into **48,605 chunks** for the dense index.
- 100% of `relevant_passage_ids` exist in the corpus, but many point at the empty passages: of 42,608 gold IDs, **29,657 are usable** (non-empty). **333 questions have no usable gold passage** and are excluded, leaving **4,386 usable questions**.
- All metrics use `gold_in_corpus` = gold IDs that are non-empty passages.
- The corpus is the union of the gold passages of all questions, so every indexed passage is relevant to some question; retrieval scores will be higher than on an open-domain corpus. The report will state this.
- No question-type column exists; types are derived by rules (`qtype_v2`) and spot-checked.
- Passage `id` is a PubMed ID → source link `https://pubmed.ncbi.nlm.nih.gov/<id>/`.

## 5. Functional requirements

Priority: **P0** = required by the brief, **P1** = strongly desirable, **P2** = nice to have.

### 5.1 Retrieval (Brief Step 1 — 20 marks)

| ID | Requirement | P |
|---|---|---|
| R1 | Lexical retrieval with BM25 over the passage corpus. | P0 |
| R2 | Dense vector retrieval over the **same** passages (same IDs). | P0 |
| R3 | Hybrid retrieval combining lexical + dense signals (configurable fusion: RRF, weighted score). | P0 |
| R4 | Query expansion before retrieval: generate N (default 3) alternate queries; the original query is always kept and searched. | P0 |
| R5 | Every search records query, expansions, config, retrieved passage IDs and per-retriever + fused scores. | P0 |
| R6 | Retrieval is configurable by a single config object (mode, k, fusion params, expansion on/off, reranker on/off). | P0 |
| R7 | Optional cross-encoder re-ranking stage (candidate for "strongest variant"). | P1 |
| R8 | Long passages are chunked for dense indexing; results are aggregated back to passage IDs. | P1 |

### 5.2 User interface (Brief Step 2 — 20 marks)

| ID | Requirement | P |
|---|---|---|
| U1 | Biomedical question input and a Search action. | P0 |
| U2 | Display the original query and its expansions. | P0 |
| U3 | Ranked evidence list: rank, passage ID, fused score (+ lexical/dense component scores), passage text snippet, PubMed source link. | P0 |
| U4 | Final LLM answer with inline citations `[PMID]` that link/scroll to the evidence card. | P0 |
| U5 | Explicit **"Insufficient evidence"** state when the retrieved context cannot support an answer. | P0 |
| U6 | Final LLM call answers **only** from the selected retrieved passages. | P0 |
| U7 | Mode selector (lexical / dense / hybrid / hybrid+QE / best) for side-by-side exploration. | P1 |
| U8 | Show latency and estimated cost per request. | P1 |
| U9 | Sample-question chips from the eval set. | P2 |
| U10 | Hostable locally; frontend deployable to Netlify. | P0 |

### 5.3 Offline evaluation (Brief Steps 3 & 4 — 40 marks)

| ID | Requirement | P |
|---|---|---|
| E1 | Fixed set of **100 queries** sampled from the dataset with a fixed seed, mixing yes/no, definition, list, treatment, mechanism, effect where possible. Committed to the repo. | P0 |
| E2 | Same 100 query IDs used for every run. | P0 |
| E3 | Configurations: Lexical, Dense, Hybrid, Hybrid + QE, Strongest variant. | P0 |
| E4 | Retrieval metrics: Recall@5, Recall@10, MRR@10, nDCG@10 using `relevant_passage_ids`. | P0 |
| E5 | Latency (p50/p95 per stage) and cost (tokens × price) per run. | P0 |
| E6 | LLM-judge metrics via **RAGAS** (or DeepEval): answer correctness, groundedness/faithfulness, context relevance. | P0 |
| E7 | Judge prompt, model and settings **fixed and versioned** across runs. | P0 |
| E8 | Citation validity checked **in code** (cited IDs ⊆ provided context IDs; answer sentences carry citations). | P0 |
| E9 | Manual review sheet of cases where judge and retrieval metrics disagree. | P0 |
| E10 | One command reproduces all runs and the comparison table. | P0 |

### 5.4 System design & docs (Brief Step 5 — 20 marks)

| ID | Requirement | P |
|---|---|---|
| D1 | System-design diagram showing offline data path, online query path, user-facing path, experiment path. | P0 |
| D2 | README explaining: storage/vector DB, embedding model, fusion method, final-answer model, where citation validation is enforced. | P0 |

## 6. Non-functional requirements

| Area | Target |
|---|---|
| **Latency** (local, warm, excl. LLM answer) | Retrieval p95 < 500 ms without QE; QE adds one LLM call. End-to-end answer p95 < 10 s. |
| **Cost** | Full 5-config eval (100 queries, answers + judge) should be runnable on a small budget; report actual $ per run. |
| **Reproducibility** | Fixed seed, pinned model names/versions, cached LLM outputs (expansions, answers) keyed by prompt+model+input, config hash per run. |
| **Provider-agnostic LLM** | All LLM calls go through one adapter; provider/model chosen by config/env. No provider-specific code in pipeline logic. |
| **Portability** | Runs on a laptop (CPU OK; GPU optional). Index build is a one-time offline step. |
| **Safety** | Disclaimer; no answer without citations; "insufficient evidence" preferred over guessing. |
| **Secrets** | API keys only via `.env`, never committed. |

## 7. Evaluation plan (summary)

1. **Query set**: parse all 4,720 questions → keep those with ≥1 relevant ID present in the corpus → classify question type (rule-based, optionally LLM-assisted, then spot-checked) → stratified sample of 100 with seed 42 → save `eval/queries_100.jsonl` (id, question, gold answer, relevant IDs in corpus, type).
2. **Runs**: each config is a YAML file; the runner executes the *same* online pipeline used by the API and logs everything under `runs/<run_id>/`.
3. **Metrics**: retrieval metrics per query and averaged; RAGAS metrics with a fixed judge; latency and cost.
4. **Selection**: best hybrid config = highest nDCG@10 / Recall@10 with judge metrics not worse, then trade-off vs latency/cost explicitly stated.
5. **Manual review**: ≥10 disagreement cases (high recall + low judge score and vice versa), plus a few clear successes and failures for the report.

## 8. Deliverables (from the brief)

1. Working UI + source code (this repo).
2. Fixed 100-query eval set + reproducible evaluation script.
3. Short report with **one comparison table**.
4. Selected **best hybrid configuration**, backed by results — what won, why, latency/cost trade-off, a few successes and failures.
5. System-design diagram + README choices.

## 9. Success criteria

- All P0 requirements met and demonstrable.
- `make eval` (or equivalent) regenerates the comparison table from scratch.
- Chosen configuration beats plain Lexical and plain Dense on Recall@10 **and** nDCG@10, or the report honestly explains why not.
- Zero answers in the eval with invalid citations (enforced in code; any LLM-produced invalid ID is stripped or triggers the insufficient-evidence path, and is counted).

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Empty placeholder passages listed as gold → unreachable targets | Excluded from indexes and from gold (`gold_in_corpus`); 333 questions dropped; numbers reported. |
| Very long passages dilute embeddings | Chunk for dense index; max-pool chunk scores per passage. |
| Query expansion drifts off-topic | Keep original query with higher weight; limit to 3 expansions; prompt for synonyms/abbreviations only. |
| LLM-judge variance | temperature 0, fixed judge model and prompts, cache judge outputs, report judge version. |
| Hallucinated citations | Structured output + code-level validation (§ SYSTEM_DESIGN). |
| API cost for 5 configs × 100 queries × judge | Cache expansions/answers; reuse expansions across configs; small models for generation. |
| Netlify frontend needs a reachable backend | Default to local; if deployed, host FastAPI separately and set `VITE_API_BASE_URL`. |

## 11. Open questions

1. ~~Which LLM provider/models to use?~~ **Decided (2026-09-30): one OpenRouter API key**, called through LiteLLM. Roles: a cheap OpenAI model (Luna tier) for query expansion and final answers; a Google model (Gemini 3.8 Flash) as the fixed RAGAS judge, so the judge is from a different vendor than the answerer. Exact OpenRouter model IDs are set in `.env` and must be confirmed on openrouter.ai/models before the first run; the chosen IDs are recorded in each run's `env.json`.
2. ~~Embedding model final pick~~ **Decided:** `BAAI/bge-small-en-v1.5` for CPU speed. MedCPT and other biomedical models were not compared (listed as untried in REPORT.md).
3. ~~Deploy backend publicly, or local-only demo?~~ **Decided (2026-09-30): run locally.** Public deployment (Netlify frontend + hosted backend) only if required later.

## 12. Milestones

| # | Milestone | Output |
|---|---|---|
| M0 | Repo scaffold, docs (this PRD, architecture, system design) | `docs/` |
| M1 | Data prep + BM25 + Chroma indexes | `data/`, index build script |
| M2 | Retrieval pipeline: lexical/dense/hybrid + QE + logging | `backend/app/retrieval` |
| M3 | Answer generation + citation validation + insufficient-evidence | `backend/app/generation` |
| M4 | FastAPI + React UI | `backend/app/api`, `frontend/` |
| M5 | Eval set + runner + retrieval metrics | `eval/` |
| M6 | RAGAS judge + manual review | `eval/` |
| M7 | Report, diagram, README, best config | `REPORT.md`, `README.md` |
