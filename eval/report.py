"""Aggregate runs into results/comparison.md|csv (the single comparison table), paired bootstrap CIs, per-type table,
citation-validity audit, and the manual-review sheet for the judge-vs-retrieval disagreements. No LLM calls."""
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
ORDER = ["lexical", "dense", "hybrid", "hybrid_qe", "best"]
LABEL = {"lexical": "Lexical (BM25)", "dense": "Dense (bge-small)", "hybrid": "Hybrid (RRF)", "hybrid_qe": "Hybrid + query expansion",
         "best": "Best: weighted hybrid + QE"}
rng = np.random.default_rng(0)

runs = {}
for d in (ROOT / "runs").iterdir():
    if (d / "metrics.json").exists():
        m = json.loads((d / "metrics.json").read_text()); m["latency"] = json.loads((d / "latency.json").read_text())
        m["rows"] = [json.loads(l) for l in open(d / "results.jsonl")]; runs[m["config_name"]] = m
assert set(ORDER) <= set(runs), runs.keys()
qids = [r["qid"] for r in runs["best"]["rows"]]
assert all([r["qid"] for r in runs[c]["rows"]] == qids for c in ORDER), "configs must use identical query IDs in the same order"


def col(c, k):
    def val(r):
        v = r[k] if k in r else r["judge"][k]
        return np.nan if v is None else v
    return np.array([val(r) for r in runs[c]["rows"]], dtype=float)


def boot(a, b, n=5000):  # paired bootstrap of mean(a-b) over queries, ignoring pairs with NaN
    d = (a - b); d = d[~np.isnan(d)]
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


L = ["# Comparison table (fixed 100 queries, identical IDs for every run)", ""]
L += ["| Config | R@5 | R@10 | MRR@10 | nDCG@10 | Answer correctness | Groundedness | Context relevance | Insufficient % | Latency p50 / p95 (ms) | LLM cost / 100 q |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
rows_csv = []
for c in ORDER:
    m = runs[c]; lat = m["latency"]["end_to_end_ms"]
    L.append(f"| {LABEL[c]} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | {m['mrr@10']:.3f} | {m['ndcg@10']:.3f} | {m['answer_correctness']:.3f} | "
             f"{m['groundedness']:.3f} (n={m['groundedness_n']}) | {m['context_relevance']:.3f} | {m['insufficient_pct']:.0f} | "
             f"{lat['p50']:.0f} / {lat['p95']:.0f} | ${m['cost_usd']['total']:.2f} |")
    rows_csv.append({"config": c, **{k: m[k] for k in ["recall@5", "recall@10", "mrr@10", "ndcg@10", "answer_correctness", "groundedness", "context_relevance", "insufficient_pct"]},
                     "latency_p50_ms": lat["p50"], "latency_p95_ms": lat["p95"], "retrieval_only_p50_ms": m["latency"]["retrieval_only_ms"]["p50"],
                     "cost_expansion": m["cost_usd"]["expansion"], "cost_answer": m["cost_usd"]["answer"], "cost_judge": m["cost_usd"]["judge"], "cost_total": m["cost_usd"]["total"]})
with open(ROOT / "results" / "comparison.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_csv[0])); w.writeheader(); w.writerows(rows_csv)

L += ["", "Latency = retrieval + query-expansion LLM + answer LLM (LLM stage times are the original call latencies). Cost = expansion + answers + judge for the 100 queries.",
      "Groundedness is only scored for answered questions (n shown). Insufficient-evidence answers score 0 on correctness. Judge: " + runs["best"]["judge_model"] + ", " + runs["best"]["judge_version"] + ".", ""]

L += ["## Paired differences (mean, 95% bootstrap CI over the 100 queries)", "", "| Comparison | dnDCG@10 | dRecall@10 | dCorrectness |", "|---|---|---|---|"]
for a, b in [("best", "lexical"), ("best", "hybrid"), ("best", "hybrid_qe"), ("hybrid", "lexical"), ("hybrid_qe", "hybrid"), ("hybrid", "dense")]:
    cells = []
    for k in ["ndcg@10", "recall@10", "answer_correctness"]:
        mu, lo, hi = boot(col(a, k), col(b, k)); cells.append(f"{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]" + ("" if lo <= 0 <= hi else " *"))
    L.append(f"| {LABEL[a]} vs {LABEL[b]} | " + " | ".join(cells) + " |")
L += ["", "`*` = the 95% interval excludes 0.", ""]

L += ["## nDCG@10 by question type", "", "| Type (n) | " + " | ".join(LABEL[c] for c in ORDER) + " |", "|---|" + "---|" * len(ORDER)]
types = sorted({r["qtype"] for r in runs["best"]["rows"]})
for t in types:
    n = sum(r["qtype"] == t for r in runs["best"]["rows"])
    L.append(f"| {t} ({n}) | " + " | ".join(f"{runs[c]['by_qtype'][t]['ndcg@10']:.3f}" for c in ORDER) + " |")

L += ["", "## Citation validity audit (code-enforced)", "", "| Config | Answered | Insufficient | Answers with an invalid citation removed | Cited PMIDs outside context (after validation) |", "|---|---|---|---|---|"]
for c in ORDER:
    rows = runs[c]["rows"]; ans = [r for r in rows if r["status"] == "answered"]
    outside = sum(len(set(r["cited_valid"]) - set(r["context_pmids"])) for r in rows)
    L.append(f"| {LABEL[c]} | {len(ans)} | {len(rows)-len(ans)} | {sum(1 for r in rows if r['cited_invalid'])} | {outside} |")
(ROOT / "results" / "comparison.md").write_text("\n".join(L) + "\n"); print("\n".join(L))

# manual-review sheet for the best config: judge and retrieval metrics disagree
best = runs["best"]["rows"]
hi_ret_lo_judge = [r for r in best if r["recall@10"] >= 0.5 and (r["status"] != "answered" or (r["judge"]["answer_correctness"] or 0) <= 0.25)]
lo_ret_hi_judge = [r for r in best if r["recall@10"] <= 0.1 and r["status"] == "answered" and (r["judge"]["answer_correctness"] or 0) >= 0.75]
inv = [r for r in best if r["cited_invalid"]]
sheet = [("high_retrieval_low_judge", r) for r in hi_ret_lo_judge] + [("low_retrieval_high_judge", r) for r in lo_ret_hi_judge] + [("invalid_citation_removed", r) for r in inv]
with open(ROOT / "results" / "review_best.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["case_type", "qid", "qtype", "question", "gold_answer_from_dataset?", "model_answer", "status", "reason", "recall@10", "ndcg@10", "correctness", "groundedness", "cited_valid", "cited_invalid", "gold_pmids", "top10", "reviewer_verdict", "notes"])
    qa = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / "eval" / "queries_100.jsonl")}
    for t, r in sheet:
        w.writerow([t, r["qid"], r["qtype"], r["question"], qa[r["qid"]]["answer"], r["answer"], r["status"], r["reason"], r["recall@10"], round(r["ndcg@10"], 3),
                    r["judge"]["answer_correctness"], r["judge"]["groundedness"], r["cited_valid"], r["cited_invalid"], r["gold"][:15], r["ranked"], "", ""])
print(f"\nreview sheet: {len(hi_ret_lo_judge)} high-retrieval/low-judge, {len(lo_ret_hi_judge)} low-retrieval/high-judge, {len(inv)} invalid-citation")
