"""Full reproducible evaluation: retrieval metrics + answers + RAGAS judge + latency + cost, for each config on the
fixed 100 queries. Everything goes through the same SearchPipeline / AnswerPipeline the API uses.

Usage:  python -m eval.evaluate --configs lexical dense hybrid hybrid_qe best [--limit N] [--out runs]
Costs: all LLM calls are cached on disk and count toward the persisted spend cap (LLM_BUDGET_USD), so re-running is
free and the cap holds. Cost attribution uses the original cost stored with each (possibly cached) response.
"""
import argparse, asyncio, json, os, sys, time
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend")); load_dotenv(ROOT / ".env")
from app.llm.client import LLMClient
from app.retrieval.pipeline import SearchPipeline, load_config, config_hash
from app.retrieval.expansion import make_expander
from app.generation.answer import select_context, answer
from eval.metrics import all_metrics
from eval.judge import make_metrics, judge_one, JUDGE_VERSION
from eval.run import load_queries

CONC = 6


def pct(x, p): return float(np.percentile(x, p)) if len(x) else None
def mean(x): x = [v for v in x if v is not None]; return float(np.mean(x)) if x else None


async def judge_all(metrics, items):
    sem = asyncio.Semaphore(CONC)

    async def one(it):
        async with sem:
            try:
                return await judge_one(metrics, it["question"], it["answer"], it["contexts"], it["reference"], it["status"])
            except Exception as e:  # keep going; counted and reported
                return {"context_relevance": None, "answer_correctness": None, "groundedness": None, "error": str(e)[:200]}
    return await asyncio.gather(*[one(it) for it in items])


def run_config(name, queries, client, out_root):
    cfg = load_config(name)
    expander = make_expander(client) if cfg["expansion"]["enabled"] else None
    pipe = SearchPipeline(cfg, expander=expander)
    run_id = f"{cfg['name']}-{config_hash(cfg)}"
    out = out_root / run_id; out.mkdir(parents=True, exist_ok=True)
    rows, judge_items = [], []
    for q in queries:
        tr = pipe.run(q["question"], top=10)
        ranked = [r["pmid"] for r in tr["results"]]
        ctx = select_context(tr["results"], cfg["context"]["top_c"], cfg["context"]["token_budget"])
        t = time.perf_counter(); ans = answer(client, q["question"], ctx); ans_wall = (time.perf_counter() - t) * 1000
        eu = tr.get("expansion_usage") or {}
        rows.append({"qid": q["qid"], "qtype": q["qtype"], "question": q["question"], "gold": q["gold_in_corpus"], "ranked": ranked,
                     "scores": [r["score"] for r in tr["results"]], "expansions": tr["expansions"], **all_metrics(ranked, q["gold_in_corpus"]),
                     "retrieval_ms": tr["timings_ms"]["total_ms"], "answer_ms": ans["usage"].get("latency_ms", 0.0) if not ans["usage"].get("cached") else None,
                     "answer_wall_ms": ans_wall, "expansion_cost": eu.get("cost_usd", 0.0), "answer_cost": ans["usage"].get("cost_usd", 0.0),
                     "status": ans["status"], "reason": ans["reason"], "answer": ans["answer"], "cited_valid": ans["valid"],
                     "cited_invalid": ans["invalid"], "context_pmids": ans.get("context_pmids", [])})
        judge_items.append({"question": q["question"], "answer": ans["answer"], "contexts": [t for _, t in ctx],
                            "reference": q["answer"], "status": ans["status"]})
    print(f"  [{name}] retrieval+answers done, judging {len(judge_items)} ...", flush=True)
    metrics = make_metrics(client)
    j = asyncio.run(judge_all(metrics, judge_items))
    for r, s in zip(rows, j): r["judge"] = s
    with open(out / "results.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

    keys = ["recall@5", "recall@10", "mrr@10", "ndcg@10"]
    answered = [r for r in rows if r["status"] == "answered"]
    ans_wall_total = [r["retrieval_ms"] + (r["answer_wall_ms"] if r["answer_ms"] is not None else r["answer_ms"] or 0) for r in rows if r["answer_ms"] is not None]
    summ = {"run_id": run_id, "config_name": name, "config": cfg, "n": len(rows),
            **{k: float(np.mean([r[k] for r in rows])) for k in keys},
            "answer_correctness": mean([r["judge"]["answer_correctness"] for r in rows]),
            "groundedness": mean([r["judge"]["groundedness"] for r in rows]),
            "groundedness_n": sum(r["judge"]["groundedness"] is not None for r in rows),
            "context_relevance": mean([r["judge"]["context_relevance"] for r in rows]),
            "judge_errors": sum("error" in r["judge"] for r in rows),
            "insufficient_pct": 100 * (1 - len(answered) / len(rows)),
            "invalid_citation_answers": sum(1 for r in rows if r["cited_invalid"]),
            "answers_with_zero_invalid_pct": 100 * sum(1 for r in rows if not r["cited_invalid"]) / len(rows),
            "retrieval_ms_p50": pct([r["retrieval_ms"] for r in rows], 50), "retrieval_ms_p95": pct([r["retrieval_ms"] for r in rows], 95),
            "cost_usd": {"expansion": sum(r["expansion_cost"] for r in rows), "answer": sum(r["answer_cost"] for r in rows), "judge": metrics[0].cost},
            "judge_calls": metrics[0].calls, "judge_version": JUDGE_VERSION, "judge_model": os.environ["LLM_JUDGE_MODEL"],
            "answer_model": os.environ["LLM_ANSWER_MODEL"], "expander_model": os.environ["LLM_EXPANDER_MODEL"],
            "by_qtype": {t: {k: float(np.mean([r[k] for r in rows if r["qtype"] == t])) for k in keys} for t in sorted({r["qtype"] for r in rows})}}
    summ["cost_usd"]["total"] = sum(summ["cost_usd"].values())
    (out / "metrics.json").write_text(json.dumps(summ, indent=2))
    print(f"  [{name}] R@5={summ['recall@5']:.3f} R@10={summ['recall@10']:.3f} MRR={summ['mrr@10']:.3f} nDCG={summ['ndcg@10']:.3f} "
          f"corr={summ['answer_correctness']} ground={summ['groundedness']} ctxrel={summ['context_relevance']} "
          f"insuff={summ['insufficient_pct']:.0f}% cost=${summ['cost_usd']['total']:.4f} spent_total=${client.spent():.4f}", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["lexical", "dense", "hybrid", "hybrid_qe", "best"])
    ap.add_argument("--queries", default="eval/queries_100.jsonl"); ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    qs = load_queries(ROOT / a.queries)[: a.limit]
    client = LLMClient()
    print(f"{len(qs)} queries; spend so far ${client.spent():.4f}, cap ${client.budget}", flush=True)
    out_root = ROOT / a.out; out_root.mkdir(exist_ok=True)
    allres = [run_config(c, qs, client, out_root) for c in a.configs]
    (out_root / "summary.json").write_text(json.dumps(allres, indent=2))


if __name__ == "__main__":
    main()
