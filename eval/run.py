"""Run retrieval configs over a fixed query file, log traces and metrics. (Answer + judge stages are added later.)

Usage: python -m eval.run --configs lexical dense hybrid --queries eval/queries_100.jsonl
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.retrieval.pipeline import SearchPipeline, load_config, config_hash
from eval.metrics import all_metrics


def load_queries(path):
    return [json.loads(l) for l in open(path)]


def run_config(name, queries, out_root, overrides=None, expander=None, tag=""):
    cfg = load_config(name, overrides)
    pipe = SearchPipeline(cfg, expander=expander)
    run_id = f"{cfg['name']}{tag}-{config_hash(cfg)}"
    out = out_root / run_id; out.mkdir(parents=True, exist_ok=True)
    per_q, lat = [], []
    with open(out / "traces.jsonl", "w") as f:
        for q in queries:
            tr = pipe.run(q["question"], top=10)
            ranked = [r["pmid"] for r in tr["results"]]
            m = all_metrics(ranked, q["gold_in_corpus"])
            per_q.append({"qid": q["qid"], "qtype": q["qtype"], **m}); lat.append(tr["timings_ms"]["total_ms"])
            f.write(json.dumps({"qid": q["qid"], **tr}) + "\n")
    keys = ["recall@5", "recall@10", "mrr@10", "ndcg@10"]
    summary = {"run_id": run_id, "config": cfg, "n_queries": len(queries),
               **{k: float(np.mean([r[k] for r in per_q])) for k in keys},
               "latency_ms_p50": float(np.percentile(lat, 50)), "latency_ms_p95": float(np.percentile(lat, 95)),
               "by_qtype": {t: {k: float(np.mean([r[k] for r in per_q if r["qtype"] == t])) for k in keys}
                            for t in sorted({r["qtype"] for r in per_q})}}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2)); (out / "per_query.json").write_text(json.dumps(per_q))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True); ap.add_argument("--queries", default="eval/queries_100.jsonl")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    qs = load_queries(ROOT / a.queries)
    for c in a.configs:
        s = run_config(c, qs, ROOT / a.out)
        print(f"{s['run_id']:<24} R@5={s['recall@5']:.3f} R@10={s['recall@10']:.3f} MRR@10={s['mrr@10']:.3f} "
              f"nDCG@10={s['ndcg@10']:.3f} p50={s['latency_ms_p50']:.0f}ms p95={s['latency_ms_p95']:.0f}ms")


if __name__ == "__main__":
    main()
