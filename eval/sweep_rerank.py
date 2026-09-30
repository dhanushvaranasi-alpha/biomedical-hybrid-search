"""Dev-set test of cross-encoder reranking on top-N fused candidates. Caches rerank scores so blends are free.
Usage: python -m eval.sweep_rerank BAAI/bge-reranker-base [top_n]"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import weighted
from eval.metrics import all_metrics
from eval.run import load_queries
from sentence_transformers import CrossEncoder

model = sys.argv[1]; top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
qs = load_queries(ROOT / "eval" / "dev_200.jsonl")
text = dict(zip(*[pd.read_parquet(ROOT / "data" / "passages.parquet")[c] for c in ("pmid", "text")]))
lex, den = LexicalRetriever(ROOT / "data" / "bm25"), DenseRetriever(ROOT / "data")
ce = CrossEncoder(model, max_length=384, device="cpu")
cache_f = ROOT / "runs_dev" / f"rerank_{model.replace('/', '_')}_{top_n}.json"
cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
t0 = time.time(); cands = {}
for n, q in enumerate(qs):
    L, D = lex.search(q["question"], 50), den.search(q["question"], 50)
    fused = weighted([(1.0, L, D)], 0.4)[:top_n]; cands[q["qid"]] = fused
    if str(q["qid"]) not in cache:
        sc = ce.predict([(q["question"], text[p]) for p, _ in fused], batch_size=16, show_progress_bar=False)
        cache[str(q["qid"])] = {str(p): float(s) for (p, _), s in zip(fused, sc)}
    if n % 20 == 19:
        cache_f.write_text(json.dumps(cache)); print(f"{n+1}/{len(qs)} {time.time()-t0:.0f}s", flush=True)
cache_f.write_text(json.dumps(cache))


def z(d):
    v = np.array(list(d.values())); return {k: (x - v.mean()) / (v.std() + 1e-9) for k, x in zip(d.keys(), v)}


def ev(beta):  # beta = weight on reranker z-score; 1-beta on fused z-score
    ms = []
    for q in qs:
        f = {str(p): s for p, s in cands[q["qid"]]}; r = cache[str(q["qid"])]
        zf, zr = z(f), z(r)
        order = sorted(f, key=lambda p: -(beta * zr[p] + (1 - beta) * zf[p]))
        ms.append(all_metrics([int(p) for p in order[:10]], q["gold_in_corpus"]))
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}


base = {k: float(np.mean([all_metrics([p for p, _ in cands[q["qid"]][:10]], q["gold_in_corpus"])[k] for q in qs])) for k in ("recall@5", "recall@10", "mrr@10", "ndcg@10")}
print(f"{model} top{top_n}\nfused only        ", {k: round(v, 3) for k, v in base.items()})
for b in [0.3, 0.5, 0.7, 1.0]:
    print(f"beta={b:<3} ", {k: round(v, 3) for k, v in ev(b).items()})
