"""Free fusion sweep on the DEV set (never the 100 eval queries): fetch lexical + dense lists once, fuse offline."""
import itertools, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import rrf, weighted
from eval.metrics import all_metrics
from eval.run import load_queries

qs = load_queries(ROOT / "eval" / "dev_200.jsonl")
lex, den = LexicalRetriever(ROOT / "data" / "bm25"), DenseRetriever(ROOT / "data")
L = [lex.search(q["question"], 50) for q in qs]
D = [den.search(q["question"], 50) for q in qs]
gold = [q["gold_in_corpus"] for q in qs]


def score(fn):
    ms = [all_metrics([p for p, _ in fn(i)[:10]], gold[i]) for i in range(len(qs))]
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}


rows = []
for wl, k in itertools.product([1.0, 1.5, 2.0, 3.0], [10, 30, 60]):
    rows.append((f"rrf wl={wl} k={k}", score(lambda i: rrf([(wl, L[i]), (1.0, D[i])], k))))
for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    rows.append((f"weighted alpha={a}", score(lambda i: weighted([(1.0, L[i], D[i])], a))))
rows.append(("lexical only", score(lambda i: L[i]))); rows.append(("dense only", score(lambda i: D[i])))
rows.sort(key=lambda r: -r[1]["ndcg@10"])
for n, m in rows:
    print(f"{n:<22} R@5={m['recall@5']:.3f} R@10={m['recall@10']:.3f} MRR={m['mrr@10']:.3f} nDCG={m['ndcg@10']:.3f}")
