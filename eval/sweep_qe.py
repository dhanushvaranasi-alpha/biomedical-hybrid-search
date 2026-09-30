"""Dev-set test of LLM query expansion (small LLM cost, cached). Compares fusion variants with/without expansion."""
import os, sys, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend")); load_dotenv(ROOT / ".env")
from app.llm.client import LLMClient
from app.retrieval.expansion import make_expander
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import weighted, rrf
from eval.metrics import all_metrics
from eval.run import load_queries

qs = load_queries(ROOT / "eval" / "dev_200.jsonl")
client = LLMClient(); expand = make_expander(client)
with ThreadPoolExecutor(8) as ex:
    E = list(ex.map(lambda q: expand(q["question"], 3), qs))
print(f"expansions: {sum(1 for e in E if e)}/{len(E)} non-empty, avg {np.mean([len(e) for e in E]):.2f}; "
      f"LLM calls={client.stats['calls']} cache_hits={client.stats['cache_hits']} spent=${client.spent():.5f}", flush=True)

lex, den = LexicalRetriever(ROOT / "data" / "bm25"), DenseRetriever(ROOT / "data")
L0 = [lex.search(q["question"], 50) for q in qs]; D0 = [den.search(q["question"], 50) for q in qs]
LA = [[lex.search(a, 50) for a in e] for e in E]
DA = [den.search_many(e, 50) if e else [] for e in E]
gold = [q["gold_in_corpus"] for q in qs]


def score(fn):
    ms = [all_metrics([p for p, _ in fn(i)[:10]], gold[i]) for i in range(len(qs))]
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}, ms


def w_groups(i, aw):  # weighted fusion across original + alternates
    return [(1.0, L0[i], D0[i])] + [(aw, l, d) for l, d in zip(LA[i], DA[i])]


rows = [("hybrid weighted a=0.4 (no QE)", lambda i: weighted([(1.0, L0[i], D0[i])], 0.4))]
for aw in (0.3, 0.5, 1.0):
    rows.append((f"+QE weighted a=0.4 alt_w={aw}", lambda i, aw=aw: weighted(w_groups(i, aw), 0.4)))
rows.append(("hybrid RRF equal (no QE)", lambda i: rrf([(1.0, L0[i]), (1.0, D0[i])], 60)))
rows.append(("+QE RRF equal alt_w=0.5", lambda i: rrf([(1.0, L0[i]), (1.0, D0[i])] + [(0.5, l) for l in LA[i]] + [(0.5, d) for d in DA[i]], 60)))
base = None
for name, fn in rows:
    m, per = score(fn)
    if base is None: base = per
    d = np.array([a["ndcg@10"] - b["ndcg@10"] for a, b in zip(per, base)])
    print(f"{name:<34} R@5={m['recall@5']:.3f} R@10={m['recall@10']:.3f} MRR={m['mrr@10']:.3f} nDCG={m['ndcg@10']:.3f}"
          f"  dnDCG_vs_first={d.mean():+.3f} (se {d.std(ddof=1)/np.sqrt(len(d)):.3f})")
