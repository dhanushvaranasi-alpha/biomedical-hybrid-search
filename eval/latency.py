"""Latency breakdown per config, reconstructed without new LLM spend.

Retrieval-only time is re-measured (no LLM, warm process). LLM stage times (query expansion, answer generation) use the
ORIGINAL call latency stored with each cached response, so cached reruns do not understate them.
End-to-end = retrieval-only + expansion LLM + answer LLM.   Usage: python -m eval.latency
"""
import json, sys
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend")); load_dotenv(ROOT / ".env")
from app.llm.client import LLMClient
from app.retrieval.pipeline import SearchPipeline, load_config, config_hash
from app.retrieval.expansion import make_expander
from app.generation.answer import select_context, answer
from eval.run import load_queries

client = LLMClient(); qs = load_queries(ROOT / "eval" / "queries_100.jsonl")
spent0 = client.spent()
for name in ["lexical", "dense", "hybrid", "hybrid_qe", "best"]:
    cfg = load_config(name); ex = make_expander(client) if cfg["expansion"]["enabled"] else None
    pipe = SearchPipeline(cfg, expander=ex); rid = f"{cfg['name']}-{config_hash(cfg)}"
    pipe.run(qs[0]["question"])  # warm-up (model load, caches)
    ret, exp_l, ans_l = [], [], []
    for q in qs:
        tr = pipe.run(q["question"], top=10); tm = tr["timings_ms"]
        ret.append(tm["total_ms"] - tm.get("expand_ms", 0.0))
        eu = tr.get("expansion_usage") or {}; exp_l.append(eu.get("latency_ms", 0.0))
        a = answer(client, q["question"], select_context(tr["results"], cfg["context"]["top_c"], cfg["context"]["token_budget"]))
        assert a["usage"].get("cached", False), "answer not cached - would spend"
        ans_l.append(a["usage"]["latency_ms"])
    e2e = np.array(ret) + np.array(exp_l) + np.array(ans_l); p = lambda x, q: float(np.percentile(x, q))
    out = {"run_id": rid, "retrieval_only_ms": {"p50": p(ret, 50), "p95": p(ret, 95)}, "expansion_llm_ms": {"p50": p(exp_l, 50), "p95": p(exp_l, 95)},
           "answer_llm_ms": {"p50": p(ans_l, 50), "p95": p(ans_l, 95)}, "end_to_end_ms": {"p50": p(e2e, 50), "p95": p(e2e, 95)}}
    (ROOT / "runs" / rid / "latency.json").write_text(json.dumps(out, indent=2))
    print(name, {k: {a: round(b) for a, b in v.items()} for k, v in out.items() if k != "run_id"}, flush=True)
print("new spend during latency reconstruction: $%.6f" % (client.spent() - spent0))
