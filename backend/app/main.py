"""FastAPI service. Uses the same SearchPipeline / AnswerPipeline as the offline evaluation."""
import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from .retrieval.pipeline import SearchPipeline, load_config, config_hash
from .generation.answer import answer as run_answer, select_context, passage_text
from .llm.client import LLMClient, BudgetExceeded

app = FastAPI(title="Biomedical Hybrid Search")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
NAMED = ["lexical", "dense", "hybrid", "hybrid_qe", "best"]


class Req(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    config: str = "best"


@lru_cache(maxsize=1)
def client() -> LLMClient:
    return LLMClient()


@lru_cache(maxsize=8)
def pipeline(name: str) -> SearchPipeline:
    cfg = load_config(name)
    exp = None
    if cfg["expansion"]["enabled"]:
        from .retrieval.expansion import make_expander
        exp = make_expander(client())
    return SearchPipeline(cfg, expander=exp)


def _search(req: Req):
    if req.config not in NAMED:
        raise HTTPException(400, f"unknown config {req.config}")
    try:
        p = pipeline(req.config)
        tr = p.run(req.query, top=10)
    except FileNotFoundError:
        raise HTTPException(503, f"config '{req.config}' or indexes not available; build indexes first")
    for r in tr["results"]:
        t = passage_text(r["pmid"])
        r["snippet"] = t[:400] + ("..." if len(t) > 400 else "")
        r["source_url"] = f"https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/"
    return p, tr


@app.on_event("startup")
def warm():
    """Load indexes and the embedding model once at startup (no LLM call), so the first search is not slow."""
    try:
        p = pipeline("hybrid")
        p.den.search("warm up", 1)
    except Exception as e:  # indexes not built yet: the API still starts and reports a clear error per request
        print("warm-up skipped:", e)


@app.get("/api/health")
def health():
    return {"status": "ok", "llm_spent_usd": client().spent(), "llm_budget_usd": client().budget}


@app.get("/api/configs")
def configs():
    return [c for c in NAMED if (ROOT / "configs" / f"{c}.yaml").exists()]


@app.post("/api/search")
def search(req: Req):
    return _search(req)[1]


@app.post("/api/ask")
def ask(req: Req):
    p, tr = _search(req)
    c = p.cfg["context"]
    ctx = select_context(tr["results"], c["top_c"], c["token_budget"])
    used = {pm for pm, _ in ctx}
    for r in tr["results"]:
        r["selected_for_context"] = r["pmid"] in used
    try:
        ans = run_answer(client(), req.query, ctx)
    except BudgetExceeded as e:
        raise HTTPException(429, str(e))
    except KeyError as e:
        raise HTTPException(503, f"LLM not configured: missing {e} (see .env.example)")
    return {**tr, "answer": ans}
