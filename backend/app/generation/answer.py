"""AnswerPipeline: build grounded prompt from selected passages -> LLM (JSON) -> citation validation."""
import json, os, re
import pandas as pd
from pathlib import Path
from .prompts import ANSWER_SYSTEM, ANSWER_VERSION, answer_user
from .citations import validate

ROOT = Path(__file__).resolve().parents[3]
_TEXT: dict[int, str] | None = None


def passage_text(pmid: int) -> str:
    global _TEXT
    if _TEXT is None:
        df = pd.read_parquet(ROOT / "data" / "passages.parquet")
        _TEXT = dict(zip(df["pmid"].astype(int), df["text"]))
    return _TEXT[int(pmid)]


def select_context(results: list[dict], top_c: int, token_budget: int) -> list[tuple[int, str]]:
    """Top-c passages, truncating text so the total stays within ~token_budget (approx 1.4 tokens/word)."""
    max_words, out, used = int(token_budget / 1.4), [], 0
    for r in results[:top_c]:
        words = passage_text(r["pmid"]).split()
        take = words[: max(0, min(len(words), max_words - used))]
        if not take:
            break
        out.append((r["pmid"], " ".join(take))); used += len(take)
    return out


def answer(client, question: str, context: list[tuple[int, str]], model: str | None = None) -> dict:
    model = model or os.environ["LLM_ANSWER_MODEL"]
    if not context:
        return {"status": "insufficient_evidence", "reason": "no_passages", "answer": "", "valid": [], "invalid": [],
                "valid_rate": 0.0, "uncited_sentences": [], "usage": {}}
    msgs = [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": answer_user(question, context)}]
    r = client.chat(model, msgs, ANSWER_VERSION, json_mode=True, max_tokens=500)
    raw = re.sub(r"^```(?:json)?|```$", "", r["text"].strip(), flags=re.M).strip()
    try:
        d = json.loads(raw); ans, insuff = str(d.get("answer", "")), bool(d.get("insufficient_evidence", False))
    except Exception:
        ans, insuff = "", True  # unparseable output is never shown as an answer
        v = validate("", {p for p, _ in context}, True); v["reason"] = "parse_failed"
    else:
        v = validate(ans, {p for p, _ in context}, insuff)
    v["usage"] = {k: r[k] for k in ("prompt_tokens", "completion_tokens", "cost_usd", "latency_ms", "cached")}
    v["context_pmids"] = [p for p, _ in context]
    return v
