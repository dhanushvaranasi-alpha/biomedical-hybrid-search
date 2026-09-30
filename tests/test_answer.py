import json
from app.generation import answer as A
from app.llm.client import LLMClient


def client(tmp_path, payload):
    def f(model, messages, **p):
        return {"choices": [{"message": {"content": payload if isinstance(payload, str) else json.dumps(payload)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0}}
    return LLMClient(tmp_path / "c.sqlite", budget_usd=1, completion_fn=f)


CTX = [(1, "Papilin is secreted."), (2, "Other.")]


def test_answered(tmp_path):
    r = A.answer(client(tmp_path, {"insufficient_evidence": False, "answer": "Yes [PMID:1]."}), "q", CTX, model="m")
    assert r["status"] == "answered" and r["valid"] == [1]


def test_hallucinated_citation_rejected(tmp_path):
    r = A.answer(client(tmp_path, {"insufficient_evidence": False, "answer": "Yes [PMID:999]."}), "q", CTX, model="m")
    assert r["status"] == "insufficient_evidence" and r["invalid"] == [999]


def test_parse_failure(tmp_path):
    r = A.answer(client(tmp_path, "not json"), "q", CTX, model="m")
    assert r["status"] == "insufficient_evidence" and r["reason"] == "parse_failed"


def test_no_context_skips_llm(tmp_path):
    r = A.answer(client(tmp_path, {}), "q", [], model="m")
    assert r["reason"] == "no_passages"
