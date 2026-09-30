import pytest
from app.llm.client import LLMClient, BudgetExceeded


def fake(cost):
    calls = []
    def f(model, messages, **p):
        calls.append(1)
        return {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost": cost}}
    f.calls = calls
    return f


def test_cache_hit_costs_nothing(tmp_path):
    f = fake(0.01); c = LLMClient(tmp_path / "c.sqlite", budget_usd=1, completion_fn=f)
    a = c.chat("m", [{"role": "user", "content": "x"}], "v1"); b = c.chat("m", [{"role": "user", "content": "x"}], "v1")
    assert len(f.calls) == 1 and b["cached"] and not a["cached"] and abs(c.spent() - 0.01) < 1e-9


def test_prompt_version_busts_cache(tmp_path):
    f = fake(0.01); c = LLMClient(tmp_path / "c.sqlite", budget_usd=1, completion_fn=f)
    c.chat("m", [{"role": "user", "content": "x"}], "v1"); c.chat("m", [{"role": "user", "content": "x"}], "v2")
    assert len(f.calls) == 2


def test_budget_cap_blocks_call_and_persists(tmp_path):
    f = fake(0.6); p = tmp_path / "c.sqlite"; c = LLMClient(p, budget_usd=1.0, completion_fn=f)
    c.chat("m", [{"role": "user", "content": "1"}], "v"); c.chat("m", [{"role": "user", "content": "2"}], "v")
    with pytest.raises(BudgetExceeded):
        c.chat("m", [{"role": "user", "content": "3"}], "v")
    c2 = LLMClient(p, budget_usd=1.0, completion_fn=f)  # new process sees same spend
    with pytest.raises(BudgetExceeded):
        c2.chat("m", [{"role": "user", "content": "4"}], "v")
    assert len(f.calls) == 2


def test_empty_completion_not_cached_but_cost_counted(tmp_path):
    def f(model, messages, **p):
        return {"choices": [{"message": {"content": None}}], "usage": {"prompt_tokens": 5, "completion_tokens": 200, "cost": 0.01}}
    c = LLMClient(tmp_path / "c.sqlite", budget_usd=1, completion_fn=f)
    a = c.chat("m", [{"role": "user", "content": "x"}], "v"); b = c.chat("m", [{"role": "user", "content": "x"}], "v")
    assert a["empty"] and b["empty"] and not b["cached"] and abs(c.spent() - 0.02) < 1e-9
