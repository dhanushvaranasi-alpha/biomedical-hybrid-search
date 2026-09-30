"""Single adapter for every LLM call (expansion, answer, judge). Routed through OpenRouter via LiteLLM.

Cost control (the OpenRouter balance is small):
  * every call is cached on disk, keyed by (model, prompt_version, messages, params) - a repeat costs nothing;
  * cumulative spend is persisted in the same SQLite file and a hard cap (LLM_BUDGET_USD, default 8.0) is enforced
    BEFORE each network call;
  * per-call cost comes from OpenRouter's usage accounting when present, else from configs/pricing.yaml.
"""
import hashlib, json, os, sqlite3, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / ".cache" / "llm_cache.sqlite"


class BudgetExceeded(RuntimeError):
    pass


class LLMClient:
    def __init__(self, db_path: Path = DB_PATH, budget_usd: float | None = None, completion_fn=None):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = threading.Lock()
        self.db.execute("CREATE TABLE IF NOT EXISTS cache(k TEXT PRIMARY KEY, resp TEXT, cost REAL, ts REAL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS spend(id INTEGER PRIMARY KEY CHECK(id=1), usd REAL)")
        self.db.execute("INSERT OR IGNORE INTO spend VALUES(1, 0)")
        self.db.commit()
        self.budget = float(os.getenv("LLM_BUDGET_USD", "8.0")) if budget_usd is None else budget_usd
        self._completion = completion_fn
        self.stats = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}

    def spent(self) -> float:
        return self.db.execute("SELECT usd FROM spend WHERE id=1").fetchone()[0]

    @staticmethod
    def key(model, prompt_version, messages, params) -> str:
        blob = json.dumps([model, prompt_version, messages, params], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _call(self, model, messages, params):
        if self._completion is not None:
            return self._completion(model=model, messages=messages, **params)
        import litellm
        params = dict(params)
        extra = {"usage": {"include": True}}
        if "reasoning" in params:  # OpenRouter unified reasoning control, e.g. {"effort": "low"}
            extra["reasoning"] = params.pop("reasoning")
        return litellm.completion(model=model, messages=messages, api_key=os.environ["OPENROUTER_API_KEY"],
                                  extra_body=extra, **params)

    def chat(self, model: str, messages: list[dict], prompt_version: str, json_mode: bool = False,
             max_tokens: int = 600, temperature: float = 0.0, reasoning: dict | None = None) -> dict:
        params = {"temperature": temperature, "max_tokens": max_tokens}
        if reasoning:
            params["reasoning"] = reasoning
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        k = self.key(model, prompt_version, messages, params)
        with self.lock:
            row = self.db.execute("SELECT resp FROM cache WHERE k=?", (k,)).fetchone()
        if row:
            self.stats["cache_hits"] += 1
            out = json.loads(row[0]); out["cached"] = True
            return out
        if self.spent() >= self.budget:
            raise BudgetExceeded(f"LLM budget cap ${self.budget:.2f} reached (spent ${self.spent():.4f}); raise LLM_BUDGET_USD to continue")
        t = time.perf_counter()
        r = self._call(model, messages, params)
        ms = (time.perf_counter() - t) * 1000
        u = getattr(r, "usage", None) or r.get("usage", {}) if not hasattr(r, "usage") else r.usage
        get = (lambda o, n: (o.get(n) if isinstance(o, dict) else getattr(o, n, None)) or 0)
        cost = float(get(u, "cost"))
        msg = r["choices"][0]["message"] if isinstance(r, dict) else r.choices[0].message
        text = msg["content"] if isinstance(msg, dict) else msg.content
        out = {"text": text, "prompt_tokens": int(get(u, "prompt_tokens")), "completion_tokens": int(get(u, "completion_tokens")),
               "cost_usd": cost, "latency_ms": ms, "model": model, "cached": False}
        out["empty"] = not (text and text.strip())
        out["text"] = text or ""
        with self.lock:
            if not out["empty"]:  # never cache an empty completion (e.g. all tokens spent on hidden reasoning)
                self.db.execute("INSERT OR REPLACE INTO cache VALUES(?,?,?,?)", (k, json.dumps(out), cost, time.time()))
            self.db.execute("UPDATE spend SET usd = usd + ? WHERE id=1", (cost,))
            self.db.commit()
        self.stats["calls"] += 1; self.stats["prompt_tokens"] += out["prompt_tokens"]
        self.stats["completion_tokens"] += out["completion_tokens"]; self.stats["cost_usd"] += cost
        return out
