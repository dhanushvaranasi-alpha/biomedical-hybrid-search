"""LLM query expansion: n alternates, original always retained by the pipeline. Falls back to [] on any failure."""
import json, os, re
from ..generation.prompts import EXPANSION_SYSTEM, EXPANSION_VERSION


def make_expander(client, model: str | None = None):
    model = model or os.environ["LLM_EXPANDER_MODEL"]

    def expand(query: str, n: int = 3) -> list[str]:
        msgs = [{"role": "system", "content": EXPANSION_SYSTEM},
                {"role": "user", "content": f"N = {n}\nQuestion: {query}"}]
        try:
            r = client.chat(model, msgs, EXPANSION_VERSION, json_mode=True, max_tokens=200)
            txt = re.sub(r"^```(?:json)?|```$", "", r["text"].strip(), flags=re.M).strip()
            qs = json.loads(txt)["queries"]
            return [q.strip() for q in qs if isinstance(q, str) and q.strip()][:n]
        except Exception:
            return []
    return expand
