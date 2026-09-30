"""SearchPipeline: expand -> retrieve (lexical/dense) -> fuse. Same class is used by the API and the eval runner."""
import hashlib, json, time
from pathlib import Path
import yaml
from .fusion import rrf, weighted

ROOT = Path(__file__).resolve().parents[3]


def load_config(name_or_path: str = "base", overrides: dict | None = None) -> dict:
    def merge(a, b):
        for k, v in (b or {}).items():
            a[k] = merge(a.get(k, {}), v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
        return a
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    if name_or_path != "base":
        p = Path(name_or_path) if name_or_path.endswith(".yaml") else ROOT / "configs" / f"{name_or_path}.yaml"
        cfg = merge(cfg, yaml.safe_load(p.read_text()))
    return merge(cfg, overrides)


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


class SearchPipeline:
    def __init__(self, cfg: dict, data_dir: Path | None = None, expander=None):
        self.cfg, self.expander = cfg, expander
        d = data_dir or ROOT / "data"
        self.lex = self.den = None
        if cfg["mode"] in ("lexical", "hybrid"):
            from .lexical import LexicalRetriever
            self.lex = LexicalRetriever(d / "bm25")
        if cfg["mode"] in ("dense", "hybrid"):
            from .dense import DenseRetriever
            self.den = DenseRetriever(d)

    def run(self, query: str, top: int = 10) -> dict:
        c, t0, tm = self.cfg, time.perf_counter(), {}
        expansion_usage = None
        queries, weights = [query], [c["expansion"]["orig_weight"]]
        if c["expansion"]["enabled"]:
            s = time.perf_counter()
            if self.expander is not None:
                self.expander.usage = None
            alts = self.expander(query, c["expansion"]["n"]) if self.expander else []
            expansion_usage = getattr(self.expander, "usage", None)
            tm["expand_ms"] = (time.perf_counter() - s) * 1000
            for a in alts:
                if a.strip().lower() not in {q.lower() for q in queries}:
                    queries.append(a); weights.append(c["expansion"]["alt_weight"])
        k = c["k_retrieve"]
        lex_l = den_l = None
        if self.lex:
            s = time.perf_counter(); lex_l = [self.lex.search(q, k) for q in queries]; tm["lexical_ms"] = (time.perf_counter() - s) * 1000
        if self.den:
            s = time.perf_counter(); den_l = self.den.search_many(queries, k); tm["dense_ms"] = (time.perf_counter() - s) * 1000
        s = time.perf_counter()
        if c["mode"] == "lexical":
            fused = rrf([(w, l) for w, l in zip(weights, lex_l)], c["fusion"]["k_rrf"])
        elif c["mode"] == "dense":
            fused = rrf([(w, l) for w, l in zip(weights, den_l)], c["fusion"]["k_rrf"])
        elif c["fusion"]["method"] == "rrf":
            fused = rrf([(w, l) for w, l in zip(weights, lex_l)] + [(w, l) for w, l in zip(weights, den_l)], c["fusion"]["k_rrf"])
        else:
            fused = weighted(list(zip(weights, lex_l, den_l)), c["fusion"]["alpha"])
        tm["fuse_ms"] = (time.perf_counter() - s) * 1000
        tm["total_ms"] = (time.perf_counter() - t0) * 1000
        # per-retriever rank/score for the original query, for logging + UI
        lex0 = {p: (i + 1, sc) for i, (p, sc) in enumerate(lex_l[0])} if lex_l else {}
        den0 = {p: (i + 1, sc) for i, (p, sc) in enumerate(den_l[0])} if den_l else {}
        results = [{"rank": i + 1, "pmid": p, "score": sc,
                    "lexical": lex0.get(p), "dense": den0.get(p)} for i, (p, sc) in enumerate(fused[:top])]
        return {"query": query, "expansions": queries[1:], "config_hash": config_hash(c),
                "results": results, "timings_ms": tm, "expansion_usage": expansion_usage}
