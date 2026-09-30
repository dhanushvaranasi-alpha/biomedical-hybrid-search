"""BM25 over full passages (bm25s). Tokenizer keeps alphanumerics; English stopwords + Porter-style stemming."""
from pathlib import Path
import json
import bm25s, Stemmer

_stemmer = Stemmer.Stemmer("english")


def tokenize_docs(texts):
    return bm25s.tokenize(texts, stopwords="en", stemmer=_stemmer, show_progress=False)


def build_bm25(pmids: list[int], texts: list[str], out: Path):
    out.mkdir(parents=True, exist_ok=True)
    r = bm25s.BM25(k1=1.2, b=0.75)
    r.index(tokenize_docs(texts), show_progress=False)
    r.save(str(out / "index"))
    (out / "pmids.json").write_text(json.dumps([int(p) for p in pmids]))


class LexicalRetriever:
    def __init__(self, path: Path):
        self.r = bm25s.BM25.load(str(path / "index"))
        self.pmids = json.loads((path / "pmids.json").read_text())

    def search(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        toks = bm25s.tokenize([query], stopwords="en", stemmer=_stemmer, show_progress=False)
        if not toks.ids[0]:
            return []
        docs, scores = self.r.retrieve(toks, k=min(k, len(self.pmids)), show_progress=False)
        return [(self.pmids[int(d)], float(s)) for d, s in zip(docs[0], scores[0]) if s > 0]
