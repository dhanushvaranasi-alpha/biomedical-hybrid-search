"""Dense retrieval over chunk embeddings in Chroma; chunk hits are aggregated to passages (max similarity)."""
from pathlib import Path
import json
import chromadb
from .embedder import encode_queries


class DenseRetriever:
    def __init__(self, data_dir: Path):
        m = json.loads((data_dir / "index_manifest.json").read_text())
        self.model = m["embed_model"]
        self.col = chromadb.PersistentClient(path=str(data_dir / "chroma")).get_collection(m["collection"])

    def search(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        return self.search_many([query], k)[0]

    def search_many(self, queries: list[str], k: int = 50) -> list[list[tuple[int, float]]]:
        embs = encode_queries(self.model, queries)
        res = self.col.query(query_embeddings=embs.tolist(), n_results=k * 3, include=["metadatas", "distances"])
        out = []
        for metas, dists in zip(res["metadatas"], res["distances"]):
            best: dict[int, float] = {}
            for md, d in zip(metas, dists):
                s = 1.0 - float(d)  # cosine similarity
                p = int(md["pmid"])
                if s > best.get(p, -1.0):
                    best[p] = s
            out.append(sorted(best.items(), key=lambda x: -x[1])[:k])
        return out
