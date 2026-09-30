"""Sentence-transformers wrapper. bge models use a query instruction prefix; documents get none."""
from functools import lru_cache
import numpy as np

QUERY_PREFIX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: "}


@lru_cache(maxsize=2)
def _model(name: str):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name, device="cpu")
    m.max_seq_length = 384
    return m


def encode_docs(name: str, texts: list[str], batch_size: int = 32) -> np.ndarray:
    return _model(name).encode(texts, batch_size=batch_size, normalize_embeddings=True,
                               show_progress_bar=False).astype("float32")


def encode_queries(name: str, texts: list[str]) -> np.ndarray:
    pre = QUERY_PREFIX.get(name, "")
    return _model(name).encode([pre + t for t in texts], normalize_embeddings=True,
                               show_progress_bar=False).astype("float32")
