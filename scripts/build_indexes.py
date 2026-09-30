"""Build BM25 (full passages) and Chroma (chunks) indexes. Resumable: embeddings are cached in data/emb/.

Usage: python scripts/build_indexes.py [--limit N]   (limit is for smoke tests only)
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.retrieval.embedder import encode_docs
from app.retrieval.lexical import tokenize_docs, build_bm25

DATA = ROOT / "data"
BATCH = 2000


def slug(m): return m.replace("/", "_")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int); ap.add_argument("--skip-dense", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    model = cfg["embedding"]["model"]
    passages = pd.read_parquet(DATA / "passages.parquet")
    chunks = pd.read_parquet(DATA / "chunks.parquet")
    if a.limit:
        passages = passages.head(a.limit); chunks = chunks[chunks.pmid.isin(set(passages.pmid))]

    t = time.time()
    build_bm25(passages["pmid"].tolist(), passages["text"].tolist(), DATA / "bm25")
    print(f"BM25 built over {len(passages)} passages in {time.time()-t:.1f}s", flush=True)
    if a.skip_dense:
        return

    embdir = DATA / "emb" / slug(model) / ("limit%d" % a.limit if a.limit else "full"); embdir.mkdir(parents=True, exist_ok=True)
    texts = chunks["text"].tolist(); n = len(texts)
    t = time.time()
    for i in range(0, n, BATCH):
        f = embdir / f"{i:07d}.npy"
        if f.exists(): continue
        np.save(f, encode_docs(model, texts[i:i + BATCH]))
        done = min(i + BATCH, n); el = time.time() - t
        print(f"embedded {done}/{n} chunks, {el/60:.1f} min elapsed", flush=True)
    emb = np.concatenate([np.load(embdir / f"{i:07d}.npy") for i in range(0, n, BATCH)])
    assert emb.shape[0] == n

    import chromadb
    client = chromadb.PersistentClient(path=str(DATA / "chroma"))
    name = f"chunks_{slug(model)}"
    if a.limit: name += f"_limit{a.limit}"
    try: client.delete_collection(name)
    except Exception: pass
    col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    ids = chunks["chunk_id"].tolist(); pm = chunks["pmid"].tolist()
    for i in range(0, n, 5000):
        col.add(ids=ids[i:i+5000], embeddings=emb[i:i+5000].tolist(),
                metadatas=[{"pmid": int(p)} for p in pm[i:i+5000]])
    manifest = {"embed_model": model, "dims": int(emb.shape[1]), "n_passages": len(passages), "n_chunks": n,
                "collection": name, "bm25": "bm25s k1=1.2 b=0.75 stemmer=english stopwords=en", "built_at": time.strftime("%F %T")}
    (DATA / "index_manifest.json").write_text(json.dumps(manifest, indent=2)); print(manifest)


if __name__ == "__main__":
    main()
