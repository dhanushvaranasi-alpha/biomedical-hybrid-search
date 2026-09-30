"""Download rag-mini-bioasq parquet files into data/raw/ (no login needed)."""
import sys, urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq/resolve/main/data"
FILES = {"qa.parquet": f"{BASE}/test.parquet/part.0.parquet", "passages.parquet": f"{BASE}/passages.parquet/part.0.parquet"}
out = Path(__file__).resolve().parents[1] / "data" / "raw"
out.mkdir(parents=True, exist_ok=True)
for name, url in FILES.items():
    dest = out / name
    if dest.exists() and dest.stat().st_size > 0:
        print("exists:", dest); continue
    print("downloading", url); urllib.request.urlretrieve(url, dest)
    print("saved", dest, dest.stat().st_size, "bytes")
