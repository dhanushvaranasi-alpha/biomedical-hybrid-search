"""Prepare rag-mini-bioasq: parse gold IDs, drop empty passages, chunk, classify question types.

Usage: python scripts/prepare_data.py   (expects data/raw/{qa,passages}.parquet, see README to download)
"""
import ast, json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.data.chunking import chunk_passage
from app.data.qtype import classify, RULES_VERSION

RAW, OUT = ROOT / "data" / "raw", ROOT / "data"
MIN_CHARS = 20  # passages shorter than this are empty placeholders ("nan", "1.") in the source


def main():
    qa = pd.read_parquet(RAW / "qa.parquet").reset_index()  # 'id' stored as index
    ps = pd.read_parquet(RAW / "passages.parquet").reset_index()
    ps["passage"] = ps["passage"].astype(str)
    n_total = len(ps)
    ps["text"] = ps["passage"].str.split().str.join(" ")
    empty = ps["text"].str.len() < MIN_CHARS
    n_empty = int(empty.sum())
    passages = ps.loc[~empty, ["id", "text"]].rename(columns={"id": "pmid"}).reset_index(drop=True)
    valid = set(passages["pmid"])

    qa["gold_pmids"] = qa["relevant_passage_ids"].map(ast.literal_eval)
    qa["gold_in_corpus"] = qa["gold_pmids"].map(lambda g: [x for x in g if x in valid])
    qa["qtype"] = qa["question"].map(classify)
    qa = qa.rename(columns={"id": "qid"})[["qid", "question", "answer", "gold_pmids", "gold_in_corpus", "qtype"]]
    n_no_gold = int((qa["gold_in_corpus"].map(len) == 0).sum())

    rows = []
    for pmid, text in zip(passages["pmid"], passages["text"]):
        for i, c in enumerate(chunk_passage(text)):
            rows.append((f"{pmid}:{i}", pmid, i, c))
    chunks = pd.DataFrame(rows, columns=["chunk_id", "pmid", "idx", "text"])

    passages.to_parquet(OUT / "passages.parquet", index=False)
    chunks.to_parquet(OUT / "chunks.parquet", index=False)
    qa.to_parquet(OUT / "qa.parquet", index=False)

    n_gold = int(qa["gold_pmids"].map(len).sum())
    n_gold_ok = int(qa["gold_in_corpus"].map(len).sum())
    report = {
        "passages_raw": n_total, "passages_empty_dropped": n_empty, "passages_indexed": len(passages),
        "chunks": len(chunks), "questions": len(qa),
        "questions_with_no_usable_gold": n_no_gold, "questions_usable": len(qa) - n_no_gold,
        "gold_ids_total": n_gold, "gold_ids_usable": n_gold_ok,
        "qtype_counts_usable": qa[qa["gold_in_corpus"].map(len) > 0]["qtype"].value_counts().to_dict(),
        "qtype_rules_version": RULES_VERSION, "min_chars": MIN_CHARS,
    }
    (OUT / "prep_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
