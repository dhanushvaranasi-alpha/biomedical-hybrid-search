"""Build the fixed 100-query evaluation set (+ a disjoint dev set for tuning). Deterministic (seed fixed).

Usage: python -m eval.build_query_set
"""
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.data.qtype import RULES_VERSION

TYPES = ["yesno", "definition", "list", "treatment", "mechanism", "effect"]
SEED_EVAL, SEED_DEV, N_EVAL, N_DEV = 42, 43, 100, 200


def allocate(n, k):  # 100 over 6 -> 17,17,17,17,16,16
    return [n // k + (1 if i < n % k else 0) for i in range(k)]


def sample(df, n, seed):
    parts = []
    for t, m in zip(TYPES, allocate(n, len(TYPES))):
        parts.append(df[df.qtype == t].sample(m, random_state=seed))
    return pd.concat(parts).sort_values("qid")


def dump(df, path):
    with open(path, "w") as f:
        for r in df.itertuples():
            f.write(json.dumps({"qid": int(r.qid), "question": r.question, "answer": r.answer, "qtype": r.qtype,
                                "gold_in_corpus": [int(x) for x in r.gold_in_corpus]}) + "\n")


def main():
    qa = pd.read_parquet(ROOT / "data" / "qa.parquet")
    qa = qa[qa["gold_in_corpus"].map(len) > 0]
    ev = sample(qa, N_EVAL, SEED_EVAL)
    dev = sample(qa[~qa.qid.isin(ev.qid)], N_DEV, SEED_DEV)
    assert not set(ev.qid) & set(dev.qid) and len(ev) == N_EVAL and ev.qid.is_unique
    dump(ev, ROOT / "eval" / "queries_100.jsonl"); dump(dev, ROOT / "eval" / "dev_200.jsonl")
    meta = {"seed_eval": SEED_EVAL, "seed_dev": SEED_DEV, "qtype_rules": RULES_VERSION,
            "eval_counts": ev.qtype.value_counts().to_dict(), "dev_counts": dev.qtype.value_counts().to_dict(),
            "eval_gold_mean": float(ev.gold_in_corpus.map(len).mean())}
    (ROOT / "eval" / "queries_meta.json").write_text(json.dumps(meta, indent=2)); print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
