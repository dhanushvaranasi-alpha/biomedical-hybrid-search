import math
from eval.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from app.retrieval.fusion import rrf, weighted
from app.data.chunking import chunk_passage
from app.data.qtype import classify


def test_recall():
    assert recall_at_k([1, 2, 3, 4, 5, 6], {2, 6, 9}, 5) == 1 / 3
    assert recall_at_k([1, 2, 3, 4, 5, 6], {2, 6, 9}, 10) == 2 / 3
    assert recall_at_k([1], set(), 5) == 0.0


def test_mrr():
    assert mrr_at_k([9, 8, 7, 3], {3}, 10) == 0.25
    assert mrr_at_k([9, 8, 7, 3], {3}, 3) == 0.0


def test_ndcg_hand_computed():
    # gold {a,b}; ranking hits at ranks 1 and 3: DCG = 1 + 1/log2(4) = 1.5 ; IDCG = 1 + 1/log2(3)
    val = ndcg_at_k(["a", "x", "b"], {"a", "b"}, 10)
    assert math.isclose(val, 1.5 / (1 + 1 / math.log2(3)))
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 10) == 1.0
    assert ndcg_at_k(["x"], {"a"}, 10) == 0.0


def test_rrf():
    r = rrf([(1.0, [("a", 9), ("b", 8)]), (1.0, [("b", 5), ("c", 4)])], k_rrf=60)
    d = dict(r)
    assert math.isclose(d["b"], 1 / 62 + 1 / 61) and r[0][0] == "b"
    assert math.isclose(d["a"], 1 / 61) and math.isclose(d["c"], 1 / 62)


def test_rrf_weight():
    d = dict(rrf([(0.5, [("a", 1)])], k_rrf=60))
    assert math.isclose(d["a"], 0.5 / 61)


def test_weighted_alpha():
    r = weighted([(1.0, [("a", 10), ("b", 0)], [("b", 1.0), ("a", 0.0)])], alpha=1.0)
    assert r[0][0] == "b"


def test_chunking():
    assert chunk_passage("Short text here.") == ["Short text here."]
    long = " ".join(f"Sentence number {i} has some words in it." for i in range(100))
    ch = chunk_passage(long, max_words=50)
    assert len(ch) > 3 and all(len(c.split()) <= 65 for c in ch)


def test_qtype():
    assert classify("Is the protein Papilin secreted?") == "yesno"
    assert classify("Anaplasma is a bacterium, yes or no") == "yesno"
    assert classify("List signaling molecules that interact with EGFR?") == "list"
    assert classify("What is the mechanism of action of eprotirome?") == "mechanism"
