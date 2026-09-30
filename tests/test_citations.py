from app.generation.citations import validate


def test_valid_answer():
    r = validate("Yes. Papilin is secreted [PMID:1]. It binds collagen [PMID:2, PMID:1].", {1, 2, 3})
    assert r["status"] == "answered" and r["valid"] == [1, 2] and r["invalid"] == [] and r["valid_rate"] == 1.0


def test_invalid_removed_and_counted():
    r = validate("It is secreted [PMID:1]. It is also x [PMID:99].", {1, 2})
    assert r["invalid"] == [99] and "99" not in r["answer"] and r["valid"] == [1]
    assert r["valid_rate"] == 0.5


def test_all_invalid_is_insufficient():
    r = validate("Claim [PMID:99].", {1})
    assert r["status"] == "insufficient_evidence" and r["reason"] == "no_valid_citations"


def test_uncited_is_insufficient():
    r = validate("Claim one. Claim two. Claim three [PMID:1].", {1})
    assert r["status"] == "insufficient_evidence" and r["reason"] == "mostly_uncited"


def test_model_flag():
    assert validate("Not enough info.", {1}, model_insufficient=True)["reason"] == "model_reported_insufficient"


def test_multi_id_group_with_one_invalid():
    r = validate("Fact [PMID:1, PMID:7].", {1})
    assert r["valid"] == [1] and r["invalid"] == [7] and "[PMID:1]" in r["answer"]
