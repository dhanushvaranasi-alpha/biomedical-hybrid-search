"""Citation validation - the single place citation correctness is enforced (used by API and eval).

Rules: cited PMIDs must be a subset of the PMIDs given to the LLM as context. Invalid citations are removed from the
displayed answer and counted. Answers left with no valid citation (or mostly uncited) become 'insufficient_evidence'.
"""
import re

CITE = re.compile(r"\[PMID:\s*(\d+(?:\s*,\s*(?:PMID:\s*)?\d+)*)\]", re.I)
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
MAX_UNCITED_FRAC = 0.5


def _ids(group: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", group)]


def validate(answer: str, context_pmids: set[int], model_insufficient: bool = False) -> dict:
    ctx = {int(p) for p in context_pmids}
    cited, invalid = [], []
    for m in CITE.finditer(answer):
        for i in _ids(m.group(1)):
            (cited if i in ctx else invalid).append(i)

    def strip_invalid(m):
        keep = [i for i in _ids(m.group(1)) if i in ctx]
        return "[" + ", ".join(f"PMID:{i}" for i in keep) + "]" if keep else ""

    clean = re.sub(r"\s+([.,;])", r"\1", CITE.sub(strip_invalid, answer)).strip()
    sentences = [s for s in SENT.split(clean) if s.strip()]
    uncited = [s for s in sentences if not CITE.search(s)]
    n_cit = len(cited) + len(invalid)
    reason = None
    if model_insufficient:
        reason = "model_reported_insufficient"
    elif not cited:
        reason = "no_valid_citations"
    elif sentences and len(uncited) / len(sentences) > MAX_UNCITED_FRAC:
        reason = "mostly_uncited"
    return {"status": "insufficient_evidence" if reason else "answered", "reason": reason,
            "answer": clean, "valid": sorted(set(cited)), "invalid": sorted(set(invalid)),
            "valid_rate": (len(cited) / n_cit) if n_cit else 0.0, "n_sentences": len(sentences),
            "uncited_sentences": uncited}
