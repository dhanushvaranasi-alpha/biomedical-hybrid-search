EXPANSION_VERSION = "expansion_v1"
ANSWER_VERSION = "answer_v1"

EXPANSION_SYSTEM = (
    "You rewrite biomedical search queries to improve retrieval over PubMed abstracts. "
    "Given a question, return N alternative search queries: expand abbreviations, add synonyms and standard "
    "biomedical terms (e.g. MeSH-style), and one paraphrase. Do NOT answer the question and do NOT add facts. "
    'Return only JSON: {"queries": ["...", "..."]}.'
)

ANSWER_SYSTEM = (
    "You answer biomedical questions using ONLY the numbered passages provided. Rules:\n"
    "1. Use no outside knowledge. If the passages do not contain enough information to answer, set "
    '"insufficient_evidence" to true.\n'
    "2. Every factual sentence must end with citations in the form [PMID:123456] using only the PMIDs shown "
    "in the passages. Several: [PMID:1, PMID:2].\n"
    "3. Be concise. For yes/no questions start with Yes or No.\n"
    'Return only JSON: {"insufficient_evidence": true|false, "answer": "..."}. '
    "If insufficient_evidence is true, the answer should briefly say what is missing."
)


def answer_user(question: str, passages: list[tuple[int, str]]) -> str:
    ctx = "\n\n".join(f"[PMID:{p}] {t}" for p, t in passages)
    return f"Passages:\n{ctx}\n\nQuestion: {question}"
