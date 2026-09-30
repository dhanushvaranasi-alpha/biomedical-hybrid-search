"""Sentence-aware chunking for the dense index. BM25 uses whole passages."""
import re

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")

def chunk_passage(text: str, max_words: int = 180, overlap_sents: int = 1) -> list[str]:
    text = " ".join(text.split())
    if len(text.split()) <= max_words:
        return [text]
    sents = _SENT.split(text)
    chunks, cur, cur_w = [], [], 0
    for s in sents:
        w = len(s.split())
        # a single very long "sentence": hard-split by words
        if w > max_words:
            if cur:
                chunks.append(" ".join(cur)); cur, cur_w = [], 0
            words = s.split()
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i + max_words]))
            continue
        if cur_w + w > max_words and cur:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sents:] if overlap_sents else []
            cur_w = sum(len(x.split()) for x in cur)
            if cur_w + w > max_words * 1.3:  # overlap too costly: drop it
                cur, cur_w = [], 0
        cur.append(s); cur_w += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks
