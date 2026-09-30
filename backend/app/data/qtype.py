"""Rule-based question-type classifier (first matching rule wins). Version recorded in eval meta."""
import re

RULES_VERSION = "qtype_v2"

_RULES = [
    ("yesno", re.compile(r"\byes\s*(or|/)\s*no\b", re.I)),
    ("list", re.compile(r"^(list|which|name)\b|\bwhat are the\b|\bwhich are\b", re.I)),
    ("yesno", re.compile(r"^(is|are|does|do|did|can|could|was|were|has|have|had|should|will|would)\b", re.I)),
    ("treatment", re.compile(r"\b(treat|treatment|therapy|therapeutic|drug for|management of|approved for)\b", re.I)),
    ("mechanism", re.compile(r"\b(mechanism|how does|how do|pathway|mode of action|mediate|regulate)\b", re.I)),
    ("effect", re.compile(r"\b(effect of|effects of|effect on|impact|affect|associated with|cause|causes)\b", re.I)),
    ("definition", re.compile(r"^(what is|what are|what does|define|describe|who)\b|\bwhat is\b", re.I)),
]

def classify(question: str) -> str:
    q = question.strip()
    for name, rx in _RULES:
        if rx.search(q):
            return name
    return "other"
