"""LLM-judge via RAGAS with fixed judge model/prompts/settings. All judge calls go through LLMClient
(disk cache + persisted spend cap + cost accounting).

Metrics (RAGAS 0.4 legacy classes, LLM-only, no embeddings):
  answer correctness  -> AnswerAccuracy   (response vs gold answer)
  groundedness        -> Faithfulness     (claims in answer supported by the retrieved contexts)
  context relevance   -> ContextRelevance (each retrieved context vs the question)
"""
import asyncio, os, re
from langchain_core.outputs import Generation, LLMResult
from ragas.dataset_schema import SingleTurnSample
from ragas.llms.base import BaseRagasLLM
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._nv_metrics import AnswerAccuracy, ContextRelevance

JUDGE_VERSION = "judge_v1"
CITE = re.compile(r"\[PMID:[^\]]*\]")


class ClientRagasLLM(BaseRagasLLM):
    def __init__(self, client, model: str, max_tokens: int = 3000):
        super().__init__()
        self.client, self.model, self.max_tokens = client, model, max_tokens
        self.calls, self.cost = 0, 0.0
        self._lock = __import__('threading').Lock()

    def is_finished(self, response) -> bool:
        return True

    def generate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None) -> LLMResult:
        r = self.client.chat(self.model, [{"role": "user", "content": prompt.to_string()}], JUDGE_VERSION,
                             max_tokens=self.max_tokens, temperature=0.0, reasoning={"effort": "low"})
        with self._lock:
            self.calls += 1; self.cost += r["cost_usd"]
        return LLMResult(generations=[[Generation(text=r["text"])]])

    async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None) -> LLMResult:
        return await asyncio.get_running_loop().run_in_executor(None, self.generate_text, prompt, n, temperature, stop, callbacks)


def make_metrics(client, model: str | None = None):
    llm = ClientRagasLLM(client, model or os.environ["LLM_JUDGE_MODEL"])
    return llm, AnswerAccuracy(llm=llm), Faithfulness(llm=llm), ContextRelevance(llm=llm)


async def judge_one(metrics, question: str, answer: str, contexts: list[str], reference: str, status: str) -> dict:
    """Returns scores in [0,1] or None when not applicable. Insufficient-evidence answers: correctness=0, groundedness=None."""
    _, acc, faith, ctxrel = metrics
    resp = CITE.sub("", answer).strip()
    s = SingleTurnSample(user_input=question, response=resp, retrieved_contexts=contexts, reference=reference)
    out = {"context_relevance": None, "answer_correctness": None, "groundedness": None}
    out["context_relevance"] = await ctxrel.single_turn_ascore(s)
    if status == "answered":
        out["answer_correctness"] = await acc.single_turn_ascore(s)
        out["groundedness"] = await faith.single_turn_ascore(s)
    else:
        out["answer_correctness"] = 0.0
    return out
