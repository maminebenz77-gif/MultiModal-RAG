"""RAG generation chain: retrieve -> grounded prompt -> LLM -> answer
with citations.

LCEL provides orchestration/composition; the actual LLM call still goes
through get_llm(), so "same chain runs against the local OpenAI key and
the internal company LLM on the server, no code change" is exactly as
true here as everywhere else in this project — LCEL never decides which
backend runs, it only sequences the steps around it. Using LangChain's
own model-wrapper classes here instead would have reintroduced the
vendor coupling the whole providers layer exists to avoid.

State flows through the chain as a dict, each step adding a key — the
standard LCEL pattern for carrying auxiliary data (here: the context
results, needed again after the LLM call to map citation markers back
to real metadata) alongside the main value.
"""

from typing import Any, Protocol

from langchain_core.runnables import RunnableLambda

from ..providers.factory import get_llm
from ..retrieval.schema import RetrievalMethod
from ..stores.schema import SearchResult
from .context import assemble_context
from .parse import parse_answer
from .prompt import build_messages
from .schema import RagAnswer

_DEFAULT_TOKEN_BUDGET = 2000


class RetrieverLike(Protocol):
    """What RagChain actually needs from a retriever — depending on this
    instead of the concrete Retriever class means anything with a
    compatible retrieve() works here (a test stub, a future caching
    wrapper, ...), not just Retriever itself."""

    def retrieve(
        self, query: str, method: RetrievalMethod, top_k: int
    ) -> list[SearchResult]: ...


class RagChain:
    def __init__(
        self,
        retriever: RetrieverLike,
        method: RetrievalMethod = RetrievalMethod.HYBRID_RRF,
        top_k: int = 5,
        token_budget: int = _DEFAULT_TOKEN_BUDGET,
    ) -> None:
        self._retriever = retriever
        self._method = method
        self._top_k = top_k
        self._token_budget = token_budget
        self._chain = (
            RunnableLambda(self._retrieve)
            | RunnableLambda(self._build_messages)
            | RunnableLambda(self._call_llm)
            | RunnableLambda(self._parse)
        )

    def answer(self, query: str) -> RagAnswer:
        result: RagAnswer = self._chain.invoke(query)
        return result

    def _retrieve(self, query: str) -> dict[str, Any]:
        results = self._retriever.retrieve(query, method=self._method, top_k=self._top_k)
        context = assemble_context(results, self._token_budget)
        return {"query": query, "context": context}

    def _build_messages(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = build_messages(state["query"], state["context"])
        return {**state, "messages": messages}

    def _call_llm(self, state: dict[str, Any]) -> dict[str, Any]:
        raw_answer = get_llm().generate(state["messages"])
        return {**state, "raw_answer": raw_answer}

    def _parse(self, state: dict[str, Any]) -> RagAnswer:
        return parse_answer(state["raw_answer"], state["context"])
