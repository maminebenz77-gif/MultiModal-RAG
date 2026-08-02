"""RAG generation chain: rewrite -> retrieve -> grounded prompt -> LLM
-> answer with citations.

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
to real metadata) alongside the main value. The rewrite step is a no-op
LLM-call-wise when there's no conversation history, so single-turn
callers pay nothing extra.
"""

from typing import Any, Protocol

from langchain_core.runnables import RunnableLambda

from ..providers.factory import get_llm
from ..retrieval.schema import RetrievalMethod
from ..stores.schema import SearchResult
from .context import assemble_context
from .parse import parse_answer
from .prompt import build_messages
from .rewrite import rewrite_query
from .schema import RagAnswer

_DEFAULT_TOKEN_BUDGET = 2000


class RetrieverLike(Protocol):
    """What RagChain actually needs from a retriever — depending on this
    instead of the concrete Retriever class means anything with a
    compatible retrieve() works here (a test stub, a future caching
    wrapper, ...), not just Retriever itself."""

    def retrieve(
        self,
        query: str,
        method: RetrievalMethod,
        top_k: int,
        *,
        resolve_parent_context: bool = False,
        doc_ids: list[str] | None = None,
    ) -> list[SearchResult]: ...


class RagChain:
    def __init__(
        self,
        retriever: RetrieverLike,
        method: RetrievalMethod = RetrievalMethod.HYBRID_RRF,
        top_k: int = 5,
        token_budget: int = _DEFAULT_TOKEN_BUDGET,
        resolve_parent_context: bool = False,
    ) -> None:
        self._retriever = retriever
        self._method = method
        self._top_k = top_k
        self._token_budget = token_budget
        self._resolve_parent_context = resolve_parent_context
        self._chain = (
            RunnableLambda(self._rewrite)
            | RunnableLambda(self._retrieve)
            | RunnableLambda(self._build_messages)
            | RunnableLambda(self._call_llm)
            | RunnableLambda(self._parse)
        )

    def answer(
        self,
        query: str,
        history: list[tuple[str, str]] | None = None,
        doc_ids: list[str] | None = None,
    ) -> RagAnswer:
        """`history` is prior (question, answer) turns, oldest first. If
        non-empty, the query is rewritten into a standalone form before
        retrieval — see rewrite.py. Omitted or empty, this behaves exactly
        like single-turn use. `doc_ids`, if given, restricts retrieval to
        those documents."""
        result: RagAnswer = self._chain.invoke(
            {"query": query, "history": history or [], "doc_ids": doc_ids}
        )
        return result

    def _rewrite(self, state: dict[str, Any]) -> dict[str, Any]:
        query = rewrite_query(state["query"], state["history"])
        return {**state, "query": query}

    def _retrieve(self, state: dict[str, Any]) -> dict[str, Any]:
        results = self._retriever.retrieve(
            state["query"],
            method=self._method,
            top_k=self._top_k,
            resolve_parent_context=self._resolve_parent_context,
            doc_ids=state["doc_ids"],
        )
        context = assemble_context(results, self._token_budget)
        return {**state, "context": context}

    def _build_messages(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = build_messages(state["query"], state["context"])
        return {**state, "messages": messages}

    def _call_llm(self, state: dict[str, Any]) -> dict[str, Any]:
        raw_answer = get_llm().generate(state["messages"])
        return {**state, "raw_answer": raw_answer}

    def _parse(self, state: dict[str, Any]) -> RagAnswer:
        return parse_answer(state["raw_answer"], state["context"])
