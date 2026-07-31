import pytest

from multimodal_rag.generation.chain import RagChain
from multimodal_rag.generation.prompt import REFUSAL_TEXT
from multimodal_rag.providers.base import LLMProvider
from multimodal_rag.retrieval.schema import RetrievalMethod
from multimodal_rag.stores.schema import SearchResult


class FakeLLM(LLMProvider):
    def __init__(self, response: str) -> None:
        self._response = response
        self.last_messages: list[dict[str, str]] | None = None

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.last_messages = messages
        return self._response


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.last_call: dict[str, object] | None = None

    def retrieve(
        self,
        query: str,
        method: RetrievalMethod,
        top_k: int,
        resolve_parent_context: bool = False,
    ) -> list[SearchResult]:
        self.last_call = {
            "query": query,
            "method": method,
            "top_k": top_k,
            "resolve_parent_context": resolve_parent_context,
        }
        return self._results


def _result(chunk_id: str, text: str, source: str = "doc.md") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=1.0,
        text=text,
        source=source,
        doc_id=source,
        element_types=["title"],
    )


def test_answer_returns_grounded_answer_with_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = FakeLLM("The latency was 220ms ⟦1⟧.")
    monkeypatch.setattr("multimodal_rag.generation.chain.get_llm", lambda: fake_llm)

    retriever = FakeRetriever([_result("a", "gpt-4o-mini had 220ms latency")])
    chain = RagChain(retriever)

    result = chain.answer("What was the latency?")

    assert result.answer == "The latency was 220ms ⟦1⟧."
    assert result.citations[0].chunk_id == "a"
    assert result.refused is False


def test_answer_passes_query_method_and_top_k_to_retriever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = FakeLLM("An answer ⟦1⟧.")
    monkeypatch.setattr("multimodal_rag.generation.chain.get_llm", lambda: fake_llm)

    retriever = FakeRetriever([_result("a", "text")])
    chain = RagChain(retriever, method=RetrievalMethod.BM25, top_k=3)

    chain.answer("a question")

    assert retriever.last_call == {
        "query": "a question",
        "method": RetrievalMethod.BM25,
        "top_k": 3,
        "resolve_parent_context": False,
    }


def test_resolve_parent_context_is_passed_through_to_retriever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = FakeLLM("An answer ⟦1⟧.")
    monkeypatch.setattr("multimodal_rag.generation.chain.get_llm", lambda: fake_llm)

    retriever = FakeRetriever([_result("a", "text")])
    chain = RagChain(retriever, resolve_parent_context=True)

    chain.answer("a question")

    assert retriever.last_call is not None
    assert retriever.last_call["resolve_parent_context"] is True


def test_answer_sends_grounded_prompt_containing_chunk_text_to_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = FakeLLM("An answer ⟦1⟧.")
    monkeypatch.setattr("multimodal_rag.generation.chain.get_llm", lambda: fake_llm)

    retriever = FakeRetriever([_result("a", "some unique chunk text")])
    chain = RagChain(retriever)

    chain.answer("a question")

    assert fake_llm.last_messages is not None
    assert "some unique chunk text" in fake_llm.last_messages[1]["content"]


def test_refusal_response_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = FakeLLM(REFUSAL_TEXT)
    monkeypatch.setattr("multimodal_rag.generation.chain.get_llm", lambda: fake_llm)

    retriever = FakeRetriever([_result("a", "unrelated text")])
    chain = RagChain(retriever)

    result = chain.answer("a question with no answer in the corpus")

    assert result.refused is True
    assert result.citations == []


def test_no_retrieved_results_still_produces_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = FakeLLM(REFUSAL_TEXT)
    monkeypatch.setattr("multimodal_rag.generation.chain.get_llm", lambda: fake_llm)

    retriever = FakeRetriever([])
    chain = RagChain(retriever)

    result = chain.answer("anything")

    assert result.refused is True


def test_history_triggers_rewrite_and_retrieval_uses_rewritten_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rewrite_llm = FakeLLM("What was gpt-4o-mini's latency?")
    answer_llm = FakeLLM("The latency was 220ms ⟦1⟧.")
    monkeypatch.setattr("multimodal_rag.generation.rewrite.get_llm", lambda: rewrite_llm)
    monkeypatch.setattr("multimodal_rag.generation.chain.get_llm", lambda: answer_llm)

    retriever = FakeRetriever([_result("a", "gpt-4o-mini had 220ms latency")])
    chain = RagChain(retriever)

    chain.answer("What about that?", history=[("What model was used?", "gpt-4o-mini")])

    assert retriever.last_call is not None
    assert retriever.last_call["query"] == "What was gpt-4o-mini's latency?"


def test_no_history_skips_the_rewrite_llm_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_get_llm() -> FakeLLM:
        raise AssertionError("rewrite must not call get_llm() when there's no history")

    monkeypatch.setattr("multimodal_rag.generation.rewrite.get_llm", _fail_get_llm)
    answer_llm = FakeLLM("An answer ⟦1⟧.")
    monkeypatch.setattr("multimodal_rag.generation.chain.get_llm", lambda: answer_llm)

    retriever = FakeRetriever([_result("a", "text")])
    chain = RagChain(retriever)

    chain.answer("a question")

    assert retriever.last_call is not None
    assert retriever.last_call["query"] == "a question"
