import pytest

from multimodal_rag.chunking.semantic import SemanticChunker
from multimodal_rag.ingestion.schema import Element, ElementMetadata, ElementType
from multimodal_rag.providers.base import EmbeddingProvider
from multimodal_rag.providers.schema import EmbeddingVector


def _meta(position: int) -> ElementMetadata:
    return ElementMetadata(source_file="doc.md", position=position)


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self._vectors_by_text = vectors_by_text

    def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        return [
            EmbeddingVector(vector=self._vectors_by_text[t], model_id="fake-model", dimension=2)
            for t in texts
        ]


def test_empty_elements_returns_no_chunks() -> None:
    assert SemanticChunker().chunk([]) == []


def test_single_sentence_is_its_own_chunk_without_calling_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called() -> EmbeddingProvider:
        raise AssertionError("should not embed for a single sentence")

    monkeypatch.setattr("multimodal_rag.chunking.semantic.get_embedder", fail_if_called)
    elements = [Element(type=ElementType.PARAGRAPH, text="Only one sentence.", metadata=_meta(0))]
    chunks = SemanticChunker().chunk(elements)
    assert len(chunks) == 1
    assert chunks[0].text == "Only one sentence."


def test_cuts_where_similarity_drops_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    sentences = ["Sentence one.", "Sentence two.", "Totally different topic.", "Still different."]
    vectors = {
        "Sentence one.": [1.0, 0.0],
        "Sentence two.": [0.99, 0.01],
        "Totally different topic.": [0.0, 1.0],
        "Still different.": [0.01, 0.99],
    }
    monkeypatch.setattr(
        "multimodal_rag.chunking.semantic.get_embedder",
        lambda: FakeEmbeddingProvider(vectors),
    )

    text = " ".join(sentences)
    elements = [Element(type=ElementType.PARAGRAPH, text=text, metadata=_meta(0))]
    chunks = SemanticChunker(similarity_threshold=0.5).chunk(elements)

    assert len(chunks) == 2
    assert chunks[0].text == "Sentence one. Sentence two."
    assert chunks[1].text == "Totally different topic. Still different."


def test_element_positions_not_tracked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "multimodal_rag.chunking.semantic.get_embedder",
        lambda: FakeEmbeddingProvider({}),
    )
    elements = [Element(type=ElementType.PARAGRAPH, text="Only one sentence.", metadata=_meta(4))]
    chunks = SemanticChunker().chunk(elements)
    assert chunks[0].metadata.element_positions == []


def test_regex_sentence_splitter_is_fooled_by_abbreviations() -> None:
    # Documented limitation: a real NLP sentence tokenizer would know "Dr."
    # doesn't end a sentence; our lightweight regex doesn't.
    text = "Dr. Smith wrote the report. It was thorough."
    sentences = SemanticChunker._split_sentences(text)
    assert sentences == ["Dr.", "Smith wrote the report.", "It was thorough."]
