from multimodal_rag.chunking.recursive import RecursiveCharacterChunker
from multimodal_rag.ingestion.schema import Element, ElementMetadata, ElementType


def _meta(position: int) -> ElementMetadata:
    return ElementMetadata(source_file="doc.md", position=position)


def test_empty_elements_returns_no_chunks() -> None:
    assert RecursiveCharacterChunker().chunk([]) == []


def test_short_text_stays_in_one_chunk() -> None:
    elements = [Element(type=ElementType.PARAGRAPH, text="short text", metadata=_meta(0))]
    chunks = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50).chunk(elements)
    assert len(chunks) == 1


def test_prefers_paragraph_boundary_over_hard_cut() -> None:
    text = "First paragraph is short.\n\nSecond paragraph is also fairly short."
    elements = [Element(type=ElementType.PARAGRAPH, text=text, metadata=_meta(0))]
    chunks = RecursiveCharacterChunker(chunk_size=30, chunk_overlap=0).chunk(elements)
    # A paragraph-boundary-respecting split should isolate the first
    # sentence cleanly rather than cutting mid-word.
    assert chunks[0].text.strip() == "First paragraph is short."


def test_element_positions_not_tracked() -> None:
    elements = [Element(type=ElementType.PARAGRAPH, text="hello world", metadata=_meta(2))]
    chunks = RecursiveCharacterChunker().chunk(elements)
    assert chunks[0].metadata.element_positions == []
