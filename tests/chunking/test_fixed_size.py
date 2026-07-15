from multimodal_rag.chunking.fixed_size import FixedSizeChunker
from multimodal_rag.ingestion.schema import Element, ElementMetadata, ElementType


def _meta(position: int) -> ElementMetadata:
    return ElementMetadata(source_file="doc.md", position=position)


def test_empty_elements_returns_no_chunks() -> None:
    assert FixedSizeChunker().chunk([]) == []


def test_short_text_stays_in_one_chunk() -> None:
    elements = [Element(type=ElementType.PARAGRAPH, text="short text", metadata=_meta(0))]
    chunks = FixedSizeChunker(chunk_size=500, chunk_overlap=50).chunk(elements)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"


def test_long_text_is_split_into_multiple_uniform_chunks() -> None:
    long_text = "A" * 40 + " " + "B" * 40 + " " + "C" * 40
    elements = [Element(type=ElementType.PARAGRAPH, text=long_text, metadata=_meta(0))]
    chunks = FixedSizeChunker(chunk_size=20, chunk_overlap=5).chunk(elements)
    assert len(chunks) > 1
    assert all(len(c.text) <= 20 for c in chunks)


def test_chunk_ids_are_sequential_and_stable() -> None:
    elements = [Element(type=ElementType.PARAGRAPH, text="A" * 50, metadata=_meta(0))]
    chunks = FixedSizeChunker(chunk_size=20, chunk_overlap=5).chunk(elements)
    assert [c.id for c in chunks] == [f"doc.md::fixed::{i}" for i in range(len(chunks))]


def test_element_positions_not_tracked() -> None:
    elements = [Element(type=ElementType.PARAGRAPH, text="hello world", metadata=_meta(3))]
    chunks = FixedSizeChunker().chunk(elements)
    assert chunks[0].metadata.element_positions == []


def test_cuts_through_a_table_boundary_content_blind() -> None:
    # Demonstrates the strategy's core downside directly: it has no idea
    # this text is a markdown table and will happily cut it mid-row.
    table_markdown = "| A | B |\n| --- | --- |\n| " + ("x" * 60) + " | y |"
    elements = [Element(type=ElementType.TABLE, text=table_markdown, metadata=_meta(0))]
    chunks = FixedSizeChunker(chunk_size=20, chunk_overlap=0).chunk(elements)
    assert len(chunks) > 1
    assert not any(c.text.strip() == table_markdown.strip() for c in chunks)
