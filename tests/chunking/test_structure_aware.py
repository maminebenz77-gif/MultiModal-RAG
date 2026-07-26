from multimodal_rag.chunking.structure_aware import StructureAwareChunker
from multimodal_rag.ingestion.schema import Element, ElementMetadata, ElementType


def _el(el_type: ElementType, position: int, text: str | None = None) -> Element:
    return Element(
        type=el_type, text=text, metadata=ElementMetadata(source_file="doc.md", position=position)
    )


def test_empty_elements_returns_no_chunks() -> None:
    assert StructureAwareChunker().chunk([]) == []


def test_starts_new_section_at_each_title() -> None:
    elements = [
        _el(ElementType.TITLE, 0, "Section A"),
        _el(ElementType.PARAGRAPH, 1, "Body of A."),
        _el(ElementType.TITLE, 2, "Section B"),
        _el(ElementType.PARAGRAPH, 3, "Body of B."),
    ]
    chunks = StructureAwareChunker().chunk(elements)
    assert len(chunks) == 2
    assert chunks[0].text == "Section A\n\nBody of A."
    assert chunks[1].text == "Section B\n\nBody of B."


def test_content_before_first_title_becomes_its_own_preamble_section() -> None:
    elements = [
        _el(ElementType.PARAGRAPH, 0, "Untitled preamble."),
        _el(ElementType.TITLE, 1, "Section A"),
        _el(ElementType.PARAGRAPH, 2, "Body of A."),
    ]
    chunks = StructureAwareChunker().chunk(elements)
    assert len(chunks) == 2
    assert chunks[0].text == "Untitled preamble."
    assert chunks[1].text == "Section A\n\nBody of A."


def test_a_table_is_never_split_across_chunks() -> None:
    table_text = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    elements = [
        _el(ElementType.TITLE, 0, "Section A"),
        _el(ElementType.TABLE, 1, table_text),
    ]
    chunks = StructureAwareChunker().chunk(elements)
    assert table_text in chunks[0].text


def test_element_positions_are_tracked_exactly() -> None:
    elements = [
        _el(ElementType.TITLE, 5, "Section A"),
        _el(ElementType.PARAGRAPH, 6, "Body."),
        _el(ElementType.TITLE, 9, "Section B"),
    ]
    chunks = StructureAwareChunker().chunk(elements)
    assert chunks[0].metadata.element_positions == [5, 6]
    assert chunks[1].metadata.element_positions == [9]


def test_no_titles_at_all_produces_a_single_section() -> None:
    elements = [
        _el(ElementType.PARAGRAPH, 0, "Para one."),
        _el(ElementType.PARAGRAPH, 1, "Para two."),
    ]
    chunks = StructureAwareChunker().chunk(elements)
    assert len(chunks) == 1


def test_element_types_reflects_every_distinct_type_in_the_section() -> None:
    table_text = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    elements = [
        _el(ElementType.TITLE, 0, "Section A"),
        _el(ElementType.PARAGRAPH, 1, "Body."),
        _el(ElementType.TABLE, 2, table_text),
    ]
    chunks = StructureAwareChunker().chunk(elements)
    assert chunks[0].metadata.element_types == ["title", "paragraph", "table"]


def test_element_types_has_no_duplicates() -> None:
    elements = [
        _el(ElementType.TITLE, 0, "Section A"),
        _el(ElementType.PARAGRAPH, 1, "First paragraph."),
        _el(ElementType.PARAGRAPH, 2, "Second paragraph."),
    ]
    chunks = StructureAwareChunker().chunk(elements)
    assert chunks[0].metadata.element_types == ["title", "paragraph"]
