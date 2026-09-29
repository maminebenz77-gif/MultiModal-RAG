from multimodal_rag.chunking.parent_child import ParentChildChunker
from multimodal_rag.chunking.structure_aware import StructureAwareChunker
from multimodal_rag.ingestion.schema import Element, ElementMetadata, ElementType


def _el(
    el_type: ElementType,
    position: int,
    text: str | None = None,
    page: int | None = None,
    heading_level: int | None = None,
) -> Element:
    return Element(
        type=el_type,
        text=text,
        metadata=ElementMetadata(
            source_file="doc.md",
            position=position,
            page=page,
            heading_level=heading_level,
        ),
    )


def test_empty_elements_returns_no_chunks() -> None:
    assert ParentChildChunker().chunk([]) == []


def test_short_section_produces_one_parent_and_one_child() -> None:
    elements = [
        _el(ElementType.TITLE, 0, "Section A"),
        _el(ElementType.PARAGRAPH, 1, "Short body."),
    ]
    chunks = ParentChildChunker(child_chunk_size=200, child_chunk_overlap=0).chunk(elements)

    parents = [c for c in chunks if c.parent_id is None]
    children = [c for c in chunks if c.parent_id is not None]
    assert len(parents) == 1
    assert len(children) == 1
    assert parents[0].id == StructureAwareChunker().chunk(elements)[0].id
    assert children[0].parent_id == parents[0].id


def test_leading_title_only_h1_is_merged_into_following_section() -> None:
    elements = [
        _el(ElementType.TITLE, 0, "MAEVA", heading_level=1),
        _el(ElementType.TITLE, 1, "Introduction", heading_level=2),
        _el(ElementType.PARAGRAPH, 2, "MAEVA is a thermal platform."),
    ]

    chunks = ParentChildChunker(child_chunk_size=200, child_chunk_overlap=0).chunk(elements)

    parents = [chunk for chunk in chunks if chunk.parent_id is None]
    children = [chunk for chunk in chunks if chunk.parent_id is not None]
    assert len(parents) == 1
    assert parents[0].text == "MAEVA\n\nIntroduction\n\nMAEVA is a thermal platform."
    assert len(children) == 1


def test_h1_with_body_remains_a_real_section() -> None:
    elements = [
        _el(ElementType.TITLE, 0, "Introduction", heading_level=1),
        _el(ElementType.PARAGRAPH, 1, "MAEVA is a thermal platform."),
        _el(ElementType.TITLE, 2, "Architecture", heading_level=2),
        _el(ElementType.PARAGRAPH, 3, "The architecture has three layers."),
    ]

    chunks = ParentChildChunker(child_chunk_size=200, child_chunk_overlap=0).chunk(elements)

    parents = [chunk for chunk in chunks if chunk.parent_id is None]
    assert [parent.text for parent in parents] == [
        "Introduction\n\nMAEVA is a thermal platform.",
        "Architecture\n\nThe architecture has three layers.",
    ]


def test_heading_only_parent_has_no_searchable_child() -> None:
    elements = [_el(ElementType.TITLE, 0, "Architecture", heading_level=2)]

    chunks = ParentChildChunker().chunk(elements)

    assert len(chunks) == 1
    assert chunks[0].is_parent
    assert chunks[0].parent_id is None


def test_long_section_produces_multiple_children_under_one_parent() -> None:
    long_body = " ".join(f"Sentence number {i}." for i in range(50))
    elements = [
        _el(ElementType.TITLE, 0, "Section A"),
        _el(ElementType.PARAGRAPH, 1, long_body),
    ]
    chunks = ParentChildChunker(child_chunk_size=100, child_chunk_overlap=0).chunk(elements)

    parents = [c for c in chunks if c.parent_id is None]
    children = [c for c in chunks if c.parent_id is not None]
    assert len(parents) == 1
    assert len(children) > 1
    assert all(c.parent_id == parents[0].id for c in children)


def test_oversized_section_produces_bounded_parents_with_linked_children() -> None:
    long_body = " ".join(f"Sentence number {i}." for i in range(100))
    elements = [
        _el(ElementType.TITLE, 0, "Section A"),
        _el(ElementType.PARAGRAPH, 1, long_body),
    ]
    chunks = ParentChildChunker(
        parent_chunk_size=300, child_chunk_size=100, child_chunk_overlap=0
    ).chunk(elements)

    parents = [c for c in chunks if c.parent_id is None]
    children = [c for c in chunks if c.parent_id is not None]
    assert len(parents) > 1
    assert all(len(parent.text) <= 300 for parent in parents)
    assert {child.parent_id for child in children} == {parent.id for parent in parents}


def test_multiple_sections_produce_independently_linked_parent_child_groups() -> None:
    elements = [
        _el(ElementType.TITLE, 0, "Section A"),
        _el(ElementType.PARAGRAPH, 1, "Body of A."),
        _el(ElementType.TITLE, 2, "Section B"),
        _el(ElementType.PARAGRAPH, 3, "Body of B."),
    ]
    chunks = ParentChildChunker(child_chunk_size=200, child_chunk_overlap=0).chunk(elements)

    parents = [c for c in chunks if c.parent_id is None]
    assert len(parents) == 2

    for parent in parents:
        children = [c for c in chunks if c.parent_id == parent.id]
        assert len(children) >= 1


def test_child_inherits_parent_element_positions() -> None:
    elements = [
        _el(ElementType.TITLE, 5, "Section A"),
        _el(ElementType.PARAGRAPH, 6, "Body."),
    ]
    chunks = ParentChildChunker(child_chunk_size=200, child_chunk_overlap=0).chunk(elements)
    child = next(c for c in chunks if c.parent_id is not None)
    assert child.metadata.element_positions == [5, 6]


def test_child_inherits_parent_element_types() -> None:
    elements = [
        _el(ElementType.TITLE, 5, "Section A"),
        _el(ElementType.PARAGRAPH, 6, "Body."),
    ]
    chunks = ParentChildChunker(child_chunk_size=200, child_chunk_overlap=0).chunk(elements)
    child = next(c for c in chunks if c.parent_id is not None)
    assert child.metadata.element_types == ["title", "paragraph"]


def test_child_inherits_parent_elements() -> None:
    elements = [
        _el(ElementType.TITLE, 5, "Section A"),
        _el(ElementType.PARAGRAPH, 6, "Body."),
    ]
    chunks = ParentChildChunker(child_chunk_size=200, child_chunk_overlap=0).chunk(elements)
    parent = next(c for c in chunks if c.parent_id is None)
    child = next(c for c in chunks if c.parent_id is not None)
    assert child.metadata.elements == parent.metadata.elements
    assert [e.type for e in child.metadata.elements] == ["title", "paragraph"]


def test_child_inherits_parent_pages() -> None:
    elements = [
        _el(ElementType.TITLE, 5, "Section A", page=2),
        _el(ElementType.PARAGRAPH, 6, "Body.", page=2),
    ]
    chunks = ParentChildChunker(child_chunk_size=200, child_chunk_overlap=0).chunk(elements)
    child = next(c for c in chunks if c.parent_id is not None)
    assert child.metadata.pages == [2]


def test_parents_appear_before_their_children_in_result_order() -> None:
    elements = [
        _el(ElementType.TITLE, 0, "Section A"),
        _el(ElementType.PARAGRAPH, 1, "Body of A."),
    ]
    chunks = ParentChildChunker(child_chunk_size=200, child_chunk_overlap=0).chunk(elements)
    assert chunks[0].parent_id is None
    assert chunks[1].parent_id == chunks[0].id


def test_only_parents_are_marked_is_parent() -> None:
    elements = [
        _el(ElementType.TITLE, 0, "Section A"),
        _el(ElementType.PARAGRAPH, 1, "Body of A."),
    ]
    chunks = ParentChildChunker(child_chunk_size=200, child_chunk_overlap=0).chunk(elements)

    parents = [c for c in chunks if c.parent_id is None]
    children = [c for c in chunks if c.parent_id is not None]
    assert all(c.is_parent for c in parents)
    assert all(not c.is_parent for c in children)
