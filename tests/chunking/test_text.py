from multimodal_rag.chunking.text import element_text, flatten_elements
from multimodal_rag.ingestion.schema import Element, ElementMetadata, ElementType


def _meta(position: int) -> ElementMetadata:
    return ElementMetadata(source_file="doc.md", position=position)


def test_element_text_uses_text_field_for_paragraph() -> None:
    el = Element(type=ElementType.PARAGRAPH, text="hello", metadata=_meta(0))
    assert element_text(el) == "hello"


def test_element_text_uses_description_for_image() -> None:
    el = Element(
        type=ElementType.IMAGE, description="a photo of a server rack", metadata=_meta(0)
    )
    assert element_text(el) == "a photo of a server rack"


def test_element_text_uses_description_for_chart() -> None:
    el = Element(type=ElementType.CHART, description="bar chart of latency", metadata=_meta(0))
    assert element_text(el) == "bar chart of latency"


def test_element_text_empty_when_nothing_set() -> None:
    el = Element(type=ElementType.IMAGE, metadata=_meta(0))
    assert element_text(el) == ""


def test_flatten_elements_joins_with_blank_line_and_skips_empty() -> None:
    elements = [
        Element(type=ElementType.TITLE, text="Title", metadata=_meta(0)),
        Element(type=ElementType.IMAGE, description=None, metadata=_meta(1)),
        Element(type=ElementType.PARAGRAPH, text="Body.", metadata=_meta(2)),
    ]
    assert flatten_elements(elements) == "Title\n\nBody."
