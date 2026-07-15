"""Shared element -> text projection, used by every chunker that needs to
turn an Element into embeddable/chunkable text.

Same multi-vector idea as ingestion: an image/chart's *description* is its
text projection for chunking/embedding purposes, not its raw bytes.
"""

from ..ingestion.schema import Element, ElementType


def element_text(element: Element) -> str:
    if element.type in (ElementType.IMAGE, ElementType.CHART):
        return element.description or ""
    return element.text or ""


def flatten_elements(elements: list[Element]) -> str:
    return "\n\n".join(text for el in elements if (text := element_text(el)))
