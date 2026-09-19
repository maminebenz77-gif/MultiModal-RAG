"""Shared element -> text projection, used by every chunker that needs to
turn an Element into embeddable/chunkable text.

Same multi-vector idea as ingestion: an image/chart's *description* is its
text projection for chunking/embedding purposes, not its raw bytes. A
table's projection is its LLM summary for the same reason -- a raw
markdown table is mostly punctuation and repeated short cell tokens,
which embeds poorly for semantic search, whereas "this table lists Q3
revenue by region" is exactly the kind of natural-language description a
query can actually match against. The real table text isn't lost: it's
still on the Element (and, after chunking, on ChunkMetadata.elements'
snapshot) for generation to use instead -- see
generation/context.py's _generation_text.
"""

from ..ingestion.schema import Element, ElementType


def element_text(element: Element) -> str:
    if element.type in (ElementType.IMAGE, ElementType.CHART):
        return element.description or ""
    if element.type == ElementType.TABLE:
        return element.table_summary or element.text or ""
    return element.text or ""


def flatten_elements(elements: list[Element]) -> str:
    return "\n\n".join(text for el in elements if (text := element_text(el)))
