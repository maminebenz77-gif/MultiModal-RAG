"""Structure-aware chunking: split on TITLE elements, using the types
ingestion already tagged.

The only strategy that uses everything ingestion built — a table or
image is never split mid-element, and chunk boundaries match a section a
human actually wrote, not an arbitrary length. Fully dependent on parsing
quality: a misdetected title silently produces a wrong boundary.

Deliberately left "pure" (no internal length-based subdivision of
oversized sections) so its real downside — very uneven chunk sizes — stays
visible in the strategy comparison rather than being hidden by a hybrid
fallback.
"""

import base64

from ..image_utils import downscale_image
from ..ingestion.schema import Element, ElementType
from .base import Chunker
from .ids import chunk_id
from .schema import Chunk, ChunkElement, ChunkMetadata
from .text import element_text

# A detail-view thumbnail needs far less resolution than a vision-model
# input (providers/vision.py uses 1024) -- this is purely for on-screen
# preview, and a smaller image keeps the store payload compact.
_THUMBNAIL_MAX_DIMENSION = 512


def _to_chunk_element(element: Element) -> ChunkElement:
    if element.type in (ElementType.IMAGE, ElementType.CHART):
        image_base64 = None
        if element.image_bytes:
            thumbnail = downscale_image(element.image_bytes, _THUMBNAIL_MAX_DIMENSION)
            image_base64 = base64.b64encode(thumbnail).decode("ascii")
        return ChunkElement(
            type=element.type.value,
            image_base64=image_base64,
            description=element.description,
            page=element.metadata.page,
            slide=element.metadata.slide,
        )
    return ChunkElement(
        type=element.type.value,
        text=element.text,
        page=element.metadata.page,
        slide=element.metadata.slide,
    )


class StructureAwareChunker(Chunker):
    def chunk(self, elements: list[Element]) -> list[Chunk]:
        if not elements:
            return []
        source_file = elements[0].metadata.source_file
        sections = self._group_into_sections(elements)

        chunks = []
        for i, section in enumerate(sections):
            text = "\n\n".join(t for el in section if (t := element_text(el)))
            positions = [el.metadata.position for el in section]
            types = list(dict.fromkeys(el.type.value for el in section))
            chunk_elements = [_to_chunk_element(el) for el in section]
            pages = list(
                dict.fromkeys(el.metadata.page for el in section if el.metadata.page is not None)
            )
            slides = list(
                dict.fromkeys(el.metadata.slide for el in section if el.metadata.slide is not None)
            )
            chunks.append(
                Chunk(
                    id=chunk_id(source_file, "structure", i, text),
                    text=text,
                    metadata=ChunkMetadata(
                        source_file=source_file,
                        element_positions=positions,
                        element_types=types,
                        elements=chunk_elements,
                        pages=pages,
                        slides=slides,
                    ),
                )
            )
        return chunks

    @staticmethod
    def _group_into_sections(elements: list[Element]) -> list[list[Element]]:
        sections: list[list[Element]] = []
        current: list[Element] = []
        for el in elements:
            if el.type == ElementType.TITLE and current:
                sections.append(current)
                current = []
            current.append(el)
        if current:
            sections.append(current)
        return sections
