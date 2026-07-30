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

from ..ingestion.schema import Element, ElementType
from .base import Chunker
from .ids import chunk_id
from .schema import Chunk, ChunkMetadata
from .text import element_text


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
