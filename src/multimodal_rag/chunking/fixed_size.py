"""Fixed-size chunking with overlap — content-blind, uniform chunk sizes.

Simplest and most predictable strategy, but has no idea a table or image
exists: it will cut straight through the middle of a table's markdown if
the table is longer than chunk_size. element_positions stays empty on
every chunk here — flattening to raw text before splitting loses the
ability to trace a chunk back to a specific source element.
"""

from langchain_text_splitters import CharacterTextSplitter

from ..ingestion.schema import Element
from .base import Chunker
from .ids import chunk_id
from .schema import Chunk, ChunkMetadata
from .text import flatten_elements


class FixedSizeChunker(Chunker):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self._splitter = CharacterTextSplitter(
            separator="", chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def chunk(self, elements: list[Element]) -> list[Chunk]:
        if not elements:
            return []
        source_file = elements[0].metadata.source_file
        pieces = self._splitter.split_text(flatten_elements(elements))
        return [
            Chunk(
                id=chunk_id(source_file, "fixed", i, piece),
                text=piece,
                metadata=ChunkMetadata(source_file=source_file),
            )
            for i, piece in enumerate(pieces)
        ]
