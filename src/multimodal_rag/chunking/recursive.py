"""Recursive character splitting — length-driven, but prefers natural
boundaries (paragraph, then line, then sentence, then word) over a hard
cut. Better default than fixed-size for prose, but still blind to our
Element types: it can still slice through a table, just less brutally.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..ingestion.schema import Element
from .base import Chunker
from .schema import Chunk, ChunkMetadata
from .text import flatten_elements


class RecursiveCharacterChunker(Chunker):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def chunk(self, elements: list[Element]) -> list[Chunk]:
        if not elements:
            return []
        source_file = elements[0].metadata.source_file
        pieces = self._splitter.split_text(flatten_elements(elements))
        return [
            Chunk(
                id=f"{source_file}::recursive::{i}",
                text=piece,
                metadata=ChunkMetadata(source_file=source_file),
            )
            for i, piece in enumerate(pieces)
        ]
