"""Parent-child / small-to-big chunking.

Not really a distinct boundary-detection algorithm — it's a way of using
two of the other strategies at two granularities at once. Parents are
whole sections (StructureAwareChunker); each parent's text is further
split into smaller children (RecursiveCharacterTextSplitter), linked back
via parent_id. In a real retrieval step, children would be what's
searched (precise matching), while their parent is what's actually
returned to the LLM (full section context) — solving the classic tension
where small chunks retrieve accurately but lack context, and large
chunks have context but retrieve imprecisely.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..ingestion.schema import Element
from .base import Chunker
from .schema import Chunk, ChunkMetadata
from .structure_aware import StructureAwareChunker


class ParentChildChunker(Chunker):
    def __init__(self, child_chunk_size: int = 200, child_chunk_overlap: int = 20) -> None:
        self._parent_chunker = StructureAwareChunker()
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap
        )

    def chunk(self, elements: list[Element]) -> list[Chunk]:
        result: list[Chunk] = []
        for parent in self._parent_chunker.chunk(elements):
            result.append(parent)
            for j, piece in enumerate(self._child_splitter.split_text(parent.text)):
                result.append(
                    Chunk(
                        id=f"{parent.id}::child::{j}",
                        text=piece,
                        parent_id=parent.id,
                        metadata=ChunkMetadata(
                            source_file=parent.metadata.source_file,
                            element_positions=parent.metadata.element_positions,
                            element_types=parent.metadata.element_types,
                        ),
                    )
                )
        return result
