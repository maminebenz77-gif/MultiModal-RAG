"""Parent-child / small-to-big chunking.

Not really a distinct boundary-detection algorithm — it's a way of using
two of the other strategies at two granularities at once. Parents are
whole sections (StructureAwareChunker); each parent's text is further
split into smaller children (RecursiveCharacterTextSplitter), linked back
via parent_id. Children are what's searched (precise matching) — parents
are marked is_parent=True specifically so VectorStore/KeywordStore.
search() can exclude them from ever being a direct hit — while a
matched child's parent is what's actually returned to the LLM (full
section context), via Retriever.resolve_parent_context. This solves the
classic tension where small chunks retrieve accurately but lack context,
and large chunks have context but retrieve imprecisely.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..ingestion.schema import Element
from .base import Chunker
from .ids import chunk_id
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
        for raw_parent in self._parent_chunker.chunk(elements):
            parent = raw_parent.model_copy(update={"is_parent": True})
            result.append(parent)
            for j, piece in enumerate(self._child_splitter.split_text(parent.text)):
                result.append(
                    Chunk(
                        id=chunk_id(parent.id, "child", j, piece),
                        text=piece,
                        parent_id=parent.id,
                        metadata=ChunkMetadata(
                            source_file=parent.metadata.source_file,
                            element_positions=parent.metadata.element_positions,
                            element_types=parent.metadata.element_types,
                            pages=parent.metadata.pages,
                            slides=parent.metadata.slides,
                        ),
                    )
                )
        return result
