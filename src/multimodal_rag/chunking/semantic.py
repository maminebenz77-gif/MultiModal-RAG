"""Semantic chunking: split text into sentences, embed each one via
get_embedder(), and cut where similarity between consecutive sentences
drops below a threshold — boundaries follow where the *topic* shifts,
not a fixed length or document markup.

Uses a lightweight regex sentence splitter rather than a full NLP
tokenizer — a known simplification (won't handle abbreviations like
"Dr." or "e.g." correctly), noted here rather than hidden.

The most expensive strategy: every sentence gets embedded here, before
the resulting chunks get embedded *again* later for the vector store.
Goes through get_embedder() like everything else that needs embeddings —
never a bypassed, direct backend call.
"""

import re

from ..ingestion.schema import Element
from ..providers.factory import get_embedder
from ..similarity import cosine_similarity
from .base import Chunker
from .ids import chunk_id
from .schema import Chunk, ChunkMetadata
from .text import flatten_elements

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class SemanticChunker(Chunker):
    def __init__(self, similarity_threshold: float = 0.5) -> None:
        self._threshold = similarity_threshold

    def chunk(self, elements: list[Element]) -> list[Chunk]:
        if not elements:
            return []
        source_file = elements[0].metadata.source_file
        sentences = self._split_sentences(flatten_elements(elements))
        if not sentences:
            return []

        groups = self._group_by_similarity(sentences)
        texts = [" ".join(group) for group in groups]
        return [
            Chunk(
                id=chunk_id(source_file, "semantic", i, text),
                text=text,
                metadata=ChunkMetadata(source_file=source_file),
            )
            for i, text in enumerate(texts)
        ]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]

    def _group_by_similarity(self, sentences: list[str]) -> list[list[str]]:
        if len(sentences) == 1:
            return [sentences]

        vectors = [ev.vector for ev in get_embedder().embed(sentences)]
        groups: list[list[str]] = [[sentences[0]]]
        for sentence, vector, previous_vector in zip(
            sentences[1:], vectors[1:], vectors[:-1], strict=True
        ):
            if cosine_similarity(vector, previous_vector) < self._threshold:
                groups.append([])
            groups[-1].append(sentence)
        return groups
