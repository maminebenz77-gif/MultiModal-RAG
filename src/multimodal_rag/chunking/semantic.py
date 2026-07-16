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
from .base import Chunker
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
        return [
            Chunk(
                id=f"{source_file}::semantic::{i}",
                text=" ".join(group),
                metadata=ChunkMetadata(source_file=source_file),
            )
            for i, group in enumerate(groups)
        ]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]

    def _group_by_similarity(self, sentences: list[str]) -> list[list[str]]:
        if len(sentences) == 1:
            return [sentences]

        vectors = get_embedder().embed(sentences)
        groups: list[list[str]] = [[sentences[0]]]
        for sentence, vector, previous_vector in zip(
            sentences[1:], vectors[1:], vectors[:-1], strict=True
        ):
            if _cosine_similarity(vector, previous_vector) < self._threshold:
                groups.append([])
            groups[-1].append(sentence)
        return groups


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
