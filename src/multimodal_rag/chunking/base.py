"""Abstract Chunker interface ("port") every strategy implements.

Same ports/adapters shape as providers and ingestion: downstream code
depends only on this interface, so strategies stay genuinely swappable.
"""

from abc import ABC, abstractmethod

from ..ingestion.schema import Element
from .schema import Chunk


class Chunker(ABC):
    @abstractmethod
    def chunk(self, elements: list[Element]) -> list[Chunk]:
        """Split a single document's elements into retrieval-sized Chunks."""
