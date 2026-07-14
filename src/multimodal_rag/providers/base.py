"""Abstract provider interfaces ("ports").

Everything outside this package should depend only on these types — never
on a concrete provider class — and should obtain instances exclusively via
multimodal_rag.providers.factory. That's what lets the LLM/embedding/vision
backend change per environment (Mac vs. air-gapped server) purely through
config, with zero changes to retrieval/generation/etc.
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: list[dict[str, str]]) -> str:
        """Send a chat-style message list and return the model's text reply."""


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors, one vector per input text."""


class VisionProvider(ABC):
    @abstractmethod
    def describe(self, image_bytes: bytes, prompt: str | None = None) -> str:
        """Produce a text description/analysis of an image."""


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: list[str]) -> list[int]:
        """Return document indices ordered from most to least relevant to the query."""
