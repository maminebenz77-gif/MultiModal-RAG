"""Abstract provider interfaces ("ports").

Everything outside this package should depend only on these types — never
on a concrete provider class — and should obtain instances exclusively via
multimodal_rag.providers.factory. That's what lets the LLM/embedding/vision
backend change per environment (Mac vs. air-gapped server) purely through
config, with zero changes to retrieval/generation/etc.
"""

from abc import ABC, abstractmethod
from typing import Any

from .schema import EmbeddingVector, ToolResponse


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: list[dict[str, str]]) -> str:
        """Send a chat-style message list and return the model's text reply."""

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ToolResponse:
        """Send a chat-style message list plus OpenAI-style function-tool
        schemas, and return either a text reply or the tool call(s) the
        model wants made. `tool_choice` optionally constrains that decision
        for calls where application control flow requires a tool invocation.

        Concrete, not abstract, with a NotImplementedError default --
        unlike generate(), not every provider needs to support this, and
        making it optional means adding it doesn't break InternalServerLLM
        or any existing test double that only implements generate()."""
        raise NotImplementedError(f"{type(self).__name__} does not support tool calling")


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        """Embed a batch of texts into vectors, one vector per input text.

        Each EmbeddingVector carries the model_id and dimension that
        produced it — never just a bare list of floats — since vectors
        from different models can never be compared or mixed.
        """


class VisionProvider(ABC):
    @abstractmethod
    def describe(self, image_bytes: bytes, prompt: str | None = None) -> str:
        """Produce a text description/analysis of an image."""


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: list[str]) -> list[int]:
        """Return document indices ordered from most to least relevant to the query."""
