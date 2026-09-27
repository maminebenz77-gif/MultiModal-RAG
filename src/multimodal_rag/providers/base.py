"""Abstract provider interfaces ("ports").

Everything outside this package should depend only on these types — never
on a concrete provider class — and should obtain instances exclusively via
multimodal_rag.providers.factory. That's what lets the LLM/embedding/vision
backend change per environment (Mac vs. air-gapped server) purely through
config, with zero changes to retrieval/generation/etc.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from .schema import EmbeddingVector, TokenChunk, ToolCallsChunk, ToolResponse


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

    def generate_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Streaming counterpart to generate() -- yields the reply as text
        fragments in arrival order instead of returning it all at once.

        Concrete with a NotImplementedError default, same rationale as
        generate_with_tools(): optional, so adding it doesn't break any
        existing provider or test double that only implements generate()."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")

    def generate_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Iterator[TokenChunk | ToolCallsChunk]:
        """Streaming counterpart to generate_with_tools() -- yields a
        TokenChunk per fragment of visible text as it's generated, and, if
        the model calls the tool, exactly one ToolCallsChunk once its
        arguments are fully reassembled (see ToolCallsChunk -- never
        interleaved fragment-by-fragment the way TokenChunk is).

        Concrete with a NotImplementedError default, same rationale as
        generate_with_tools()."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming tool calls")


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
