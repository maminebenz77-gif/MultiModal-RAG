"""Shared provider-level data types.

EmbeddingVector exists specifically so every vector is self-describing:
vectors from different embedding models live in different, incompatible
geometric spaces (even if they happen to share a dimension count), so
anything that stores or compares vectors later needs to know which model
produced each one — that's what makes it possible to detect a stale
embedding and re-embed it, e.g. after switching models on the server.
"""

from typing import Any

from pydantic import BaseModel


class EmbeddingVector(BaseModel):
    vector: list[float]
    model_id: str
    dimension: int


class ToolCall(BaseModel):
    """One function call the model asked for, as returned by
    LLMProvider.generate_with_tools() -- `arguments` is already parsed
    into a dict, never the raw JSON string providers hand back, so
    callers never re-implement that parsing themselves."""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolResponse(BaseModel):
    """A generate_with_tools() reply: either `content` (the model chose
    to answer/respond in plain text) or `tool_calls` (it wants to call
    the tool(s) first) -- a model can also return both when it narrates
    before calling a tool, so callers should check `tool_calls` first."""

    content: str | None
    tool_calls: list[ToolCall] = []


class TokenChunk(BaseModel):
    """One fragment of streamed text from generate_stream()/
    generate_with_tools_stream() -- callers append `text` in arrival
    order to reconstruct the full string. Never reordered or buffered
    here; that's the caller's business if it needs it."""

    text: str


class ToolCallsChunk(BaseModel):
    """Terminal chunk of a generate_with_tools_stream() call, yielded
    once the model has finished emitting a tool call. OpenAI-style
    providers stream a call's arguments as fragments keyed by index, so
    this is only produced after they're fully reassembled and parsed --
    never partially, the way TokenChunk is."""

    tool_calls: list[ToolCall]


def assert_single_model(vectors: list[EmbeddingVector]) -> None:
    """Raise if `vectors` mixes more than one model_id.

    Vectors from different models must never be compared or stored
    together — this makes that invariant something callers can actually
    check, not just a comment.
    """
    model_ids = {v.model_id for v in vectors}
    if len(model_ids) > 1:
        raise ValueError(
            f"Refusing to mix vectors from different embedding models: {sorted(model_ids)}"
        )
