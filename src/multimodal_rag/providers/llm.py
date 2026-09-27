"""Concrete LLMProvider implementations."""

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import litellm

from ..tracing import record_generation_result, traced_generation
from .base import LLMProvider
from .schema import TokenChunk, ToolCall, ToolCallsChunk, ToolResponse


def _ensure_current_event_loop() -> asyncio.AbstractEventLoop | None:
    try:
        asyncio.get_running_loop()
        return None
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


class LiteLLMProvider(LLMProvider):
    """Covers any OpenAI-compatible backend — real OpenAI, Ollama, vLLM, or
    an internal gateway that speaks the same schema — purely via
    model/base_url/api_key. This is what makes "switch provider by editing
    .env" possible: no code path here is provider-specific.
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None) -> None:
        # LiteLLM defaults to phoning home anonymous usage telemetry,
        # independent of whatever base_url/model we configured — that's a
        # network call our privacy guard (which only checks base_url) can't
        # see. Disable it unconditionally; there's no case where we want it.
        litellm.telemetry = False
        self._model = self._normalize_model(model, base_url)
        self._base_url = base_url
        self._api_key = api_key

    def generate(self, messages: list[dict[str, str]]) -> str:
        owned_loop = _ensure_current_event_loop()
        try:
            with traced_generation("generate", self._model, messages) as generation:
                response = litellm.completion(
                    model=self._model,
                    messages=messages,
                    base_url=self._base_url,
                    api_key=self._api_key,
                )
                content = response.choices[0].message.content or ""
                record_generation_result(generation, response, content)
        finally:
            if owned_loop is not None:
                owned_loop.close()
                asyncio.set_event_loop(None)
        return content

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ToolResponse:
        owned_loop = _ensure_current_event_loop()
        try:
            with traced_generation("generate_with_tools", self._model, messages) as generation:
                kwargs = dict(
                    model=self._model,
                    messages=messages,
                    tools=tools,
                    base_url=self._base_url,
                    api_key=self._api_key,
                )
                if tool_choice is not None:
                    kwargs["tool_choice"] = tool_choice
                response = litellm.completion(**kwargs)
                message = response.choices[0].message
                tool_calls = [
                    ToolCall(
                        id=call.id,
                        name=call.function.name,
                        # LiteLLM (like the raw OpenAI schema it mirrors) hands
                        # back arguments as a JSON string, not a parsed object
                        # -- decode it once here so nothing downstream
                        # re-implements this.
                        arguments=json.loads(call.function.arguments),
                    )
                    for call in (message.tool_calls or [])
                ]
                result = ToolResponse(content=message.content, tool_calls=tool_calls)
                record_generation_result(
                    generation, response, result.model_dump(mode="json")
                )
        finally:
            if owned_loop is not None:
                owned_loop.close()
                asyncio.set_event_loop(None)
        return result

    def generate_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        owned_loop = _ensure_current_event_loop()
        try:
            with traced_generation("generate_stream", self._model, messages) as generation:
                raw_chunks: list[Any] = []
                text_parts: list[str] = []
                for chunk in litellm.completion(
                    model=self._model,
                    messages=messages,
                    base_url=self._base_url,
                    api_key=self._api_key,
                    stream=True,
                ):
                    if generation is not None:
                        raw_chunks.append(chunk)
                    text = chunk.choices[0].delta.content
                    if text:
                        text_parts.append(text)
                        yield text
                if generation is not None:
                    record_generation_result(
                        generation,
                        self._rebuild_full_response(raw_chunks, messages),
                        "".join(text_parts),
                    )
        finally:
            if owned_loop is not None:
                owned_loop.close()
                asyncio.set_event_loop(None)

    def generate_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Iterator[TokenChunk | ToolCallsChunk]:
        owned_loop = _ensure_current_event_loop()
        try:
            with traced_generation(
                "generate_with_tools_stream", self._model, messages
            ) as generation:
                kwargs = dict(
                    model=self._model,
                    messages=messages,
                    tools=tools,
                    base_url=self._base_url,
                    api_key=self._api_key,
                    stream=True,
                )
                if tool_choice is not None:
                    kwargs["tool_choice"] = tool_choice

                raw_chunks: list[Any] = []
                text_parts: list[str] = []
                # OpenAI-style streaming fragments a tool call's arguments
                # across many chunks, keyed by its position in the
                # response -- id/name only arrive on that call's first
                # fragment, so each is recorded once and never overwritten
                # by a later, empty fragment.
                pending_calls: dict[int, dict[str, str]] = {}
                for chunk in litellm.completion(**kwargs):
                    if generation is not None:
                        raw_chunks.append(chunk)
                    delta = chunk.choices[0].delta
                    if delta.content:
                        text_parts.append(delta.content)
                        yield TokenChunk(text=delta.content)
                    for call_delta in delta.tool_calls or []:
                        entry = pending_calls.setdefault(
                            call_delta.index, {"id": "", "name": "", "arguments": ""}
                        )
                        if call_delta.id:
                            entry["id"] = call_delta.id
                        if call_delta.function and call_delta.function.name:
                            entry["name"] = call_delta.function.name
                        if call_delta.function and call_delta.function.arguments:
                            entry["arguments"] += call_delta.function.arguments

                tool_calls = [
                    ToolCall(
                        id=entry["id"],
                        name=entry["name"],
                        arguments=json.loads(entry["arguments"]) if entry["arguments"] else {},
                    )
                    for _, entry in sorted(pending_calls.items())
                ]
                if tool_calls:
                    yield ToolCallsChunk(tool_calls=tool_calls)

                if generation is not None:
                    result = ToolResponse(content="".join(text_parts) or None, tool_calls=tool_calls)
                    record_generation_result(
                        generation,
                        self._rebuild_full_response(raw_chunks, messages),
                        result.model_dump(mode="json"),
                    )
        finally:
            if owned_loop is not None:
                owned_loop.close()
                asyncio.set_event_loop(None)

    @staticmethod
    def _rebuild_full_response(raw_chunks: list[Any], messages: list[dict[str, Any]]) -> Any:
        """Reassembles the full response object (incl. usage) from raw
        stream chunks, purely for Langfuse's usage/cost logging --
        record_generation_result() already tolerates None (no usage
        attached), so a malformed/incomplete chunk list degrades tracing
        quality only, never breaks the request that already streamed
        successfully to its caller."""
        try:
            return litellm.stream_chunk_builder(raw_chunks, messages=messages)
        except Exception:
            return None

    @staticmethod
    def _normalize_model(model: str, base_url: str | None) -> str:
        # For OpenAI-compatible gateways, LiteLLM expects an explicit
        # provider prefix for bare model names. Normalizing once up front
        # avoids the noisy fail-then-retry path in every request.
        if base_url is not None and "/" not in model:
            return f"openai/{model}"
        return model


class InternalServerLLM(LLMProvider):
    """Stub for a company-internal endpoint that does NOT speak the
    OpenAI-compatible schema. LiteLLM can't paper over a bespoke API, so
    this is the explicit escape hatch: implement `generate` against the
    real internal request/response contract once it's known, without
    touching anything outside this class.
    """

    def __init__(self, base_url: str | None, api_key: str | None = None) -> None:
        if base_url is None:
            raise ValueError("InternalServerLLM requires base_url to be set")
        self._base_url = base_url
        self._api_key = api_key

    def generate(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError(
            "InternalServerLLM is a stub. Implement the request/response mapping "
            "for the internal endpoint's actual (non-OpenAI-compatible) API contract."
        )
