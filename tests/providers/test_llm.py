import litellm
import pytest

from multimodal_rag.providers.llm import InternalServerLLM, LiteLLMProvider
from multimodal_rag.providers.schema import TokenChunk, ToolCallsChunk


class _FakeFunctionDelta:
    def __init__(self, name: str | None = None, arguments: str | None = None) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCallDelta:
    def __init__(
        self,
        index: int,
        call_id: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
    ) -> None:
        self.index = index
        self.id = call_id
        self.function = _FakeFunctionDelta(name, arguments)


class _FakeDelta:
    def __init__(
        self, content: str | None = None, tool_calls: list[_FakeToolCallDelta] | None = None
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _FakeStreamChunk:
    def __init__(self, delta: _FakeDelta) -> None:
        self.choices = [type("Choice", (), {"delta": delta})()]


def test_litellm_provider_disables_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "telemetry", True)
    LiteLLMProvider(model="gpt-4o-mini")
    assert litellm.telemetry is False


def test_litellm_provider_extracts_text_from_response(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs = {}

    class FakeMessage:
        content = "hello from the model"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    def fake_completion(**kwargs: object) -> FakeResponse:
        captured_kwargs.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("multimodal_rag.providers.llm.litellm.completion", fake_completion)

    provider = LiteLLMProvider(model="gpt-4o-mini", base_url="http://localhost:11434", api_key="k")
    result = provider.generate([{"role": "user", "content": "hi"}])

    assert result == "hello from the model"
    assert captured_kwargs["model"] == "openai/gpt-4o-mini"
    assert captured_kwargs["base_url"] == "http://localhost:11434"


def test_litellm_provider_normalizes_model_with_openai_prefix_for_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs = {}

    class FakeMessage:
        content = "hello"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    def fake_completion(**kwargs: object):
        captured_kwargs.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("multimodal_rag.providers.llm.litellm.completion", fake_completion)

    provider = LiteLLMProvider(model="gemma4", base_url="http://localhost:11434")
    result = provider.generate([{"role": "user", "content": "hi"}])

    assert result == "hello"
    assert captured_kwargs["model"] == "openai/gemma4"


def test_litellm_provider_forwards_required_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs = {}

    class FakeMessage:
        content = None
        tool_calls = []

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    def fake_completion(**kwargs: object) -> FakeResponse:
        captured_kwargs.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("multimodal_rag.providers.llm.litellm.completion", fake_completion)

    provider = LiteLLMProvider(model="gpt-4o-mini")
    provider.generate_with_tools([], [], tool_choice="required")

    assert captured_kwargs["tool_choice"] == "required"


def test_internal_server_llm_requires_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        InternalServerLLM(base_url=None)


def test_internal_server_llm_generate_is_unimplemented() -> None:
    provider = InternalServerLLM(base_url="http://10.0.0.5:8080")
    with pytest.raises(NotImplementedError):
        provider.generate([{"role": "user", "content": "hi"}])


def test_generate_stream_yields_fragments_in_arrival_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        _FakeStreamChunk(_FakeDelta(content="Hel")),
        _FakeStreamChunk(_FakeDelta(content="lo")),
        _FakeStreamChunk(_FakeDelta(content=None)),
    ]
    captured_kwargs = {}

    def fake_completion(**kwargs: object):
        captured_kwargs.update(kwargs)
        return iter(chunks)

    monkeypatch.setattr("multimodal_rag.providers.llm.litellm.completion", fake_completion)

    provider = LiteLLMProvider(model="gpt-4o-mini")
    fragments = list(provider.generate_stream([{"role": "user", "content": "hi"}]))

    assert fragments == ["Hel", "lo"]
    assert captured_kwargs["stream"] is True


def test_generate_with_tools_stream_yields_tokens_then_no_tool_calls_chunk_when_none_made(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        _FakeStreamChunk(_FakeDelta(content="Hi ")),
        _FakeStreamChunk(_FakeDelta(content="there!")),
    ]
    monkeypatch.setattr(
        "multimodal_rag.providers.llm.litellm.completion", lambda **kwargs: iter(chunks)
    )

    provider = LiteLLMProvider(model="gpt-4o-mini")
    events = list(provider.generate_with_tools_stream([{"role": "user", "content": "hi"}], []))

    assert events == [TokenChunk(text="Hi "), TokenChunk(text="there!")]


def test_generate_with_tools_stream_reassembles_fragmented_tool_call_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Real OpenAI-compatible streaming splits one tool call's JSON
    # arguments across many chunks -- id/name only arrive on the first
    # fragment for that call's index, arguments accumulate across all of
    # them. This is the reassembly generate_with_tools() gets for free
    # from a non-streaming response; the streaming path has to do it by
    # hand, so it's worth pinning down directly.
    chunks = [
        _FakeStreamChunk(
            _FakeDelta(
                tool_calls=[
                    _FakeToolCallDelta(
                        index=0, call_id="call_1", name="search_knowledge_base", arguments='{"que'
                    )
                ]
            )
        ),
        _FakeStreamChunk(_FakeDelta(tool_calls=[_FakeToolCallDelta(index=0, arguments='ry": "l')])),
        _FakeStreamChunk(
            _FakeDelta(tool_calls=[_FakeToolCallDelta(index=0, arguments='atency"}')])
        ),
    ]
    monkeypatch.setattr(
        "multimodal_rag.providers.llm.litellm.completion", lambda **kwargs: iter(chunks)
    )

    provider = LiteLLMProvider(model="gpt-4o-mini")
    events = list(provider.generate_with_tools_stream([{"role": "user", "content": "hi"}], []))

    assert events == [ToolCallsChunk(tool_calls=events[0].tool_calls)]
    call = events[0].tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "search_knowledge_base"
    assert call.arguments == {"query": "latency"}


def test_generate_with_tools_stream_forwards_tool_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs = {}

    def fake_completion(**kwargs: object):
        captured_kwargs.update(kwargs)
        return iter([])

    monkeypatch.setattr("multimodal_rag.providers.llm.litellm.completion", fake_completion)

    provider = LiteLLMProvider(model="gpt-4o-mini")
    list(
        provider.generate_with_tools_stream(
            [{"role": "user", "content": "hi"}], [], tool_choice="required"
        )
    )

    assert captured_kwargs["tool_choice"] == "required"
