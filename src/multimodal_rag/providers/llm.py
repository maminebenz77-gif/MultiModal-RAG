"""Concrete LLMProvider implementations."""

import litellm

from .base import LLMProvider


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
        self._model = model
        self._base_url = base_url
        self._api_key = api_key

    def generate(self, messages: list[dict[str, str]]) -> str:
        response = litellm.completion(
            model=self._model,
            messages=messages,
            base_url=self._base_url,
            api_key=self._api_key,
        )
        content = response.choices[0].message.content
        return content or ""


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
