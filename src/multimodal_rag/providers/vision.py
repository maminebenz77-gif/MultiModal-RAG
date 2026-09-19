"""Concrete VisionProvider implementations."""

import asyncio
import base64

import litellm
import magic

from ..image_utils import downscale_image
from ..tracing import record_generation_result, traced_generation
from .base import VisionProvider

_DEFAULT_PROMPT = (
    "Describe this image factually and concisely, in 1-3 sentences, for "
    "someone who cannot see it. If it's a chart or diagram, describe its "
    "type and what it shows."
)

_DEFAULT_MAX_DIMENSION = 1024


def _ensure_current_event_loop() -> asyncio.AbstractEventLoop | None:
    try:
        asyncio.get_running_loop()
        return None
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


class LiteLLMVisionProvider(VisionProvider):
    """Covers any OpenAI-compatible *multimodal* chat backend — a vision
    call is the same LiteLLM completion as LiteLLMProvider, just with an
    image_url content block alongside the text. No new provider
    architecture needed, same reason LiteLLMProvider covers many LLM
    backends: the wire schema is shared.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_dimension: int = _DEFAULT_MAX_DIMENSION,
    ) -> None:
        litellm.telemetry = False
        self._model = self._normalize_model(model, base_url)
        self._base_url = base_url
        self._api_key = api_key
        self._max_dimension = max_dimension

    @staticmethod
    def _normalize_model(model: str, base_url: str | None) -> str:
        if base_url is not None and "/" not in model:
            return f"openai/{model}"
        return model

    def describe(self, image_bytes: bytes, prompt: str | None = None) -> str:
        owned_loop = _ensure_current_event_loop()
        try:
            image_bytes = downscale_image(image_bytes, self._max_dimension)
            mime_type = magic.from_buffer(image_bytes, mime=True)
            encoded = base64.b64encode(image_bytes).decode("ascii")
            prompt_text = prompt or _DEFAULT_PROMPT
            # Traced input is a placeholder for the image, not the real
            # base64 payload -- same reasoning retrieval tracing already
            # applies to SearchResult.elements (never traced whole): a
            # trace should stay legible in the UI, not become a payload
            # dump of image bytes.
            traced_messages = [
                {
                    "role": "user",
                    "content": f"{prompt_text}\n[image: {len(image_bytes)} bytes, {mime_type}]",
                }
            ]
            with traced_generation("describe", self._model, traced_messages) as generation:
                response = litellm.completion(
                    model=self._model,
                    base_url=self._base_url,
                    api_key=self._api_key,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                                },
                            ],
                        }
                    ],
                )
                content = response.choices[0].message.content or ""
                record_generation_result(generation, response, content)
        finally:
            if owned_loop is not None:
                owned_loop.close()
                asyncio.set_event_loop(None)
        return content


class InternalServerVisionProvider(VisionProvider):
    """Stub for a company-internal vision endpoint that does NOT speak the
    OpenAI-compatible multimodal schema. Same escape hatch as
    InternalServerLLM: implement `describe` against the real internal
    request/response contract once it's known.
    """

    def __init__(self, base_url: str | None, api_key: str | None = None) -> None:
        if base_url is None:
            raise ValueError("InternalServerVisionProvider requires base_url to be set")
        self._base_url = base_url
        self._api_key = api_key

    def describe(self, image_bytes: bytes, prompt: str | None = None) -> str:
        raise NotImplementedError(
            "InternalServerVisionProvider is a stub. Implement the request/response "
            "mapping for the internal endpoint's actual (non-OpenAI-compatible) API contract."
        )
