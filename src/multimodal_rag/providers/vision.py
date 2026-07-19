"""Concrete VisionProvider implementations."""

import base64

import litellm
import magic

from .base import VisionProvider

_DEFAULT_PROMPT = (
    "Describe this image factually and concisely, in 1-3 sentences, for "
    "someone who cannot see it. If it's a chart or diagram, describe its "
    "type and what it shows."
)


class LiteLLMVisionProvider(VisionProvider):
    """Covers any OpenAI-compatible *multimodal* chat backend — a vision
    call is the same LiteLLM completion as LiteLLMProvider, just with an
    image_url content block alongside the text. No new provider
    architecture needed, same reason LiteLLMProvider covers many LLM
    backends: the wire schema is shared.
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None) -> None:
        litellm.telemetry = False
        self._model = model
        self._base_url = base_url
        self._api_key = api_key

    def describe(self, image_bytes: bytes, prompt: str | None = None) -> str:
        mime_type = magic.from_buffer(image_bytes, mime=True)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        response = litellm.completion(
            model=self._model,
            base_url=self._base_url,
            api_key=self._api_key,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or _DEFAULT_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                        },
                    ],
                }
            ],
        )
        content = response.choices[0].message.content
        return content or ""


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
