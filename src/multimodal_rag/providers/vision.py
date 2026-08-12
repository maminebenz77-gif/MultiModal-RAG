"""Concrete VisionProvider implementations."""

import base64
import io

import litellm
import magic
from PIL import Image

from .base import VisionProvider

_DEFAULT_PROMPT = (
    "Describe this image factually and concisely, in 1-3 sentences, for "
    "someone who cannot see it. If it's a chart or diagram, describe its "
    "type and what it shows."
)

_DEFAULT_MAX_DIMENSION = 1024


def _downscale_if_needed(image_bytes: bytes, max_dimension: int) -> bytes:
    """Cap the image's longest edge at max_dimension before it gets
    base64-inlined into a request. Vision API cost scales with image
    resolution, not with how large the source file happens to be, so a
    4000px screenshot pays for detail a caption doesn't need.

    Returns the original bytes unchanged if the image is already small
    enough, or if it can't be decoded (fails closed to "send as-is"
    rather than breaking the whole describe() call over a resize step
    that was only ever meant to save money).
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception:
        return image_bytes

    if max(image.width, image.height) <= max_dimension:
        return image_bytes

    scale = max_dimension / max(image.width, image.height)
    new_size = (round(image.width * scale), round(image.height * scale))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    resized.save(buffer, format=image.format or "PNG")
    return buffer.getvalue()


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
        image_bytes = _downscale_if_needed(image_bytes, self._max_dimension)
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
