"""Concrete EmbeddingProvider implementations."""

import litellm
from sentence_transformers import SentenceTransformer

from .base import EmbeddingProvider
from .schema import EmbeddingVector

_DEFAULT_BATCH_SIZE = 64


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local, device-aware embedding model — no network call once the
    model is cached, which is what makes it safe to use on the air-gapped
    server profile.
    """

    def __init__(self, model_name: str, device: str, batch_size: int = _DEFAULT_BATCH_SIZE) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        # sentence-transformers batches internally given batch_size; no
        # manual chunking needed here, unlike the API-backed provider.
        vectors = self._model.encode(
            texts, batch_size=self._batch_size, convert_to_numpy=True
        ).tolist()
        return [
            EmbeddingVector(vector=v, model_id=self._model_name, dimension=len(v)) for v in vectors
        ]


class LiteLLMEmbeddingProvider(EmbeddingProvider):
    """Covers any OpenAI-compatible embedding backend — same LiteLLM
    pattern as LiteLLMProvider/LiteLLMVisionProvider, this time via
    litellm.embedding() instead of litellm.completion().

    Manually chunks the input into batches: unlike sentence-transformers,
    LiteLLM doesn't batch for us, and a real embedding API has a practical
    per-request input-count limit.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        litellm.telemetry = False
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        vectors: list[EmbeddingVector] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = litellm.embedding(
                model=self._model, input=batch, base_url=self._base_url, api_key=self._api_key
            )
            vectors.extend(
                EmbeddingVector(
                    vector=item["embedding"], model_id=self._model, dimension=len(item["embedding"])
                )
                for item in response.data
            )
        return vectors
