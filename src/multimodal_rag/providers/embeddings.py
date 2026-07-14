"""Concrete EmbeddingProvider implementations."""

from typing import cast

from sentence_transformers import SentenceTransformer

from .base import EmbeddingProvider


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local, device-aware embedding model — no network call once the
    model is cached, which is what makes it safe to use on the air-gapped
    server profile.
    """

    def __init__(self, model_name: str, device: str) -> None:
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return cast(list[list[float]], vectors.tolist())


class APIEmbeddingProvider(EmbeddingProvider):
    """Stub for a remote embedding API. Implement once we pick a concrete
    API-based embedding backend to support alongside the local model.
    """

    def __init__(self, model: str, base_url: str | None, api_key: str | None = None) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = api_key

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "APIEmbeddingProvider is a stub. Implement against the chosen embedding API."
        )
