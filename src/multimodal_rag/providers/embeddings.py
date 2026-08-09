"""Concrete EmbeddingProvider implementations."""

import logging

import litellm
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from ..retry import retry_with_backoff
from .base import EmbeddingProvider
from .schema import EmbeddingVector

_DEFAULT_BATCH_SIZE = 64
_logger = logging.getLogger(__name__)


class EmbeddingBatchError(RuntimeError):
    """Raised when one or more batches failed to embed even after
    retries. Carries the vectors that DID succeed and the texts that
    didn't, so a caller can use the partial results or retry just the
    failures later, rather than losing already-completed (and already
    paid-for) work.
    """

    def __init__(
        self, message: str, succeeded: list[EmbeddingVector], failed_texts: list[str]
    ) -> None:
        super().__init__(message)
        self.succeeded = succeeded
        self.failed_texts = failed_texts


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

    Each batch is retried (exponential backoff) before being considered
    failed, since most real failures here are transient (rate limits,
    network blips). If a batch still fails after retries, embedding keeps
    going with the remaining batches rather than aborting — a single bad
    batch shouldn't discard every other batch's already-completed,
    already-paid-for work.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        litellm.telemetry = False
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds

        # litellm builds its own internal OpenAI client for embedding
        # calls, which on the corporate-network profile doesn't reliably
        # pick up the truststore SSL patch load_settings() applies when
        # trust_system_certs=True (see config.py). Building our own
        # client here and handing it to litellm explicitly sidesteps
        # that, and is a no-op change everywhere trust_system_certs is
        # off -- a plain OpenAI client either way.
        #
        # The openai package's own client (unlike litellm's more lenient
        # internal one) refuses to construct at all without a non-empty
        # api_key, even for backends that don't check auth (e.g. an
        # internal, unauthenticated OpenAI-compatible server) -- a
        # placeholder satisfies that without changing what's actually
        # sent when a real key is configured.
        self._openai_client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")

    def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        vectors: list[EmbeddingVector] = []
        failed_texts: list[str] = []

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                vectors.extend(self._embed_batch(batch))
            except Exception:
                _logger.warning("Embedding batch failed after retries", exc_info=True)
                failed_texts.extend(batch)

        if failed_texts:
            raise EmbeddingBatchError(
                f"{len(failed_texts)} of {len(texts)} texts failed to embed after "
                f"{self._max_retries} attempts per batch; {len(vectors)} succeeded.",
                succeeded=vectors,
                failed_texts=failed_texts,
            )
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[EmbeddingVector]:
        def call() -> list[EmbeddingVector]:
            response = litellm.embedding(
                model=self._model,
                input=batch,
                client=self._openai_client,
            )
            return [
                EmbeddingVector(
                    vector=item["embedding"], model_id=self._model, dimension=len(item["embedding"])
                )
                for item in response.data
            ]

        return retry_with_backoff(call, self._max_retries, self._retry_backoff_seconds)