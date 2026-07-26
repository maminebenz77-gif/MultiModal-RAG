"""Concrete Reranker implementations.

A cross-encoder scores a query and ONE candidate document together, in a
single transformer forward pass, letting attention directly compare
query and document tokens — meaningfully more accurate than comparing
independently-computed embeddings, but infeasible to run over an entire
corpus (one forward pass per candidate). It's applied to a small,
already-retrieved candidate set, not used for retrieval itself.
"""

from sentence_transformers import CrossEncoder

from .base import Reranker


class CrossEncoderReranker(Reranker):
    """Local, device-aware cross-encoder — same reasoning as
    SentenceTransformerEmbeddingProvider: no network call once the model
    is cached, safe on the air-gapped server profile.
    """

    def __init__(self, model_name: str, device: str) -> None:
        self._model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, documents: list[str]) -> list[int]:
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)
        return sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
