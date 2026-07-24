import pytest

from multimodal_rag.providers.embeddings import (
    LiteLLMEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)


class FakeVectors:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors

    def tolist(self) -> list[list[float]]:
        return self._vectors


def test_sentence_transformer_provider_wires_model_device_and_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str) -> None:
            captured["model_name"] = model_name
            captured["device"] = device

        def encode(self, texts: list[str], batch_size: int, convert_to_numpy: bool) -> FakeVectors:
            captured["texts"] = texts
            captured["batch_size"] = batch_size
            return FakeVectors([[0.1, 0.2], [0.3, 0.4]])

    monkeypatch.setattr(
        "multimodal_rag.providers.embeddings.SentenceTransformer", FakeSentenceTransformer
    )

    provider = SentenceTransformerEmbeddingProvider(
        model_name="all-MiniLM-L6-v2", device="cpu", batch_size=16
    )
    result = provider.embed(["a", "b"])

    assert captured["model_name"] == "all-MiniLM-L6-v2"
    assert captured["device"] == "cpu"
    assert captured["batch_size"] == 16
    assert captured["texts"] == ["a", "b"]

    assert [v.vector for v in result] == [[0.1, 0.2], [0.3, 0.4]]
    assert all(v.model_id == "all-MiniLM-L6-v2" for v in result)
    assert [v.dimension for v in result] == [2, 2]


class FakeEmbeddingResponse:
    def __init__(self, dimension: int, count: int) -> None:
        self.data = [{"embedding": [float(i)] * dimension, "index": i} for i in range(count)]


def test_litellm_embedding_provider_wires_model_and_returns_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_embedding(**kwargs: object) -> FakeEmbeddingResponse:
        captured_kwargs.update(kwargs)
        texts = kwargs["input"]
        assert isinstance(texts, list)
        return FakeEmbeddingResponse(dimension=3, count=len(texts))

    monkeypatch.setattr("multimodal_rag.providers.embeddings.litellm.embedding", fake_embedding)

    provider = LiteLLMEmbeddingProvider(model="text-embedding-3-small", base_url="https://api.openai.com/v1")
    result = provider.embed(["a", "b"])

    assert captured_kwargs["model"] == "text-embedding-3-small"
    assert len(result) == 2
    assert all(v.model_id == "text-embedding-3-small" for v in result)
    assert all(v.dimension == 3 for v in result)


def test_litellm_embedding_provider_chunks_into_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    call_sizes: list[int] = []

    def fake_embedding(**kwargs: object) -> FakeEmbeddingResponse:
        texts = kwargs["input"]
        assert isinstance(texts, list)
        call_sizes.append(len(texts))
        return FakeEmbeddingResponse(dimension=2, count=len(texts))

    monkeypatch.setattr("multimodal_rag.providers.embeddings.litellm.embedding", fake_embedding)

    provider = LiteLLMEmbeddingProvider(model="text-embedding-3-small", batch_size=3)
    texts = [f"text {i}" for i in range(7)]
    result = provider.embed(texts)

    assert call_sizes == [3, 3, 1]
    assert len(result) == 7
