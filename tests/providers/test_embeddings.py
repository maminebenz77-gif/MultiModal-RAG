import pytest

from multimodal_rag.providers.embeddings import (
    APIEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)


def test_sentence_transformer_provider_wires_model_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeVectors:
        def tolist(self) -> list[list[float]]:
            return [[0.1, 0.2], [0.3, 0.4]]

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str) -> None:
            captured["model_name"] = model_name
            captured["device"] = device

        def encode(self, texts: list[str], convert_to_numpy: bool) -> FakeVectors:
            captured["texts"] = texts
            return FakeVectors()

    monkeypatch.setattr(
        "multimodal_rag.providers.embeddings.SentenceTransformer", FakeSentenceTransformer
    )

    provider = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2", device="cpu")
    result = provider.embed(["a", "b"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["model_name"] == "all-MiniLM-L6-v2"
    assert captured["device"] == "cpu"
    assert captured["texts"] == ["a", "b"]


def test_api_embedding_provider_embed_is_unimplemented() -> None:
    provider = APIEmbeddingProvider(model="text-embed-3", base_url="https://api.example.com")
    with pytest.raises(NotImplementedError):
        provider.embed(["a"])
