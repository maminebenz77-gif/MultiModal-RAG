import pytest

from multimodal_rag.providers.rerank import CrossEncoderReranker


def test_rerank_orders_documents_by_descending_score(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeCrossEncoder:
        def __init__(self, model_name: str, device: str) -> None:
            captured["model_name"] = model_name
            captured["device"] = device

        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            captured["pairs"] = pairs
            # doc at index 1 is most relevant, then 2, then 0
            return [0.1, 0.9, 0.5]

    monkeypatch.setattr("multimodal_rag.providers.rerank.CrossEncoder", FakeCrossEncoder)

    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    order = reranker.rerank("a query", ["doc a", "doc b", "doc c"])

    assert order == [1, 2, 0]
    assert captured["model_name"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert captured["device"] == "cpu"
    assert captured["pairs"] == [("a query", "doc a"), ("a query", "doc b"), ("a query", "doc c")]
