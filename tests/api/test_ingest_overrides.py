import json

import httpx
import pytest

from multimodal_rag.providers.base import EmbeddingProvider
from multimodal_rag.providers.schema import EmbeddingVector

from .conftest import SAMPLE_DOC


class _FakeEmbedder(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        return [EmbeddingVector(vector=[0.1, 0.2], model_id="fake", dimension=2) for _ in texts]


async def test_ingest_rejects_invalid_runtime_overrides_json(client: httpx.AsyncClient) -> None:
    with open(SAMPLE_DOC, "rb") as f:
        response = await client.post(
            "/ingest",
            files={
                "file": ("chunking_demo.md", f, "text/markdown"),
                "runtime_overrides_json": (None, "{bad json"),
            },
        )
    assert response.status_code == 400
    assert "Invalid runtime overrides" in response.json()["detail"]


async def test_ingest_uses_embedder_runtime_override(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "multimodal_rag.api.routers.ingest.embedder_from_override",
        lambda **_: _FakeEmbedder(),
    )

    with open(SAMPLE_DOC, "rb") as f:
        response = await client.post(
            "/ingest",
            files={
                "file": ("chunking_demo.md", f, "text/markdown"),
                "runtime_overrides_json": (
                    None,
                    json.dumps(
                        {
                            "embedder": {
                                "provider": "litellm",
                                "model": "text-embedding-3-small",
                                "base_url": "http://localhost:1234",
                                "api_key": "k",
                            }
                        }
                    ),
                ),
            },
        )

    assert response.status_code == 422
    assert "incompatible with the existing corpus" in response.json()["detail"]