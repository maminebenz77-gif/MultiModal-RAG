import httpx


def test_optional_reranker_network_failure_does_not_block_startup(monkeypatch) -> None:
    from multimodal_rag.api import main

    def _raise_network_error(_settings):
        raise httpx.RemoteProtocolError("model server disconnected")

    monkeypatch.setattr(main, "get_reranker", _raise_network_error)

    assert main._load_optional_reranker(object()) is None

async def test_health_reports_ok_when_the_store_is_reachable(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "elasticsearch": "up"}


async def test_health_reports_degraded_when_the_store_is_unreachable(
    client: httpx.AsyncClient,
) -> None:
    # vector_store and keyword_store are two role-views onto the SAME
    # Elasticsearch backend now (see stores/elasticsearch_store.py) --
    # /health only pings one of them (see routers/health.py), so either
    # attribute would do; keyword_store matches what the route actually
    # depends on.
    client.app.state.app_state.keyword_store.ping = lambda: False  # type: ignore[attr-defined]

    response = await client.get("/health")

    body = response.json()
    assert body == {"status": "degraded", "elasticsearch": "down"}
