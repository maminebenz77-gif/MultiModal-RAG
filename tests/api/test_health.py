import httpx


async def test_health_reports_ok_when_both_stores_are_reachable(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "qdrant": "up", "elasticsearch": "up"}


async def test_health_reports_degraded_when_a_store_is_unreachable(
    client: httpx.AsyncClient,
) -> None:
    client.app.state.app_state.vector_store.ping = lambda: False  # type: ignore[attr-defined]

    response = await client.get("/health")

    body = response.json()
    assert body["status"] == "degraded"
    assert body["qdrant"] == "down"
    assert body["elasticsearch"] == "up"
