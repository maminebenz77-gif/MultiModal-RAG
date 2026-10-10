"""End-to-end proof that the docgen graph can be driven entirely over
HTTP -- the same graph docgen/cli.py drives from a terminal, started
and resumed instead via POST /docgen/runs and
POST /docgen/runs/{thread_id}/resume, with status polled via GET
/docgen/runs/{thread_id} rather than printed to a terminal.

LLM faked via monkeypatch.setattr("multimodal_rag.docgen.graph.get_llm", ...)
-- the same pattern test_query.py/test_query_scoping.py already use for
AgentChain -- since build_graph() always resolves get_llm() from its
OWN module namespace at call time (inside the background thread this
router starts), patching that name before a request is issued is
enough; the test's own poll loop blocks until the background thread
finishes before the test function returns, so the patch is still in
effect for the whole window it's needed.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from multimodal_rag.api.identity import get_principal
from multimodal_rag.api.routers import docgen as docgen_router
from multimodal_rag.identity import Principal
from multimodal_rag.providers.base import LLMProvider

from .conftest import SAMPLE_DOC


class _ScriptedLLM(LLMProvider):
    def __init__(self, replies: list[str]) -> None:
        self._replies = iter(replies)

    def generate(self, messages: list[dict[str, str]]) -> str:
        try:
            return next(self._replies)
        except StopIteration:
            raise AssertionError("LLM was called more times than the test scripted") from None


class _FailingLLM(LLMProvider):
    def generate(self, messages: list[dict[str, str]]) -> str:
        raise RuntimeError("simulated LLM outage")


def _use_llm(monkeypatch: pytest.MonkeyPatch, llm: LLMProvider) -> None:
    monkeypatch.setattr("multimodal_rag.docgen.graph.get_llm", lambda: llm)


def _as(client: httpx.AsyncClient, principal: Principal) -> None:
    client.app.dependency_overrides[get_principal] = lambda: principal  # type: ignore[attr-defined]


async def _ingest_tagged_doc(client: httpx.AsyncClient, tag: str) -> None:
    metadata_json = json.dumps({"classification": "public", "tags": [tag]})
    with open(SAMPLE_DOC, "rb") as f:
        response = await client.post(
            "/ingest",
            files={"file": ("chunking_demo.md", f, "text/markdown")},
            data={"metadata_json": metadata_json},
        )
    assert response.status_code == 200, response.text


async def _poll_until_not_running(
    client: httpx.AsyncClient, thread_id: str, timeout: float = 10.0
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = await client.get(f"/docgen/runs/{thread_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] != "running":
            return body
        await asyncio.sleep(0.05)
    raise AssertionError(f"run {thread_id!r} never left 'running' within {timeout}s")


_INTERPRETATION = (
    '{"questions": [{"text": "What was the hosted API average latency?", '
    '"sources_required": ["task_docs"]}], "format": "pptx", "template": ""}'
)
_FULL_SCRIPT = [
    _INTERPRETATION,  # interpret_request
    "hosted api latency",  # formulate_query
    "120ms on average.",  # generate_answer
    '{"valid": true, "reason": "fully grounded"}',  # validate_answer
    '{"q1": "The hosted API averaged 120ms of latency."}',  # harmonize_answers
]


async def test_full_run_confirm_to_approve_to_download(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_llm(monkeypatch, _ScriptedLLM(list(_FULL_SCRIPT)))
    await _ingest_tagged_doc(client, "docgen:task:demo")

    create = await client.post(
        "/docgen/runs",
        json={
            "request_text": "What was the hosted API's average latency?",
            "sources": [{"role": "task_docs", "tag": "docgen:task:demo", "required": True}],
        },
    )
    assert create.status_code == 200, create.text
    thread_id = create.json()["thread_id"]

    paused = await _poll_until_not_running(client, thread_id)
    assert paused["status"] == "paused"
    assert paused["pending"]["kind"] == "confirm_configuration"

    resumed = await client.post(
        f"/docgen/runs/{thread_id}/resume",
        json={"action": "confirm", "text": "", "question_ids": []},
    )
    assert resumed.status_code == 200, resumed.text

    paused_for_review = await _poll_until_not_running(client, thread_id)
    assert paused_for_review["status"] == "paused"
    assert paused_for_review["pending"]["kind"] == "human_review"
    assert paused_for_review["questions"][0]["id"] == "q1"
    # Already harmonized by the time human_review pauses -- harmonize_answers
    # runs right before it.
    assert paused_for_review["answers"]["q1"]["text"] == "The hosted API averaged 120ms of latency."

    approved = await client.post(
        f"/docgen/runs/{thread_id}/resume",
        json={"action": "approve", "text": "", "question_ids": []},
    )
    assert approved.status_code == 200, approved.text

    done = await _poll_until_not_running(client, thread_id)
    assert done["status"] == "done"
    assert done["answers"]["q1"]["text"] == "The hosted API averaged 120ms of latency."
    assert done["output_path"]

    download = await client.get(f"/docgen/runs/{thread_id}/download")
    assert download.status_code == 200
    assert download.headers["content-type"] == (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )
    assert len(download.content) > 0


async def test_get_run_404s_for_an_unknown_thread_id(client: httpx.AsyncClient) -> None:
    response = await client.get("/docgen/runs/does-not-exist")
    assert response.status_code == 404


async def test_resume_404s_for_an_unknown_thread_id(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/docgen/runs/does-not-exist/resume",
        json={"action": "confirm", "text": "", "question_ids": []},
    )
    assert response.status_code == 404


async def test_download_404s_for_an_unknown_thread_id(client: httpx.AsyncClient) -> None:
    response = await client.get("/docgen/runs/does-not-exist/download")
    assert response.status_code == 404


async def test_another_principals_run_is_invisible_not_just_forbidden(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same convention as api/db.py's DocumentNotFoundError: "can't see
    it" and "doesn't exist" are deliberately the same 404, never a 403
    that would confirm someone else's run exists."""
    _use_llm(monkeypatch, _ScriptedLLM(list(_FULL_SCRIPT)))
    await _ingest_tagged_doc(client, "docgen:task:demo2")

    _as(client, Principal(principal_id="user:alice", clearance="c1"))
    create = await client.post(
        "/docgen/runs",
        json={
            "request_text": "What was the hosted API's average latency?",
            "sources": [{"role": "task_docs", "tag": "docgen:task:demo2", "required": True}],
        },
    )
    assert create.status_code == 200, create.text
    thread_id = create.json()["thread_id"]
    await _poll_until_not_running(client, thread_id)

    _as(client, Principal(principal_id="user:bob", clearance="c1"))
    assert (await client.get(f"/docgen/runs/{thread_id}")).status_code == 404
    assert (await client.get("/docgen/runs")).json()["runs"] == []
    resume = await client.post(
        f"/docgen/runs/{thread_id}/resume",
        json={"action": "confirm", "text": "", "question_ids": []},
    )
    assert resume.status_code == 404


async def test_resume_rejects_an_action_that_does_not_match_the_pending_kind(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_llm(monkeypatch, _ScriptedLLM(list(_FULL_SCRIPT)))
    await _ingest_tagged_doc(client, "docgen:task:demo3")

    create = await client.post(
        "/docgen/runs",
        json={
            "request_text": "What was the hosted API's average latency?",
            "sources": [{"role": "task_docs", "tag": "docgen:task:demo3", "required": True}],
        },
    )
    thread_id = create.json()["thread_id"]
    paused = await _poll_until_not_running(client, thread_id)
    assert paused["pending"]["kind"] == "confirm_configuration"

    # "approve" is only valid for a human_review pause, not
    # confirm_configuration -- the router must reject the mismatch, not
    # silently forward it into a ConfirmationResponse-shaped resume.
    mismatched = await client.post(
        f"/docgen/runs/{thread_id}/resume",
        json={"action": "approve", "text": "", "question_ids": []},
    )
    assert mismatched.status_code == 400


async def test_resume_409s_when_nothing_is_pending(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_llm(monkeypatch, _ScriptedLLM(list(_FULL_SCRIPT)))
    await _ingest_tagged_doc(client, "docgen:task:demo4")

    create = await client.post(
        "/docgen/runs",
        json={
            "request_text": "What was the hosted API's average latency?",
            "sources": [{"role": "task_docs", "tag": "docgen:task:demo4", "required": True}],
        },
    )
    thread_id = create.json()["thread_id"]
    await _poll_until_not_running(client, thread_id)  # paused at confirm_configuration

    # Immediately resume twice in a row without waiting: the second
    # call either lands on a thread_id with no pending interrupt yet
    # (409 "no pending interrupt") or on one already being resumed
    # (409 "already being resumed") -- both are the same guarantee,
    # that two invoke()s never run concurrently on one thread_id.
    first, second = await asyncio.gather(
        client.post(
            f"/docgen/runs/{thread_id}/resume",
            json={"action": "confirm", "text": "", "question_ids": []},
        ),
        client.post(
            f"/docgen/runs/{thread_id}/resume",
            json={"action": "confirm", "text": "", "question_ids": []},
        ),
    )
    statuses = sorted([first.status_code, second.status_code])
    assert statuses == [200, 409]


def test_lock_registry_rejects_a_second_concurrent_acquire() -> None:
    """Direct, timing-independent coverage of the locking primitive
    itself (the HTTP-level test above exercises it through real
    concurrent requests, but that's inherently timing-sensitive)."""
    thread_id = "lock-test-thread"
    assert docgen_router._try_acquire(thread_id) is True
    assert docgen_router._try_acquire(thread_id) is False
    docgen_router._release(thread_id)
    assert docgen_router._try_acquire(thread_id) is True
    docgen_router._release(thread_id)


async def test_a_background_thread_crash_is_reported_as_failed_not_stuck_running(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _broken_checkpointer(*args, **kwargs):
        raise RuntimeError("simulated checkpointer outage")

    monkeypatch.setattr(docgen_router, "build_checkpointer", _broken_checkpointer)

    create = await client.post(
        "/docgen/runs",
        json={
            "request_text": "What was the hosted API's average latency?",
            "sources": [{"role": "task_docs", "tag": "docgen:task:anything", "required": True}],
        },
    )
    assert create.status_code == 200, create.text
    thread_id = create.json()["thread_id"]

    failed = await _poll_until_not_running(client, thread_id)
    assert failed["status"] == "failed"
    assert "simulated checkpointer outage" in failed["last_error"]
