"""The HTTP surface for docgen: start a run, poll its status, resume a
paused one, and download the finished document. Lets the frontend (or
any caller) drive the exact same graph docgen/cli.py already drives --
this router never reimplements any node logic, it just orchestrates.

Execution model: a docgen run can take many minutes and pause several
times. Tying that to one open HTTP connection (blocking or streaming)
would mean the job dies the moment a browser tab closes or a proxy
times out -- fragile for something explicitly meant to run unattended.
Instead, POST /docgen/runs and POST /docgen/runs/{thread_id}/resume
each launch ONE segment of graph execution on a dedicated background
thread and return immediately; a caller polls GET
/docgen/runs/{thread_id} to see how it's going. The checkpointer
(docgen/checkpointer.py) is what makes this safe: it already persists
every intermediate step to disk, so "what's the status of this run"
can always be answered fresh from graph.get_state(), independent of
which process or thread is asking.

Status is NEVER tracked as a stored enum -- only two things are hard to
derive live: "has a background thread crashed outright" (there's no
checkpoint for that) and "whose run is this, and what's it called" (the
checkpointer has no notion of listing threads). Both live in
docgen/runs_db.py's tiny table; everything else comes from
graph.get_state() each time, so there's exactly one source of truth for
running/paused/done.
"""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from langgraph.types import Command, RunnableConfig
from starlette.concurrency import run_in_threadpool

from ...chunking.schema import ChunkElement
from ...docgen.checkpointer import build_checkpointer
from ...docgen.graph import build_graph
from ...docgen.nodes.retrieval import RetrievedChunk
from ...docgen.runs_db import DocgenRunsDB, RunRow
from ...docgen.sources import SourceSpec
from ...docgen.stack import DocgenStack
from ...docgen.state import DocGenState
from ...identity import Principal
from ..dependencies import get_docgen_checkpoint_path, get_docgen_runs_db, get_docgen_stack
from ..identity import get_principal
from ..schemas import (
    ChunkElementOut,
    CreateDocgenRunRequest,
    DocgenAnswerOut,
    DocgenAttemptOut,
    DocgenPendingOut,
    DocgenQuestionOut,
    DocgenResumeRequest,
    DocgenRunAccepted,
    DocgenRunListResponse,
    DocgenRunStatusResponse,
    DocgenRunSummary,
    RetrievedChunkOut,
)

router = APIRouter()

_logger = logging.getLogger(__name__)

_VALID_ACTIONS: dict[str, set[str]] = {
    "confirm_configuration": {"confirm", "revise"},
    "human_review": {"approve", "edit"},
    "ask_human": {"answer", "reformulate", "skip"},
}
"""Which DocgenResumeRequest.action values are meaningful for each
interrupt `kind` -- the three TypedDicts (ConfirmationResponse,
ReviewResponse, HumanResponse) this maps onto are never imported here
directly; resume_run builds their matching dict shape by hand (see
_resume_payload) since there's one flat request body for all three."""

_run_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()
"""In-process only -- consistent with the rest of this app's
single-process assumption (AppState is already an in-process
singleton). Prevents two concurrent invoke() calls against the same
thread_id (a double-click, or a retried request), not a cross-process
or cross-worker guarantee."""


def _try_acquire(thread_id: str) -> bool:
    with _registry_lock:
        lock = _run_locks.setdefault(thread_id, threading.Lock())
    return lock.acquire(blocking=False)


def _release(thread_id: str) -> None:
    _run_locks[thread_id].release()


def _is_running(thread_id: str) -> bool:
    """True the instant a segment is dispatched (_try_acquire succeeds
    in the request handler), not just once its background thread
    actually starts executing -- closing a real race: right after a
    resume is submitted, the new segment's first checkpoint hasn't been
    written yet, so a naive checkpoint read would still show the OLD
    pause a caller just resumed, indistinguishable from a genuine new
    one without this."""
    with _registry_lock:
        lock = _run_locks.get(thread_id)
    if lock is None:
        return False
    acquired = lock.acquire(blocking=False)
    if not acquired:
        return True
    lock.release()
    return False


def _run_segment(
    thread_id: str, run_input: Any, stack: DocgenStack, runs_db: DocgenRunsDB, checkpoint_path: Path
) -> None:
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    runs_db.clear_error(thread_id)
    try:
        with build_checkpointer(checkpoint_path) as checkpointer:
            graph = build_graph(stack, checkpointer=checkpointer)
            graph.invoke(run_input, config)
    except Exception as exc:
        _logger.exception("docgen run %r crashed", thread_id)
        runs_db.record_error(thread_id, str(exc))
    finally:
        _release(thread_id)


def _run_status(
    thread_id: str, stack: DocgenStack, checkpoint_path: Path, last_error: str | None
) -> tuple[str, Any | None, dict[str, Any]]:
    """The one place "what's this run's status" is decided -- returns
    (status, the pending Interrupt or None, the state values dict).

    _is_running() is checked FIRST, before even a recorded error: a
    segment that's actively in flight right now is unambiguously
    "running" regardless of what the last checkpoint says (see its own
    docstring for the race this closes), and _run_segment always
    clears any old error at the start of a new segment anyway.

    A recorded crash short-circuits BEFORE ever touching the
    checkpointer again: once a background thread has already failed,
    nothing should depend on that same checkpointer/graph being healthy
    (if a broken checkpointer is WHY it crashed, calling it again here
    would just crash the status check too, instead of cleanly reporting
    "failed").

    Past that: snapshot.next is NOT a reliable "still running" signal on
    its own -- confirmed empirically, not just in theory: between two
    nodes joined by a plain unconditional edge (e.g.
    harmonize_answers -> human_review), the checkpoint written right
    after the first one can transiently show next=() for exactly one
    tick, even though the graph is nowhere near done. An actual
    interrupt() showing up in snapshot.tasks[...].interrupts is the
    only reliable "paused, waiting on a human" signal, checked
    unconditionally (not gated behind snapshot.next being non-empty).
    "Done" is decided by output_path being set -- the one field only
    ever written by generate_document, the sole path to END in this
    graph -- rather than by snapshot.next being empty, for the same
    reason.
    """
    running_now = _is_running(thread_id)
    if last_error is not None and not running_now:
        return "failed", None, {}
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    try:
        with build_checkpointer(checkpoint_path) as checkpointer:
            graph = build_graph(stack, checkpointer=checkpointer)
            snapshot = graph.get_state(config)
    except Exception:
        # A status CHECK must never crash the endpoint, even if the
        # checkpointer itself is having trouble -- if this is the same
        # failure a running segment is about to hit, its own
        # _run_segment will record last_error shortly and a later poll
        # will correctly report "failed" without touching the
        # checkpointer again at all.
        _logger.exception("Failed to read docgen checkpoint for %r", thread_id)
        return "running", None, {}
    values: dict[str, Any] = snapshot.values or {}
    if running_now:
        return "running", None, values  # a segment is in flight right now
    if not values:
        return "running", None, values  # no checkpoint written yet at all
    for task in snapshot.tasks:
        if task.interrupts:
            return "paused", task.interrupts[0], values
    if values.get("output_path") is not None:
        return "done", None, values  # generate_document ran; the graph reached END
    return "running", None, values  # mid-segment, more steps queued


def _to_chunk_element_out(e: ChunkElement) -> ChunkElementOut:
    return ChunkElementOut(
        type=e.type,
        text=e.text,
        image_base64=e.image_base64,
        description=e.description,
        page=e.page,
        slide=e.slide,
    )


def _to_retrieved_chunk_out(rc: RetrievedChunk) -> RetrievedChunkOut:
    c = rc.chunk
    return RetrievedChunkOut(
        chunk_id=c.chunk_id,
        score=c.score,
        text=c.text,
        source=c.source,
        doc_id=c.doc_id,
        pages=c.pages,
        slides=c.slides,
        elements=[_to_chunk_element_out(e) for e in c.elements],
    )


def _pending_out(interrupt_value: dict[str, Any]) -> DocgenPendingOut:
    kind = interrupt_value["kind"]
    if kind == "ask_human":
        return DocgenPendingOut(
            kind=kind,
            question=interrupt_value.get("question"),
            chunks=[_to_retrieved_chunk_out(c) for c in interrupt_value.get("chunks", [])],
            attempts=[
                DocgenAttemptOut(query=a["query"], answer=a["answer"], reason=a["reason"])
                for a in interrupt_value.get("attempts", [])
            ],
            previous_guidance=interrupt_value.get("previous_guidance"),
        )
    return DocgenPendingOut(kind=kind, summary=interrupt_value.get("summary"))


def _resume_payload(kind: str, body: DocgenResumeRequest) -> dict[str, Any]:
    if kind == "human_review":
        return {"action": body.action, "text": body.text, "question_ids": body.question_ids}
    return {"action": body.action, "text": body.text}


def _status_response(
    thread_id: str, run: RunRow, stack: DocgenStack, checkpoint_path: Path
) -> DocgenRunStatusResponse:
    run_status, interrupt, values = _run_status(thread_id, stack, checkpoint_path, run.last_error)
    questions = [
        DocgenQuestionOut(
            id=q["id"], text=q["text"], sources_required=q["sources_required"], status=q["status"]
        )
        for q in values.get("questions", [])
    ]
    answers = {
        qid: DocgenAnswerOut(text=a["text"], accepted_by=a["accepted_by"])
        for qid, a in values.get("answers", {}).items()
    }
    return DocgenRunStatusResponse(
        thread_id=thread_id,
        title=run.title,
        status=run_status,
        pending=_pending_out(interrupt.value) if interrupt is not None else None,
        questions=questions,
        answers=answers,
        llm_calls=values.get("usage", {}).get("llm_calls", 0),
        output_path=values.get("output_path"),
        last_error=run.last_error,
    )


def _initial_state(body: CreateDocgenRunRequest) -> DocGenState:
    return {
        "sources": [SourceSpec(role=s.role, tag=s.tag, required=s.required) for s in body.sources],
        "questions": [],
        "answers": {},
        "configuration": {
            "template": "",
            "format": "pptx",
            "confirmed": False,
            "max_retries": body.max_retries,
        },
        "review": {"decision": "pending", "flagged_question_ids": [], "guidance": None},
        "current": None,
        "usage": {"llm_calls": 0},
        "request": {"text": body.request_text, "corrections": []},
        "output_path": None,
    }


@router.post("/docgen/runs", response_model=DocgenRunAccepted)
async def create_run(
    body: CreateDocgenRunRequest,
    principal: Principal = Depends(get_principal),
    stack: DocgenStack = Depends(get_docgen_stack),
    runs_db: DocgenRunsDB = Depends(get_docgen_runs_db),
    checkpoint_path: Path = Depends(get_docgen_checkpoint_path),
) -> DocgenRunAccepted:
    thread_id = uuid.uuid4().hex
    title = body.title or body.request_text[:80]
    await run_in_threadpool(runs_db.create_run, principal, thread_id, title)
    if not _try_acquire(thread_id):
        # Practically impossible for a freshly generated uuid, but keep
        # the run discoverable as failed rather than silently stuck.
        await run_in_threadpool(runs_db.record_error, thread_id, "Could not start run")
        raise HTTPException(status_code=409, detail="Could not start run")
    threading.Thread(
        target=_run_segment,
        args=(thread_id, _initial_state(body), stack, runs_db, checkpoint_path),
        daemon=True,
    ).start()
    return DocgenRunAccepted(thread_id=thread_id, status="running")


@router.get("/docgen/runs", response_model=DocgenRunListResponse)
async def list_runs(
    principal: Principal = Depends(get_principal),
    stack: DocgenStack = Depends(get_docgen_stack),
    runs_db: DocgenRunsDB = Depends(get_docgen_runs_db),
    checkpoint_path: Path = Depends(get_docgen_checkpoint_path),
) -> DocgenRunListResponse:
    runs = await run_in_threadpool(runs_db.list_runs, principal)

    def _build_summaries() -> list[DocgenRunSummary]:
        summaries = []
        for run in runs:
            run_status, _, _ = _run_status(run.thread_id, stack, checkpoint_path, run.last_error)
            summaries.append(
                DocgenRunSummary(
                    thread_id=run.thread_id,
                    title=run.title,
                    status=run_status,
                    created_at=run.created_at,
                    last_error=run.last_error,
                )
            )
        return summaries

    return DocgenRunListResponse(runs=await run_in_threadpool(_build_summaries))


@router.get("/docgen/runs/{thread_id}", response_model=DocgenRunStatusResponse)
async def get_run(
    thread_id: str,
    principal: Principal = Depends(get_principal),
    stack: DocgenStack = Depends(get_docgen_stack),
    runs_db: DocgenRunsDB = Depends(get_docgen_runs_db),
    checkpoint_path: Path = Depends(get_docgen_checkpoint_path),
) -> DocgenRunStatusResponse:
    run = await run_in_threadpool(runs_db.get_run, principal, thread_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"No such docgen run: {thread_id!r}")
    return await run_in_threadpool(_status_response, thread_id, run, stack, checkpoint_path)


@router.post("/docgen/runs/{thread_id}/resume", response_model=DocgenRunAccepted)
async def resume_run(
    thread_id: str,
    body: DocgenResumeRequest,
    principal: Principal = Depends(get_principal),
    stack: DocgenStack = Depends(get_docgen_stack),
    runs_db: DocgenRunsDB = Depends(get_docgen_runs_db),
    checkpoint_path: Path = Depends(get_docgen_checkpoint_path),
) -> DocgenRunAccepted:
    run = await run_in_threadpool(runs_db.get_run, principal, thread_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"No such docgen run: {thread_id!r}")

    def _pending_kind() -> str | None:
        _, interrupt, _ = _run_status(thread_id, stack, checkpoint_path, run.last_error)
        return None if interrupt is None else interrupt.value["kind"]

    kind = await run_in_threadpool(_pending_kind)
    if kind is None:
        raise HTTPException(status_code=409, detail="No pending interrupt to resume")
    if body.action not in _VALID_ACTIONS[kind]:
        raise HTTPException(
            status_code=400,
            detail=f"Action {body.action!r} is not valid while a {kind!r} interrupt is pending",
        )

    if not _try_acquire(thread_id):
        raise HTTPException(status_code=409, detail="This run is already being resumed")
    run_input: Command[Any] = Command(resume=_resume_payload(kind, body))
    threading.Thread(
        target=_run_segment,
        args=(thread_id, run_input, stack, runs_db, checkpoint_path),
        daemon=True,
    ).start()
    return DocgenRunAccepted(thread_id=thread_id, status="running")


@router.get("/docgen/runs/{thread_id}/download")
async def download_run(
    thread_id: str,
    principal: Principal = Depends(get_principal),
    stack: DocgenStack = Depends(get_docgen_stack),
    runs_db: DocgenRunsDB = Depends(get_docgen_runs_db),
    checkpoint_path: Path = Depends(get_docgen_checkpoint_path),
) -> FileResponse:
    run = await run_in_threadpool(runs_db.get_run, principal, thread_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"No such docgen run: {thread_id!r}")

    def _finished_output_path() -> str | None:
        run_status, _, values = _run_status(thread_id, stack, checkpoint_path, run.last_error)
        if run_status != "done":
            return None
        output_path: str | None = values.get("output_path")
        return output_path

    output_path = await run_in_threadpool(_finished_output_path)
    if not output_path:
        raise HTTPException(status_code=404, detail="This run has no finished document yet")

    path = Path(output_path)
    media_type = (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        if path.suffix == ".pptx"
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    return FileResponse(path, media_type=media_type, filename=path.name)
