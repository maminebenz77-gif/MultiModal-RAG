import logging
import shutil
from pathlib import Path

import pytest
from langgraph.types import Command

from multimodal_rag.docgen.checkpointer import build_checkpointer
from multimodal_rag.docgen.graph import build_graph
from multimodal_rag.docgen.ingest import ingest_folder
from multimodal_rag.docgen.sources import SourceSpec
from multimodal_rag.docgen.stack import DocgenStack
from multimodal_rag.docgen.state import DocGenState

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"


class _ScriptedLLM:
    def __init__(self, replies: list[str]) -> None:
        self._replies = iter(replies)

    def generate(self, messages: list[dict[str, str]]) -> str:
        try:
            return next(self._replies)
        except StopIteration:
            raise AssertionError("LLM was called more times than the test scripted") from None


def _ingest_task_docs(stack: DocgenStack, tmp_path: Path) -> SourceSpec:
    folder = tmp_path / "task"
    folder.mkdir()
    shutil.copy(SAMPLES_DIR / "sample.md", folder / "sample.md")
    ingest_folder(folder, "docgen:task:demo", "public", stack)
    return SourceSpec(role="task_docs", tag="docgen:task:demo", required=True)


def _initial_state(source: SourceSpec) -> DocGenState:
    return {
        "sources": [source],
        "questions": [
            {
                "id": "q1",
                "text": "What was the hosted API's average latency?",
                "sources_required": ["task_docs"],
                "status": "pending",
            }
        ],
        "answers": {},
        "configuration": {"template": "", "format": "pptx", "confirmed": True},
        "review": {"decision": "pending", "flagged_question_ids": []},
        "current": None,
    }


def test_build_checkpointer_creates_the_sqlite_file(tmp_path: Path) -> None:
    db_path = tmp_path / "checkpoints.sqlite"

    with build_checkpointer(db_path):
        pass

    assert db_path.exists()


def test_resuming_does_not_warn_about_unregistered_checkpoint_types(
    stack: DocgenStack, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Guards against regressing back to the default, unrestricted
    deserialization behavior -- real run output before
    _ALLOWED_MSGPACK_MODULES was added: "Deserializing unregistered
    type ...SourceSpec from checkpoint. This will be blocked in a
    future version." Registering the types we actually put in state
    (checkpointer.py's _ALLOWED_MSGPACK_MODULES) silences this AND
    switches the serializer into strict mode (blocking anything not
    registered), rather than warn-but-allow-anything."""
    source = _ingest_task_docs(stack, tmp_path)
    rejection = '{"valid": false, "reason": "Still not grounded."}'
    llm = _ScriptedLLM(["a search query", "a bad answer", rejection] * 3)
    config = {"configurable": {"thread_id": "test-thread"}}

    with caplog.at_level(logging.WARNING):
        with build_checkpointer(tmp_path / "checkpoints.sqlite") as checkpointer:
            graph = build_graph(stack, llm=llm, checkpointer=checkpointer)
            graph.invoke(_initial_state(source), config)
            graph.get_state(config)  # the step that actually deserializes `current`

    assert "unregistered type" not in caplog.text.lower()
    assert "blocked deserialization" not in caplog.text.lower()


def test_escalation_resumes_with_a_human_provided_answer(
    stack: DocgenStack, tmp_path: Path
) -> None:
    source = _ingest_task_docs(stack, tmp_path)
    rejection = '{"valid": false, "reason": "Still not grounded."}'
    llm = _ScriptedLLM(["a search query", "a bad answer", rejection] * 3)
    config = {"configurable": {"thread_id": "test-thread"}}

    with build_checkpointer(tmp_path / "checkpoints.sqlite") as checkpointer:
        graph = build_graph(stack, llm=llm, checkpointer=checkpointer)

        paused = graph.invoke(_initial_state(source), config)
        assert "__interrupt__" in paused
        assert paused["questions"][0]["status"] == "escalated"

        resume = {"action": "answer", "text": "120ms, from a human who checked manually."}
        result = graph.invoke(Command(resume=resume), config)

    assert "__interrupt__" not in result
    assert result["questions"][0]["status"] == "answered"
    assert result["answers"]["q1"]["text"] == "120ms, from a human who checked manually."
    assert result["answers"]["q1"]["accepted_by"] == "human"
    assert len(result["answers"]["q1"]["attempts"]) == 3
    assert result["current"] is None


def test_escalation_resumes_after_a_simulated_process_restart(
    stack: DocgenStack, tmp_path: Path
) -> None:
    """The real point of a SQLite (not in-memory) checkpointer: state
    must survive the checkpointer object itself being destroyed and a
    brand new one opened against the same file -- standing in for the
    original process dying and a fresh one picking the thread back up."""
    source = _ingest_task_docs(stack, tmp_path)
    rejection = '{"valid": false, "reason": "Still not grounded."}'
    llm = _ScriptedLLM(["a search query", "a bad answer", rejection] * 3)
    config = {"configurable": {"thread_id": "test-thread"}}
    db_path = tmp_path / "checkpoints.sqlite"

    with build_checkpointer(db_path) as checkpointer:
        graph = build_graph(stack, llm=llm, checkpointer=checkpointer)
        paused = graph.invoke(_initial_state(source), config)
        assert "__interrupt__" in paused
    # The `with` block has exited -- that connection is closed. Nothing
    # Python-level ties the next checkpointer to this one except the
    # file on disk.

    with build_checkpointer(db_path) as checkpointer:
        graph = build_graph(stack, llm=llm, checkpointer=checkpointer)
        snapshot = graph.get_state(config)
        pending_question = snapshot.values["current"]["question_id"]
        interrupt_payload = snapshot.tasks[0].interrupts[0].value

        resume = {"action": "answer", "text": "Resumed after restart: 120ms."}
        result = graph.invoke(Command(resume=resume), config)

    assert pending_question == "q1"
    assert interrupt_payload["question"] == "What was the hosted API's average latency?"
    assert result["answers"]["q1"]["text"] == "Resumed after restart: 120ms."
    assert result["answers"]["q1"]["accepted_by"] == "human"


def test_escalation_can_skip_the_question_entirely(stack: DocgenStack, tmp_path: Path) -> None:
    source = _ingest_task_docs(stack, tmp_path)
    rejection = '{"valid": false, "reason": "Still not grounded."}'
    llm = _ScriptedLLM(["a search query", "a bad answer", rejection] * 3)
    config = {"configurable": {"thread_id": "test-thread"}}

    with build_checkpointer(tmp_path / "checkpoints.sqlite") as checkpointer:
        graph = build_graph(stack, llm=llm, checkpointer=checkpointer)
        paused = graph.invoke(_initial_state(source), config)
        assert "__interrupt__" in paused

        result = graph.invoke(Command(resume={"action": "skip", "text": ""}), config)

    assert "__interrupt__" not in result
    assert result["questions"][0]["status"] == "skipped"
    assert "q1" not in result["answers"]
    assert result["current"] is None


def test_escalation_can_reformulate_instead_of_answering_directly(
    stack: DocgenStack, tmp_path: Path
) -> None:
    source = _ingest_task_docs(stack, tmp_path)
    rejection = '{"valid": false, "reason": "Still not grounded."}'
    accepted = '{"valid": true, "reason": "Matches the context now."}'
    llm = _ScriptedLLM(
        ["a search query", "a bad answer", rejection] * 3
        + ["a human-steered search query", "120ms on average.", accepted]
    )
    config = {"configurable": {"thread_id": "test-thread"}}

    with build_checkpointer(tmp_path / "checkpoints.sqlite") as checkpointer:
        graph = build_graph(stack, llm=llm, checkpointer=checkpointer)
        paused = graph.invoke(_initial_state(source), config)
        assert "__interrupt__" in paused

        resume = {
            "action": "reformulate",
            "text": "Search specifically for the hosted API's row in the latency table.",
        }
        result = graph.invoke(Command(resume=resume), config)

    assert "__interrupt__" not in result
    assert result["questions"][0]["status"] == "answered"
    assert result["answers"]["q1"]["text"] == "120ms on average."
    assert result["answers"]["q1"]["accepted_by"] == "validation"
    # The reformulated attempt gets a FRESH budget -- the 3 pre-escalation
    # rejections belonged to the attempt that just got replaced, not to
    # this one, so they don't carry into the final Answer's attempts.
    assert result["answers"]["q1"]["attempts"] == []
