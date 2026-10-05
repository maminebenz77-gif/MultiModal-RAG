import shutil
from pathlib import Path

from multimodal_rag.docgen.graph import build_graph
from multimodal_rag.docgen.ingest import ingest_folder
from multimodal_rag.docgen.sources import SourceSpec
from multimodal_rag.docgen.stack import DocgenStack
from multimodal_rag.docgen.state import DocGenState

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"


class _ScriptedLLM:
    """Returns one scripted reply per call, in order -- call order
    through this graph is deterministic (formulate_query, then
    generate_answer, then validate_answer, every attempt), so a plain
    script is enough; no need to branch on prompt content."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = iter(replies)

    def generate(self, messages: list[dict[str, str]]) -> str:
        try:
            return next(self._replies)
        except StopIteration:
            raise AssertionError("LLM was called more times than the test scripted") from None


class _FailingLLM:
    """Simulates a persistent provider failure -- RuntimeError is one
    of the exception types langgraph.types.default_retry_on does NOT
    automatically retry, so this reaches docgen's own try/except
    immediately rather than after LangGraph's internal retries."""

    def generate(self, messages: list[dict[str, str]]) -> str:
        raise RuntimeError("simulated LLM outage")


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


def test_graph_accepts_an_answer_on_the_first_try(stack: DocgenStack, tmp_path: Path) -> None:
    source = _ingest_task_docs(stack, tmp_path)
    llm = _ScriptedLLM(
        [
            "hosted api latency",  # formulate_query
            "120ms on average.",  # generate_answer
            '{"valid": true, "reason": "fully grounded"}',  # validate_answer
        ]
    )
    graph = build_graph(stack, llm=llm)

    result = graph.invoke(_initial_state(source))

    assert result["questions"][0]["status"] == "answered"
    assert result["answers"]["q1"]["text"] == "120ms on average."
    assert result["answers"]["q1"]["attempts"] == []
    assert result["current"] is None


def test_graph_retries_once_then_accepts(stack: DocgenStack, tmp_path: Path) -> None:
    source = _ingest_task_docs(stack, tmp_path)
    llm = _ScriptedLLM(
        [
            "hosted api latency",  # formulate_query (attempt 1)
            "80ms, probably.",  # generate_answer (attempt 1)
            '{"valid": false, "reason": "The context says 120ms, not 80ms."}',
            "hosted api average latency ms",  # formulate_query (attempt 2)
            "120ms on average.",  # generate_answer (attempt 2)
            '{"valid": true, "reason": "Now matches the context."}',
        ]
    )
    graph = build_graph(stack, llm=llm)

    result = graph.invoke(_initial_state(source))

    assert result["questions"][0]["status"] == "answered"
    assert result["answers"]["q1"]["text"] == "120ms on average."
    assert len(result["answers"]["q1"]["attempts"]) == 1
    assert result["answers"]["q1"]["attempts"][0]["answer"] == "80ms, probably."
    assert "120ms" in result["answers"]["q1"]["attempts"][0]["reason"]


def test_graph_pauses_for_a_human_after_exhausting_retries(
    stack: DocgenStack, tmp_path: Path
) -> None:
    source = _ingest_task_docs(stack, tmp_path)
    rejection = '{"valid": false, "reason": "Still not grounded."}'
    llm = _ScriptedLLM(["a search query", "a bad answer", rejection] * 3)
    graph = build_graph(stack, llm=llm)

    result = graph.invoke(_initial_state(source))

    assert result["questions"][0]["status"] == "escalated"
    assert "q1" not in result["answers"]
    # current is deliberately preserved (not cleared) so ask_human can
    # show a human exactly what was tried and why it was rejected.
    assert result["current"] is not None
    assert len(result["current"]["attempts"]) == 3
    # Reaching the interrupt doesn't need a checkpointer -- only
    # RESUMING it does (see test_checkpointer.py for that).
    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value["question"] == (
        "What was the hosted API's average latency?"
    )


def test_graph_pauses_for_a_human_instead_of_crashing_on_a_persistent_technical_failure(
    stack: DocgenStack, tmp_path: Path
) -> None:
    source = _ingest_task_docs(stack, tmp_path)
    graph = build_graph(stack, llm=_FailingLLM())

    result = graph.invoke(_initial_state(source))

    assert result["questions"][0]["status"] == "escalated"
    assert len(result["current"]["attempts"]) == 3
    assert all("Technical failure" in a["reason"] for a in result["current"]["attempts"])
    assert "__interrupt__" in result
