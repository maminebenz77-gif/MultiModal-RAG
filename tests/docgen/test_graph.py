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
    through this graph is deterministic (generate_answer always runs
    right before validate_answer), so a plain script is enough; no need
    to branch on prompt content."""

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


def test_graph_accepts_an_answer_on_the_first_try(stack: DocgenStack, tmp_path: Path) -> None:
    source = _ingest_task_docs(stack, tmp_path)
    llm = _ScriptedLLM(["120ms on average.", '{"valid": true, "reason": "fully grounded"}'])
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
            "80ms, probably.",
            '{"valid": false, "reason": "The context says 120ms, not 80ms."}',
            "120ms on average.",
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


def test_graph_escalates_after_exhausting_retries(stack: DocgenStack, tmp_path: Path) -> None:
    source = _ingest_task_docs(stack, tmp_path)
    rejection = '{"valid": false, "reason": "Still not grounded."}'
    llm = _ScriptedLLM(["a bad answer", rejection] * 3)
    graph = build_graph(stack, llm=llm)

    result = graph.invoke(_initial_state(source))

    assert result["questions"][0]["status"] == "escalated"
    assert "q1" not in result["answers"]
    assert result["current"] is None
