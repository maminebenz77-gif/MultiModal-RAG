import shutil
from pathlib import Path

from multimodal_rag.docgen.ingest import ingest_folder
from multimodal_rag.docgen.nodes.retrieval import retrieve_for_question
from multimodal_rag.docgen.sources import SourceSpec
from multimodal_rag.docgen.stack import DocgenStack

SAMPLES_DIR = Path(__file__).resolve().parents[3] / "data" / "samples"

QUERY = "latency benchmark across LLM providers"


def _ingest_one(stack: DocgenStack, tmp_path: Path, filename: str, tag: str) -> None:
    folder = tmp_path / tag.replace(":", "_")
    folder.mkdir()
    shutil.copy(SAMPLES_DIR / filename, folder / filename)
    ingest_folder(folder, tag, "public", stack)


def test_retrieve_for_question_scopes_results_to_the_given_sources_tag(
    stack: DocgenStack, tmp_path: Path
) -> None:
    _ingest_one(stack, tmp_path, "sample.md", "docgen:task:demo")
    _ingest_one(stack, tmp_path, "chunking_demo.md", "docgen:ref:demo")

    task_only = [SourceSpec(name="task_docs", tag="docgen:task:demo", required=True)]
    results = retrieve_for_question(QUERY, task_only, stack.retriever)

    assert results  # the task doc is actually about latency benchmarking
    assert all(r.source_name == "task_docs" for r in results)
    assert all("sample.md" == r.chunk.source for r in results)


def test_retrieve_for_question_merges_and_labels_multiple_sources(
    stack: DocgenStack, tmp_path: Path
) -> None:
    _ingest_one(stack, tmp_path, "sample.md", "docgen:task:demo")
    _ingest_one(stack, tmp_path, "chunking_demo.md", "docgen:ref:demo")

    both = [
        SourceSpec(name="task_docs", tag="docgen:task:demo", required=True),
        SourceSpec(name="reference_kb", tag="docgen:ref:demo", required=False),
    ]
    results = retrieve_for_question(QUERY, both, stack.retriever)

    labels = {r.source_name for r in results}
    assert labels == {"task_docs", "reference_kb"}


def test_retrieve_for_question_returns_no_chunks_for_a_source_matching_nothing(
    stack: DocgenStack,
) -> None:
    unused_source = [SourceSpec(name="reference_kb", tag="docgen:ref:nothing-here", required=False)]

    results = retrieve_for_question(QUERY, unused_source, stack.retriever)

    assert results == []
