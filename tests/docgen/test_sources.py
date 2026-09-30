import shutil
from pathlib import Path

from multimodal_rag.docgen.sources import IngestNew, ReuseExisting, resolve_source
from multimodal_rag.docgen.stack import DocgenStack
from multimodal_rag.docgen.tags import list_known_tags

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"


def test_resolve_source_with_reuse_existing_does_no_ingestion(stack: DocgenStack) -> None:
    spec = resolve_source("reference_kb", ReuseExisting(tag="some-existing-tag"), stack)

    assert spec.name == "reference_kb"
    assert spec.tag == "some-existing-tag"
    assert spec.required is True
    assert list_known_tags(stack.db) == []  # nothing was ingested


def test_resolve_source_with_ingest_new_ingests_the_folder_under_a_prefixed_tag(
    stack: DocgenStack, tmp_path: Path
) -> None:
    folder = tmp_path / "batch"
    folder.mkdir()
    shutil.copy(SAMPLES_DIR / "sample.md", folder / "sample.md")

    spec = resolve_source(
        "task_docs",
        IngestNew(folder=folder, label="q3-audit", classification="public"),
        stack,
    )

    assert spec.tag == "docgen:task:q3-audit"
    assert spec.tag in list_known_tags(stack.db)


def test_resolve_source_prefixes_reference_kb_with_ref_not_task(
    stack: DocgenStack, tmp_path: Path
) -> None:
    folder = tmp_path / "batch"
    folder.mkdir()
    shutil.copy(SAMPLES_DIR / "sample.md", folder / "sample.md")

    spec = resolve_source(
        "reference_kb",
        IngestNew(folder=folder, label="iso-27001", classification="public"),
        stack,
    )

    assert spec.tag == "docgen:ref:iso-27001"


def test_resolve_source_honors_the_required_flag(stack: DocgenStack) -> None:
    spec = resolve_source(
        "reference_kb", ReuseExisting(tag="some-tag"), stack, required=False
    )

    assert spec.required is False
