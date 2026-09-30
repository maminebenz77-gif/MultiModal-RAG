import shutil
from pathlib import Path

import pytest

from multimodal_rag.docgen.ingest import ingest_folder
from multimodal_rag.docgen.stack import DocgenStack
from multimodal_rag.docgen.tags import build_docgen_tag, list_known_tags

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"


def test_build_docgen_tag_prefixes_by_role() -> None:
    assert build_docgen_tag("task", "q3-audit") == "docgen:task:q3-audit"
    assert build_docgen_tag("ref", "iso-27001") == "docgen:ref:iso-27001"


def test_build_docgen_tag_rejects_an_empty_label() -> None:
    with pytest.raises(ValueError, match="empty"):
        build_docgen_tag("task", "   ")


def test_build_docgen_tag_rejects_a_colon_in_the_label() -> None:
    with pytest.raises(ValueError, match="separator"):
        build_docgen_tag("task", "a:b")


def test_list_known_tags_returns_tags_from_ingested_documents(
    stack: DocgenStack, tmp_path: Path
) -> None:
    folder = tmp_path / "batch"
    folder.mkdir()
    shutil.copy(SAMPLES_DIR / "sample.md", folder / "sample.md")
    tag = build_docgen_tag("task", "q3-audit")

    ingest_folder(folder, tag, "public", stack)

    assert tag in list_known_tags(stack.db)


def test_list_known_tags_can_be_narrowed_by_prefix(stack: DocgenStack, tmp_path: Path) -> None:
    task_folder = tmp_path / "task"
    task_folder.mkdir()
    shutil.copy(SAMPLES_DIR / "sample.md", task_folder / "sample.md")
    ref_folder = tmp_path / "ref"
    ref_folder.mkdir()
    shutil.copy(SAMPLES_DIR / "chunking_demo.md", ref_folder / "chunking_demo.md")

    task_tag = build_docgen_tag("task", "q3-audit")
    ref_tag = build_docgen_tag("ref", "iso-27001")
    ingest_folder(task_folder, task_tag, "public", stack)
    ingest_folder(ref_folder, ref_tag, "public", stack)

    task_only = list_known_tags(stack.db, prefix="docgen:task:")

    assert task_tag in task_only
    assert ref_tag not in task_only
