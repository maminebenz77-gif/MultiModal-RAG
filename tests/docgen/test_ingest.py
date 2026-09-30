import shutil
from pathlib import Path

import pytest

from multimodal_rag.docgen.ingest import ingest_folder
from multimodal_rag.docgen.stack import DocgenStack

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"


def test_ingest_folder_tags_every_document_with_the_given_tag(
    stack: DocgenStack, tmp_path: Path
) -> None:
    folder = tmp_path / "batch"
    folder.mkdir()
    shutil.copy(SAMPLES_DIR / "chunking_demo.md", folder / "chunking_demo.md")
    shutil.copy(SAMPLES_DIR / "sample.md", folder / "sample.md")

    summary = ingest_folder(folder, "docgen:task:demo", "public", stack)

    assert summary.num_files == 2
    assert summary.tag == "docgen:task:demo"
    for result in summary.results:
        assert result.status == "ingested"
        assert result.metadata.tags == ["docgen:task:demo"]
        assert result.metadata.classification == "public"


def test_ingest_folder_skips_dotfiles_and_subfolders(stack: DocgenStack, tmp_path: Path) -> None:
    folder = tmp_path / "batch"
    folder.mkdir()
    shutil.copy(SAMPLES_DIR / "sample.md", folder / "sample.md")
    (folder / ".hidden.md").write_text("should be skipped")
    (folder / "nested").mkdir()
    shutil.copy(SAMPLES_DIR / "chunking_demo.md", folder / "nested" / "chunking_demo.md")

    summary = ingest_folder(folder, "docgen:task:demo", "public", stack)

    assert summary.num_files == 1
    assert summary.results[0].filename == "sample.md"


def test_ingest_folder_raises_on_a_folder_with_no_files(stack: DocgenStack, tmp_path: Path) -> None:
    folder = tmp_path / "empty"
    folder.mkdir()

    with pytest.raises(ValueError, match="No files found"):
        ingest_folder(folder, "docgen:task:demo", "public", stack)


def test_ingest_folder_adds_the_new_tag_to_a_document_already_ingested_under_another_tag(
    stack: DocgenStack, tmp_path: Path
) -> None:
    first_folder = tmp_path / "first"
    first_folder.mkdir()
    shutil.copy(SAMPLES_DIR / "sample.md", first_folder / "sample.md")
    second_folder = tmp_path / "second"
    second_folder.mkdir()
    shutil.copy(SAMPLES_DIR / "sample.md", second_folder / "sample.md")

    first = ingest_folder(first_folder, "docgen:task:first-batch", "public", stack)
    second = ingest_folder(second_folder, "docgen:ref:second-batch", "public", stack)

    assert first.results[0].status == "ingested"
    assert second.results[0].status == "already_ingested"
    assert second.results[0].doc_id == first.results[0].doc_id
    assert set(second.results[0].metadata.tags) == {
        "docgen:task:first-batch",
        "docgen:ref:second-batch",
    }


def test_ingest_folder_does_not_duplicate_an_already_present_tag(
    stack: DocgenStack, tmp_path: Path
) -> None:
    folder = tmp_path / "batch"
    folder.mkdir()
    shutil.copy(SAMPLES_DIR / "sample.md", folder / "sample.md")

    ingest_folder(folder, "docgen:task:demo", "public", stack)
    second = ingest_folder(folder, "docgen:task:demo", "public", stack)

    assert second.results[0].metadata.tags == ["docgen:task:demo"]
