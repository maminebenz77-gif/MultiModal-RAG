from pathlib import Path

import pytest

from multimodal_rag.ingestion.markdown import parse_markdown
from multimodal_rag.ingestion.schema import ElementType
from multimodal_rag.providers.base import VisionProvider


class FakeVisionProvider(VisionProvider):
    def describe(self, image_bytes: bytes, prompt: str | None = None) -> str:
        return f"a description of {len(image_bytes)} bytes"


@pytest.fixture(autouse=True)
def _fake_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "multimodal_rag.ingestion.vision.get_vision", lambda: FakeVisionProvider()
    )


def _write_sample(tmp_path: Path) -> Path:
    image_path = tmp_path / "pic.png"
    image_path.write_bytes(b"\x89PNG-fake-bytes")

    md_path = tmp_path / "doc.md"
    md_path.write_text(
        "# A Title\n"
        "\n"
        "A paragraph of body text.\n"
        "\n"
        "| A | B |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n"
        "\n"
        "![a figure](pic.png)\n",
        encoding="utf-8",
    )
    return md_path


def test_parses_title_paragraph_table_and_image(tmp_path: Path) -> None:
    elements = parse_markdown(_write_sample(tmp_path))

    types = [el.type for el in elements]
    assert types == [
        ElementType.TITLE,
        ElementType.PARAGRAPH,
        ElementType.TABLE,
        ElementType.IMAGE,
    ]


def test_title_and_paragraph_text(tmp_path: Path) -> None:
    elements = parse_markdown(_write_sample(tmp_path))
    assert elements[0].text == "A Title"
    assert elements[1].text == "A paragraph of body text."


def test_table_rendered_as_markdown(tmp_path: Path) -> None:
    elements = parse_markdown(_write_sample(tmp_path))
    table = elements[2]
    assert table.text == "| A | B |\n| --- | --- |\n| 1 | 2 |"
    assert table.table_summary is None  # opt-in, not requested


def test_image_element_has_bytes_and_description(tmp_path: Path) -> None:
    elements = parse_markdown(_write_sample(tmp_path))
    image = elements[3]
    assert image.image_bytes == b"\x89PNG-fake-bytes"
    assert image.description == "a description of 15 bytes"
    assert image.description_status == "generated"


def test_positions_are_sequential(tmp_path: Path) -> None:
    elements = parse_markdown(_write_sample(tmp_path))
    assert [el.metadata.position for el in elements] == [0, 1, 2, 3]
    assert all(el.metadata.source_file for el in elements)


def test_summarize_tables_flag_calls_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "multimodal_rag.ingestion.markdown.summarize_table", lambda markdown_table: "a summary"
    )
    elements = parse_markdown(_write_sample(tmp_path), summarize_tables=True)
    table = next(el for el in elements if el.type == ElementType.TABLE)
    assert table.table_summary == "a summary"
