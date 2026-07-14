from pathlib import Path

import pytest

from multimodal_rag.ingestion import parse_document
from multimodal_rag.ingestion.schema import ElementType
from multimodal_rag.providers.base import VisionProvider

_SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"


class FakeVisionProvider(VisionProvider):
    def describe(self, image_bytes: bytes, prompt: str | None = None) -> str:
        return "a description"


@pytest.fixture(autouse=True)
def _fake_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "multimodal_rag.ingestion.vision.get_vision", lambda: FakeVisionProvider()
    )


def test_routes_markdown_by_extension_since_magic_reports_text_plain() -> None:
    elements = parse_document(_SAMPLES / "sample.md")
    assert elements[0].type == ElementType.TITLE


def test_routes_docx_by_content_mime_type() -> None:
    elements = parse_document(_SAMPLES / "sample.docx")
    assert elements[0].type == ElementType.TITLE


def test_routes_pptx_by_content_mime_type() -> None:
    elements = parse_document(_SAMPLES / "sample.pptx")
    assert elements[0].type == ElementType.TITLE


def test_routes_pdf_by_content_mime_type(monkeypatch: pytest.MonkeyPatch) -> None:
    # partition_pdf's hi_res strategy is too slow for a unit test — mock it,
    # just to confirm the dispatcher routes .pdf to parse_pdf at all.
    monkeypatch.setattr("multimodal_rag.ingestion.pdf.partition_pdf", lambda **kwargs: [])
    elements = parse_document(_SAMPLES / "sample.pdf")
    assert elements == []


def test_unsupported_file_type_raises(tmp_path: Path) -> None:
    image_path = tmp_path / "not_a_document.png"
    image_path.write_bytes((_SAMPLES / "latency_chart.png").read_bytes())

    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_document(image_path)
