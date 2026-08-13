import base64
from pathlib import Path
from typing import Any

import pytest

from multimodal_rag.ingestion.pdf import parse_pdf
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


class FakeMetadata:
    def __init__(
        self,
        page_number: int = 1,
        image_base64: str | None = None,
        text_as_html: str | None = None,
    ) -> None:
        self.page_number = page_number
        self.image_base64 = image_base64
        self.text_as_html = text_as_html


class FakeUnstructuredElement:
    def __init__(self, category: str, text: str, metadata: FakeMetadata) -> None:
        self.category = category
        self.text = text
        self.metadata = metadata


def _patch_partition_pdf(monkeypatch: pytest.MonkeyPatch, elements: list[Any]) -> None:
    monkeypatch.setattr(
        "multimodal_rag.ingestion.pdf.partition_pdf", lambda **kwargs: elements
    )


def test_title_and_narrative_text_map_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_partition_pdf(
        monkeypatch,
        [
            FakeUnstructuredElement("Title", "A Title", FakeMetadata()),
            FakeUnstructuredElement("NarrativeText", "Some body text.", FakeMetadata()),
            FakeUnstructuredElement("UncategorizedText", "Uncategorized bit.", FakeMetadata()),
        ],
    )
    elements = parse_pdf(Path("doc.pdf"))
    assert [el.type for el in elements] == [
        ElementType.TITLE,
        ElementType.PARAGRAPH,
        ElementType.PARAGRAPH,
    ]
    assert elements[0].text == "A Title"


def test_empty_text_elements_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_partition_pdf(
        monkeypatch,
        [
            FakeUnstructuredElement("NarrativeText", "   ", FakeMetadata()),
            FakeUnstructuredElement("NarrativeText", "Real text.", FakeMetadata()),
        ],
    )
    elements = parse_pdf(Path("doc.pdf"))
    assert len(elements) == 1
    assert elements[0].text == "Real text."


def test_table_uses_text_as_html_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    _patch_partition_pdf(
        monkeypatch,
        [FakeUnstructuredElement("Table", "A B 1 2", FakeMetadata(text_as_html=html))],
    )
    elements = parse_pdf(Path("doc.pdf"))
    assert elements[0].type == ElementType.TABLE
    assert elements[0].text == "| A | B |\n| --- | --- |\n| 1 | 2 |"


def test_table_falls_back_to_flattened_text_without_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_partition_pdf(
        monkeypatch,
        [FakeUnstructuredElement("Table", "A B 1 2", FakeMetadata(text_as_html=None))],
    )
    elements = parse_pdf(Path("doc.pdf"))
    assert elements[0].type == ElementType.TABLE
    assert elements[0].text == "A B 1 2"


def test_image_element_decodes_base64_and_gets_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_bytes = b"fake-image-bytes"
    encoded = base64.b64encode(raw_bytes).decode("ascii")
    _patch_partition_pdf(
        monkeypatch,
        [FakeUnstructuredElement("Image", "", FakeMetadata(image_base64=encoded))],
    )
    elements = parse_pdf(Path("doc.pdf"))
    assert elements[0].type == ElementType.IMAGE
    assert elements[0].image_bytes == raw_bytes
    assert elements[0].description == f"a description of {len(raw_bytes)} bytes"


def test_image_without_extracted_bytes_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_partition_pdf(
        monkeypatch,
        [
            FakeUnstructuredElement("Image", "", FakeMetadata(image_base64=None)),
            FakeUnstructuredElement("NarrativeText", "Real text.", FakeMetadata()),
        ],
    )
    elements = parse_pdf(Path("doc.pdf"))
    assert len(elements) == 1
    assert elements[0].type == ElementType.PARAGRAPH


def test_page_numbers_are_carried_through(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_partition_pdf(
        monkeypatch,
        [
            FakeUnstructuredElement("Title", "Page 1 title", FakeMetadata(page_number=1)),
            FakeUnstructuredElement("Title", "Page 2 title", FakeMetadata(page_number=2)),
        ],
    )
    elements = parse_pdf(Path("doc.pdf"))
    assert [el.metadata.page for el in elements] == [1, 2]


def test_pdf_falls_back_to_fast_strategy_if_hi_res_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_elements = [
        FakeUnstructuredElement("NarrativeText", "Fallback text.", FakeMetadata(page_number=1))
    ]
    calls: list[dict[str, Any]] = []

    def _fake_partition_pdf(**kwargs: Any) -> list[Any]:
        calls.append(kwargs)
        if kwargs.get("strategy") == "hi_res":
            raise RuntimeError("hi_res unavailable")
        return fallback_elements

    monkeypatch.setattr("multimodal_rag.ingestion.pdf.partition_pdf", _fake_partition_pdf)

    elements = parse_pdf(Path("doc.pdf"))

    assert len(elements) == 1
    assert elements[0].text == "Fallback text."
    assert calls[0]["strategy"] == "hi_res"
    assert calls[1]["strategy"] == "fast"


def test_pdf_hi_res_fallback_emits_actionable_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_elements = [
        FakeUnstructuredElement("NarrativeText", "Fallback text.", FakeMetadata(page_number=1))
    ]

    def _fake_partition_pdf(**kwargs: Any) -> list[Any]:
        if kwargs.get("strategy") == "hi_res":
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        return fallback_elements

    monkeypatch.setattr("multimodal_rag.ingestion.pdf.partition_pdf", _fake_partition_pdf)
    monkeypatch.setattr("multimodal_rag.ingestion.pdf.shutil.which", lambda _name: None)

    with pytest.warns(RuntimeWarning, match="PDF hi_res parsing failed") as warns:
        elements = parse_pdf(Path("doc.pdf"))

    assert len(elements) == 1
    msg = str(warns.list[0].message)
    assert "tesseract not found on PATH" in msg
    assert "poppler (pdftoppm) not found on PATH" in msg
    assert "table/image quality may degrade" in msg


def test_pdf_uses_local_hi_res_model_directory_when_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_dir = tmp_path / ".model-cache" / "unstructuredio" / "yolo_x_layout"
    local_dir.mkdir(parents=True)
    (local_dir / "yolox_l0.05.onnx").write_bytes(b"fake")

    captured: list[dict[str, Any]] = []

    def _fake_partition_pdf(**kwargs: Any) -> list[Any]:
        captured.append(kwargs)
        return [FakeUnstructuredElement("NarrativeText", "ok", FakeMetadata())]

    monkeypatch.setattr("multimodal_rag.ingestion.pdf.partition_pdf", _fake_partition_pdf)
    monkeypatch.setattr(
        "multimodal_rag.ingestion.pdf._LOCAL_HI_RES_MODEL_CANDIDATES",
        [local_dir / "yolox_l0.05.onnx"],
    )
    monkeypatch.setattr(
        "multimodal_rag.ingestion.pdf._LOCAL_HI_RES_MODEL_NAME",
        "test_local_yolox",
    )

    elements = parse_pdf(Path("doc.pdf"))

    assert elements[0].text == "ok"
    assert captured[0]["hi_res_model_name"] == "test_local_yolox"
