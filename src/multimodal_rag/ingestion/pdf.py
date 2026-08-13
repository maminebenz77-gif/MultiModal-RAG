"""PDF parser: unstructured (poppler for rendering, tesseract for OCR) ->
common Element schema.

Unlike DOCX/PPTX, a PDF has no native structural markup — it's just
positioned glyphs on a page. unstructured's hi_res strategy renders each
page (via poppler) and runs a layout-detection model to classify regions
as Title/NarrativeText/Table/Image, then OCRs image regions with
tesseract. This is the one parser in the pipeline that's genuinely
*inferring* structure rather than reading it that was already there, so
it's also the one most likely to misclassify things — expect rougher
edges here than in the DOCX/PPTX parsers (e.g. table header rows
misdetected, OCR character errors in extracted image text).

PDF has no native chart signal (unlike PPTX) — every embedded graphic,
chart or otherwise, comes back as a generic "Image" element.
"""

import base64
from pathlib import Path
import shutil
from typing import Any
import warnings

from bs4 import BeautifulSoup
from unstructured.partition.pdf import partition_pdf
from unstructured_inference.models.base import register_new_model
from unstructured_inference.models.yolox import YOLOX_LABEL_MAP, UnstructuredYoloXModel

from .schema import Element, ElementMetadata, ElementType
from .tables import rows_to_markdown, summarize_table
from .vision import ImageDescriber

_TYPE_MAP = {
    "Title": ElementType.TITLE,
    "NarrativeText": ElementType.PARAGRAPH,
    "UncategorizedText": ElementType.PARAGRAPH,
    "ListItem": ElementType.PARAGRAPH,
}

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_HI_RES_MODEL_CANDIDATES = [
    _PROJECT_ROOT / ".model-cache" / "unstructuredio" / "yolo_x_layout" / "yolox_l0.05.onnx",
    _PROJECT_ROOT / ".model-cacheunstructuredioyolo_x_layout" / "yolox_l0.05.onnx",
]
_LOCAL_HI_RES_MODEL_NAME = "multimodal_rag_local_yolox"


def parse_pdf(path: Path, summarize_tables: bool = False) -> list[Element]:
    raw_elements = _partition_pdf_with_fallback(path)
    describer = ImageDescriber()

    elements: list[Element] = []
    for position, raw in enumerate(raw_elements):
        metadata = ElementMetadata(
            source_file=str(path), page=raw.metadata.page_number, position=position
        )

        if raw.category == "Table":
            markdown_table = _table_to_markdown(raw)
            summary = summarize_table(markdown_table) if summarize_tables else None
            elements.append(
                Element(
                    type=ElementType.TABLE,
                    text=markdown_table,
                    table_summary=summary,
                    metadata=metadata,
                )
            )
            continue

        if raw.category == "Image":
            if raw.metadata.image_base64 is None:
                continue  # layout model flagged an image region but extraction failed
            image_bytes = base64.b64decode(raw.metadata.image_base64)
            description, status = describer.describe(image_bytes)
            elements.append(
                Element(
                    type=ElementType.IMAGE,
                    image_bytes=image_bytes,
                    description=description,
                    description_status=status,
                    metadata=metadata,
                )
            )
            continue

        text = (raw.text or "").strip()
        if not text:
            continue
        elements.append(
            Element(
                type=_TYPE_MAP.get(raw.category, ElementType.PARAGRAPH),
                text=text,
                metadata=metadata,
            )
        )

    return elements


def _partition_pdf_with_fallback(path: Path) -> list[Any]:
    try:
        hi_res_kwargs = {}
        if local_model_name := _local_hi_res_model_name():
            hi_res_kwargs["hi_res_model_name"] = local_model_name
        return partition_pdf(
            filename=str(path),
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            **hi_res_kwargs,
        )
    except Exception as exc:
        # Some environments cannot download/initialize the hi_res layout
        # model (e.g. restricted network). Fall back to text-first parsing
        # instead of failing the entire ingest request.
        warnings.warn(_hi_res_fallback_message(exc), RuntimeWarning, stacklevel=2)
        return partition_pdf(
            filename=str(path),
            strategy="fast",
            infer_table_structure=False,
            extract_images_in_pdf=False,
        )


def _local_hi_res_model_path() -> Path | None:
    for candidate in _LOCAL_HI_RES_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _local_hi_res_model_name() -> str | None:
    local_model_path = _local_hi_res_model_path()
    if local_model_path is None:
        return None

    register_new_model(
        {
            _LOCAL_HI_RES_MODEL_NAME: {
                "model_path": str(local_model_path),
                "label_map": YOLOX_LABEL_MAP,
            }
        },
        UnstructuredYoloXModel,
    )
    return _LOCAL_HI_RES_MODEL_NAME


def _hi_res_fallback_message(exc: Exception) -> str:
    hints: list[str] = []

    if shutil.which("tesseract") is None:
        hints.append("tesseract not found on PATH")
    if shutil.which("pdftoppm") is None:
        hints.append("poppler (pdftoppm) not found on PATH")
    if _local_hi_res_model_path() is None:
        hints.append("local YOLOX hi_res model not found in .model-cache")

    message = (
        "PDF hi_res parsing failed; falling back to fast mode (table/image quality may degrade). "
        f"Cause: {type(exc).__name__}: {exc}."
    )
    if hints:
        message += " Environment checks: " + ", ".join(hints) + "."
    message += (
        " If your network uses custom TLS/proxy, ensure model downloads for "
        "unstructured layout detection are reachable and trusted."
    )
    return message


def _table_to_markdown(raw: Any) -> str:
    html = getattr(raw.metadata, "text_as_html", None)
    if not html:
        # The layout model didn't produce structured HTML for this table
        # (can happen for irregular tables) — fall back to flattened text.
        return raw.text or ""
    soup = BeautifulSoup(html, "html.parser")
    rows = [
        [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
        for tr in soup.find_all("tr")
    ]
    return rows_to_markdown(rows)
