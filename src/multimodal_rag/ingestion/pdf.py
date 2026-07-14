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
from typing import Any

from bs4 import BeautifulSoup
from unstructured.partition.pdf import partition_pdf

from .schema import Element, ElementMetadata, ElementType
from .tables import rows_to_markdown, summarize_table
from .vision import ImageDescriber

_TYPE_MAP = {
    "Title": ElementType.TITLE,
    "NarrativeText": ElementType.PARAGRAPH,
    "UncategorizedText": ElementType.PARAGRAPH,
    "ListItem": ElementType.PARAGRAPH,
}


def parse_pdf(path: Path, summarize_tables: bool = False) -> list[Element]:
    raw_elements = partition_pdf(
        filename=str(path),
        strategy="hi_res",
        infer_table_structure=True,
        extract_images_in_pdf=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )
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
