"""Dispatcher: routes a document to the right format-specific parser.

File type is detected from actual file content via libmagic, not just the
extension — a renamed or extensionless PDF/DOCX/PPTX still routes
correctly. Markdown is the one exception: plain text has no distinguishing
magic bytes, so libmagic can only ever report "text/plain" for it, and we
fall back to the file extension in that one case.
"""

from pathlib import Path

import magic

from .docx import parse_docx
from .markdown import parse_markdown
from .pdf import parse_pdf
from .pptx import parse_pptx
from .schema import Element

_MIME_PARSERS = {
    "application/pdf": parse_pdf,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": parse_docx,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": parse_pptx,
}

_EXTENSION_PARSERS = {
    ".md": parse_markdown,
    ".markdown": parse_markdown,
}


def parse_document(path: Path, summarize_tables: bool = False) -> list[Element]:
    mime_type = magic.from_file(str(path), mime=True)

    parser = _MIME_PARSERS.get(mime_type)
    if parser is None and mime_type == "text/plain":
        parser = _EXTENSION_PARSERS.get(path.suffix.lower())

    if parser is None:
        raise ValueError(
            f"Unsupported file type for {path}: detected MIME type {mime_type!r} "
            "(supported: PDF, DOCX, PPTX, Markdown)"
        )

    return parser(path, summarize_tables=summarize_tables)
