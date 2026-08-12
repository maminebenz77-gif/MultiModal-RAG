"""Dispatcher: routes a document to the right format-specific parser.

File type is detected from actual file content via libmagic, not just the
extension — a renamed or extensionless PDF/DOCX/PPTX still routes
correctly. Markdown is the one exception: plain text has no distinguishing
magic bytes, so libmagic can only ever report "text/plain" for it, and we
fall back to the file extension in that one case.
"""

from pathlib import Path
import zipfile

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
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".pptx": parse_pptx,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
}

_AMBIGUOUS_MIME_TYPES = {
    "text/plain",
    "application/octet-stream",
    "application/zip",
}


def parse_document(path: Path, summarize_tables: bool = False) -> list[Element]:
    mime_type = magic.from_file(str(path), mime=True)

    parser = _MIME_PARSERS.get(mime_type)
    if parser is None and mime_type in _AMBIGUOUS_MIME_TYPES:
        parser = _EXTENSION_PARSERS.get(path.suffix.lower())
    if parser is None and mime_type in _AMBIGUOUS_MIME_TYPES:
        parser = _parser_from_content_signature(path)

    if parser is None:
        raise ValueError(
            f"Unsupported file type for {path}: detected MIME type {mime_type!r} "
            "(supported: PDF, DOCX, PPTX, Markdown)"
        )

    return parser(path, summarize_tables=summarize_tables)


def _parser_from_content_signature(path: Path):
    # PDF magic bytes, independent of filename extension.
    try:
        if path.read_bytes().startswith(b"%PDF"):
            return parse_pdf
    except OSError:
        return None

    # DOCX/PPTX are ZIP containers; detect by canonical internal paths.
    try:
        if not zipfile.is_zipfile(path):
            return None
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
    except (OSError, zipfile.BadZipFile):
        return None

    if "word/document.xml" in names:
        return parse_docx
    if "ppt/presentation.xml" in names:
        return parse_pptx
    return None
