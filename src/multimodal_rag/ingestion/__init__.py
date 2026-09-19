"""Dispatcher: routes a document to the right format-specific parser.

File type is detected from actual file content via libmagic, not just the
extension — a renamed or extensionless PDF/DOCX/PPTX still routes
correctly. Markdown is the one exception: plain text has no distinguishing
magic bytes, so libmagic can only ever report "text/plain" for it, and we
fall back to the file extension in that one case.
"""

import zipfile
from pathlib import Path

import magic

from .csv_ import parse_csv
from .docx import parse_docx
from .markdown import parse_markdown
from .pdf import parse_pdf
from .pptx import parse_pptx
from .schema import Element

# Lambdas, not the parser functions themselves -- a dict literal captures
# function objects once, at import time, which would silently ignore a
# test's monkeypatch.setattr("multimodal_rag.ingestion.parse_pptx", fake)
# (that only reassigns the module attribute, not these dicts' stale
# references). A lambda's body does a free-variable lookup against the
# module's current globals at CALL time instead, so it honors monkeypatching
# -- the same reason _parser_from_content_signature's `return parse_docx`
# already does.
_MIME_PARSERS = {
    "application/pdf": lambda path, summarize_tables=False: parse_pdf(path, summarize_tables),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        lambda path, summarize_tables=False: parse_docx(path, summarize_tables)
    ),
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": (
        lambda path, summarize_tables=False: parse_pptx(path, summarize_tables)
    ),
    # Some libmagic builds do report this distinctly for well-formed CSV;
    # most report the plain "text/plain" ambiguous case instead, handled
    # via the extension fallback below.
    "text/csv": lambda path, summarize_tables=False: parse_csv(path, summarize_tables),
}

_EXTENSION_PARSERS = {
    ".pdf": lambda path, summarize_tables=False: parse_pdf(path, summarize_tables),
    ".docx": lambda path, summarize_tables=False: parse_docx(path, summarize_tables),
    ".pptx": lambda path, summarize_tables=False: parse_pptx(path, summarize_tables),
    ".md": lambda path, summarize_tables=False: parse_markdown(path, summarize_tables),
    ".markdown": lambda path, summarize_tables=False: parse_markdown(path, summarize_tables),
    ".csv": lambda path, summarize_tables=False: parse_csv(path, summarize_tables),
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
            "(supported: PDF, DOCX, PPTX, Markdown, CSV)"
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
