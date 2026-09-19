"""Dispatcher: routes a document to the right format-specific parser.

File type is detected from actual file content via libmagic, not just the
extension — a renamed or extensionless PDF/DOCX/PPTX still routes
correctly. Markdown is the one exception: plain text has no distinguishing
magic bytes, so libmagic can only ever report "text/plain" for it, and we
fall back to the file extension in that one case.

parse_document() is also the one place every caller of every parser goes
through -- the real /ingest endpoint, run_eval.py, run_expert_eval.py, or
a one-off script alike -- so it's where ingestion gets wrapped in a
Langfuse span, same "instrument at the source, not reactively from one
caller" choice retrieval/retriever.py already made. Every LLM call a
parser triggers while it runs (a table's summarize_table(), a chart's
describe_chart(), a picture's vision describe() -- all already traced
individually, see providers/llm.py and providers/vision.py) nests under
this span automatically via OpenTelemetry context propagation, so opening
one document's ingestion trace in Langfuse shows every summary/
description made while parsing it, instead of each becoming its own
disconnected root trace with nothing tying them to the document or to
each other.
"""

import zipfile
from pathlib import Path

import magic

from ..tracing import traced_span, update_span_output
from .csv_ import parse_csv
from .docx import parse_docx
from .excel import parse_excel
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
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": (
        lambda path, summarize_tables=False: parse_excel(path, summarize_tables)
    ),
}

_EXTENSION_PARSERS = {
    ".pdf": lambda path, summarize_tables=False: parse_pdf(path, summarize_tables),
    ".docx": lambda path, summarize_tables=False: parse_docx(path, summarize_tables),
    ".pptx": lambda path, summarize_tables=False: parse_pptx(path, summarize_tables),
    ".md": lambda path, summarize_tables=False: parse_markdown(path, summarize_tables),
    ".markdown": lambda path, summarize_tables=False: parse_markdown(path, summarize_tables),
    ".csv": lambda path, summarize_tables=False: parse_csv(path, summarize_tables),
    ".xlsx": lambda path, summarize_tables=False: parse_excel(path, summarize_tables),
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
            "(supported: PDF, DOCX, PPTX, Markdown, CSV, Excel)"
        )

    with traced_span("ingest", input=str(path), metadata={"mime_type": mime_type}) as span:
        elements = parser(path, summarize_tables=summarize_tables)
        element_counts: dict[str, int] = {}
        for element in elements:
            element_counts[element.type.value] = element_counts.get(element.type.value, 0) + 1
        update_span_output(span, {"element_count": len(elements), "by_type": element_counts})
        return elements


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
    if "xl/workbook.xml" in names:
        return parse_excel
    return None
