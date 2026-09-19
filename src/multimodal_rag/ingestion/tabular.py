"""Shared row-batching for row-oriented tabular sources (CSV, Excel) --
distinct from ingestion/tables.py's whole-table-as-one-Element handling,
which fits a document's small, hand-authored table but breaks down at
real spreadsheet sizes: a single Element holding thousands of rows would
either blow past a reasonable LLM summary call, or get mid-row-sliced by
the character-based chunker downstream with no idea where a row boundary
is -- misaligned columns and no header, useless for retrieval.

Each batch of ROWS_PER_ELEMENT rows becomes its own Element instead --
self-contained (the header is repeated in every batch), never split
mid-row by construction, and already close to a reasonable chunk size
before chunking even runs. Deliberately does NOT try to detect a "real"
header row among leading title/note rows some spreadsheets have --
that's unreliable to guess in general, so every batch is rendered as-is
(whatever rows are actually in it) and the LLM summary (see
ingestion/tables.py's summarize_table, reused here) is what's relied on
to describe what a batch actually contains, not a hard-coded structural
assumption.
"""

from .schema import Element, ElementMetadata, ElementType
from .tables import rows_to_markdown, summarize_table

ROWS_PER_ELEMENT = 30
"""Rows per batch/Element. Same "small vs large chunk" tension as prose
chunking (see chunking_strategy_notes.md): fewer, larger batches give a
retrieval hit more surrounding rows of context but dilute how precisely
a query matches one row; more, smaller batches are the opposite. Picked
as a reasonable middle ground, not derived from measurement the way
child_chunk_size (chunking/parent_child.py) was -- no golden-set-driven
tuning has been done for tabular data yet."""


def rows_to_elements(
    header: list[str],
    data_rows: list[list[str]],
    *,
    source_file: str,
    sheet: str | None,
    start_position: int,
    summarize_tables: bool,
) -> list[Element]:
    """Batches data_rows into ROWS_PER_ELEMENT-row groups, each rendered as
    its own TABLE Element with `header` repeated at the top of every batch
    -- so a batch from anywhere in a large sheet is independently
    self-describing, not just the first one.
    """
    elements: list[Element] = []
    for batch_index, batch_start in enumerate(range(0, len(data_rows), ROWS_PER_ELEMENT)):
        batch = data_rows[batch_start : batch_start + ROWS_PER_ELEMENT]
        markdown_table = rows_to_markdown([header, *batch])
        summary = summarize_table(markdown_table) if summarize_tables else None
        elements.append(
            Element(
                type=ElementType.TABLE,
                text=markdown_table,
                table_summary=summary,
                metadata=ElementMetadata(
                    source_file=source_file,
                    sheet=sheet,
                    position=start_position + batch_index,
                ),
            )
        )
    return elements
