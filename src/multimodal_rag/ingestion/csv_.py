"""CSV parsing -- see tabular.py for the shared row-batching logic this
shares with excel.py. Named csv_.py (trailing underscore) so it doesn't
shadow the stdlib `csv` module it imports.

No new dependency: Python's stdlib `csv` module is sufficient to read a
CSV's rows; there's no chart/image/multi-sheet concept for this format,
unlike Excel.
"""

import csv
from pathlib import Path

from .schema import Element
from .tabular import rows_to_elements


def parse_csv(path: Path, summarize_tables: bool = False) -> list[Element]:
    # utf-8-sig strips a BOM if present (common from spreadsheet
    # exports) without affecting plain UTF-8 files, which have none.
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))

    if not rows:
        return []

    header, *data_rows = rows
    return rows_to_elements(
        header,
        data_rows,
        source_file=str(path),
        sheet=None,
        start_position=0,
        summarize_tables=summarize_tables,
    )
