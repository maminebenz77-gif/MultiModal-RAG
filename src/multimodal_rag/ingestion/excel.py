"""Excel (.xlsx) parsing -- see tabular.py for the shared row-batching
logic this shares with csv_.py. Deliberately .xlsx only: legacy .xls
needs a different, largely unmaintained library (xlrd), and openpyxl
itself dropped .xls support entirely.

Loaded without read_only=True on purpose, even though this file only
reads row data today -- read-only mode drops access to embedded images,
which a later pass over this same parser needs (openpyxl's own
documented limitation), and reusing one workbook load for both is worth
more than read-only mode's memory savings for files this project's scale
actually sees.
"""

from pathlib import Path

import openpyxl

from .schema import Element
from .tabular import rows_to_elements


def parse_excel(path: Path, summarize_tables: bool = False) -> list[Element]:
    # data_only=True reads a formula cell's last-calculated VALUE (what
    # Excel itself cached the last time the file was saved), not the
    # formula string -- "42", not "=SUM(A1:A10)". The value is what's
    # worth citing or matching against a query; the formula isn't.
    workbook = openpyxl.load_workbook(path, data_only=True)

    elements: list[Element] = []
    position = 0
    for worksheet in workbook.worksheets:
        rows = [
            [_cell_to_str(value) for value in row] for row in worksheet.iter_rows(values_only=True)
        ]
        # Fully-blank rows are common in real spreadsheets (formatting
        # artifacts, deleted rows) -- dropped so they don't pollute every
        # batch's rendering. Doesn't attempt to detect more than one
        # logical table on a sheet separated by a blank row -- same
        # "don't guess at structure" choice as the header row (see
        # tabular.py).
        rows = [row for row in rows if any(row)]
        if not rows:
            continue

        header, *data_rows = rows
        sheet_elements = rows_to_elements(
            header,
            data_rows,
            source_file=str(path),
            sheet=worksheet.title,
            start_position=position,
            summarize_tables=summarize_tables,
        )
        elements.extend(sheet_elements)
        position += len(sheet_elements)

    return elements


def _cell_to_str(value: object) -> str:
    return "" if value is None else str(value)
