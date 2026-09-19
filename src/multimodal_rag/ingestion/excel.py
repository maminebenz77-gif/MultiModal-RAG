"""Excel (.xlsx) parsing -- see tabular.py for the shared row-batching
logic this shares with csv_.py. Deliberately .xlsx only: legacy .xls
needs a different, largely unmaintained library (xlrd), and openpyxl
itself dropped .xls support entirely.

Loaded without read_only=True on purpose: read-only mode drops access to
embedded pictures (openpyxl's own documented limitation), which this
parser also extracts, via the same vision-description pipeline
docx.py/pptx.py already use -- generally where the actual insight in a
real spreadsheet lives, in a chart someone pasted in as a picture, not
just the raw numbers.

Known gap, same shape as docx.py's: a native Excel CHART (built from
cell data, editable, as opposed to a pasted-in raster image of one) is
NOT extracted here -- openpyxl can't read an existing chart's definition
at all (its chart classes are for creating new ones, not parsing
existing files), so there's a separate, later pass for that (see
excel_charts.py) that reads the chart's raw XML directly instead of
going through openpyxl.
"""

from pathlib import Path

import openpyxl

from .schema import Element, ElementMetadata, ElementType
from .tabular import rows_to_elements
from .vision import ImageDescriber


def parse_excel(path: Path, summarize_tables: bool = False) -> list[Element]:
    # data_only=True reads a formula cell's last-calculated VALUE (what
    # Excel itself cached the last time the file was saved), not the
    # formula string -- "42", not "=SUM(A1:A10)". The value is what's
    # worth citing or matching against a query; the formula isn't.
    workbook = openpyxl.load_workbook(path, data_only=True)
    describer = ImageDescriber()

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
        if rows:
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

        # worksheet._images / image._data() are openpyxl's own PRIVATE
        # API -- there's no public accessor for embedded pictures at all
        # (confirmed: openpyxl's chart/image classes are built for
        # writing, not reading existing files). Could break silently on
        # an openpyxl upgrade; worth re-checking if one ever changes
        # this parser's behavior unexpectedly.
        for image in worksheet._images:
            image_bytes = image._data()
            description, status = describer.describe(image_bytes)
            elements.append(
                Element(
                    type=ElementType.IMAGE,
                    image_bytes=image_bytes,
                    description=description,
                    description_status=status,
                    metadata=ElementMetadata(
                        source_file=str(path), sheet=worksheet.title, position=position
                    ),
                )
            )
            position += 1

    return elements


def _cell_to_str(value: object) -> str:
    return "" if value is None else str(value)
