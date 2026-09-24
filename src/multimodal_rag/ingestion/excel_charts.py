"""Native Excel chart extraction -- a genuinely different problem from
embedded pictures (excel.py). openpyxl can CREATE charts but cannot read
an existing chart's definition at all, confirmed via its own
documentation, not assumed: reading `worksheet._charts` on a loaded file
returns nothing, since openpyxl's chart classes are built for writing.

So this reads a chart's raw XML directly instead: a .xlsx is a ZIP of
OOXML parts, and a chart's own definition lives at
xl/charts/chartN.xml, in the standard, documented chart schema --
readable via stdlib zipfile + ElementTree, no new dependency.

Describes what the chart's DATA says (type, title, series, referenced
cell ranges -- resolved against the sheet's own already-loaded values)
via a text-LLM summary, not what it visually looks like (colors, manual
annotations aren't part of the data model at all). That's the right
fidelity for "what insight does this communicate," not a substitute for
rendering it as an actual image. Considered and rejected: commercial
rendering libraries (real free-tier limits on Excel-to-image conversion)
and LibreOffice headless conversion (no confirmed way to export one
isolated chart rather than a whole rendered sheet, and a full
office-suite runtime dependency) -- see docs/technical-decisions.md.

Scoped to the common "category axis" chart shape (bar/column/line/area/
pie -- all share the same <cat>/<val>/<tx> series structure). Other
types (scatter, bubble, stock, surface, radar -- which reference cells
under different tag names, e.g. <xVal>/<yVal>) fall back to a minimal
type+title description rather than being silently dropped or crashing.
"""

import posixpath
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from openpyxl.utils.cell import range_boundaries
from openpyxl.worksheet.worksheet import Worksheet

from ..providers.factory import get_llm

_CHART_NS = "http://schemas.openxmlformats.org/drawingml/2006/chart"
_DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_DRAWING_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing"
_CHART_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart"

_CATEGORY_AXIS_CHART_TYPES = {
    "barChart",
    "bar3DChart",
    "lineChart",
    "line3DChart",
    "areaChart",
    "area3DChart",
    "pieChart",
    "pie3DChart",
    "doughnutChart",
    "radarChart",
    "ofPieChart",
}
_OTHER_CHART_TYPES = {
    "scatterChart",
    "bubbleChart",
    "stockChart",
    "surfaceChart",
    "surface3DChart",
}


def charts_for_worksheet(path: Path, worksheet: Worksheet) -> list[str]:
    """Chart part names (e.g. "xl/charts/chart1.xml") anchored to this
    specific worksheet -- charts aren't named per-sheet, so this follows
    the real OOXML relationship chain: worksheet -> its drawing part
    (via worksheet._rels, openpyxl's own private but already-resolved
    link -- no need to hand-parse workbook.xml's sheet-name-to-part-name
    mapping) -> that drawing's own _rels file -> each chart it embeds.
    """
    drawing_targets = [
        rel.Target for rel in worksheet._rels if getattr(rel, "Type", None) == _DRAWING_REL_TYPE
    ]
    if not drawing_targets:
        return []

    chart_paths: list[str] = []
    with zipfile.ZipFile(path) as zf:
        for drawing_target in drawing_targets:
            drawing_path = drawing_target.lstrip("/")
            drawing_dir, drawing_name = drawing_path.rsplit("/", 1)
            rels_path = f"{drawing_dir}/_rels/{drawing_name}.rels"
            if rels_path not in zf.namelist():
                continue
            root = ET.fromstring(zf.read(rels_path))
            for rel in root.findall(f"{{{_REL_NS}}}Relationship"):
                if rel.get("Type") == _CHART_REL_TYPE:
                    target = rel.get("Target")
                    if target:
                        chart_paths.append(_resolve_part_path(drawing_dir, target))
    return chart_paths


def _resolve_part_path(source_dir: str, target: str) -> str:
    """An OPC relationship Target is either package-absolute
    ("/xl/charts/chart1.xml") or -- the spec-correct, and far more
    common, form -- relative to the SOURCE part's own directory (e.g.
    "../charts/chart1.xml" from "xl/drawings/"), never relative to the
    zip archive's flat root. The old code did `target.lstrip("/")`
    unconditionally, which only happens to produce a real archive name
    for an absolute target; every relative target silently became a
    literal, nonexistent entry name ("../charts/chart1.xml") and
    zipfile raised KeyError. That went unnoticed because openpyxl's own
    writer -- the only thing this module's tests ever built workbooks
    with -- always emits absolute targets; real Excel/LibreOffice write
    relative ones. See test_excel_charts.py for the regression built
    from an actual Excel-written relative target."""
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join(source_dir, target))


def describe_chart(path: Path, chart_xml_path: str, worksheet: Worksheet) -> str | None:
    """Returns an LLM-written description of one chart, or None if the
    LLM is unconfigured, the call fails, or the chart's definition
    couldn't be parsed at all -- a chart description is a nice-to-have,
    same fail-soft contract as ingestion/tables.py's summarize_table().
    """
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read(chart_xml_path))

    title = _extract_title(root)
    type_element = _find_chart_type_element(root)
    if type_element is None:
        return None
    chart_type = _local_name(type_element.tag)

    if chart_type not in _CATEGORY_AXIS_CHART_TYPES:
        return _summarize(_fallback_description(chart_type, title))

    series_lines = [
        line
        for ser in type_element.findall(f"{{{_CHART_NS}}}ser")
        if (line := _describe_series(ser, worksheet)) is not None
    ]
    if not series_lines:
        return _summarize(_fallback_description(chart_type, title))

    lines = [f"Chart type: {chart_type}"]
    if title:
        lines.append(f"Title: {title}")
    lines.extend(series_lines)
    return _summarize("\n".join(lines))


def _fallback_description(chart_type: str, title: str | None) -> str:
    lines = [f"Chart type: {chart_type}"]
    if title:
        lines.append(f"Title: {title}")
    lines.append("(Underlying data series could not be parsed for this chart type.)")
    return "\n".join(lines)


def _extract_title(root: ET.Element) -> str | None:
    title_element = root.find(f".//{{{_CHART_NS}}}title")
    if title_element is None:
        return None
    texts = [t.text for t in title_element.findall(f".//{{{_DRAWING_NS}}}t") if t.text]
    return "".join(texts) if texts else None


def _find_chart_type_element(root: ET.Element) -> ET.Element | None:
    plot_area = root.find(f".//{{{_CHART_NS}}}plotArea")
    if plot_area is None:
        return None
    for child in plot_area:
        if _local_name(child.tag) in _CATEGORY_AXIS_CHART_TYPES | _OTHER_CHART_TYPES:
            return child
    return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _describe_series(ser: ET.Element, worksheet: Worksheet) -> str | None:
    name = _resolve_single_ref(ser, "tx", worksheet)
    values = _resolve_range_ref(ser, "val", worksheet)
    if not values:
        return None
    categories = _resolve_range_ref(ser, "cat", worksheet)
    label = name or "series"
    if categories and len(categories) == len(values):
        pairs = ", ".join(f"{c}: {v}" for c, v in zip(categories, values, strict=True))
        return f"{label} -- {pairs}"
    return f"{label} -- {', '.join(values)}"


def _resolve_single_ref(ser: ET.Element, tag: str, worksheet: Worksheet) -> str | None:
    values = _resolve_ref(ser, tag, worksheet)
    return values[0] if values else None


def _resolve_range_ref(ser: ET.Element, tag: str, worksheet: Worksheet) -> list[str] | None:
    return _resolve_ref(ser, tag, worksheet)


def _resolve_ref(ser: ET.Element, tag: str, worksheet: Worksheet) -> list[str] | None:
    container = ser.find(f"{{{_CHART_NS}}}{tag}")
    if container is None:
        return None
    formula = container.find(f".//{{{_CHART_NS}}}f")
    if formula is None or not formula.text:
        return None
    return _resolve_range(formula.text, worksheet)


def _resolve_range(reference: str, worksheet: Worksheet) -> list[str]:
    # A reference looks like "'Sheet1'!$B$2:$B$4" -- the sheet name
    # prefix is dropped rather than validated, since `worksheet` is
    # already the specific sheet this chart part belongs to (a chart
    # referencing a DIFFERENT sheet's data isn't something this project
    # needs to handle).
    cell_range = reference.split("!", 1)[-1]
    try:
        min_col, min_row, max_col, max_row = range_boundaries(cell_range)
    except ValueError:
        return []
    values: list[str] = []
    for row in worksheet.iter_rows(
        min_row=min_row, max_row=max_row, min_col=min_col, max_col=max_col, values_only=True
    ):
        values.extend(str(value) for value in row if value is not None)
    return values


def _summarize(chart_description: str) -> str | None:
    try:
        llm = get_llm()
    except Exception:
        return None
    try:
        return llm.generate(
            [
                {
                    "role": "user",
                    "content": (
                        "Summarize what this chart shows in one or two short sentences, for "
                        "someone deciding whether to look at it in full. Here is the chart's "
                        "underlying data, not a picture of it:\n\n" + chart_description
                    ),
                }
            ]
        )
    except Exception:
        return None
