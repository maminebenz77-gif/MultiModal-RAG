from pathlib import Path

import openpyxl
import pytest

from multimodal_rag.ingestion.excel import parse_excel
from multimodal_rag.ingestion.schema import ElementType


def _write_workbook(path: Path, sheets: dict[str, list[list[object]]]) -> Path:
    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    for name, rows in sheets.items():
        worksheet = workbook.create_sheet(name)
        for row in rows:
            worksheet.append(row)
    workbook.save(path)
    return path


def test_parses_a_single_sheet_into_a_table_element(tmp_path: Path) -> None:
    path = _write_workbook(
        tmp_path / "wb.xlsx",
        {"Sheet1": [["Region", "Revenue"], ["North", 120], ["South", 340]]},
    )
    elements = parse_excel(path)

    assert len(elements) == 1
    assert elements[0].type == ElementType.TABLE
    assert elements[0].text is not None
    assert "Region" in elements[0].text
    assert "North" in elements[0].text


def test_sheet_name_is_recorded_on_metadata(tmp_path: Path) -> None:
    path = _write_workbook(tmp_path / "wb.xlsx", {"Revenue": [["A"], ["1"]]})
    elements = parse_excel(path)
    assert elements[0].metadata.sheet == "Revenue"


def test_multiple_sheets_each_produce_their_own_elements(tmp_path: Path) -> None:
    path = _write_workbook(
        tmp_path / "wb.xlsx",
        {
            "Revenue": [["Region", "Revenue"], ["North", 120]],
            "Headcount": [["Team", "Count"], ["Eng", 12]],
        },
    )
    elements = parse_excel(path)

    sheets = [el.metadata.sheet for el in elements]
    assert sheets == ["Revenue", "Headcount"]
    assert [el.metadata.position for el in elements] == [0, 1]


def test_blank_rows_are_dropped(tmp_path: Path) -> None:
    path = _write_workbook(
        tmp_path / "wb.xlsx",
        {"Sheet1": [["A", "B"], ["1", "2"], [None, None], ["3", "4"]]},
    )
    elements = parse_excel(path)
    assert elements[0].text is not None
    lines = elements[0].text.splitlines()
    assert len(lines) == 4  # header + separator + 2 data rows, no blank row
    assert "None" not in elements[0].text


def test_empty_sheet_produces_no_elements(tmp_path: Path) -> None:
    path = _write_workbook(tmp_path / "wb.xlsx", {"Empty": []})
    assert parse_excel(path) == []


def test_summarize_tables_flag_calls_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "multimodal_rag.ingestion.tabular.summarize_table", lambda markdown_table: "a summary"
    )
    path = _write_workbook(tmp_path / "wb.xlsx", {"Sheet1": [["A"], ["1"]]})
    elements = parse_excel(path, summarize_tables=True)
    assert elements[0].table_summary == "a summary"


def test_many_rows_are_batched_across_multiple_elements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from multimodal_rag.ingestion import tabular

    monkeypatch.setattr(tabular, "ROWS_PER_ELEMENT", 10)
    rows = [["A", "B"]] + [[i, i * 2] for i in range(25)]
    path = _write_workbook(tmp_path / "wb.xlsx", {"Sheet1": rows})

    elements = parse_excel(path)

    assert len(elements) == 3  # 25 data rows / 10 per batch -> 10, 10, 5
    assert all(el.metadata.sheet == "Sheet1" for el in elements)
