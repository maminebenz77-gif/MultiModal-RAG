from pathlib import Path

import openpyxl
import pytest
from openpyxl.chart import BarChart, Reference

from multimodal_rag.ingestion import excel_charts
from multimodal_rag.ingestion.excel_charts import charts_for_worksheet, describe_chart


def _write_bar_chart_workbook(path: Path) -> Path:
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "Sheet1"
    worksheet.append(["Region", "Revenue"])
    worksheet.append(["North", 120])
    worksheet.append(["South", 340])
    worksheet.append(["East", 275])

    chart = BarChart()
    chart.type = "col"
    chart.title = "Q3 Revenue by Region"
    data = Reference(worksheet, min_col=2, min_row=1, max_row=4)
    categories = Reference(worksheet, min_col=1, min_row=2, max_row=4)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(categories)
    worksheet.add_chart(chart, "D2")

    workbook.save(path)
    return path


@pytest.fixture
def _stub_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    # Not autouse -- test_summarize_returns_none_when_* need the REAL
    # _summarize. Every describe_chart test that only cares about correct
    # XML extraction/resolution (not the LLM call itself) requests this
    # explicitly to get the raw structured description back instead.
    monkeypatch.setattr(excel_charts, "_summarize", lambda description: description)


def test_charts_for_worksheet_finds_the_chart_on_its_own_sheet(tmp_path: Path) -> None:
    path = _write_bar_chart_workbook(tmp_path / "wb.xlsx")
    workbook = openpyxl.load_workbook(path, data_only=True)

    chart_paths = charts_for_worksheet(path, workbook["Sheet1"])

    assert chart_paths == ["xl/charts/chart1.xml"]


def test_charts_for_worksheet_empty_when_sheet_has_no_chart(tmp_path: Path) -> None:
    workbook = openpyxl.Workbook()
    workbook.active.title = "Sheet1"
    workbook.active.append(["A"])
    path = tmp_path / "wb.xlsx"
    workbook.save(path)
    reopened = openpyxl.load_workbook(path)

    assert charts_for_worksheet(path, reopened["Sheet1"]) == []


def test_describe_chart_extracts_type_title_and_resolved_series(
    tmp_path: Path, _stub_llm: None
) -> None:
    path = _write_bar_chart_workbook(tmp_path / "wb.xlsx")
    workbook = openpyxl.load_workbook(path, data_only=True)
    worksheet = workbook["Sheet1"]
    [chart_path] = charts_for_worksheet(path, worksheet)

    description = describe_chart(path, chart_path, worksheet)

    assert description is not None
    assert "barChart" in description
    assert "Q3 Revenue by Region" in description
    assert "North: 120" in description
    assert "South: 340" in description
    assert "East: 275" in description


def test_llm_is_used_to_summarize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(excel_charts, "_summarize", lambda description: "a chart summary")
    path = _write_bar_chart_workbook(tmp_path / "wb.xlsx")
    workbook = openpyxl.load_workbook(path, data_only=True)
    worksheet = workbook["Sheet1"]
    [chart_path] = charts_for_worksheet(path, worksheet)

    assert describe_chart(path, chart_path, worksheet) == "a chart summary"


def test_describe_chart_returns_none_when_summarize_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(excel_charts, "_summarize", lambda description: None)
    path = _write_bar_chart_workbook(tmp_path / "wb.xlsx")
    workbook = openpyxl.load_workbook(path, data_only=True)
    worksheet = workbook["Sheet1"]
    [chart_path] = charts_for_worksheet(path, worksheet)

    assert describe_chart(path, chart_path, worksheet) is None


def test_summarize_returns_none_when_llm_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> None:
        raise NotImplementedError("no provider configured")

    monkeypatch.setattr("multimodal_rag.ingestion.excel_charts.get_llm", _raise)
    assert excel_charts._summarize("some description") is None


def test_summarize_returns_none_when_llm_call_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingLLM:
        def generate(self, messages: list[dict[str, str]]) -> str:
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr("multimodal_rag.ingestion.excel_charts.get_llm", lambda: _FailingLLM())
    assert excel_charts._summarize("some description") is None
