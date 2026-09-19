from pathlib import Path

import pytest

from multimodal_rag.ingestion.csv_ import parse_csv
from multimodal_rag.ingestion.schema import ElementType


def _write_csv(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "data.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_parses_header_and_rows_into_a_table_element(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, "Region,Revenue\nNorth,120\nSouth,340\n")
    elements = parse_csv(path)

    assert len(elements) == 1
    assert elements[0].type == ElementType.TABLE
    assert elements[0].text is not None
    assert "Region" in elements[0].text
    assert "North" in elements[0].text
    assert "South" in elements[0].text


def test_sheet_is_none_for_csv(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, "A,B\n1,2\n")
    elements = parse_csv(path)
    assert elements[0].metadata.sheet is None


def test_empty_file_produces_no_elements(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, "")
    assert parse_csv(path) == []


def test_header_only_no_data_rows_produces_no_elements(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, "A,B\n")
    assert parse_csv(path) == []


def test_strips_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.csv"
    path.write_bytes("A,B\n1,2\n".encode("utf-8-sig"))
    elements = parse_csv(path)
    assert elements[0].text is not None
    assert elements[0].text.startswith("| A | B |")


def test_summarize_tables_flag_calls_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "multimodal_rag.ingestion.tabular.summarize_table", lambda markdown_table: "a summary"
    )
    path = _write_csv(tmp_path, "A,B\n1,2\n")
    elements = parse_csv(path, summarize_tables=True)
    assert elements[0].table_summary == "a summary"


def test_many_rows_are_batched_across_multiple_elements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from multimodal_rag.ingestion import tabular

    monkeypatch.setattr(tabular, "ROWS_PER_ELEMENT", 10)
    rows = "\n".join(f"{i},{i * 2}" for i in range(25))
    path = _write_csv(tmp_path, f"A,B\n{rows}\n")

    elements = parse_csv(path)

    assert len(elements) == 3  # 25 rows / 10 per batch -> 10, 10, 5
    assert [el.metadata.position for el in elements] == [0, 1, 2]
