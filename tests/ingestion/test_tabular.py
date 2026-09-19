import pytest

from multimodal_rag.ingestion import tabular
from multimodal_rag.ingestion.schema import ElementType
from multimodal_rag.ingestion.tabular import rows_to_elements


def test_batches_rows_into_groups_of_rows_per_element(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tabular, "ROWS_PER_ELEMENT", 2)
    header = ["A", "B"]
    data_rows = [["1", "2"], ["3", "4"], ["5", "6"]]  # 3 rows -> batches of 2, 1

    elements = rows_to_elements(
        header,
        data_rows,
        source_file="f.csv",
        sheet=None,
        start_position=0,
        summarize_tables=False,
    )

    assert len(elements) == 2
    assert all(el.type == ElementType.TABLE for el in elements)


def test_header_is_repeated_in_every_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tabular, "ROWS_PER_ELEMENT", 1)
    header = ["A", "B"]
    data_rows = [["1", "2"], ["3", "4"]]

    elements = rows_to_elements(
        header,
        data_rows,
        source_file="f.csv",
        sheet=None,
        start_position=0,
        summarize_tables=False,
    )

    assert all(el.text is not None and el.text.startswith("| A | B |") for el in elements)


def test_positions_are_sequential_from_start_position() -> None:
    elements = rows_to_elements(
        ["A"],
        [["1"]],
        source_file="f.csv",
        sheet=None,
        start_position=5,
        summarize_tables=False,
    )
    assert elements[0].metadata.position == 5


def test_sheet_is_carried_onto_metadata() -> None:
    elements = rows_to_elements(
        ["A"],
        [["1"]],
        source_file="f.xlsx",
        sheet="Sheet1",
        start_position=0,
        summarize_tables=False,
    )
    assert elements[0].metadata.sheet == "Sheet1"


def test_no_data_rows_produces_no_elements() -> None:
    elements = rows_to_elements(
        ["A", "B"],
        [],
        source_file="f.csv",
        sheet=None,
        start_position=0,
        summarize_tables=False,
    )
    assert elements == []


def test_summarize_tables_flag_calls_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "multimodal_rag.ingestion.tabular.summarize_table", lambda markdown_table: "a summary"
    )
    elements = rows_to_elements(
        ["A"],
        [["1"]],
        source_file="f.csv",
        sheet=None,
        start_position=0,
        summarize_tables=True,
    )
    assert elements[0].table_summary == "a summary"


def test_summarize_tables_false_leaves_summary_none() -> None:
    elements = rows_to_elements(
        ["A"],
        [["1"]],
        source_file="f.csv",
        sheet=None,
        start_position=0,
        summarize_tables=False,
    )
    assert elements[0].table_summary is None
