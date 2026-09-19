from multimodal_rag.chunking.schema import ChunkElement
from multimodal_rag.generation.context import assemble_context, count_tokens, format_context_block
from multimodal_rag.stores.schema import SearchResult


def _result(
    chunk_id: str,
    text: str,
    pages: list[int] | None = None,
    elements: list[ChunkElement] | None = None,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=1.0,
        text=text,
        source="doc.md",
        doc_id="doc.md",
        element_types=["title"],
        pages=pages or [],
        elements=elements or [],
    )


def test_count_tokens_is_positive_for_nonempty_text() -> None:
    assert count_tokens("hello world") > 0


def test_count_tokens_zero_for_empty_text() -> None:
    assert count_tokens("") == 0


def test_assemble_context_includes_all_when_under_budget() -> None:
    results = [_result("a", "short"), _result("b", "also short")]
    included = assemble_context(results, token_budget=1000)
    assert [r.chunk_id for r in included] == ["a", "b"]


def test_assemble_context_drops_the_tail_when_over_budget() -> None:
    long_text = "word " * 500  # comfortably over a small budget
    results = [_result("a", "short"), _result("b", long_text), _result("c", "short too")]
    included = assemble_context(results, token_budget=10)
    # "a" fits, "b" alone blows the budget so nothing after it is added.
    assert [r.chunk_id for r in included] == ["a"]


def test_assemble_context_always_includes_the_top_result_even_if_it_alone_exceeds_budget() -> None:
    long_text = "word " * 500
    results = [_result("a", long_text)]
    included = assemble_context(results, token_budget=1)
    assert [r.chunk_id for r in included] == ["a"]


def test_assemble_context_empty_input_returns_empty() -> None:
    assert assemble_context([], token_budget=1000) == []


def test_format_context_block_includes_index_source_and_text() -> None:
    block = format_context_block(1, _result("a", "the chunk text"))
    assert block.startswith("⟦1⟧ (source: doc.md)")
    assert "the chunk text" in block


def test_format_context_block_includes_page_when_present() -> None:
    block = format_context_block(2, _result("a", "text", pages=[3, 4]))
    assert "page 3, 4" in block


def test_format_context_block_omits_location_when_no_page_or_slide() -> None:
    block = format_context_block(1, _result("a", "text"))
    assert "(source: doc.md)" in block


def test_format_context_block_uses_raw_table_not_the_embedded_summary() -> None:
    raw_table = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    result = _result(
        "a",
        text="A table of two columns, A and B.",  # what got embedded
        elements=[ChunkElement(type="table", text=raw_table)],
    )
    block = format_context_block(1, result)
    assert raw_table in block
    assert "A table of two columns" not in block


def test_format_context_block_falls_back_to_result_text_without_elements() -> None:
    result = _result("a", "plain text", elements=[])
    block = format_context_block(1, result)
    assert "plain text" in block


def test_format_context_block_unchanged_for_a_non_table_chunk_with_elements() -> None:
    result = _result(
        "a",
        text="hello",
        elements=[ChunkElement(type="paragraph", text="hello")],
    )
    block = format_context_block(1, result)
    assert "hello" in block


def test_assemble_context_budgets_against_the_raw_table_not_the_summary() -> None:
    long_raw_table = "| A |\n| --- |\n" + "\n".join(f"| {i} |" for i in range(500))
    result = _result(
        "a",
        text="short summary",  # cheap if this were budgeted instead
        elements=[ChunkElement(type="table", text=long_raw_table)],
    )
    included = assemble_context([result, _result("b", "short too")], token_budget=10)
    # "a" alone already blows the budget once the real table is counted,
    # so "b" never gets added -- this would fail if budgeting still used
    # the cheap embedded summary instead of the real generation text.
    assert [r.chunk_id for r in included] == ["a"]
