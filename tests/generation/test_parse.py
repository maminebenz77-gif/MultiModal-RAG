from multimodal_rag.generation.parse import parse_answer
from multimodal_rag.generation.prompt import REFUSAL_TEXT
from multimodal_rag.stores.schema import SearchResult


def _result(chunk_id: str, source: str = "doc.md", pages: list[int] | None = None) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=1.0,
        text="text",
        source=source,
        doc_id=source,
        element_types=["title"],
        pages=pages or [],
    )


def test_extracts_a_single_citation() -> None:
    context = [_result("a"), _result("b")]
    result = parse_answer("The answer is X ⟦1⟧.", context)
    assert len(result.citations) == 1
    assert result.citations[0].marker == 1
    assert result.citations[0].chunk_id == "a"


def test_extracts_multiple_citations_deduplicated_and_sorted() -> None:
    context = [_result("a"), _result("b"), _result("c")]
    result = parse_answer("Claim one ⟦2⟧. Claim two ⟦1⟧. Repeated ⟦2⟧ again.", context)
    assert [c.marker for c in result.citations] == [1, 2]
    assert [c.chunk_id for c in result.citations] == ["a", "b"]


def test_citation_carries_real_source_and_page_metadata() -> None:
    context = [_result("a", source="report.pdf", pages=[4])]
    result = parse_answer("Answer ⟦1⟧.", context)
    assert result.citations[0].source == "report.pdf"
    assert result.citations[0].pages == [4]


def test_out_of_range_citation_marker_is_ignored() -> None:
    context = [_result("a")]
    result = parse_answer("Answer ⟦1⟧ and also ⟦99⟧.", context)
    assert [c.marker for c in result.citations] == [1]


def test_no_citations_in_answer_produces_empty_list() -> None:
    context = [_result("a")]
    result = parse_answer("An answer with no markers.", context)
    assert result.citations == []


def test_plain_square_bracket_numbers_are_not_mistaken_for_citations() -> None:
    """The whole reason for the ⟦N⟧ marker: source text or the model's own
    enumerated lists routinely contain plain "[1]"-style numbers (footnotes,
    step lists, array indices) that must NOT be parsed as citations."""
    context = [_result("a")]
    result = parse_answer("See step [1] and item [2] in the referenced list.", context)
    assert result.citations == []


def test_refusal_text_sets_refused_true() -> None:
    result = parse_answer(REFUSAL_TEXT, [])
    assert result.refused is True
    assert result.citations == []


def test_normal_answer_sets_refused_false() -> None:
    result = parse_answer("Here is a real answer ⟦1⟧.", [_result("a")])
    assert result.refused is False


def test_raw_answer_text_is_preserved_verbatim() -> None:
    result = parse_answer("Exactly this text ⟦1⟧.", [_result("a")])
    assert result.answer == "Exactly this text ⟦1⟧."
