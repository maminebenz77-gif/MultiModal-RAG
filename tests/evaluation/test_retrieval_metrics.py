import pytest

from multimodal_rag.evaluation.retrieval_metrics import mrr, ndcg_at_k, recall_at_k
from multimodal_rag.stores.schema import SearchResult


def _result(source: str, score: float = 1.0) -> SearchResult:
    return SearchResult(
        chunk_id=f"{source}::0::hash",
        score=score,
        text="irrelevant",
        source=source,
        doc_id=source,
        element_types=["paragraph"],
    )


def test_recall_at_k_is_one_when_the_only_expected_source_is_present() -> None:
    retrieved = [_result("a.md"), _result("b.md")]
    assert recall_at_k(retrieved, ["a.md"]) == 1.0


def test_recall_at_k_gives_partial_credit_for_multiple_expected_sources() -> None:
    retrieved = [_result("a.md"), _result("c.md")]
    assert recall_at_k(retrieved, ["a.md", "b.md"]) == 0.5


def test_recall_at_k_is_zero_when_nothing_relevant_was_retrieved() -> None:
    retrieved = [_result("x.md"), _result("y.md")]
    assert recall_at_k(retrieved, ["a.md"]) == 0.0


def test_recall_at_k_raises_for_an_empty_expected_sources_list() -> None:
    with pytest.raises(ValueError):
        recall_at_k([_result("a.md")], [])


def test_mrr_is_one_when_the_first_result_is_relevant() -> None:
    retrieved = [_result("a.md"), _result("b.md")]
    assert mrr(retrieved, ["a.md"]) == 1.0


def test_mrr_is_lower_the_further_down_the_first_relevant_result_is() -> None:
    retrieved = [_result("x.md"), _result("y.md"), _result("a.md")]
    assert mrr(retrieved, ["a.md"]) == pytest.approx(1 / 3)


def test_mrr_is_zero_when_nothing_relevant_was_retrieved() -> None:
    retrieved = [_result("x.md"), _result("y.md")]
    assert mrr(retrieved, ["a.md"]) == 0.0


def test_mrr_only_counts_the_first_relevant_result_not_later_ones() -> None:
    # Same "first relevant at rank 1" for both -- a second relevant result
    # further down shouldn't change the score, unlike ndcg_at_k.
    assert mrr([_result("a.md"), _result("b.md")], ["a.md", "b.md"]) == 1.0
    assert mrr([_result("a.md"), _result("x.md")], ["a.md", "b.md"]) == 1.0


def test_ndcg_at_k_is_one_for_a_perfectly_ordered_ideal_ranking() -> None:
    retrieved = [_result("a.md"), _result("b.md")]
    assert ndcg_at_k(retrieved, ["a.md", "b.md"]) == pytest.approx(1.0)


def test_ndcg_at_k_penalizes_relevant_results_ranked_lower() -> None:
    ranked_first = ndcg_at_k([_result("a.md"), _result("x.md")], ["a.md"])
    ranked_second = ndcg_at_k([_result("x.md"), _result("a.md")], ["a.md"])
    assert ranked_first == pytest.approx(1.0)
    assert ranked_second < ranked_first


def test_ndcg_at_k_sees_second_relevant_result_position_that_mrr_cannot() -> None:
    # Both orderings have their FIRST relevant result ("a.md") at rank 1 --
    # mrr only ever looks at that, so it can't distinguish them. The second
    # relevant result ("b.md") is ranked higher in the first case, which
    # ndcg_at_k -- unlike mrr -- actually rewards.
    expected_sources = ["a.md", "b.md"]
    b_ranked_higher = [_result("a.md"), _result("b.md"), _result("x.md")]
    b_ranked_lower = [_result("a.md"), _result("x.md"), _result("b.md")]

    assert mrr(b_ranked_higher, expected_sources) == mrr(b_ranked_lower, expected_sources)
    assert ndcg_at_k(b_ranked_higher, expected_sources) > ndcg_at_k(
        b_ranked_lower, expected_sources
    )


def test_ndcg_at_k_rewards_finding_more_of_multiple_relevant_sources() -> None:
    one_of_two = ndcg_at_k([_result("a.md"), _result("x.md")], ["a.md", "b.md"])
    two_of_two = ndcg_at_k([_result("a.md"), _result("b.md")], ["a.md", "b.md"])
    assert two_of_two > one_of_two


def test_ndcg_at_k_is_zero_when_nothing_relevant_was_retrieved() -> None:
    retrieved = [_result("x.md"), _result("y.md")]
    assert ndcg_at_k(retrieved, ["a.md"]) == 0.0


def test_ndcg_at_k_raises_for_an_empty_expected_sources_list() -> None:
    with pytest.raises(ValueError):
        ndcg_at_k([_result("a.md")], [])


def test_ndcg_at_k_never_exceeds_one_when_one_source_has_several_retrieved_chunks() -> None:
    # Regression: a document contributing multiple child chunks to the
    # top-k (routine with ParentChildChunker) must not score more than the
    # single ideal slot IDCG budgets for it -- caught live, where
    # chunking_demo.md legitimately placed 3+ of its own chunks in one
    # query's top-5.
    retrieved = [_result("a.md"), _result("a.md"), _result("a.md")]
    assert ndcg_at_k(retrieved, ["a.md"]) == pytest.approx(1.0)
