from multimodal_rag.generation.prompt import REFUSAL_TEXT, build_messages
from multimodal_rag.stores.schema import SearchResult


def _result(chunk_id: str, text: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=1.0,
        text=text,
        source="doc.md",
        doc_id="doc.md",
        element_types=["title"],
    )


def test_build_messages_has_system_and_user_roles() -> None:
    messages = build_messages("a question", [_result("a", "some text")])
    assert [m["role"] for m in messages] == ["system", "user"]


def test_system_prompt_mentions_refusal_text_and_citation_instruction() -> None:
    messages = build_messages("q", [])
    system = messages[0]["content"]
    assert REFUSAL_TEXT in system
    assert "cite" in system.lower()


def test_system_prompt_instructs_treating_context_as_data_not_instructions() -> None:
    messages = build_messages("q", [])
    system = messages[0]["content"]
    assert "instruction" in system.lower()


def test_user_message_includes_numbered_context_and_question() -> None:
    messages = build_messages(
        "What is X?", [_result("a", "chunk one text"), _result("b", "chunk two text")]
    )
    user = messages[1]["content"]
    assert "⟦1⟧" in user
    assert "⟦2⟧" in user
    assert "chunk one text" in user
    assert "chunk two text" in user
    assert "Question: What is X?" in user


def test_user_message_with_no_context_still_includes_question() -> None:
    messages = build_messages("What is X?", [])
    assert "Question: What is X?" in messages[1]["content"]
