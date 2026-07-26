from multimodal_rag.chunking.ids import chunk_id


def test_same_inputs_produce_the_same_id() -> None:
    a = chunk_id("doc.md", "structure", 0, "some text")
    b = chunk_id("doc.md", "structure", 0, "some text")
    assert a == b


def test_different_text_produces_a_different_id() -> None:
    a = chunk_id("doc.md", "structure", 0, "original text")
    b = chunk_id("doc.md", "structure", 0, "updated text")
    assert a != b
    # ...but the doc/strategy/index prefix stays recognizable.
    assert a.startswith("doc.md::structure::0::")
    assert b.startswith("doc.md::structure::0::")


def test_different_index_produces_a_different_id_even_with_same_text() -> None:
    a = chunk_id("doc.md", "structure", 0, "same text")
    b = chunk_id("doc.md", "structure", 1, "same text")
    assert a != b


def test_different_doc_produces_a_different_id_even_with_same_text_and_index() -> None:
    a = chunk_id("doc-a.md", "structure", 0, "same text")
    b = chunk_id("doc-b.md", "structure", 0, "same text")
    assert a != b
