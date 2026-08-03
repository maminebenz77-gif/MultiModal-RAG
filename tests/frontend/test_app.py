"""Regression coverage for the Streamlit app's own script logic --
NOT retrieval/generation quality, which is already covered elsewhere.
Deliberately avoids depending on a real running API: these tests exist
to catch script-level bugs (import errors, bad widget wiring, guard
clauses) the way AppTest caught a real one during development --
`from config import get_frontend_settings` relied on `streamlit run`'s
implicit sys.path behavior, which AppTest.run() does NOT provide, so
the app raised ModuleNotFoundError under it until fixed.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

_APP_PATH = str(Path(__file__).resolve().parents[2] / "frontend" / "app.py")


def test_app_loads_without_exceptions() -> None:
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    assert not at.exception


def test_app_renders_title_and_sidebar() -> None:
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    assert at.title[0].value == "Multimodal RAG Demo"
    assert at.sidebar.header[0].value == "Ingest a document"


def test_ingest_without_a_file_shows_a_warning() -> None:
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)

    at.button(key="FormSubmitter:ingest_form-Ingest").click()
    at.run(timeout=30)

    assert not at.exception
    assert any("Choose a file first" in w.value for w in at.warning)


def test_asking_with_a_blank_question_does_not_query_or_raise() -> None:
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)

    at.text_input[0].set_value("")
    at.button(key="FormSubmitter:query_form-Ask").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["last_result"] is None
