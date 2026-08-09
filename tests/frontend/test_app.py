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


def test_bulk_ingest_button_is_disabled_without_any_files() -> None:
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)

    ingest_button = next(b for b in at.sidebar.button if b.label == "Ingest files")
    assert ingest_button.disabled is True


def test_bulk_ingest_shows_a_live_preview_of_skipped_junk_before_submitting() -> None:
    """Streamlit's own type= allowlist already rejects extensionless
    junk like .DS_Store at the widget level (confirmed: passing one to
    set_value raises before the app's own code even runs). What it
    canNOT catch is junk with a technically-valid extension: Word's
    ~$-prefixed lock file (~$report.docx -- created while a document is
    open for editing) and a dotfile that happens to end in .md. Both
    need the app's own filename-prefix filter -- and since the uploader
    lives outside the form, this preview appears as soon as files are
    selected, before the (disabled, since nothing survives filtering)
    Ingest button could even be clicked."""
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)

    uploader = at.sidebar.file_uploader[1]
    uploader.set_value(
        [
            ("~$report.docx", b"garbage", "application/octet-stream"),
            (".hidden.md", b"garbage", "text/markdown"),
        ]
    )
    at.run(timeout=30)

    assert not at.exception
    assert any("Will skip 2 file(s)" in c.value for c in at.sidebar.caption)
    ingest_button = next(b for b in at.sidebar.button if b.label == "Ingest files")
    assert ingest_button.disabled is True


def test_bulk_ingest_uploader_uses_the_native_folder_picker() -> None:
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)

    bulk_uploader = at.sidebar.file_uploader[1]
    assert bulk_uploader.accept_directory is True


def test_wipe_button_shows_a_confirmation_before_actually_wiping() -> None:
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)

    wipe_button = next(b for b in at.main.button if "Wipe all" in b.label)
    wipe_button.click()
    at.run(timeout=30)

    assert not at.exception
    assert any("cannot be undone" in w.value for w in at.warning)
    assert any(b.label == "Yes, wipe everything" for b in at.main.button)


def test_wipe_confirmation_cancel_returns_to_the_normal_view() -> None:
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)

    next(b for b in at.main.button if "Wipe all" in b.label).click()
    at.run(timeout=30)
    next(b for b in at.main.button if b.label == "Cancel").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["confirm_wipe"] is False
    assert any("Wipe all" in b.label for b in at.main.button)


def test_metrics_and_documents_panels_render_without_exceptions() -> None:
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)

    labels = [e.label for e in at.main.expander]
    assert "📈 Metrics" in labels
    assert "📚 Documents in the corpus" in labels
    assert not at.exception


def test_asking_with_a_blank_question_does_not_query_or_raise() -> None:
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)

    at.text_input[0].set_value("")
    at.button(key="FormSubmitter:query_form-Ask").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["last_result"] is None
