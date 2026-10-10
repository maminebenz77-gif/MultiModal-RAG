"""Regression coverage for the docgen Streamlit page's own script logic
-- same discipline as test_app.py: no real API is listening in this
test environment, so every API call must degrade gracefully (empty
lists, a caught httpx.HTTPError) rather than raising, and this is what
actually catches that if a future edit breaks it.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

_PAGE_PATH = str(Path(__file__).resolve().parents[2] / "frontend" / "pages" / "1_Docgen.py")


def test_page_loads_without_exceptions() -> None:
    at = AppTest.from_file(_PAGE_PATH)
    at.run(timeout=30)
    assert not at.exception


def test_page_renders_title_and_sidebar_sections() -> None:
    at = AppTest.from_file(_PAGE_PATH)
    at.run(timeout=30)
    assert at.title[0].value == "Document Generator"
    assert any(h.value == "Start a new run" for h in at.sidebar.header)
    assert any(h.value == "Your runs" for h in at.sidebar.header)


def test_run_list_falls_back_gracefully_when_api_is_unreachable() -> None:
    at = AppTest.from_file(_PAGE_PATH)
    at.run(timeout=30)
    assert any("No docgen runs yet" in c.value for c in at.sidebar.caption)


def test_no_run_selected_shows_an_informational_message() -> None:
    at = AppTest.from_file(_PAGE_PATH)
    at.run(timeout=30)
    assert any("Start a run from the sidebar" in i.value for i in at.info)


def test_starting_a_run_without_describing_it_shows_a_warning() -> None:
    at = AppTest.from_file(_PAGE_PATH)
    at.run(timeout=30)

    start_button = next(b for b in at.sidebar.button if b.label == "Start run")
    start_button.click()
    at.run(timeout=30)

    assert not at.exception
    assert any("Describe what the document should answer" in w.value for w in at.sidebar.warning)


def test_starting_a_run_without_a_task_source_shows_a_warning() -> None:
    at = AppTest.from_file(_PAGE_PATH)
    at.run(timeout=30)

    at.sidebar.text_area[0].set_value("What was the hosted API's average latency?")
    at.run(timeout=30)

    start_button = next(b for b in at.sidebar.button if b.label == "Start run")
    start_button.click()
    at.run(timeout=30)

    assert not at.exception
    assert any("Task documents source is required" in w.value for w in at.sidebar.warning)


def test_task_docs_folder_upload_accepts_multiple_files_and_filters_junk() -> None:
    """Regression: the source picker originally used a plain single-file
    st.file_uploader, so a whole folder couldn't be selected at all --
    this locks in the fix (accept_multiple_files="directory", same
    junk-filtering convention as app.py's own bulk-folder ingest)."""
    at = AppTest.from_file(_PAGE_PATH)
    at.run(timeout=30)

    at.sidebar.radio(key="task_docs_choice").set_value("Ingest a new folder")
    at.run(timeout=30)

    uploader = at.sidebar.file_uploader(key="task_docs_upload")
    uploader.set_value(
        [
            ("real-doc.md", b"content", "text/markdown"),
            ("another-doc.md", b"more content", "text/markdown"),
            # A technically-valid extension is what makes these actually
            # reach the uploader widget at all (Streamlit's own `type=`
            # check rejects an extensionless file like .DS_Store before
            # the script's own logic ever runs) -- same junk examples
            # app.py's own bulk-ingest test uses, for the same reason.
            ("~$report.docx", b"junk", "application/octet-stream"),
            (".hidden.md", b"junk", "text/markdown"),
        ]
    )
    at.run(timeout=30)

    assert not at.exception
    assert any("2 file(s) ready to ingest" in c.value for c in at.sidebar.caption)
    assert any("Will skip 2 file(s)" in c.value for c in at.sidebar.caption)
    ingest_button = next(
        b for b in at.sidebar.button if b.label == "Ingest & use as source" and not b.disabled
    )
    assert ingest_button is not None
