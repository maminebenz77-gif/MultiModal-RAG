"""Streamlit page for the docgen workflow -- launch and drive a run
entirely from the browser instead of docgen/cli.py's hand-written JSON
state files and terminal input() prompts.

First use of Streamlit's native multipage `pages/` directory in this
project: a docgen run is a genuinely different interaction model from
the main chat page (a job you check back on over minutes, with forms
that answer a pause, not an instant back-and-forth turn), so it gets
its own page rather than another sidebar expander on top of the
chat's.

Same boundary rule as app.py: talks to the API only over HTTP (httpx),
never imports `multimodal_rag` directly.

No autorefresh/polling loop -- docgen segments run for minutes, so an
auto-poll would mean every open tab continuously re-fetches for a job
explicitly meant to be checked back on later. A manual "Refresh"
button, plus the free fetch that already happens on a normal page
load, is enough.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import streamlit as st
from config import get_frontend_settings

st.set_page_config(page_title="LIBRA AI -- Docgen", layout="wide")

api_base_url = get_frontend_settings().api_base_url

_CLASSIFICATION_OPTIONS: dict[str, str] = {"Public": "public", "C1": "c1", "C2": "c2", "C3": "c3"}

_SOURCE_ROLES: dict[str, str] = {
    "task_docs": "Task documents",
    "reference_kb": "Reference/comparison",
}


def _http_error_detail(exc: httpx.HTTPError) -> str:
    """Duplicated from app.py rather than imported -- the two pages
    don't otherwise share a module, and this is the cheapest way to
    keep this page's diff self-contained."""
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    try:
        body = response.json()
    except ValueError:
        body = response.text
    if isinstance(body, dict) and "detail" in body:
        return f"{exc} -- {body['detail']}"
    if isinstance(body, str) and body.strip():
        return f"{exc} -- {body.strip()}"
    return str(exc)


def _distinct_tags() -> list[str]:
    try:
        documents = httpx.get(f"{api_base_url}/documents", timeout=10.0).json()["documents"]
    except (httpx.HTTPError, ValueError, KeyError):
        return []
    seen: set[str] = set()
    for doc in documents:
        seen.update(doc["metadata"]["tags"])
    return sorted(seen)


def _list_runs() -> list[dict]:
    try:
        response = httpx.get(f"{api_base_url}/docgen/runs", timeout=10.0)
        response.raise_for_status()
        runs: list[dict] = response.json()["runs"]
        return runs
    except httpx.HTTPError:
        return []


def _get_run(thread_id: str) -> dict | None:
    try:
        response = httpx.get(f"{api_base_url}/docgen/runs/{thread_id}", timeout=10.0)
        response.raise_for_status()
        run: dict = response.json()
        return run
    except httpx.HTTPError:
        return None


if "docgen_task_tag" not in st.session_state:
    st.session_state.docgen_task_tag = None
if "docgen_ref_tag" not in st.session_state:
    st.session_state.docgen_ref_tag = None
if "docgen_thread_id" not in st.session_state:
    st.session_state.docgen_thread_id = st.query_params.get("t")


def _ingest_source(role: str, label: str, uploaded_file, classification_label: str) -> str | None:
    """Ingests one file under a fresh docgen:<role>:<label> tag via the
    EXISTING /ingest endpoint -- no new ingestion path, just a tag
    convention this page follows. Returns the tag on success."""
    prefix = "task" if role == "task_docs" else "ref"
    tag = f"docgen:{prefix}:{label.strip()}"
    try:
        metadata_json = json.dumps(
            {"classification": _CLASSIFICATION_OPTIONS[classification_label], "tags": [tag]}
        )
        response = httpx.post(
            f"{api_base_url}/ingest",
            files={
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type),
                "metadata_json": (None, metadata_json),
            },
            timeout=600.0,
        )
        response.raise_for_status()
        st.toast(f"Ingested {uploaded_file.name} under tag {tag!r}.")
        return tag
    except httpx.HTTPError as exc:
        st.error(f"Ingest failed: {_http_error_detail(exc)}")
        return None


def _source_picker(role: str, tag_options: list[str], *, required: bool) -> None:
    state_key = "docgen_task_tag" if role == "task_docs" else "docgen_ref_tag"
    st.subheader(_SOURCE_ROLES[role] + (" (required)" if required else " (optional)"))
    if not required:
        include = st.checkbox("Include this source", key=f"{role}_include")
        if not include:
            st.session_state[state_key] = None
            return

    choice = st.radio(
        "Where from?",
        ["Reuse an existing tag", "Ingest a new file"],
        key=f"{role}_choice",
        horizontal=True,
    )
    if choice == "Reuse an existing tag":
        selected = st.selectbox(
            "Tag",
            options=tag_options,
            key=f"{role}_tag_select",
            index=None,
            placeholder="Choose a tag...",
        )
        st.session_state[state_key] = selected
    else:
        uploaded = st.file_uploader(
            "File", type=["pdf", "docx", "pptx", "md", "csv", "xlsx"], key=f"{role}_upload"
        )
        label = st.text_input("Label for the new tag", key=f"{role}_label")
        classification_label = st.selectbox(
            "Classification", options=list(_CLASSIFICATION_OPTIONS), key=f"{role}_classification"
        )
        if st.button("Ingest & use as source", key=f"{role}_ingest_button"):
            if uploaded is None or not label.strip():
                st.warning("Choose a file and a label first.")
            else:
                tag = _ingest_source(role, label, uploaded, classification_label)
                if tag:
                    st.session_state[state_key] = tag
        if st.session_state[state_key]:
            st.caption(f"Will use tag: {st.session_state[state_key]!r}")


st.title("Document Generator")

with st.sidebar:
    st.header("Start a new run")
    tag_options = _distinct_tags()
    request_text = st.text_area(
        "What should this document answer?",
        help="Free text -- the model turns this into a confirmed question list "
        "before anything else runs, and you'll be asked to confirm or revise it.",
    )
    _source_picker("task_docs", tag_options, required=True)
    _source_picker("reference_kb", tag_options, required=False)
    title_input = st.text_input("Title (optional)")

    if st.button("Start run", type="primary"):
        task_tag = st.session_state.docgen_task_tag
        if not request_text.strip():
            st.warning("Describe what the document should answer first.")
        elif not task_tag:
            st.warning("Task documents source is required.")
        else:
            sources = [{"role": "task_docs", "tag": task_tag, "required": True}]
            if st.session_state.docgen_ref_tag:
                sources.append(
                    {
                        "role": "reference_kb",
                        "tag": st.session_state.docgen_ref_tag,
                        "required": True,
                    }
                )
            try:
                response = httpx.post(
                    f"{api_base_url}/docgen/runs",
                    json={
                        "request_text": request_text,
                        "sources": sources,
                        "title": title_input.strip() or None,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                thread_id = response.json()["thread_id"]
                st.session_state.docgen_thread_id = thread_id
                st.query_params["t"] = thread_id
                st.toast("Run started.")
                st.rerun()
            except httpx.HTTPError as exc:
                st.error(f"Could not start run: {_http_error_detail(exc)}")

    st.divider()
    st.header("Your runs")
    runs = _list_runs()
    if not runs:
        st.caption("No docgen runs yet.")
    for run in runs:
        is_current = run["thread_id"] == st.session_state.docgen_thread_id
        label = f"{run['title'][:30]} [{run['status']}]"
        if st.button(
            label, key=f"select_run_{run['thread_id']}", disabled=is_current, width="stretch"
        ):
            st.session_state.docgen_thread_id = run["thread_id"]
            st.query_params["t"] = run["thread_id"]
            st.rerun()


def _resume(
    thread_id: str, action: str, text: str = "", question_ids: list[str] | None = None
) -> None:
    try:
        response = httpx.post(
            f"{api_base_url}/docgen/runs/{thread_id}/resume",
            json={"action": action, "text": text, "question_ids": question_ids or []},
            timeout=30.0,
        )
        response.raise_for_status()
        st.toast("Submitted -- refresh in a moment to see what happens next.")
        st.rerun()
    except httpx.HTTPError as exc:
        st.error(f"Resume failed: {_http_error_detail(exc)}")


def _render_pending(thread_id: str, run: dict) -> None:
    pending = run["pending"]
    kind = pending["kind"]
    st.subheader("Waiting for you")

    if kind == "confirm_configuration":
        st.markdown(pending["summary"])
        col1, col2 = st.columns(2)
        if col1.button("Confirm", key="confirm_btn", type="primary"):
            _resume(thread_id, "confirm")
        with col2.popover("Revise instead"):
            correction = st.text_area("What should change?", key="revise_text")
            if st.button("Submit revision", key="revise_submit"):
                _resume(thread_id, "revise", text=correction)

    elif kind == "human_review":
        st.markdown(pending["summary"])
        col1, col2 = st.columns(2)
        if col1.button("Approve", key="approve_btn", type="primary"):
            _resume(thread_id, "approve")
        with col2.popover("Request edits instead"):
            question_ids = st.multiselect(
                "Which question(s) need redoing?",
                options=[q["id"] for q in run["questions"]],
                format_func=lambda qid: next(q["text"] for q in run["questions"] if q["id"] == qid),
                key="edit_question_ids",
            )
            guidance = st.text_area("Guidance for redoing them", key="edit_guidance")
            if st.button("Submit edit request", key="edit_submit"):
                if not question_ids:
                    st.warning("Pick at least one question to flag.")
                else:
                    _resume(thread_id, "edit", text=guidance, question_ids=question_ids)

    elif kind == "ask_human":
        st.markdown(f"**Question:** {pending['question']}")
        if pending["attempts"]:
            with st.expander(f"{len(pending['attempts'])} rejected attempt(s) so far"):
                for attempt in pending["attempts"]:
                    st.markdown(f"- *{attempt['answer']}* -- {attempt['reason']}")
        if pending["chunks"]:
            with st.expander(f"{len(pending['chunks'])} retrieved chunk(s)"):
                for chunk in pending["chunks"]:
                    st.caption(chunk["source"])
                    st.text(chunk["text"])
        action = st.radio(
            "How do you want to handle this?",
            ["answer", "reformulate", "skip"],
            key="ask_human_action",
            format_func=lambda a: {
                "answer": "Answer it myself",
                "reformulate": "Give guidance, let it try again",
                "skip": "Skip this question",
            }[a],
        )
        text = ""
        if action != "skip":
            text = st.text_area(
                "Your answer" if action == "answer" else "Guidance for the next attempt",
                key="ask_human_text",
            )
        if st.button("Submit", key="ask_human_submit", type="primary"):
            _resume(thread_id, action, text=text)


thread_id = st.session_state.docgen_thread_id
if not thread_id:
    st.info("Start a run from the sidebar, or select an existing one.")
else:
    run = _get_run(thread_id)
    if run is None:
        st.error(f"Could not load run {thread_id!r}.")
    else:
        header_col, refresh_col = st.columns([5, 1])
        header_col.subheader(f"{run['title']} -- {run['status']}")
        if refresh_col.button("Refresh"):
            st.rerun()

        if run["questions"]:
            question_count = len(run["questions"])
            with st.expander(f"{question_count} question(s)", expanded=run["status"] == "done"):
                for question in run["questions"]:
                    answer = run["answers"].get(question["id"])
                    st.markdown(f"**[{question['status']}] {question['text']}**")
                    if answer:
                        st.write(answer["text"])

        st.caption(f"LLM calls so far: {run['llm_calls']}")

        if run["status"] == "paused":
            _render_pending(thread_id, run)
        elif run["status"] == "failed":
            st.error(f"This run failed: {run['last_error']}")
        elif run["status"] == "done":
            st.success("Done.")
            download_key = f"docgen_download_{thread_id}"
            if download_key not in st.session_state:
                if st.button("Prepare download"):
                    try:
                        response = httpx.get(
                            f"{api_base_url}/docgen/runs/{thread_id}/download", timeout=30.0
                        )
                        response.raise_for_status()
                        st.session_state[download_key] = (
                            response.content,
                            response.headers.get("content-type", "application/octet-stream"),
                        )
                        st.rerun()
                    except httpx.HTTPError as exc:
                        st.error(f"Download failed: {_http_error_detail(exc)}")
            else:
                content, content_type = st.session_state[download_key]
                is_pptx = "presentation" in content_type
                filename = "docgen-report.pptx" if is_pptx else "docgen-report.docx"
                st.download_button(
                    "Download document", data=content, file_name=filename, mime=content_type
                )
        else:
            st.caption("Still running -- check back in a moment.")
