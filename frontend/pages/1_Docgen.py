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

Auto-refresh ONLY while a segment is actively running (st_autorefresh,
a small timer-driven rerun -- see its one call site below) -- a run
that's paused or done isn't going to change on its own, so polling it
would be pure waste; one that's running will, and a user watching a
spinner that only updates on manual click (the ORIGINAL design here)
reads as broken, not as "being careful with resources." Reported
directly by a live user test: after starting a run, the page kept
showing "running" long after it had actually paused, until a manual
refresh revealed it.
"""

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import streamlit as st
from config import get_frontend_settings
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="LIBRA AI -- Docgen", layout="wide")

api_base_url = get_frontend_settings().api_base_url

_CLASSIFICATION_OPTIONS: dict[str, str] = {"Public": "public", "C1": "c1", "C2": "c2", "C3": "c3"}

# User-facing labels for the two source roles -- "task_docs"/
# "reference_kb" are internal identifiers a backend engineer would
# recognize, not something to show someone filling out this form.
# Reused everywhere a role name would otherwise leak into the UI (the
# source picker, the question list, the confirmation plan).
#
# A real, caught-live bug lives in this exact spot: a bare
# triple-quoted string placed AFTER a module-level assignment, the
# convention this whole project otherwise uses for inline
# documentation, is invisible in a normal Python module -- but
# Streamlit's "magic" output auto-displays any bare top-level
# expression in the main script, so that convention silently rendered
# this explanation text onto the live page. Plain `#` comments don't
# trigger magic; every other note in this file must stay a `#`
# comment or a real docstring attached to a function/class, never a
# bare string sitting at module level on its own.
_SOURCE_ROLES: dict[str, str] = {
    "task_docs": "Documents to analyze",
    "reference_kb": "Reference documents",
}

# Same junk-filtering convention as app.py's own bulk-folder ingest --
# accept_multiple_files="directory" hands back EVERY file in the
# chosen folder, including OS/Office cruft (.DS_Store, a ~$-prefixed
# Word lock file) that Streamlit's own `type=` allowlist can't catch
# since some of it has a technically-valid extension.
_JUNK_PREFIXES = ("~$", ".")
_JUNK_NAMES = {"thumbs.db", "desktop.ini"}
_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".md", ".markdown", ".csv", ".xlsx"}


def _is_junk_file(filename: str) -> bool:
    name = Path(filename).name
    if name.startswith(_JUNK_PREFIXES):
        return True
    if name.lower() in _JUNK_NAMES:
        return True
    return Path(name).suffix.lower() not in _ALLOWED_EXTENSIONS


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


def _location_suffix(pages: list[int], slides: list[int]) -> str:
    """Also duplicated from app.py, same reasoning as _http_error_detail
    -- renders a chunk's real location (a page or slide number) next to
    its source filename."""
    if pages:
        return f", page {', '.join(str(p) for p in pages)}"
    if slides:
        return f", slide {', '.join(str(s) for s in slides)}"
    return ""


def _render_chunk_element(element: dict) -> None:
    """A chunk's REAL constituent parts (title/paragraph/table/image),
    not its flattened text -- a table renders as an actual table
    (markdown), an image as an actual image, same as the main chat
    page's own chunk detail view. Without this, a retrieved excerpt
    that's mostly a table or a picture looked cut off or garbled when
    shown as flattened text instead -- reported directly as "not all
    the chunk is shown"."""
    element_type = element["type"]
    if element_type in ("image", "chart"):
        if element.get("image_base64"):
            st.image(base64.b64decode(element["image_base64"]))
        if element.get("description"):
            st.caption(element["description"])
    elif element_type == "title":
        st.markdown(f"**{element.get('text') or ''}**")
    else:
        st.markdown(element.get("text") or "")


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


def _ingest_source(
    role: str, label: str, uploaded_files: list, classification_label: str
) -> str | None:
    """Ingests every file in uploaded_files (a whole folder's worth,
    already junk-filtered by the caller) under ONE fresh
    docgen:<role>:<label> tag via the EXISTING /ingest endpoint -- one
    label for the whole batch, since a docgen source is a single named
    set of documents, not per-file metadata. No new ingestion path,
    just this tag convention and a per-file loop, same as app.py's own
    bulk-folder ingest. Returns the tag if at least one file ingested
    successfully."""
    prefix = "task" if role == "task_docs" else "ref"
    tag = f"docgen:{prefix}:{label.strip()}"
    metadata_json = json.dumps(
        {"classification": _CLASSIFICATION_OPTIONS[classification_label], "tags": [tag]}
    )
    ingested = 0
    failures: list[str] = []
    for uploaded_file in uploaded_files:
        try:
            response = httpx.post(
                f"{api_base_url}/ingest",
                files={
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type),
                    "metadata_json": (None, metadata_json),
                },
                timeout=600.0,
            )
            response.raise_for_status()
            ingested += 1
        except httpx.HTTPError as exc:
            failures.append(f"{uploaded_file.name}: {_http_error_detail(exc)}")
    if ingested:
        st.toast(f"Ingested {ingested} file(s) under tag {tag!r}.")
    if failures:
        st.error("Some files failed to ingest:\n" + "\n".join(failures))
    return tag if ingested else None


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
        ["Reuse an existing tag", "Ingest a new folder"],
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
        uploaded_files = st.file_uploader(
            "Folder",
            type=["pdf", "docx", "pptx", "md", "csv", "xlsx"],
            accept_multiple_files="directory",
            key=f"{role}_upload",
            help="Opens your OS's native folder picker -- every supported file inside "
            "gets ingested under one tag for this source.",
        )
        good_files = [f for f in (uploaded_files or []) if not _is_junk_file(f.name)]
        if uploaded_files:
            skipped = len(uploaded_files) - len(good_files)
            if skipped:
                st.caption(f"Will skip {skipped} file(s) that aren't real documents.")
            st.caption(f"{len(good_files)} file(s) ready to ingest.")
        label = st.text_input("Label for the new tag", key=f"{role}_label")
        classification_label = st.selectbox(
            "Classification", options=list(_CLASSIFICATION_OPTIONS), key=f"{role}_classification"
        )
        if st.button(
            "Ingest & use as source", key=f"{role}_ingest_button", disabled=not good_files
        ):
            if not label.strip():
                st.warning("Choose a label first.")
            else:
                tag = _ingest_source(role, label, good_files, classification_label)
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
        st.toast("Submitted -- working on it now.")
        st.rerun()
    except httpx.HTTPError as exc:
        st.error(f"Resume failed: {_http_error_detail(exc)}")


def _format_sources(roles: list[str]) -> str:
    return ", ".join(_SOURCE_ROLES.get(role, role) for role in roles) or "no sources"


def _render_plan(run: dict) -> None:
    """The CURRENT proposed plan, as structured data (run["questions"]/
    run["configuration"]) -- never the backend's own flat summary
    string. That string IS still generated server-side (for a caller
    with no richer UI), but rendering it verbatim via st.markdown was a
    real, reported bug: a single "\\n" is not a markdown paragraph
    break, so the output-format line ran onto the same visual line as
    the last question with no space between them. Structured data
    sidesteps that by construction -- each piece is its own markdown
    call, not one joined string."""
    questions = run["questions"]
    if not questions:
        st.write("I wasn't able to turn that into any questions yet.")
        return
    st.markdown(f"Here's the plan -- **{len(questions)} question(s)**:")
    for i, question in enumerate(questions, start=1):
        st.markdown(f"{i}. {question['text']}")
        st.caption(f"Using: {_format_sources(question['sources_required'])}")
    configuration = run.get("configuration")
    if configuration:
        template = configuration["template"] or "the default template"
        st.caption(f"Output: a **{configuration['format']}** file, using {template}.")


def _render_configuration_conversation(thread_id: str, run: dict) -> None:
    """Renders the free-text request, every correction given so far,
    and the resulting plan as a conversation -- reported directly by a
    live user test: submitting a correction (e.g. "also use the
    reference docs") produced no visible acknowledgment at all, just
    the next plan appearing with no connection back to what was asked
    for. request/corrections come from the backend's own
    RequestContext (never discarded, only appended to), so this is a
    real history, not a guess at one."""
    request = run.get("request")
    if request:
        with st.chat_message("user"):
            st.write(request["text"])
        for correction in request["corrections"]:
            with st.chat_message("user"):
                st.caption("Correction")
                st.write(correction)
    with st.chat_message("assistant"):
        _render_plan(run)

    col1, col2 = st.columns(2)
    if col1.button("Confirm", key="confirm_btn", type="primary"):
        _resume(thread_id, "confirm")
    with col2.popover("Revise instead"):
        correction_text = st.text_area("What should change?", key="revise_text")
        if st.button("Submit revision", key="revise_submit"):
            _resume(thread_id, "revise", text=correction_text)


def _render_pending(thread_id: str, run: dict) -> None:
    pending = run["pending"]
    kind = pending["kind"]

    if kind == "confirm_configuration":
        st.subheader("Confirm the plan")
        _render_configuration_conversation(thread_id, run)
        return

    st.subheader("Waiting for you")
    if kind == "human_review":
        st.caption("Review the answers above, then decide:")
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
        st.write(
            "It tried this question several times and couldn't put together an answer "
            "it was confident is actually backed by the documents. It needs your help "
            "to move on."
        )
        st.markdown(f"**Question:** {pending['question']}")

        if pending.get("previous_guidance"):
            st.info(
                f"You already gave it a hint on an earlier try: “{pending['previous_guidance']}”"
            )

        attempts = pending.get("attempts") or []
        if attempts:
            with st.expander(f"What it already tried ({len(attempts)} attempt(s))"):
                for i, attempt in enumerate(attempts, start=1):
                    st.markdown(f"**Attempt {i}:** {attempt['answer'] or '_(no usable answer)_'}")
                    st.caption(f"Rejected because: {attempt['reason']}")
                    if i < len(attempts):
                        st.divider()

        chunks = pending.get("chunks") or []
        if chunks:
            with st.expander(
                f"What it found in the documents ({len(chunks)} excerpt(s))", expanded=True
            ):
                for i, chunk in enumerate(chunks, start=1):
                    location = _location_suffix(chunk["pages"], chunk["slides"])
                    st.caption(f"{chunk['source']}{location}")
                    if chunk.get("elements"):
                        for element in chunk["elements"]:
                            _render_chunk_element(element)
                    else:
                        st.write(chunk["text"])
                    if i < len(chunks):
                        st.divider()
        else:
            st.warning(
                "It didn't find anything relevant in the documents for this question at all."
            )

        st.divider()
        st.markdown("**What would you like to do?**")
        action = st.radio(
            "What would you like to do?",
            ["answer", "reformulate", "skip"],
            key="ask_human_action",
            label_visibility="collapsed",
            format_func=lambda a: {
                "answer": "I'll answer this question myself",
                "reformulate": "Give it a hint (e.g. where to look) and let it try again",
                "skip": "Skip this question -- leave it unanswered in the final document",
            }[a],
        )
        text = ""
        if action != "skip":
            text = st.text_area(
                "Your answer" if action == "answer" else "Your hint for the next attempt",
                key="ask_human_text",
            )
        if st.button("Submit", key="ask_human_submit", type="primary"):
            if action != "skip" and not text.strip():
                st.warning("Write something first.")
            else:
                _resume(thread_id, action, text=text)


_STATUS_BADGES = {
    "running": "🔄 running",
    "paused": "⏸️ waiting on you",
    "done": "✅ done",
    "failed": "❌ failed",
}

thread_id = st.session_state.docgen_thread_id
if not thread_id:
    st.info("Start a run from the sidebar, or select an existing one.")
else:
    run = _get_run(thread_id)
    if run is None:
        st.error(f"Could not load run {thread_id!r}.")
    else:
        # Only while a segment is actually in flight -- a paused or
        # finished run isn't going to change on its own, so there's
        # nothing to re-fetch early for.
        if run["status"] == "running":
            st_autorefresh(interval=3000, key=f"docgen_autorefresh_{thread_id}")

        header_col, refresh_col = st.columns([5, 1])
        header_col.subheader(f"{run['title']} -- {_STATUS_BADGES[run['status']]}")
        if refresh_col.button("Refresh"):
            st.rerun()

        pending_kind = (run.get("pending") or {}).get("kind")
        # confirm_configuration gets its own conversation render below
        # (via _render_pending), which already shows the full plan --
        # repeating it here would just be the same questions twice,
        # once with no answers at all yet.
        if run["questions"] and pending_kind != "confirm_configuration":
            question_count = len(run["questions"])
            with st.expander(
                f"{question_count} question(s)",
                expanded=run["status"] in ("done", "failed") or pending_kind == "human_review",
            ):
                for question in run["questions"]:
                    answer = run["answers"].get(question["id"])
                    st.markdown(f"**[{question['status']}] {question['text']}**")
                    st.caption(f"Using: {_format_sources(question['sources_required'])}")
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
            st.caption("Still running -- this page updates on its own, no need to refresh.")
