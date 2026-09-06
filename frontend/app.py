"""Streamlit demo UI for the RAG API -- a real chat, not a single
question/answer form.

Deliberately talks to the API only over HTTP (httpx), never imports
`multimodal_rag` directly -- the API is the real boundary between
backend and frontend, not just an internal layering convention, and
this keeps that true in practice, not just on paper.

Run: `uv run streamlit run frontend/app.py`

Streamlit re-runs this whole script top-to-bottom on every widget
interaction -- the conversation transcript has to live in
st.session_state, or it would vanish the moment you touched an
unrelated widget (like the retrieval-method dropdown).
"""

import base64
import json
import sys
from pathlib import Path

# `streamlit run` puts this script's own directory on sys.path
# automatically, but other ways of executing this file (Streamlit's own
# AppTest harness, `python -m` invocations, ...) don't -- inserting it
# explicitly makes the local `config` import work regardless of how the
# script was launched, rather than depending on that implicit behavior.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx
import streamlit as st
from config import get_frontend_provider_defaults, get_frontend_settings

_METHOD_OPTIONS: dict[str, tuple[str, bool]] = {
    "Cosine similarity": ("cosine", False),
    "MMR (diversity)": ("mmr", False),
    "BM25 (keyword)": ("bm25", False),
    "Hybrid (RRF)": ("hybrid_rrf", False),
    "Hybrid + Rerank": ("hybrid_rrf", True),
}

# The OS's native folder picker (accept_multiple_files="directory") has
# no per-file filtering UI of its own -- once you pick a folder, every
# file inside comes through, recursively. Streamlit's own `type=`
# allowlist already rejects anything with a disallowed (or missing)
# extension, like .DS_Store, at the widget level, before this code ever
# sees it -- but it can't catch junk with a technically-valid
# extension, which is exactly what Word/Office lock files are:
# ~$report.docx, created while a document is open for editing, has a
# real .docx extension. That's what this filter exists for.
_JUNK_PREFIXES = ("~$", ".")
_JUNK_NAMES = {"thumbs.db", "desktop.ini"}
_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".md", ".markdown"}
_PROVIDER_CATALOG_PATH = Path(__file__).resolve().parent / "provider_catalog.json"


def _load_provider_catalog() -> dict:
    with _PROVIDER_CATALOG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _provider_options(catalog: dict, family: str) -> dict[str, dict]:
    return catalog[family]["providers"]


def _runtime_overrides_payload(state: dict) -> dict:
    return {
        "llm": {
            "provider": state["llm_provider"],
            "model": state["llm_model"],
            "base_url": state.get("llm_base_url") or None,
            "api_key": state.get("llm_api_key") or None,
        },
        "embedder": {
            "provider": state["embed_provider"],
            "model": state["embed_model"],
            "base_url": state.get("embed_base_url") or None,
            "api_key": state.get("embed_api_key") or None,
        },
    }


def _select_index(options: list[str], preferred: str) -> int:
    if preferred in options:
        return options.index(preferred)
    return 0


def _is_junk_file(filename: str) -> bool:
    name = Path(filename).name
    if name.startswith(_JUNK_PREFIXES):
        return True
    if name.lower() in _JUNK_NAMES:
        return True
    return Path(name).suffix.lower() not in _ALLOWED_EXTENSIONS

st.set_page_config(page_title="Multimodal RAG Demo", layout="wide")

api_base_url = get_frontend_settings().api_base_url
provider_defaults = get_frontend_provider_defaults()


def _load_conversation(conversation_id: str) -> bool:
    """Fetches a conversation's turn history and loads it into session
    state; returns False (leaving state untouched) if the id doesn't
    resolve. Shared by URL-based resume on page load and the sidebar's
    "previous conversations" picker -- one fetch-and-populate path, not
    two copies of it."""
    try:
        resp = httpx.get(f"{api_base_url}/conversations/{conversation_id}", timeout=10.0)
        resp.raise_for_status()
    except httpx.HTTPError:
        return False
    st.session_state.conversation_id = conversation_id
    # retrieved_chunks isn't persisted server-side (only citations are --
    # see api/db.py) so a reloaded turn simply has none; a turn generated
    # later this session still does.
    st.session_state.turns = [
        {**message, "retrieved_chunks": []} for message in resp.json()["messages"]
    ]
    return True


if "conversation_id" not in st.session_state:
    # The conversation_id lives in the URL (?c=...), not just session
    # state, so a browser refresh can resume it. A stale/unknown id (e.g.
    # a bookmarked link after the DB was wiped) must not break the page:
    # fall back to a fresh conversation exactly like the /health check
    # below already falls back to "Status: unreachable" rather than
    # raising.
    _url_conversation_id = st.query_params.get("c")
    st.session_state.conversation_id = None
    st.session_state.turns = []
    if _url_conversation_id:
        if not _load_conversation(_url_conversation_id):
            st.query_params.pop("c", None)
if "confirm_wipe" not in st.session_state:
    st.session_state.confirm_wipe = False
if "confirm_delete_conversation_id" not in st.session_state:
    st.session_state.confirm_delete_conversation_id = None

provider_catalog = _load_provider_catalog()


def _location_suffix(pages: list[int], slides: list[int]) -> str:
    if pages:
        return f", page {', '.join(str(p) for p in pages)}"
    if slides:
        return f", slide {', '.join(str(s) for s in slides)}"
    return ""


def _render_chunk_element(element: dict) -> None:
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


@st.dialog("Chunk detail", width="large")
def _show_chunk_detail(chunk: dict) -> None:
    location = _location_suffix(chunk["pages"], chunk["slides"])
    st.caption(f"{chunk['source']}{location}")
    st.caption(chunk["chunk_id"])
    st.divider()
    if chunk["elements"]:
        # Reconstructed from the chunk's actual elements (see
        # chunking/schema.py's ChunkElement) -- a table renders as an
        # actual table, an image as an actual image, not the flattened
        # text blob below.
        for element in chunk["elements"]:
            _render_chunk_element(element)
    else:
        # Pre-existing corpus content, ingested before elements existed,
        # or produced by a flatten-first chunking strategy -- falls back
        # to the flattened text rather than showing nothing.
        st.text(chunk["text"])


def _submit_feedback(query_id: str, rating: str) -> None:
    try:
        response = httpx.post(
            f"{api_base_url}/feedback",
            json={"query_id": query_id, "rating": rating, "comment": None},
            timeout=30.0,
        )
        response.raise_for_status()
        st.toast("Feedback recorded, thank you!")
        # The Metrics panel (in script order) already ran and read the
        # pre-feedback counts this render -- st.toast() is specifically
        # designed to survive an immediate rerun, so the confirmation
        # still shows even though the script restarts right after.
        st.rerun()
    except httpx.HTTPError as exc:
        st.error(f"Feedback failed: {exc}")


def _http_error_detail(exc: httpx.HTTPError) -> str:
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


with st.sidebar:
    with st.expander("💬 Previous conversations", expanded=True):
        try:
            _conversations_response = httpx.get(f"{api_base_url}/conversations", timeout=10.0)
            _conversations_response.raise_for_status()
            _conversations = _conversations_response.json()["conversations"]
        except httpx.HTTPError:
            _conversations = []

        if not _conversations:
            st.caption("No previous conversations yet.")
        else:
            for _conversation in _conversations:
                _cid = _conversation["conversation_id"]
                _preview = _conversation["preview"]
                _label = _preview if len(_preview) <= 40 else _preview[:37] + "..."
                _is_current = _cid == st.session_state.conversation_id

                if st.session_state.confirm_delete_conversation_id == _cid:
                    st.caption(f'Delete "{_label}"? This cannot be undone.')
                    _confirm_col, _cancel_col = st.columns(2)
                    if _confirm_col.button(
                        "Yes, delete", key=f"confirm_delete_{_cid}", type="primary"
                    ):
                        try:
                            _del_response = httpx.delete(
                                f"{api_base_url}/conversations/{_cid}", timeout=10.0
                            )
                            _del_response.raise_for_status()
                            st.session_state.confirm_delete_conversation_id = None
                            if _is_current:
                                st.session_state.conversation_id = None
                                st.session_state.turns = []
                                st.query_params.pop("c", None)
                            st.rerun()
                        except httpx.HTTPError as exc:
                            st.error(f"Delete failed: {_http_error_detail(exc)}")
                    if _cancel_col.button("Cancel", key=f"cancel_delete_{_cid}"):
                        st.session_state.confirm_delete_conversation_id = None
                        st.rerun()
                else:
                    _load_col, _delete_col = st.columns([5, 1])
                    if _load_col.button(
                        _label,
                        key=f"load_conversation_{_cid}",
                        disabled=_is_current,
                        width="stretch",
                    ) and _load_conversation(_cid):
                        st.query_params["c"] = _cid
                        st.rerun()
                    if _delete_col.button("🗑️", key=f"delete_conversation_{_cid}"):
                        st.session_state.confirm_delete_conversation_id = _cid
                        st.rerun()

    st.divider()

    with st.expander("Runtime providers", expanded=False):
        st.caption("These values override backend .env defaults for this UI session only.")

        llm_options = _provider_options(provider_catalog, "llm")
        embed_options = _provider_options(provider_catalog, "embedder")

        llm_provider_labels = {
            key: f"{key} - {value['label']}" for key, value in llm_options.items()
        }
        llm_default = provider_defaults.llm_provider
        llm_provider = st.selectbox(
            "LLM provider",
            options=list(llm_options.keys()),
            index=_select_index(list(llm_options.keys()), llm_default),
            key="llm_provider",
            format_func=lambda p: llm_provider_labels[p],
        )
        llm_models = llm_options[llm_provider]["models"]
        llm_default_model = provider_defaults.llm_model
        st.selectbox(
            "LLM model",
            options=llm_models,
            index=_select_index(llm_models, llm_default_model),
            key="llm_model",
        )
        st.text_input(
            "LLM base URL",
            key="llm_base_url",
            value=provider_defaults.llm_base_url or "",
        )
        st.text_input(
            "LLM API key",
            key="llm_api_key",
            value=provider_defaults.llm_api_key or "",
            type="password",
        )

        embed_provider_labels = {
            key: f"{key} - {value['label']}" for key, value in embed_options.items()
        }
        embed_default = provider_defaults.embed_provider
        embed_provider = st.selectbox(
            "Embedder provider",
            options=list(embed_options.keys()),
            index=_select_index(list(embed_options.keys()), embed_default),
            key="embed_provider",
            format_func=lambda p: embed_provider_labels[p],
        )
        embed_models = embed_options[embed_provider]["models"]
        embed_default_model = provider_defaults.embed_model
        st.selectbox(
            "Embedder model",
            options=embed_models,
            index=_select_index(embed_models, embed_default_model),
            key="embed_model",
        )
        st.text_input(
            "Embedder base URL",
            key="embed_base_url",
            value=provider_defaults.embed_base_url or "",
        )
        st.text_input(
            "Embedder API key",
            key="embed_api_key",
            value=provider_defaults.embed_api_key or "",
            type="password",
        )

        apply_runtime_overrides = st.checkbox(
            "Use runtime provider overrides from this page",
            value=False,
        )

    runtime_overrides = _runtime_overrides_payload(st.session_state)

    with st.expander("Retrieval settings", expanded=False):
        method_label = st.selectbox(
            "Retrieval method", list(_METHOD_OPTIONS.keys()), index=3
        )
        top_k = st.number_input("top_k", min_value=1, max_value=20, value=5)

    retrieval_method, rerank = _METHOD_OPTIONS[method_label]

    st.divider()
    st.header("Ingest a document")
    st.caption(f"API: {api_base_url}")
    try:
        health = httpx.get(f"{api_base_url}/health", timeout=5.0).json()
        st.caption(
            f"Status: {health['status']} "
            f"(qdrant={health['qdrant']}, es={health['elasticsearch']})"
        )
    except httpx.HTTPError:
        st.caption("Status: unreachable")

    with st.form("ingest_form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "pptx", "md"])
        ingest_submitted = st.form_submit_button("Ingest")

    if ingest_submitted:
        if uploaded_file is None:
            st.warning("Choose a file first.")
        else:
            try:
                file_payload = (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
                response = httpx.post(
                    f"{api_base_url}/ingest",
                    files=(
                        {
                            "file": file_payload,
                            "runtime_overrides_json": (
                                None,
                                json.dumps({"embedder": runtime_overrides["embedder"]}),
                            ),
                        }
                        if apply_runtime_overrides
                        else {"file": file_payload}
                    ),
                    timeout=120.0,
                )
                response.raise_for_status()
                body = response.json()
                if body["status"] == "duplicate_content":
                    st.info(
                        f"Skipped {body['filename']}: identical content is already "
                        f"ingested as {body['duplicate_of']}."
                    )
                else:
                    verb = (
                        "Already ingested" if body["status"] == "already_ingested" else "Ingested"
                    )
                    st.success(
                        f"{verb} {body['filename']}: {body['num_parent_chunks']} parent "
                        f"chunks, {body['num_child_chunks']} child chunks."
                    )
            except httpx.HTTPError as exc:
                st.error(f"Ingest failed: {_http_error_detail(exc)}")

    st.divider()
    st.header("Bulk ingest a folder")
    st.caption("Opens your OS's native folder picker -- every supported file inside gets ingested.")

    # Deliberately NOT in a form: a form only exposes uploaded_files to
    # this script after the submit button is clicked, so there'd be no
    # way to show which files are junk (and will be skipped) before the
    # user commits to clicking Ingest -- the .DS_Store/~$lock-file
    # entries would just sit in the picker's own file list looking like
    # they're about to be ingested, even though they never would be.
    uploaded_files = st.file_uploader(
        "Choose a folder",
        type=["pdf", "docx", "pptx", "md"],
        accept_multiple_files="directory",
    )

    good_files = [f for f in (uploaded_files or []) if not _is_junk_file(f.name)]
    if uploaded_files:
        skipped_names = [Path(f.name).name for f in uploaded_files if _is_junk_file(f.name)]
        if skipped_names:
            st.caption(
                f"Will skip {len(skipped_names)} file(s) that aren't real documents: "
                + ", ".join(skipped_names)
            )
        st.caption(f"{len(good_files)} file(s) ready to ingest.")

    bulk_submitted = st.button("Ingest files", disabled=not good_files)

    if bulk_submitted:
        progress = st.progress(0.0)
        status_line = st.empty()
        counts = {"ingested": 0, "already_ingested": 0, "duplicate_content": 0}
        duplicates: list[str] = []
        failures: list[str] = []

        for i, uploaded_file in enumerate(good_files, start=1):
            status_line.text(f"({i}/{len(good_files)}) {uploaded_file.name}...")
            try:
                file_payload = (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
                response = httpx.post(
                    f"{api_base_url}/ingest",
                    files=(
                        {
                            "file": file_payload,
                            "runtime_overrides_json": (
                                None,
                                json.dumps({"embedder": runtime_overrides["embedder"]}),
                            ),
                        }
                        if apply_runtime_overrides
                        else {"file": file_payload}
                    ),
                    timeout=120.0,
                )
                response.raise_for_status()
                body = response.json()
                counts[body["status"]] += 1
                if body["status"] == "duplicate_content":
                    duplicates.append(f"{uploaded_file.name} (same as {body['duplicate_of']})")
            except httpx.HTTPError as exc:
                failures.append(f"{uploaded_file.name}: {_http_error_detail(exc)}")
            progress.progress(i / len(good_files))

        status_line.empty()
        progress.empty()
        st.success(
            f"Done: {counts['ingested']} newly ingested, "
            f"{counts['already_ingested']} already up to date, "
            f"{counts['duplicate_content']} duplicate content, "
            f"{len(failures)} failed."
        )
        if duplicates:
            st.caption("Duplicates skipped: " + ", ".join(duplicates))
        if failures:
            st.error("Failed:\n" + "\n".join(failures))

    st.divider()
    with st.expander("📈 Metrics"):
        try:
            metrics = httpx.get(f"{api_base_url}/metrics", timeout=10.0).json()
            tile_cols = st.columns(2)
            tile_cols[0].metric("Documents", metrics["total_documents"])
            tile_cols[1].metric("Chunks", metrics["total_chunks"])
            tile_cols = st.columns(2)
            tile_cols[0].metric("Queries", metrics["total_queries"])
            tile_cols[1].metric("Refusal rate", f"{metrics['refusal_rate']:.0%}")
            tile_cols = st.columns(2)
            tile_cols[0].metric("👍 Helpful", metrics["feedback_up"])
            tile_cols[1].metric("👎 Not helpful", metrics["feedback_down"])
        except httpx.HTTPError as exc:
            st.error(f"Could not load metrics: {exc}")

    with st.expander("📚 Documents in the corpus"):
        try:
            documents = httpx.get(f"{api_base_url}/documents", timeout=10.0).json()["documents"]
            if documents:
                st.dataframe(
                    [
                        {
                            "Filename": d["filename"],
                            "Parent chunks": d["num_parent_chunks"],
                            "Child chunks": d["num_child_chunks"],
                            "Ingested at": d["ingested_at"],
                        }
                        for d in documents
                    ],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.caption("No documents ingested yet.")
        except httpx.HTTPError as exc:
            st.error(f"Could not load documents: {exc}")

        if st.session_state.confirm_wipe:
            st.warning("This deletes every ingested document and chunk. This cannot be undone.")
            confirm_col, cancel_col = st.columns(2)
            if confirm_col.button("Yes, wipe everything", type="primary"):
                try:
                    response = httpx.delete(f"{api_base_url}/documents", timeout=60.0)
                    response.raise_for_status()
                    body = response.json()
                    st.session_state.confirm_wipe = False
                    st.toast(
                        f"Wiped {body['documents_deleted']} document(s), "
                        f"{body['chunks_deleted']} chunk(s)."
                    )
                    st.rerun()
                except httpx.HTTPError as exc:
                    st.error(f"Wipe failed: {exc}")
            if cancel_col.button("Cancel"):
                st.session_state.confirm_wipe = False
                st.rerun()
        elif st.button("🗑️ Wipe all ingested documents"):
            st.session_state.confirm_wipe = True
            st.rerun()

_title_col, _new_conversation_col = st.columns([5, 1])
_title_col.title("Multimodal RAG Demo")
with _new_conversation_col:
    st.write("")  # vertical nudge so the button lines up with the title text
    if st.button("New conversation"):
        st.session_state.conversation_id = None
        st.session_state.turns = []
        st.query_params.pop("c", None)
        st.rerun()

for turn in st.session_state.turns:
    with st.chat_message("user", avatar="🔵"):
        st.write(turn["question"])
    with st.chat_message("assistant", avatar="🤖"):
        if turn["needs_clarification"]:
            st.info(turn["answer"])
        elif turn["refused"]:
            st.warning(turn["answer"])
        else:
            st.write(turn["answer"])

        if turn["citations"]:
            st.markdown("**Citations**")
            _chunks_by_id = {c["chunk_id"]: c for c in turn["retrieved_chunks"]}
            for c in turn["citations"]:
                location = _location_suffix(c["pages"], c["slides"])
                _cited_chunk = _chunks_by_id.get(c["chunk_id"])
                if st.button(
                    f"⟦{c['marker']}⟧ {c['source']}{location}",
                    key=f"cite_{turn['query_id']}_{c['marker']}",
                    disabled=_cited_chunk is None,
                    help=(
                        None
                        if _cited_chunk is not None
                        else "Chunk detail isn't available for a reloaded conversation."
                    ),
                ):
                    _show_chunk_detail(_cited_chunk)

        if turn["retrieved_chunks"]:
            with st.expander(f"Retrieved chunks ({len(turn['retrieved_chunks'])})"):
                # Retrieved chunks share list position with citation markers
                # (parse_answer() resolves ⟦N⟧ against this same list, so
                # retrieved_chunks[i] IS marker i+1) -- reusing that number
                # here keeps it consistent with the Citations section above,
                # for free. Deliberately just a summary line: the raw
                # flattened text and chunk_id aren't useful to see by
                # default now that "Details" opens the real thing.
                for i, chunk in enumerate(turn["retrieved_chunks"], start=1):
                    location = _location_suffix(chunk["pages"], chunk["slides"])
                    summary_col, details_col = st.columns([5, 1])
                    summary_col.markdown(
                        f"⟦{i}⟧ **{chunk['source']}**{location} — score {chunk['score']:.3f}"
                    )
                    if details_col.button(
                        "Details", key=f"view_{turn['query_id']}_{chunk['chunk_id']}"
                    ):
                        _show_chunk_detail(chunk)

        fb_up, fb_down = st.columns(2)
        if fb_up.button("👍", key=f"fb_up_{turn['query_id']}"):
            _submit_feedback(turn["query_id"], "up")
        if fb_down.button("👎", key=f"fb_down_{turn['query_id']}"):
            _submit_feedback(turn["query_id"], "down")

prompt = st.chat_input("Ask a question")
if prompt and prompt.strip():
    query_payload = {
        "question": prompt,
        "conversation_id": st.session_state.conversation_id,
        "retrieval_method": retrieval_method,
        "top_k": top_k,
        "rerank": rerank,
    }
    if apply_runtime_overrides:
        query_payload["runtime_overrides"] = runtime_overrides
    try:
        response = httpx.post(
            f"{api_base_url}/query",
            json=query_payload,
            timeout=120.0,
        )
        response.raise_for_status()
        body = response.json()
        st.session_state.conversation_id = body["conversation_id"]
        st.query_params["c"] = body["conversation_id"]
        st.session_state.turns.append({**body, "question": prompt})
        # Same reason as the feedback rerun above: the backend recorded
        # this query (and its refusal/method) before this response came
        # back, but the Metrics panel already rendered earlier in this
        # same script run, before the query even started -- only a fresh
        # rerun picks up the updated count.
        st.rerun()
    except httpx.HTTPError as exc:
        st.error(f"Query failed: {_http_error_detail(exc)}")
