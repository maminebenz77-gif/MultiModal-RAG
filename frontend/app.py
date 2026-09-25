"""Streamlit UI for the RAG API -- a real chat, not a single
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
from datetime import date
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

# /ingest requires a classification on every upload -- no default (see
# multimodal_rag.metadata.DocumentMetadata) -- so the picker must always
# have a real value selected, never a blank placeholder option.
_NO_REPLACEMENT = "(none -- a new document)"


def _replace_options(api_base_url: str) -> dict[str, str | None]:
    """Label -> doc_id for the "this replaces..." picker: the CURRENT
    documents only (replacing an already-retired one means nothing). The
    API decides whether the caller may actually replace one; this only
    offers them. Falls back to no choices if the API can't be reached --
    replacing is optional, the upload form must still render."""
    options: dict[str, str | None] = {_NO_REPLACEMENT: None}
    try:
        documents = httpx.get(f"{api_base_url}/documents", timeout=5.0).json()["documents"]
    except (httpx.HTTPError, ValueError, KeyError):
        return options
    for doc in documents:
        metadata = doc["metadata"]
        if metadata["status"] != "current":
            continue
        # The short id keeps two same-named documents from collapsing into
        # one choice.
        label = f"{doc['filename']} (v{metadata['version']}, {doc['doc_id'][:8]})"
        options[label] = doc["doc_id"]
    return options


_CLASSIFICATION_OPTIONS: dict[str, str] = {
    "Public": "public",
    "C1": "c1",
    "C2": "c2",
    "C3": "c3",
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
_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".md", ".markdown", ".csv", ".xlsx"}
_PROVIDER_CATALOG_PATH = Path(__file__).resolve().parent / "provider_catalog.json"
_LIBRA_LOGO_PATH = Path(__file__).resolve().parents[1] / "images" / "icone_LIBRA_AI.png"


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


def _parse_tags(raw: str) -> list[str]:
    """"runbook, q3-2026, " -> ["runbook", "q3-2026"] -- stripped, and
    empty entries (a trailing comma, or the field left blank) dropped
    rather than sent through as a literal "" tag."""
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


st.set_page_config(page_title="LIBRA AI", layout="wide")

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
    # retrieved_chunks (the full, mostly-uncited candidate set) isn't
    # persisted server-side -- only citations are, but citations carry a
    # full text/elements snapshot of their own (see api/db.py), so a
    # reloaded turn's citation buttons still work; only the "Retrieved
    # chunks (N)" expander is empty for a reloaded turn.
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
if "confirm_delete_doc_id" not in st.session_state:
    st.session_state.confirm_delete_doc_id = None
if "filter_tags" not in st.session_state:
    st.session_state.filter_tags: list[str] = []
if "filter_authors" not in st.session_state:
    st.session_state.filter_authors: list[str] = []
if "filter_date_from" not in st.session_state:
    st.session_state.filter_date_from: str | None = None
if "filter_date_to" not in st.session_state:
    st.session_state.filter_date_to: str | None = None

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
        method_label = st.selectbox("Retrieval method", list(_METHOD_OPTIONS.keys()), index=3)
        top_k = st.number_input("top_k", min_value=1, max_value=20, value=5)

    retrieval_method, rerank = _METHOD_OPTIONS[method_label]

    st.divider()
    st.header("Ingest a document")
    st.caption(f"API: {api_base_url}")
    try:
        health = httpx.get(f"{api_base_url}/health", timeout=5.0).json()
        st.caption(
            f"Status: {health['status']} (qdrant={health['qdrant']}, es={health['elasticsearch']})"
        )
    except httpx.HTTPError:
        st.caption("Status: unreachable")

    replace_options = _replace_options(api_base_url)

    # Deliberately OUTSIDE the form, same reasoning as the bulk
    # uploader below: a form only exposes a widget's value to the
    # script after its own submit button is clicked, which would make
    # it impossible to fetch tag suggestions (or preview skipped junk,
    # in the bulk case) before the user commits to that click.
    uploaded_file = st.file_uploader(
        "Choose a file", type=["pdf", "docx", "pptx", "md", "csv", "xlsx"], key="ingest_file"
    )

    # Fetched once per newly-selected file, not on every rerun a widget
    # inside the form below triggers -- the fingerprint is what makes
    # "still the same file" cheap to check without re-reading its bytes.
    if uploaded_file is None:
        st.session_state.suggested_tags_for = None
        st.session_state.suggested_tags_text = ""
    else:
        file_fingerprint = (uploaded_file.name, uploaded_file.size)
        if st.session_state.get("suggested_tags_for") != file_fingerprint:
            try:
                suggest_response = httpx.post(
                    f"{api_base_url}/suggest-tags",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    },
                    timeout=30.0,
                )
                suggest_response.raise_for_status()
                suggested = suggest_response.json()["tags"]
            except httpx.HTTPError:
                # A nice-to-have preview failing must never block the
                # actual ingest form below -- same fail-soft contract
                # the backend's own suggest_tags() already keeps.
                suggested = []
            st.session_state.suggested_tags_text = ", ".join(suggested)
            st.session_state.suggested_tags_for = file_fingerprint

    with st.form("ingest_form", clear_on_submit=True):
        replaces_label = st.selectbox(
            "This replaces (optional)",
            options=list(replace_options),
            help="Pick the document this new file supersedes. The old one stops "
            "appearing in answers but is kept, and the replacement can be undone.",
        )
        classification_label = st.selectbox(
            "Classification", options=list(_CLASSIFICATION_OPTIONS), index=0
        )
        is_private = st.checkbox(
            "Only visible to me",
            help="Enforced in search and the document list -- but this deployment's "
            "default auth_mode is 'disabled', where every caller shares one "
            "unrestricted identity, so there's no OTHER principal to hide it from "
            "yet. Meaningful once real distinct users exist (auth_mode='oidc').",
        )
        author_input = st.text_input(
            "Author (optional)", help="Who actually wrote this document, for the Filters panel."
        )
        doc_date_input = st.date_input(
            "Document date (optional)",
            value=None,
            help="When this document's CONTENT was written or published -- what the "
            "Filters panel's date range narrows on. Leave blank if unknown.",
        )
        tags_input = st.text_input(
            "Tags (optional, comma-separated)",
            value=st.session_state.get("suggested_tags_text", ""),
            help='Free-form labels for the Filters panel later -- e.g. "runbook, q3-2026". '
            "Pre-filled with AI-suggested tags once a file is chosen above, when available -- "
            "edit or clear them before submitting; nothing is saved until you click Ingest.",
        )
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
                metadata_json = json.dumps(
                    {
                        "classification": _CLASSIFICATION_OPTIONS[classification_label],
                        "private": is_private,
                        "author": author_input.strip() or None,
                        "doc_date": doc_date_input.isoformat() if doc_date_input else None,
                        "tags": _parse_tags(tags_input),
                    }
                )
                form_fields: dict = {
                    "file": file_payload,
                    "metadata_json": (None, metadata_json),
                }
                if apply_runtime_overrides:
                    form_fields["runtime_overrides_json"] = (
                        None,
                        json.dumps({"embedder": runtime_overrides["embedder"]}),
                    )
                replaces_id = replace_options[replaces_label]
                if replaces_id is not None:
                    form_fields["supersedes_doc_id"] = (None, replaces_id)
                response = httpx.post(
                    f"{api_base_url}/ingest",
                    files=form_fields,
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
        type=["pdf", "docx", "pptx", "md", "csv", "xlsx"],
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

    bulk_classification_label = st.selectbox(
        "Classification for every file in this batch",
        options=list(_CLASSIFICATION_OPTIONS),
        index=0,
        key="bulk_classification",
    )
    bulk_is_private = st.checkbox(
        "Only visible to me",
        key="bulk_private",
        help="Enforced in search and the document list -- but this deployment's "
        "default auth_mode is 'disabled', where every caller shares one "
        "unrestricted identity, so there's no OTHER principal to hide it from "
        "yet. Meaningful once real distinct users exist (auth_mode='oidc').",
    )
    bulk_author_input = st.text_input(
        "Author for every file in this batch (optional)", key="bulk_author"
    )
    bulk_doc_date_input = st.date_input(
        "Document date for every file in this batch (optional)",
        value=None,
        key="bulk_doc_date",
        help="Ingest files one at a time above if they need different dates.",
    )
    bulk_tags_input = st.text_input(
        "Tags for every file in this batch (optional, comma-separated)",
        key="bulk_tags",
        help='Applied to every file in this batch identically -- e.g. "runbook, q3-2026". '
        "Ingest files one at a time above if they need different tags.",
    )
    bulk_submitted = st.button("Ingest files", disabled=not good_files)

    if bulk_submitted:
        progress = st.progress(0.0)
        status_line = st.empty()
        counts = {"ingested": 0, "already_ingested": 0, "duplicate_content": 0}
        duplicates: list[str] = []
        failures: list[str] = []
        bulk_metadata_json = json.dumps(
            {
                "classification": _CLASSIFICATION_OPTIONS[bulk_classification_label],
                "private": bulk_is_private,
                "author": bulk_author_input.strip() or None,
                "doc_date": bulk_doc_date_input.isoformat() if bulk_doc_date_input else None,
                "tags": _parse_tags(bulk_tags_input),
            }
        )

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
                            "metadata_json": (None, bulk_metadata_json),
                            "runtime_overrides_json": (
                                None,
                                json.dumps({"embedder": runtime_overrides["embedder"]}),
                            ),
                        }
                        if apply_runtime_overrides
                        else {
                            "file": file_payload,
                            "metadata_json": (None, bulk_metadata_json),
                        }
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
            _metrics_response = httpx.get(f"{api_base_url}/metrics", timeout=10.0)
            _metrics_response.raise_for_status()  # a 500's body isn't JSON
            metrics = _metrics_response.json()
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

    # Bound here, not only inside the try below -- the Filters panel
    # (near the chat input, further down the script) reads this same
    # list to populate its tags/author/date options, and must still have
    # SOMETHING to iterate even when /documents itself failed to load.
    documents: list = []

    with st.expander("📚 Documents in the corpus"):
        try:
            _documents_response = httpx.get(f"{api_base_url}/documents", timeout=10.0)
            _documents_response.raise_for_status()  # a 500's body isn't JSON
            documents = _documents_response.json()["documents"]
            if documents:
                for _doc in documents:
                    _doc_id = _doc["doc_id"]
                    if st.session_state.confirm_delete_doc_id == _doc_id:
                        st.caption(f'Delete "{_doc["filename"]}"? This cannot be undone.')
                        _confirm_col, _cancel_col = st.columns(2)
                        if _confirm_col.button(
                            "Yes, delete", key=f"confirm_delete_doc_{_doc_id}", type="primary"
                        ):
                            try:
                                _del_response = httpx.delete(
                                    f"{api_base_url}/documents/{_doc_id}", timeout=60.0
                                )
                                _del_response.raise_for_status()
                                st.session_state.confirm_delete_doc_id = None
                                st.toast(
                                    f"Deleted {_del_response.json()['chunks_deleted']} chunk(s)."
                                )
                                st.rerun()
                            except httpx.HTTPError as exc:
                                st.error(f"Delete failed: {_http_error_detail(exc)}")
                        if _cancel_col.button("Cancel", key=f"cancel_delete_doc_{_doc_id}"):
                            st.session_state.confirm_delete_doc_id = None
                            st.rerun()
                    else:
                        _meta = _doc["metadata"]
                        _retired = " · ⚠️ superseded" if _meta["status"] == "superseded" else ""
                        _summary = (
                            f"{_doc['filename']} — v{_meta['version']}{_retired} — "
                            f"{_doc['num_parent_chunks']} parent, "
                            f"{_doc['num_child_chunks']} child chunks"
                        )
                        # An expander, not a flat row -- the tags/author/dates a
                        # document carries are useful but crowd out the list at
                        # a glance if always shown; click one to see them.
                        with st.expander(_summary):
                            st.caption(
                                f"Classification: **{_meta['classification']}**"
                                + (" · 🔒 only visible to its owner" if _meta["private"] else "")
                            )
                            if _meta.get("owner"):
                                st.caption(f"Owner: {_meta['owner']}")
                            if _meta.get("author"):
                                st.caption(f"Author: {_meta['author']}")
                            if _meta.get("doc_date"):
                                st.caption(f"Document date: {_meta['doc_date']}")
                            if _meta.get("data_type"):
                                st.caption(f"Type: {_meta['data_type']}")
                            if _meta.get("effective_from"):
                                st.caption(f"Effective from: {_meta['effective_from']}")
                            if _meta.get("effective_to"):
                                st.caption(f"Effective to: {_meta['effective_to']}")

                            _tags_input = st.text_input(
                                "Tags (comma-separated)",
                                value=", ".join(_meta["tags"]),
                                key=f"tags_input_{_doc_id}",
                                help="Editable here at any time -- a tag change is a "
                                "metadata-only patch, nothing gets re-embedded.",
                            )
                            _save_col, _delete_col = st.columns(2)
                            if _save_col.button("Save tags", key=f"save_tags_{_doc_id}"):
                                try:
                                    _patch_response = httpx.patch(
                                        f"{api_base_url}/documents/{_doc_id}",
                                        json={"tags": _parse_tags(_tags_input)},
                                        timeout=30.0,
                                    )
                                    _patch_response.raise_for_status()
                                    st.toast("Tags updated.")
                                    st.rerun()
                                except httpx.HTTPError as exc:
                                    st.error(f"Could not update tags: {_http_error_detail(exc)}")
                            if _delete_col.button("🗑️ Delete document", key=f"delete_doc_{_doc_id}"):
                                st.session_state.confirm_delete_doc_id = _doc_id
                                st.rerun()
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

_logo_col, _title_col, _new_conversation_col = st.columns([1, 5, 1])
with _logo_col:
    st.image(str(_LIBRA_LOGO_PATH), width=88)
with _title_col:
    st.title("LIBRA AI")
    st.caption("Library Intelligence & Reasoning Agent")
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
            # Citations now carry their own text/elements snapshot (see
            # api/db.py), so this works identically whether the turn is
            # live or reloaded -- no more looking a chunk up in
            # retrieved_chunks (empty for a reloaded turn) or disabling
            # the button when that lookup comes up empty.
            for c in turn["citations"]:
                location = _location_suffix(c["pages"], c["slides"])
                if st.button(
                    f"⟦{c['marker']}⟧ {c['source']}{location}",
                    key=f"cite_{turn['query_id']}_{c['marker']}",
                ):
                    _show_chunk_detail(c)

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

def _distinct_tags(documents: list) -> list[str]:
    seen: set[str] = set()
    for doc in documents:
        seen.update(doc["metadata"]["tags"])
    return sorted(seen)


def _distinct_authors(documents: list) -> list[str]:
    return sorted({doc["metadata"]["author"] for doc in documents if doc["metadata"]["author"]})


def _doc_date_bounds(documents: list) -> tuple[date, date] | None:
    """(earliest, latest) doc_date across the corpus, or None if no
    ingested document has one set at all -- the date filter has nothing
    real to bound itself by in that case, so it isn't shown rather than
    offering a meaningless single-point range."""
    dates = [
        date.fromisoformat(doc["metadata"]["doc_date"])
        for doc in documents
        if doc["metadata"]["doc_date"]
    ]
    return (min(dates), max(dates)) if dates else None


@st.dialog("🔍 Filters")
def _filters_dialog(documents: list) -> None:
    tag_options = _distinct_tags(documents)
    author_options = _distinct_authors(documents)
    bounds = _doc_date_bounds(documents)

    selected_tags = st.multiselect(
        "Tags",
        options=tag_options,
        # Filtered against options, not used as-is -- a tag/author
        # chosen in an earlier session can vanish from the corpus (its
        # last document deleted), and st.multiselect raises if `default`
        # contains a value `options` no longer has.
        default=[t for t in st.session_state.filter_tags if t in tag_options],
    )
    selected_authors = st.multiselect(
        "Author",
        options=author_options,
        default=[a for a in st.session_state.filter_authors if a in author_options],
    )

    if bounds is None:
        st.caption("No ingested document has a document date recorded yet.")
        selected_from, selected_to = None, None
    else:
        earliest, latest = bounds
        default_from = (
            date.fromisoformat(st.session_state.filter_date_from)
            if st.session_state.filter_date_from
            else earliest
        )
        default_to = (
            date.fromisoformat(st.session_state.filter_date_to)
            if st.session_state.filter_date_to
            else latest
        )
        date_range = st.date_input(
            "Document date range",
            value=(default_from, default_to),
            min_value=earliest,
            max_value=latest,
        )
        # A range date_input returns a 1-tuple while the user has picked
        # only the start of the pair (mid-selection, before the second
        # click) -- not yet a real range to filter by.
        if isinstance(date_range, tuple) and len(date_range) == 2:
            selected_from, selected_to = date_range
        else:
            selected_from, selected_to = earliest, latest

    apply_col, clear_col = st.columns(2)
    if apply_col.button("Apply", type="primary"):
        st.session_state.filter_tags = selected_tags
        st.session_state.filter_authors = selected_authors
        st.session_state.filter_date_from = (
            selected_from.isoformat() if selected_from is not None else None
        )
        st.session_state.filter_date_to = (
            selected_to.isoformat() if selected_to is not None else None
        )
        st.rerun()
    if clear_col.button("Clear filters"):
        st.session_state.filter_tags = []
        st.session_state.filter_authors = []
        st.session_state.filter_date_from = None
        st.session_state.filter_date_to = None
        st.rerun()


_active_filter_parts = []
if st.session_state.filter_tags:
    _active_filter_parts.append(f"tags: {', '.join(st.session_state.filter_tags)}")
if st.session_state.filter_authors:
    _active_filter_parts.append(f"author: {', '.join(st.session_state.filter_authors)}")
if st.session_state.filter_date_from or st.session_state.filter_date_to:
    _active_filter_parts.append(
        f"{st.session_state.filter_date_from or '…'} → {st.session_state.filter_date_to or '…'}"
    )

_filter_button_col, _filter_caption_col = st.columns([1, 5])
if _filter_button_col.button("🔍 Filters"):
    _filters_dialog(documents)
if _active_filter_parts:
    _filter_caption_col.caption("Filtering by " + " · ".join(_active_filter_parts))

prompt = st.chat_input("Ask a question")
if prompt and prompt.strip():
    _metadata_filter = {
        "tags": st.session_state.filter_tags or None,
        "author": st.session_state.filter_authors or None,
        "date_from": st.session_state.filter_date_from,
        "date_to": st.session_state.filter_date_to,
    }
    query_payload = {
        "question": prompt,
        "conversation_id": st.session_state.conversation_id,
        "metadata_filter": _metadata_filter,
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
