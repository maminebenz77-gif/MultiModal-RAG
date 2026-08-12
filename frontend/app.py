"""Streamlit demo UI for the RAG API.

Deliberately talks to the API only over HTTP (httpx), never imports
`multimodal_rag` directly -- the API is the real boundary between
backend and frontend, not just an internal layering convention, and
this keeps that true in practice, not just on paper.

Run: `uv run streamlit run frontend/app.py`

Streamlit re-runs this whole script top-to-bottom on every widget
interaction -- the last query's result has to live in
st.session_state, or it would vanish the moment you touched an
unrelated widget (like the retrieval-method dropdown).
"""

import sys
import json
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

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "confirm_wipe" not in st.session_state:
    st.session_state.confirm_wipe = False

provider_catalog = _load_provider_catalog()


def _location_suffix(pages: list[int], slides: list[int]) -> str:
    if pages:
        return f", page {', '.join(str(p) for p in pages)}"
    if slides:
        return f", slide {', '.join(str(s) for s in slides)}"
    return ""


def _submit_feedback(query_id: str, rating: str, comment: str) -> None:
    try:
        response = httpx.post(
            f"{api_base_url}/feedback",
            json={"query_id": query_id, "rating": rating, "comment": comment or None},
            timeout=30.0,
        )
        response.raise_for_status()
        st.toast("Feedback recorded, thank you!")
        # The Metrics panel above (in script order) already ran and read
        # the pre-feedback counts this render -- st.toast() is specifically
        # designed to survive an immediate rerun, so the confirmation still
        # shows even though the script restarts right after.
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

st.title("Multimodal RAG Demo")

with st.expander("📈 Metrics"):
    try:
        metrics = httpx.get(f"{api_base_url}/metrics", timeout=10.0).json()
        tile_cols = st.columns(6)
        tile_cols[0].metric("Documents", metrics["total_documents"])
        tile_cols[1].metric("Chunks", metrics["total_chunks"])
        tile_cols[2].metric("Queries", metrics["total_queries"])
        tile_cols[3].metric("Refusal rate", f"{metrics['refusal_rate']:.0%}")
        tile_cols[4].metric("👍 Helpful", metrics["feedback_up"])
        tile_cols[5].metric("👎 Not helpful", metrics["feedback_down"])
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

with st.form("query_form"):
    question = st.text_input("Question")
    col1, col2 = st.columns(2)
    with col1:
        method_label = st.selectbox(
            "Retrieval method", list(_METHOD_OPTIONS.keys()), index=3
        )
    with col2:
        top_k = st.number_input("top_k", min_value=1, max_value=20, value=5)
    ask_submitted = st.form_submit_button("Ask")

if ask_submitted and question.strip():
    retrieval_method, rerank = _METHOD_OPTIONS[method_label]
    try:
        response = httpx.post(
            f"{api_base_url}/query",
            json=(
                {
                    "question": question,
                    "retrieval_method": retrieval_method,
                    "top_k": top_k,
                    "rerank": rerank,
                    "runtime_overrides": runtime_overrides,
                }
                if apply_runtime_overrides
                else {
                    "question": question,
                    "retrieval_method": retrieval_method,
                    "top_k": top_k,
                    "rerank": rerank,
                }
            ),
            timeout=120.0,
        )
        response.raise_for_status()
        st.session_state.last_result = response.json()
        # Same reason as the feedback rerun above: the backend recorded
        # this query (and its refusal/method) before this response came
        # back, but the Metrics panel already rendered earlier in this
        # same script run, before the query even started -- only a fresh
        # rerun picks up the updated count.
        st.rerun()
    except httpx.HTTPError as exc:
        st.error(f"Query failed: {_http_error_detail(exc)}")

result = st.session_state.last_result
if result is not None:
    st.subheader("Answer")
    if result["refused"]:
        st.warning(result["answer"])
    else:
        st.write(result["answer"])

    if result["citations"]:
        st.markdown("**Citations**")
        for c in result["citations"]:
            location = _location_suffix(c["pages"], c["slides"])
            st.markdown(f"⟦{c['marker']}⟧ **{c['source']}**{location}")

    st.markdown("**Was this helpful?**")
    comment = st.text_input("Comment (optional)", key="feedback_comment")
    fb_col1, fb_col2 = st.columns(2)
    if fb_col1.button("👍 Helpful"):
        _submit_feedback(result["query_id"], "up", comment)
    if fb_col2.button("👎 Not helpful"):
        _submit_feedback(result["query_id"], "down", comment)

    with st.expander(f"Retrieved chunks ({len(result['retrieved_chunks'])})"):
        for chunk in result["retrieved_chunks"]:
            location = _location_suffix(chunk["pages"], chunk["slides"])
            st.markdown(f"**{chunk['source']}**{location} (score={chunk['score']:.3f})")
            st.caption(chunk["chunk_id"])
            st.text(chunk["text"])
            st.divider()
