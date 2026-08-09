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
from pathlib import Path

# `streamlit run` puts this script's own directory on sys.path
# automatically, but other ways of executing this file (Streamlit's own
# AppTest harness, `python -m` invocations, ...) don't -- inserting it
# explicitly makes the local `config` import work regardless of how the
# script was launched, rather than depending on that implicit behavior.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx
import streamlit as st
from config import get_frontend_settings

_METHOD_OPTIONS: dict[str, tuple[str, bool]] = {
    "Cosine similarity": ("cosine", False),
    "MMR (diversity)": ("mmr", False),
    "BM25 (keyword)": ("bm25", False),
    "Hybrid (RRF)": ("hybrid_rrf", False),
    "Hybrid + Rerank": ("hybrid_rrf", True),
}

# Matches what parse_document() actually supports (see
# multimodal_rag.ingestion) -- .markdown alongside .md since the
# dispatcher's extension fallback recognizes both.
_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".md", ".markdown"}
_EXTENSION_TO_MIME = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}

st.set_page_config(page_title="Multimodal RAG Demo", layout="wide")

api_base_url = get_frontend_settings().api_base_url

if "last_result" not in st.session_state:
    st.session_state.last_result = None


def _location_suffix(pages: list[int], slides: list[int]) -> str:
    if pages:
        return f", page {', '.join(str(p) for p in pages)}"
    if slides:
        return f", slide {', '.join(str(s) for s in slides)}"
    return ""


def _ingest_file(path: Path) -> dict:
    mime = _EXTENSION_TO_MIME.get(path.suffix.lower(), "application/octet-stream")
    with open(path, "rb") as f:
        response = httpx.post(
            f"{api_base_url}/ingest",
            files={"file": (path.name, f, mime)},
            timeout=120.0,
        )
    response.raise_for_status()
    result: dict = response.json()
    return result


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


with st.sidebar:
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
                    files={"file": file_payload},
                    timeout=120.0,
                )
                response.raise_for_status()
                body = response.json()
                verb = "Already ingested" if body["status"] == "already_ingested" else "Ingested"
                st.success(
                    f"{verb} {body['filename']}: {body['num_parent_chunks']} parent "
                    f"chunks, {body['num_child_chunks']} child chunks."
                )
            except httpx.HTTPError as exc:
                st.error(f"Ingest failed: {exc}")

    st.divider()
    st.header("Bulk ingest a folder")
    st.caption("Top-level files only (.pdf, .docx, .pptx, .md) -- subfolders aren't walked.")

    if "browse_dir" not in st.session_state:
        # Home, not the project directory -- this browser can already
        # navigate anywhere on disk (nothing restricts "Up" past the
        # project root), but starting inside the project meant reaching
        # anywhere else took several clicks before you could even start
        # heading toward it.
        st.session_state.browse_dir = str(Path.home())

    with st.expander("📁 Browse for a folder"):
        current_dir = Path(st.session_state.browse_dir)
        st.caption(str(current_dir))
        try:
            entries = sorted(current_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            entries = []
            st.warning("Can't read this folder.")

        subdirs = [p for p in entries if p.is_dir() and not p.name.startswith(".")]
        supported_here = [p for p in entries if p.suffix.lower() in _SUPPORTED_EXTENSIONS]
        st.caption(f"{len(supported_here)} supported file(s) here.")

        up_col, use_col = st.columns(2)
        if up_col.button(
            "⬆️ Up", disabled=current_dir.parent == current_dir, width="stretch"
        ):
            st.session_state.browse_dir = str(current_dir.parent)
            st.rerun()
        if use_col.button("Use this folder", type="primary", width="stretch"):
            st.session_state.folder_path_text = str(current_dir)

        for sub in subdirs:
            if st.button(f"📁 {sub.name}", key=f"browse_into_{sub}", width="stretch"):
                st.session_state.browse_dir = str(sub)
                st.rerun()

    with st.form("bulk_ingest_form"):
        folder_path_input = st.text_input("Folder path", key="folder_path_text")
        bulk_submitted = st.form_submit_button("Ingest folder")

    if bulk_submitted:
        folder = Path(folder_path_input).expanduser()
        if not folder.is_dir():
            st.error(f"Not a folder: {folder}")
        else:
            files = sorted(p for p in folder.iterdir() if p.suffix.lower() in _SUPPORTED_EXTENSIONS)
            if not files:
                st.warning(f"No supported files found in {folder}")
            else:
                progress = st.progress(0.0)
                status_line = st.empty()
                counts = {"ingested": 0, "already_ingested": 0}
                failures: list[str] = []

                for i, path in enumerate(files, start=1):
                    status_line.text(f"({i}/{len(files)}) {path.name}...")
                    try:
                        body = _ingest_file(path)
                        counts[body["status"]] += 1
                    except httpx.HTTPError as exc:
                        failures.append(f"{path.name}: {exc}")
                    progress.progress(i / len(files))

                status_line.empty()
                progress.empty()
                st.success(
                    f"Done: {counts['ingested']} newly ingested, "
                    f"{counts['already_ingested']} already up to date, "
                    f"{len(failures)} failed."
                )
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
            json={
                "question": question,
                "retrieval_method": retrieval_method,
                "top_k": top_k,
                "rerank": rerank,
            },
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
        st.error(f"Query failed: {exc}")

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
