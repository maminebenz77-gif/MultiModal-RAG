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


def _submit_feedback(query_id: str, rating: str, comment: str) -> None:
    try:
        response = httpx.post(
            f"{api_base_url}/feedback",
            json={"query_id": query_id, "rating": rating, "comment": comment or None},
            timeout=30.0,
        )
        response.raise_for_status()
        st.toast("Feedback recorded, thank you!")
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
                st.success(
                    f"Ingested {body['filename']}: {body['num_parent_chunks']} parent "
                    f"chunks, {body['num_child_chunks']} child chunks."
                )
            except httpx.HTTPError as exc:
                st.error(f"Ingest failed: {exc}")

st.title("Multimodal RAG Demo")

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
