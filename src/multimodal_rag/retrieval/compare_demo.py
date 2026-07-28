"""Comparison demo: runs the same technical query through every retrieval
method and prints the top chunks side by side.

Run: `uv run python -m multimodal_rag.retrieval.compare_demo`
"""

from pathlib import Path

from ..chunking.structure_aware import StructureAwareChunker
from ..ingestion import parse_document
from ..providers.factory import get_embedder, get_reranker
from ..stores.factory import get_keyword_store, get_vector_store
from ..stores.indexer import HybridIndexer
from ..stores.schema import SearchResult
from .retriever import Retriever
from .schema import RetrievalMethod

_DOC = Path(__file__).resolve().parents[3] / "data" / "samples" / "chunking_demo.md"
_COLLECTION = "retrieval_demo"
_QUERY = "How does local inference latency compare to the internal gateway?"


def _print_results(label: str, results: list[SearchResult]) -> None:
    print(f"\n--- {label} ---")
    if not results:
        print("  (no results)")
        return
    for rank, r in enumerate(results, start=1):
        preview = r.text[:90].replace("\n", " ")
        print(f"  #{rank}  score={r.score:.4f}  chunk_id=...{r.chunk_id[-30:]}")
        print(f'       "{preview}..."')


def main() -> None:
    elements = parse_document(_DOC)
    chunks = StructureAwareChunker().chunk(elements)
    print(f"Parsed {len(elements)} elements -> {len(chunks)} chunks from {_DOC.name}")

    embedder = get_embedder()
    vectors = embedder.embed([c.text for c in chunks])

    vector_store = get_vector_store(collection_name=_COLLECTION)
    vector_store.create_collection(dimension=vectors[0].dimension, indexing_threshold=0)
    keyword_store = get_keyword_store(index_name=_COLLECTION)
    keyword_store.create_index()

    HybridIndexer(vector_store, keyword_store).index(chunks, vectors)
    vector_store.publish()

    try:
        reranker = get_reranker()
    except NotImplementedError:
        reranker = None

    retriever = Retriever(vector_store, keyword_store, embedder, reranker=reranker)

    print(f'\nQuery: "{_QUERY}"')
    print("=" * 70)

    for method in RetrievalMethod:
        results = retriever.retrieve(_QUERY, method=method, top_k=2)
        _print_results(method.value, results)

    if reranker is not None:
        results = retriever.retrieve(
            _QUERY, method=RetrievalMethod.HYBRID_RRF, top_k=2, rerank=True
        )
        _print_results("hybrid_rrf + rerank", results)


if __name__ == "__main__":
    main()
