"""Comparison demo: indexes the same document into both Qdrant (vector)
and Elasticsearch (BM25), then runs the same queries against both,
printing ranked results side by side — so a query where lexical search
wins and one where semantic search wins can be observed directly rather
than assumed.

Run: `uv run python -m multimodal_rag.stores.hybrid_compare_demo`
"""

from pathlib import Path

from ..chunking.structure_aware import StructureAwareChunker
from ..ingestion import parse_document
from ..providers.factory import get_embedder
from .factory import get_keyword_store, get_vector_store
from .indexer import HybridIndexer
from .schema import SearchResult

_DOC = Path(__file__).resolve().parents[3] / "data" / "samples" / "chunking_demo.md"
_COLLECTION = "hybrid_demo"

_QUERIES = [
    "internal-llama-70b",
    "What are the downsides of running the model on my own hardware "
    "instead of using someone else's servers?",
]


def _print_results(label: str, results: list[SearchResult]) -> None:
    print(f"  {label}:")
    if not results:
        print("    (no results)")
        return
    for rank, r in enumerate(results, start=1):
        preview = r.text[:90].replace("\n", " ")
        print(f'    #{rank}  score={r.score:.4f}  chunk_id=...{r.chunk_id[-30:]}  "{preview}..."')


def main() -> None:
    elements = parse_document(_DOC)
    chunks = StructureAwareChunker().chunk(elements)
    print(f"Parsed {len(elements)} elements -> {len(chunks)} chunks from {_DOC.name}\n")

    embedder = get_embedder()
    vectors = embedder.embed([c.text for c in chunks])

    vector_store = get_vector_store(collection_name=_COLLECTION)
    vector_store.create_collection(dimension=vectors[0].dimension, indexing_threshold=0)
    keyword_store = get_keyword_store(index_name=_COLLECTION)
    keyword_store.create_index()

    # Coordinated write: if keyword indexing fails after the vector store
    # write already succeeded, this raises IndexConsistencyError naming
    # exactly which chunk_ids are now out of sync, instead of leaving the
    # two stores silently disagreeing.
    HybridIndexer(vector_store, keyword_store).index(chunks, vectors)
    vector_store.publish()

    for query in _QUERIES:
        print("=" * 70)
        print(f'Query: "{query}"')
        print("=" * 70)

        query_vector = embedder.embed([query])[0]
        vector_results = vector_store.search(query_vector, top_k=2)
        keyword_results = keyword_store.search(query, top_k=2)

        _print_results("Vector (Qdrant, cosine)", vector_results)
        _print_results("Keyword (Elasticsearch, BM25)", keyword_results)
        print()


if __name__ == "__main__":
    main()
