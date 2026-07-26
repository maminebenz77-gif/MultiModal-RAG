"""End-to-end demo: ingest a real sample document, chunk it structure-
aware (so element_types/element_positions are populated), embed it,
upsert into a real local Qdrant collection, and search it.

indexing_threshold=0 forces Qdrant to actually build an HNSW graph even
on our tiny sample corpus — Qdrant serves any collection below its
default threshold (20000 vectors) via plain brute-force regardless of
HNSW config, so without this override, changing ef_search below would
have *no observable effect at all*.

Honest expectation-setting: on a corpus this small, don't expect a
dramatic recall swing between ef_search values — HNSW's approximate
behavior shows up at real scale (hundreds of thousands+ of vectors).
This demo makes the parameters real and experimentable, not a proof
that low ef_search visibly degrades results here.

Run: `uv run python -m multimodal_rag.stores.search_demo --ef-search 16`
"""

import argparse
from pathlib import Path

from ..chunking.structure_aware import StructureAwareChunker
from ..ingestion import parse_document
from ..providers.factory import get_embedder
from .factory import get_vector_store

_DOC = Path(__file__).resolve().parents[3] / "data" / "samples" / "chunking_demo.md"
_QUERY = "How much slower is the internal gateway compared to running locally?"
_COLLECTION = "search_demo"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ef-search",
        type=int,
        default=None,
        help="query-time HNSW ef; unset = Qdrant's default",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--query", default=_QUERY)
    args = parser.parse_args()

    elements = parse_document(_DOC)
    chunks = StructureAwareChunker().chunk(elements)
    print(
        f"Parsed {len(elements)} elements -> {len(chunks)} structure-aware "
        f"chunks from {_DOC.name}"
    )

    embedder = get_embedder()
    vectors = embedder.embed([c.text for c in chunks])
    print(f"Embedded with {vectors[0].model_id} ({vectors[0].dimension}-dim)")

    store = get_vector_store(collection_name=_COLLECTION)
    store.create_collection(dimension=vectors[0].dimension, indexing_threshold=0)
    store.upsert(chunks, vectors)
    store.publish()
    print(f"Upserted {len(chunks)} chunks into Qdrant collection {_COLLECTION!r}\n")

    query_vector = embedder.embed([args.query])[0]
    results = store.search(query_vector, top_k=args.top_k, ef_search=args.ef_search)

    print(f'Query: "{args.query}"')
    print(f"ef_search: {args.ef_search if args.ef_search is not None else '(Qdrant default)'}")
    print("=" * 70)
    for rank, result in enumerate(results, start=1):
        print(f"\n#{rank}  score={result.score:.4f}  chunk_id={result.chunk_id}")
        print(
            f"    doc_id={result.doc_id}  element_types={result.element_types}  "
            f"model_id={result.model_id}"
        )
        preview = result.text[:200].replace("\n", " ")
        print(f'    "{preview}..."')


if __name__ == "__main__":
    main()
