"""End-to-end RAG demo: ingest the sample corpus, ask one question that
IS answered in the corpus (shows grounded generation with citations),
and one question that sounds right at home in the corpus but ISN'T
actually answered (shows the refusal guardrail) — chunking_demo.md's
latency table has Avg and P95 columns, no P99, so asking for P99
latency is a genuine test of grounding discipline, not just an
obviously-unrelated question that would be an easy "refuse."

Run: `uv run python -m multimodal_rag.generation.demo`
"""

from pathlib import Path

from ..chunking.structure_aware import StructureAwareChunker
from ..ingestion import parse_document
from ..providers.factory import get_embedder
from ..retrieval.retriever import Retriever
from ..stores.factory import get_keyword_store, get_vector_store
from ..stores.indexer import HybridIndexer
from .chain import RagChain

_DOC = Path(__file__).resolve().parents[3] / "data" / "samples" / "chunking_demo.md"
_COLLECTION = "generation_demo"

_ANSWERABLE_QUESTION = "How does local inference latency compare to the internal gateway?"
_UNANSWERABLE_QUESTION = "What was the P99 latency for the internal gateway?"


def _print_answer(question: str) -> None:
    print("=" * 70)
    print(f'Q: "{question}"')
    print("=" * 70)


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

    retriever = Retriever(vector_store, keyword_store, embedder)
    chain = RagChain(retriever, top_k=3)

    for question in (_ANSWERABLE_QUESTION, _UNANSWERABLE_QUESTION):
        _print_answer(question)
        result = chain.answer(question)
        print(f"\nAnswer: {result.answer}")
        print(f"Refused: {result.refused}")
        if result.citations:
            print("Citations:")
            for citation in result.citations:
                location = ""
                if citation.pages:
                    location = f", page {', '.join(str(p) for p in citation.pages)}"
                print(f"  ⟦{citation.marker}⟧ {citation.source}{location}")
        print()


if __name__ == "__main__":
    main()
