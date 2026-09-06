"""End-to-end RAG demo: ingest the sample corpus, then show four
behaviors side by side --
  1. RagChain: one retrieve + one generate, no tool calling (the simple
     baseline every AgentChain call is built on top of).
  2. AgentChain on a compound question -- one half answered by the
     Results table, the other by the Discussion section, so this is a
     genuine test of whether the agent decomposes into separate searches
     rather than one vague one. Every search query and every chunk it
     returns is printed via `on_tool_call`, not just the final answer.
  3. AgentChain on a question that sounds at home in the corpus but
     ISN'T actually answered (same trap as (1) -- chunking_demo.md's
     latency table has Avg and P95 columns, no P99) -- shows the agent
     searching for real, coming up empty, and refusing rather than
     guessing.
  4. AgentChain across two turns of an actual conversation: the first
     message is genuinely ambiguous with zero prior context ("it"/"the
     one we discussed earlier" resolve to nothing, since nothing was
     actually discussed earlier), so the agent should ask a clarifying
     question instead of guessing or searching speculatively (the
     human-in-the-loop path) -- then the follow-up reply is threaded in
     as real history, giving the agent enough to search and answer for
     real. This is what a chat turn where the agent has to ask back
     actually looks like end to end.

Run: `uv run python -m multimodal_rag.generation.demo`
"""

from pathlib import Path

from ..chunking.parent_child import ParentChildChunker
from ..ingestion import parse_document
from ..providers.factory import get_embedder
from ..retrieval.retriever import Retriever
from ..stores.factory import get_keyword_store, get_vector_store
from ..stores.indexer import HybridIndexer
from ..stores.schema import SearchResult
from .agent import AgentChain
from .chain import RagChain
from .schema import RagAnswer

_DOC = Path(__file__).resolve().parents[3] / "data" / "samples" / "chunking_demo.md"
_COLLECTION = "generation_demo"

_SINGLE_SHOT_QUESTION = "How does local inference latency compare to the internal gateway?"
_COMPOUND_QUESTION = (
    "What was the P95 latency for the OpenAI hosted model, and what real-world cost does the "
    "local on-device configuration have that a latency-only comparison doesn't capture?"
)
_UNANSWERABLE_QUESTION = "What was the P99 latency for the internal gateway?"

_AMBIGUOUS_OPENING_QUESTION = "Can you compare it with the one we discussed earlier?"
_CLARIFYING_REPLY = "I meant the local on-device configuration compared to the internal gateway."


def _print_header(label: str, question: str) -> None:
    print("=" * 70)
    print(label)
    print(f'Q: "{question}"')
    print("=" * 70)


def _location_suffix(pages: list[int], slides: list[int]) -> str:
    if pages:
        return f", page {', '.join(str(p) for p in pages)}"
    if slides:
        return f", slide {', '.join(str(s) for s in slides)}"
    return ""


def _print_tool_call(round_index: int, query: str, results: list[SearchResult]) -> None:
    print(f"\n  [search #{round_index}] query: {query!r}")
    if not results:
        print("    -> no results")
        return
    for result in results:
        location = _location_suffix(result.pages, result.slides)
        print(f"    -> {result.chunk_id} ({result.source}{location}, score={result.score:.3f})")


def _print_answer(result: RagAnswer) -> None:
    print(f"\nAnswer: {result.answer}")
    print(f"Refused: {result.refused}")
    print(f"Needs clarification: {result.needs_clarification}")
    if result.citations:
        print("Citations:")
        for citation in result.citations:
            location = _location_suffix(citation.pages, citation.slides)
            print(f"  ⟦{citation.marker}⟧ {citation.chunk_id}{location} - {citation.source}")
    print()


def main() -> None:
    elements = parse_document(_DOC)
    chunks = ParentChildChunker().chunk(elements)
    num_parents = sum(1 for c in chunks if c.parent_id is None)
    num_children = len(chunks) - num_parents
    print(
        f"Parsed {len(elements)} elements -> {num_parents} parent chunks, "
        f"{num_children} child chunks from {_DOC.name}"
    )

    embedder = get_embedder()
    vectors = embedder.embed([c.text for c in chunks])

    vector_store = get_vector_store(collection_name=_COLLECTION)
    vector_store.create_collection(dimension=vectors[0].dimension, indexing_threshold=0)
    keyword_store = get_keyword_store(index_name=_COLLECTION)
    keyword_store.create_index()

    HybridIndexer(vector_store, keyword_store).index(chunks, vectors)
    vector_store.publish()

    retriever = Retriever(vector_store, keyword_store, embedder)
    # resolve_parent_context=True: a matched child chunk (small, precise
    # -- good for retrieval) has its text swapped for its parent's fuller
    # text (good for generation) before it reaches the LLM, while the
    # citation still points at the child, not the broader parent.

    _print_header("1. RagChain -- single-shot, no tool calling", _SINGLE_SHOT_QUESTION)
    rag_chain = RagChain(retriever, top_k=3, resolve_parent_context=True)
    _print_answer(rag_chain.answer(_SINGLE_SHOT_QUESTION))

    agent = AgentChain(retriever, top_k=3, resolve_parent_context=True)

    _print_header("2. AgentChain -- compound question, real tool calls", _COMPOUND_QUESTION)
    _print_answer(agent.answer(_COMPOUND_QUESTION, on_tool_call=_print_tool_call))

    _print_header("3. AgentChain -- unanswerable question, refusal", _UNANSWERABLE_QUESTION)
    _print_answer(agent.answer(_UNANSWERABLE_QUESTION, on_tool_call=_print_tool_call))

    _print_header(
        "4. AgentChain -- conversation with a clarifying-question round-trip",
        _AMBIGUOUS_OPENING_QUESTION,
    )
    turn_1 = agent.answer(_AMBIGUOUS_OPENING_QUESTION, on_tool_call=_print_tool_call)
    _print_answer(turn_1)

    print(f'Follow-up: "{_CLARIFYING_REPLY}"')
    turn_2 = agent.answer(
        _CLARIFYING_REPLY,
        history=[(_AMBIGUOUS_OPENING_QUESTION, turn_1.answer)],
        on_tool_call=_print_tool_call,
    )
    _print_answer(turn_2)


if __name__ == "__main__":
    main()
