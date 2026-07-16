"""Comparison demo: runs the same document through all five chunking
strategies and prints chunk counts, sizes, and an example chunk from
each — meant to be read, not tested.

Run: `uv run python -m multimodal_rag.chunking.compare_demo`
"""

from pathlib import Path

from ..ingestion import parse_document
from .base import Chunker
from .fixed_size import FixedSizeChunker
from .parent_child import ParentChildChunker
from .recursive import RecursiveCharacterChunker
from .semantic import SemanticChunker
from .structure_aware import StructureAwareChunker

_DOC = Path(__file__).resolve().parents[3] / "data" / "samples" / "chunking_demo.md"

_STRATEGIES: dict[str, Chunker] = {
    "1. fixed-size": FixedSizeChunker(),
    "2. recursive": RecursiveCharacterChunker(),
    "3. structure-aware": StructureAwareChunker(),
    "4. semantic": SemanticChunker(),
    "5. parent-child": ParentChildChunker(),
}


def main() -> None:
    elements = parse_document(_DOC)
    print(f"Parsed {len(elements)} elements from {_DOC.name}")
    print("=" * 70)

    for name, chunker in _STRATEGIES.items():
        chunks = chunker.chunk(elements)
        parents = [c for c in chunks if c.parent_id is None]
        children = [c for c in chunks if c.parent_id is not None]
        sizes = [len(c.text) for c in (children or parents)]

        print(f"\n{name}")
        print(f"  chunk count: {len(chunks)}", end="")
        if children:
            print(f"  ({len(parents)} parents, {len(children)} children)")
        else:
            print()
        if sizes:
            avg = sum(sizes) // len(sizes)
            print(f"  size range: {min(sizes)}-{max(sizes)} chars, avg {avg}")
        if elements_tracked := [c for c in chunks if c.metadata.element_positions]:
            print(f"  element_positions tracked on {len(elements_tracked)}/{len(chunks)} chunks")
        else:
            print("  element_positions tracked on 0 chunks (flattened to raw text)")

        example = (children or parents)[0] if (children or parents) else None
        if example:
            preview = example.text[:180].replace("\n", " ")
            print(f'  example: "{preview}..."')

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
