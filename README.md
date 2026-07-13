# Multimodal Technical RAG

A multimodal Retrieval-Augmented Generation (RAG) system for technical documents (text, tables, diagrams, images), built incrementally as a learning project. See [`CLAUDE.md`](./CLAUDE.md) for how we work on this repo, and [`Multimodal_RAG_Build_Plan.md`](./Multimodal_RAG_Build_Plan.md) for the full build plan.

## Architecture, at a high level

A RAG system answers questions over a document corpus by retrieving relevant chunks and feeding them to an LLM as context, instead of relying on the model's training data alone. This project is *multimodal*: source documents contain text, tables, and images/diagrams, and all of those need to be ingested, indexed, and retrievable together.

The pipeline is split into independent, swappable stages:

```
raw documents
     │
     ▼
 ingestion      — load and parse source files (PDFs, docs, images) into a common representation
     │
     ▼
 chunking       — split content into retrieval-sized units (text passages, table blocks, image crops)
     │
     ▼
 embeddings     — turn chunks into vectors (text embeddings, and image/multimodal embeddings)
     │
     ▼
 stores         — persist chunks + vectors in a vector store / index
     │
     ▼
 retrieval      — given a query, find the most relevant chunks
     │
     ▼
 generation     — feed retrieved chunks + query to an LLM to produce an answer
```

Two more modules sit alongside the pipeline:

- **`providers`** — thin wrappers around external LLM/embedding APIs, so the rest of the code depends on an internal interface rather than a specific vendor SDK.
- **`evaluation`** — measures retrieval and generation quality (e.g., "did we retrieve the right chunk?", "was the answer grounded in it?").

And two layers sit on top, once the pipeline works end-to-end:

- **`api`** — a service layer exposing the pipeline (e.g., a query endpoint).
- **`frontend`** — a UI for asking questions and inspecting retrieved sources.

## Project layout

```
src/multimodal_rag/
    providers/      # LLM + embedding client wrappers
    ingestion/       # document loading/parsing
    chunking/        # splitting content into retrievable units
    embeddings/       # vectorization
    stores/          # vector store / persistence
    retrieval/       # query -> relevant chunks
    generation/      # chunks + query -> answer
    evaluation/       # quality measurement
    api/             # service layer
tests/               # test suite (mirrors the package structure)
data/                # local data artifacts (gitignored; not committed)
frontend/            # UI
```

Each module is independent and swappable by design — that's a deliberate architectural choice for a system whose components (chunking strategy, embedding model, vector store) commonly get swapped out during iteration.

## Status

Skeleton only — no pipeline logic implemented yet. Built incrementally; see `CLAUDE.md` for the process.

## Development

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management, [`ruff`](https://docs.astral.sh/ruff/) for linting/formatting, [`pytest`](https://docs.pytest.org/) for tests, and [`mypy`](https://mypy-lang.org/) for static typing.

```bash
uv sync              # install dependencies
uv run pytest        # run tests
uv run ruff check .  # lint
uv run mypy .        # type-check
```
