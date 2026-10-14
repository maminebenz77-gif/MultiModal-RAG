# Multimodal Technical RAG

A multimodal Retrieval-Augmented Generation (RAG) system for technical documents (text, tables, diagrams, images), built as a real production-minded project. See [`CLAUDE.md`](./CLAUDE.md) for how we work on this repo.

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

## Document generation (docgen)

A second, independent capability sits alongside the query pipeline above: given a free-text
request and one or two tagged document sets (a required "task" set, and an optional
"reference" set for comparison-style questions), `docgen` turns that into a confirmed list of
questions, answers each one with retrieval-augmented generation (retry on a failed groundedness
check, escalate to a human once retries are exhausted), harmonizes tone across every answer,
waits for a human to review the result, and exports a `.pptx`/`.docx` — all as one autonomous
job that can run for minutes and pause several times waiting on a person, never as part of a
chat turn.

It's built on [LangGraph](https://github.com/langchain-ai/langgraph), not the query pipeline's
own hand-rolled agent loop, and runs from three different entry points:

- **Terminal**: `uv run python -m multimodal_rag.docgen.cli` — the original, proven-first
  entry point; a paused run survives a killed-and-restarted process.
- **Web**: `POST /docgen/runs` (and `GET`/`resume`/`download`) on the same API, driven from a
  dedicated **Docgen** page in the Streamlit frontend.
- **Evaluation**: `uv run python -m multimodal_rag.evaluation.run_docgen_eval`, against real
  two-document-set test fixtures under `data/eval/` — also run automatically on every push
  (see `.github/workflows/docgen-eval.yml`).

See `docs/technical-decisions.md` (§20–22) for the architecture and the real bugs this build
caught along the way, and `data/eval/README.md` for how to add your own test set.

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
    evaluation/       # quality measurement (including the docgen eval)
    api/             # service layer (query + docgen routers)
    docgen/           # autonomous document-generation workflow (LangGraph)
tests/               # test suite (mirrors the package structure)
data/                # local data artifacts (gitignored; not committed)
frontend/            # UI (the chat page, plus pages/1_Docgen.py)
```

Each module is independent and swappable by design — that's a deliberate architectural choice for a system whose components (chunking strategy, embedding model, vector store) commonly get swapped out during iteration.

## Status

Both capabilities are fully built and tested end-to-end: the query pipeline (ingest → chunk →
embed → store → retrieve → generate, exposed via `/query`) and docgen (above). Built
incrementally; see `CLAUDE.md` for the process and `docs/technical-decisions.md` for the
decision-by-decision history.

## Development

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management, [`ruff`](https://docs.astral.sh/ruff/) for linting/formatting, [`pytest`](https://docs.pytest.org/) for tests, and [`mypy`](https://mypy-lang.org/) for static typing.

```bash
uv sync              # install dependencies
uv run pytest        # run tests
uv run ruff check .  # lint
uv run mypy .        # type-check
```

## Running with Docker

One image, two profiles — a base `docker-compose.yml` plus one override per environment:

- **Local** (`docker-compose.local.yml`) — CPU/MPS, external providers (OpenAI, litellm) allowed, points at Elasticsearch on `localhost`.
- **Server** (`docker-compose.server.yml`) — NVIDIA GPU, no internet access at runtime, embedding + reranker model weights baked into the image at build time, points at the internal LLM gateway.

**Local, from clean, one command:**

```bash
docker compose -f docker-compose.yml -f docker-compose.local.yml up --build
```

**Server**, built on a machine with internet access (the weights are downloaded once, at build time, and baked into the image so the server itself never needs internet). Use the wrapper script rather than a bare `docker compose build` — it reads the real `.env.server` so the baked-in weights always match what it configures:

```bash
uv run python scripts/build_server_image.py
docker compose -f docker-compose.yml -f docker-compose.server.yml up -d
```

The server profile also needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host for GPU access, and a real `.env.server` filled in (copy `.env.server.example`).
