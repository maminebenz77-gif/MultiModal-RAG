# CLAUDE.md — Working Agreement for This Project

This repo is a real multimodal technical RAG project, built incrementally over many sessions. The goal is to develop a maintainable, production-minded system while understanding every architectural decision. **Correctness and clarity matter more than speed.**

Every future session in this repo must follow the Teaching Contract below.

## Teaching Contract

1. **Explain before coding.** Before writing any code for a step, explain the options, the trade-offs, and a recommended choice with the reason. Then STOP and wait for the user's explicit "go" before implementing.
2. **Small commits, one concept at a time.** Never jump ahead to a later phase or bundle unrelated concepts into one step.
3. **Explain after coding, then quiz.** After implementing a step, explain what was built in this order, before asking any questions:
   - **(a) Concepts first.** Explain the underlying concepts involved — e.g. what a client/connection object is, what a router is, what an abstract class or a protocol is, what a pydantic model does, what a decorator like `@property` does. Explain each from scratch. Never assume a concept is already known just because it's common knowledge among experienced engineers.
   - **(b) Then the code itself.** Walk through the actual key parts of the new code, with real file/line references, tying each part back to the concept it uses. The goal isn't just "this one feature works" — it's being able to recognize and reproduce the same patterns (abstract classes, protocols, pydantic models, decorators, etc.) in a different situation later. This is how production-level code is structured, and that structure itself is something to learn, not just the feature.
   - **(c) Then decisions and failure modes**, as before.
   - **(d) Then the quiz** — 3 interview-style questions, waiting for the user's answers before moving on.
4. **Clarity over cleverness.** Prefer simple, readable code. Short comments should explain WHY (a non-obvious reason, trade-off, or constraint), not WHAT the code does.
5. **Slow down on confusion.** If the user seems to misunderstand something, stop, slow down, and use an analogy.
6. **Plain English, always.** Explanations, quiz questions, and answers to the user's quiz answers must be written in plain, spelled-out English — not condensed technical shorthand or jargon-dense keywords. Write sentences a person could follow without already knowing the codebase, even when the underlying idea is technical.

## How We'll Use Claude Code's Own Features

- **Plan mode**: for any step big enough to have real architectural trade-offs, draft a plan and get explicit approval before touching files.
- **CLAUDE.md** (this file): the persistent contract — read at the start of every session so the rules don't need to be repeated. The Project Map below exists for the same reason, one level down: so a session can orient itself from this file and `docs/technical-decisions.md` instead of re-reading the codebase from scratch.
- **Subagents**: reserved for isolated research/exploration (e.g., "compare vector DB options") so the main conversation stays focused on the teaching dialogue rather than filling up with raw search results.
- **Skills**: used for repeatable, well-defined chores (e.g., code review) rather than for the core teaching/build loop, which is inherently conversational and step-by-step.

## Project Map

Read this section (and `docs/technical-decisions.md` for the *why* behind anything below) before grepping/exploring the codebase cold — it exists specifically to avoid burning tokens re-discovering structure a past session already mapped out. It's a map, not the territory: file/function names below can drift as the project evolves, so treat a surprising mismatch as a signal to `git log`/read the actual file, not as this document being wrong on purpose.

### Request flow, end to end

- **Ingest**: `POST /ingest` → `ingestion/` (one parser per file type: `pdf.py`, `docx.py`, `pptx.py`, `markdown.py`, `csv_.py`/`excel.py`/`tabular.py`, images via `vision.py`) → `chunking/parent_child.py`'s `ParentChildChunker` (the only chunker actually wired into ingestion — see below) → embeddings via `providers/factory.py` → `stores/indexer.py`'s `HybridIndexer` writes both the vector and keyword roles (both Elasticsearch — see below).
- **Query**: `POST /query` (blocking) or `POST /query/stream` (NDJSON streaming) → `api/routers/query.py` → `generation/agent.py`'s `AgentChain` (the real, tool-calling agent — decides for itself how many searches a turn needs) → `retrieval/scoped.py`'s `ScopedRetriever` (mandatory access-control wrapper, never bypassed) → `retrieval/retriever.py`'s `Retriever` (method dispatch: cosine/mmr/bm25/hybrid_rrf, optional cross-encoder rerank) → Elasticsearch. Citations are resolved positionally against the assembled context (`generation/parse.py`/`context.py`), never trusted from the model's own recall.
- **Docgen** (a second, independent workflow — never triggered from a chat turn): `docgen/cli.py` (terminal) or `POST /docgen/runs` / `GET /docgen/runs/{thread_id}` / `POST /docgen/runs/{thread_id}/resume` (web, `api/routers/docgen.py`, polled — never blocking or streamed, since a run can pause for minutes waiting on a human) → `docgen/graph.py`'s compiled LangGraph. Turns one free-text request plus one or two tagged document sets (`task_docs` required, `reference_kb` optional, for comparison questions) into a confirmed question list, answers each with retrieve → generate → validate (retry, then escalate to a human via `interrupt()` after exhausting retries), harmonizes tone across all answers, waits for human review (approve or send specific questions back for a redo), then exports a `.pptx`/`.docx` via `nodes/export.py` (deterministic, no LLM call). `docgen/checkpointer.py`'s SQLite checkpointer is what lets a paused run survive a killed-and-restarted process.

### Layer by layer

| Path | What it's for |
|---|---|
| `providers/` | LLM/embedding/vision/reranker clients behind an interface (`base.py`). Only `providers/factory.py` constructs concrete instances (`LiteLLMProvider` wraps any OpenAI-compatible backend via `litellm`) — never import a concrete provider class elsewhere. |
| `ingestion/` | One parser per file type, producing a common `ChunkElement` representation (`ingestion/schema.py`). File type is detected from content (`libmagic`), not extension. |
| `chunking/` | `parent_child.py`'s `ParentChildChunker` is what `/ingest` actually uses. `fixed_size.py`/`recursive.py`/`semantic.py`/`structure_aware.py` exist for the `compare_demo.py`-style trade-off comparisons this project favors (see technical-decisions.md §5) — not used in production ingestion. |
| `embeddings/` | Mostly a `compare_demo.py`; real embedding calls go through `providers/embeddings.py`. |
| `stores/` | Elasticsearch serves both roles (`elasticsearch_store.py`) — `ElasticsearchStore` (keyword/BM25) and `ElasticsearchVectorStore` (vector/kNN, blue-green alias versioning — see technical-decisions.md §7 for the earlier design this was carried forward from) are two classes sharing one physical index, behind `base.py`'s `VectorStore`/`KeywordStore`. Two classes, not one, because `VectorStore.search()`/`KeywordStore.search()` have the same name but incompatible signatures. `indexer.py`'s `HybridIndexer` writes/deletes/updates both roles so they can't drift out of sync (though with one shared backend, "drift" is now a narrower failure mode than it used to be — see technical-decisions.md §7/§9). `filters.py`'s `SearchFilter` is the one shape both a caller's narrowing and the mandatory security filter compose through, via `merge()` — intersection-only, so a caller can never widen access. |
| `retrieval/` | `retriever.py` dispatches by `RetrievalMethod` (`schema.py`). `scoped.py`'s `ScopedRetriever` is the access-control wrapper `AgentChain` is always handed (never the plain `Retriever`) — classification/private/superseded-status filtering, enforced both as a store-level filter and a second sqlite-backed re-check. |
| `generation/` | `agent.py`'s `AgentChain` is the real `/query` and `/query/stream` path: a tool-calling loop (`search_knowledge_base`) that decides how many searches a turn needs, can stream via `answer_stream()`, and enforces refusal/citation rules from its own system prompt (identifies itself as "LIBRA AI" if asked). `chain.py`'s `RagChain` is a simpler single-shot LCEL pipeline used only by `demo.py`. `context.py` assembles token-budgeted context; `parse.py` resolves `⟦N⟧` citation markers positionally; `prompt.py` holds rules shared between `AgentChain`/`RagChain`; `title.py` generates a conversation title. |
| `api/` | FastAPI app (`main.py` wires routers + the identity dependency + `app.state` singletons built in its lifespan). `routers/query.py` is the biggest one — `/query` and `/query/stream` share setup via `_setup_query()`. `routers/docgen.py` wraps docgen for the web (see below) — a background `threading.Thread` per segment, status always re-derived live from the checkpointer, never tracked as a second, redundant enum. `db.py` is sqlite persistence (conversations, queries, citations, feedback, documents catalogue). `api/identity.py` builds a `Principal` from a request; the `Principal` type itself lives in the top-level `identity.py` (not under `api/`, so `retrieval/` can depend on it without depending on `api/`). `auth_mode="disabled"` (the default) still runs the real scoping code through an unrestricted principal, not a bypass branch. |
| `docgen/` | The second workflow described above — see `graph.py`'s own module docstring for the full node-by-node wiring. `state.py` is the typed `DocGenState` (full-replace-per-field, no custom reducer); `nodes/` holds one module per graph stage (`configuration.py`, `retrieval.py`, `answering.py`, `escalation.py`, `harmonize.py`, `review.py`, `export.py`); `validation.py` is the LLM-as-judge groundedness check each draft answer must pass; `checkpointer.py` is the SQLite `SqliteSaver` wrapper (with an explicit type allowlist — a real deserialization-security fix, not cosmetic); `sources.py`/`tags.py`/`ingest.py` are the reuse-an-existing-tag-vs-ingest-a-new-folder source resolution the CLI wizard uses; `runs_db.py` is a small table existing only for what the checkpointer itself can't answer (listing a caller's own runs, a title, a crash message) — status is never tracked there. `cli.py` is the terminal entry point this was built and proven against before the API/frontend wrapping existed. |
| `evaluation/` | `run_eval.py` (retrieval metrics), `run_expert_eval.py` (the real `AgentChain` over a QA set, can log to Langfuse), `judge.py` (LLM-as-judge prompts for faithfulness/relevance/correctness), `run_docgen_eval.py` (the docgen analogue — two tagged document sets instead of one, many questions per run; see `data/eval/README.md`'s "Docgen eval sets" section for the format and why its scoring works differently). |
| `frontend/app.py` | Streamlit chat UI. Talks to the API only over HTTP (`httpx`), never imports `multimodal_rag` directly — that boundary is deliberate, not incidental. Streams via `/query/stream`, showing tool calls in a `st.status` widget and answer tokens growing live in place. |
| `frontend/pages/1_Docgen.py` | A second, separate Streamlit page (the project's first use of native multipage `pages/`) for driving a docgen run from the browser — a different interaction model from the chat page's instant turns (a job checked back on over minutes, answered with small forms), so it isn't crammed into the same page. Auto-refreshes (`st_autorefresh`) only while a segment is actually running; the configuration-confirmation step renders as a real conversation (`st.chat_message`), not a raw backend summary string. |

### Full endpoint list

(from `api/main.py`; every route except `/health` sits behind the identity dependency — `tests/api/test_scoping.py` has a route-table walk that fails automatically if a new one is added without being registered there)

- `POST /ingest`, `POST /suggest-tags`
- `GET /documents`, `DELETE /documents`, `PATCH /documents/{doc_id}`, `DELETE /documents/{doc_id}`
- `POST /query`, `POST /query/stream`
- `GET /conversations`, `GET /conversations/{conversation_id}`, `DELETE /conversations/{conversation_id}`
- `POST /feedback`
- `GET /metrics`
- `POST /docgen/runs`, `GET /docgen/runs`, `GET /docgen/runs/{thread_id}`, `POST /docgen/runs/{thread_id}/resume`, `GET /docgen/runs/{thread_id}/download`
- `GET /health` (the one exemption — no principal, no application data)

### Running it locally

- `uv sync` once; `uv run pytest` for tests; `uv run ruff check .` / `uv run mypy .`.
- Full local stack, one command: `uv run python tests/live/start_all.py` — starts Elasticsearch (via Docker if available, else a native `.local-services` fallback; it's the only store now, see technical-decisions.md §7), then the backend on `:8000` and frontend on `:8501`. `tests/live/stop_stores.py` stops everything (Docker containers or native processes); data persists either way. `tests/live/wipe_db.py` clears ingested data without stopping anything.
- **On this Mac, Docker requires Colima running first** (`colima status`; `colima start` if it says stopped) — every `docker`/`docker compose` command fails at the socket level otherwise, which looks identical whether you're trying to start *or* stop the store. Check this before debugging anything store-related.
- `.env.local`/`.env.server` (gitignored, real secrets) vs their checked-in `.example` templates — `RAG_ENV` (an OS env var, read before pydantic loads anything) picks the profile (`config.py`). Local allows external providers; server is air-gapped, GPU, no runtime internet.
- **Langfuse is configured in `.env.local`** (pointing at Langfuse Cloud) **but never in the test suite** — a tracing/threading regression can pass every test and still be broken in production, invisibly (`tracing.py`'s own defensive error handling swallows a broken span rather than failing the request). See technical-decisions.md §13's "A real bug, found only against a live Langfuse project" writeup before trusting green tests alone for anything that holds a `traced_query()`/`traced_generation()` span open across a suspension point (a generator yield, a thread handoff) — restart the real local backend and check its log instead.
- **Docgen** runs either from the terminal (`uv run python -m multimodal_rag.docgen.cli --thread-id <name> --state-file <path>`, resumable after a killed process with just `--thread-id`) or through the web (`tests/live/start_all.py`'s frontend, the **Docgen** page in the sidebar nav — needs the backend's `routers/docgen.py` wired in, which it is by default). Its own eval (two tagged document sets, many questions per run, a different scoring story than `run_expert_eval.py` — see `data/eval/README.md`) runs via `uv run python -m multimodal_rag.evaluation.run_docgen_eval`, and on every push via `.github/workflows/docgen-eval.yml` — that workflow needs its own `DOCGEN_EVAL_LLM_*` repository secrets pointing at a *publicly reachable* endpoint (never this project's internal-only default), and skips gracefully with a logged warning if they're unset rather than failing the build.

### Where the real design rationale lives

`docs/technical-decisions.md` — a section-by-section retrospective (ports & adapters, every store/chunking/retrieval/generation decision, access control, streaming, known limitations, named trade-offs) written specifically so the *why* doesn't need re-deriving from a diff. Check the relevant section there before re-explaining or re-deciding something from scratch; this file is the map, that file is the terrain.

### Test conventions

- `tests/` mirrors `src/multimodal_rag/`'s package structure, plus `tests/live/` (manual convenience scripts — only `test_start_all.py` is actually collected by pytest) and `tests/frontend/` (Streamlit's `AppTest` harness, deliberately never depends on a real running API — network calls are monkeypatched or expected to fail soft).
- `tests/api/` tests spin up the real FastAPI app via `httpx.AsyncClient` + `ASGITransport` against a **real** Elasticsearch (a fresh, uniquely-named index per test) — they need the store actually running, not mocked. See "Running it locally" above.
- Trust pytest's own output/exit code, not a piped or backgrounded one — if a test run is backgrounded, read its real completion status before drawing conclusions from it.
