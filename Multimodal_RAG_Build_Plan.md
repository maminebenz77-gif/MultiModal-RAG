# Multimodal Technical RAG — Build & Learning Plan

**Goal:** build, by hand and understanding every step, a multimodal RAG that ingests PDF / Markdown / Word / PPTX (each with text, tables, charts, images), indexes them into **hybrid retrieval** (vector search + Elasticsearch), lets the user pick the retrieval method in the UI, handles a corpus that changes over time, and is shipped with Docker, FastAPI, CI/CD and KPI tracking — and that runs both on **your Mac** (development/testing) and on the **company's air-gapped GPU server** (production) with no code changes.

---

## How to use this document

This file is **your playbook**, not something Claude Code runs. Here's the loop:

1. Create an empty project folder on your Mac and open **Claude Code** inside it.
2. *(Optional)* drop this file in the folder as `BUILD_PLAN.md` so it's version-controlled and handy — but don't feed the whole thing to Claude Code.
3. Copy **Prompt 0** into Claude Code. Let it finish. Answer the 3 quiz questions it asks you.
4. Only when you understand the step, copy the **next** prompt. Repeat.
5. Never paste several prompts at once — the whole point is to go one concept at a time so you're interview-ready, not to finish fast.

Prompt 0 installs a **teaching contract** into `CLAUDE.md`, so every later prompt automatically makes Claude Code explain *before* coding and quiz you *after* — you don't repeat those instructions. For the bigger steps (3, 7, 10, 13) run Claude Code in **plan mode** (shift-tab / `/plan`) so you approve the plan before any code is written.

---

## 1. Target architecture (the mental model to hold)

```
                 ┌──────────────────────────────────────────────┐
   documents ──▶ │  INGESTION: parse per-format, extract         │
 (pdf/md/docx/   │  text + tables + images/charts (multimodal)   │
   pptx)         └──────────────────────────────────────────────┘
                                   │
                                   ▼
                 ┌──────────────────────────────────────────────┐
                 │  CHUNKING (pluggable): fixed / recursive /     │
                 │  structure-aware / semantic / parent-child     │
                 └──────────────────────────────────────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
      ┌────────────────────┐              ┌────────────────────────┐
      │ EMBEDDINGS          │              │ TEXT INDEX (keyword)    │
      │ → Qdrant (dense,    │              │ → Elasticsearch (BM25,  │
      │   HNSW index)       │              │   inverted index)       │
      └────────────────────┘              └────────────────────────┘
                 └─────────────────┬─────────────────┘
                                   ▼
                 ┌──────────────────────────────────────────────┐
                 │  RETRIEVAL (user-selectable):                 │
                 │  similarity · MMR · BM25 · hybrid(RRF) ·      │
                 │  +optional rerank · parent-doc                │
                 └──────────────────────────────────────────────┘
                                   │
                                   ▼
                 ┌──────────────────────────────────────────────┐
                 │  GENERATION: LLM via provider factory,         │
                 │  citations, "I don't know" guardrail           │
                 └──────────────────────────────────────────────┘
                                   │
        FastAPI (async) ◀──────────┘──────────▶ KPIs / eval / tracing (Langfuse + RAGAS)
              │
   Frontend (method selector) · Docker-compose · GitHub Actions CI/CD · runs local & server

   ┌───────────────────────────────────────────────────────────────────────┐
   │ EVERYTHING external (LLM, embeddings, vision, reranker, stores) sits    │
   │ behind an INTERFACE. One variable RAG_ENV=local|server picks the real   │
   │ implementation. Same code on Mac and on the air-gapped GPU server.      │
   └───────────────────────────────────────────────────────────────────────┘
```

**Stack (deliberate choices you should be able to defend):**

| Layer | Choice | Why (interview answer) |
|---|---|---|
| Parsing | `unstructured` + fallbacks (PyMuPDF, python-docx, python-pptx) | Layout-aware partitioning into elements (title/table/image); shows why naïve text extraction loses structure in technical docs |
| Multimodal | Vision-LLM summaries of images/charts/tables → embed the summary (multi-vector) | Portable + cheap; works with any text vector store; keep the raw image for display |
| Chunking | Pluggable strategies behind one interface | The whole point: compare methods empirically |
| Dense store | **Qdrant** (HNSW) | Purpose-built vector DB, payload filtering, easy Docker |
| Keyword store | **Elasticsearch** (BM25) | Industry-standard inverted index; gives the "vector + elastic" hybrid |
| Fusion | Reciprocal Rank Fusion (RRF) | Combines two ranked lists without score-scale problems |
| Orchestration | **LangChain** for loaders, splitters, retriever interface, LCEL chain | Learn standard abstractions; hand-write ES + fusion to understand the internals |
| Generation | LLM behind a provider factory (LiteLLM / OpenAI-compatible) | Swap between your API key and the internal company LLM by config alone |
| API | **FastAPI** (async) | Modern, typed, fast to demo |
| Packaging | **Docker** + compose (app + Qdrant + ES) with local/server profiles | Reproducible; one-command spin-up in either environment |
| Observability / KPIs | **Langfuse** (traces, cost, latency) + **RAGAS** (eval) | Turns "feels good" into measurable quality — pure PM signal |
| CI/CD | **GitHub Actions** (ruff, pytest, docker build) | Standard, free, demonstrable |

---

## 2. Concepts you'll master (map to interview questions)

- **Indexing** — inverted index (BM25) vs ANN index (HNSW: `M`, `ef_construction`, `ef_search`) and their recall/latency/memory trade-offs.
- **Chunking** — fixed / recursive / structure-aware / semantic / parent-child; chunk size vs retrieval quality vs context cost.
- **Embeddings** — what they are, dimensionality, why technical text may need a different model, and why you *version* them.
- **Hybrid retrieval** — dense (semantics) vs sparse/keyword (exact part numbers, acronyms — vital for technical data), fused with RRF.
- **MMR** — trading relevance for diversity.
- **Reranking** — cross-encoder second pass; precision vs cost.
- **Changing corpus** — content hashing, idempotent upsert, deletes/tombstones, embedding-model migration.
- **Evaluation** — recall@k, MRR, nDCG (retrieval); faithfulness, answer relevance, hallucination rate (generation).
- **Guardrails** — grounded-only answering, refusal, prompt-injection awareness.
- **Portability & ops** — provider abstraction, air-gapped deployment, device detection, latency/cost budgets, tracing.

---

## 3. Portability: Mac (dev) ↔ Company server (prod)

You develop and test on the Mac with cheap/external models and dummy docs; you deploy to the server where an internal GPU-served LLM and the **real** technical documents live. The rule: **your code never knows where it runs.** Every external dependency hides behind an interface; `RAG_ENV=local|server` selects the implementation.

### What actually differs

| Thing | Mac (dev / testing) | Company server (prod) |
|---|---|---|
| **Generation LLM** | Your OpenAI (ChatGPT) API key, or local Ollama/HF | Internal company LLM (own endpoint + GPU) |
| **Embeddings** | Small local model (e.g. BGE) on CPU/MPS, or an API | Same/larger model on **CUDA GPU**, or an internal endpoint |
| **Vision** (image/chart/table → text) | API with vision, or degrade/skip | Internal multimodal model or GPU captioner |
| **Reranker** | Small cross-encoder on CPU, or off | Cross-encoder on GPU |
| **Device** | `mps` (Apple Silicon) or `cpu` | `cuda` |
| **Internet** | Available | Often **air-gapped** — no downloads, no external calls |
| **Documents** | A few dummy files | Real, **sensitive** docs that must never leave the server |
| **Stores** | `localhost` via Docker | Server hostnames, bigger resources |

Two hard constraints: on the server there's **no internet** (models pre-baked/mounted, offline mode forced), and real documents **must never reach an external API** (the `server` profile makes external calls *impossible*, not just unused).

### The pattern (built in Prompt 1)

Provider **interfaces** (`LLMProvider`, `EmbeddingProvider`, `VisionProvider`, `Reranker`) → a config **profile** selected by `RAG_ENV` (via `pydantic-settings`, `.env.local` / `.env.server`) → **factories** (`get_llm()`, `get_embedder()`, …) that return the right concrete provider. The app depends only on the interfaces.

The key enabler: use an **OpenAI-compatible interface** as the common LLM protocol. Most internal stacks (vLLM, TGI, a LiteLLM proxy) expose an OpenAI-style endpoint, so "switch to the company LLM" is just changing `base_url` + model name in config. A library like **LiteLLM** (or LangChain's `init_chat_model`) gives you that across OpenAI / Anthropic / Ollama / HF / internal. If the internal LLM isn't OpenAI-compatible, you write **one** thin `LLMProvider` class and nothing else changes.

**The embedding gotcha (say this in interviews):** Mac and server may use different embedding models, and vectors from different models aren't comparable — so you **re-embed and re-index on the server** against the real docs. That's the same `model_id`-versioning + migration you build in Prompt 10, so portability and "changing DB" are the same mechanism.

---

## 4. Repo shape you're aiming for

```
rag/
├── CLAUDE.md                 # conventions + teaching contract
├── config.py                 # profiles: RAG_ENV=local|server
├── .env.local  .env.server   # per-environment settings (from .env.example)
├── docker-compose.yml + .local.yml + .server.yml
├── Dockerfile
├── pyproject.toml
├── .github/workflows/ci.yml
├── app/
│   ├── providers/            # interfaces + factories (llm, embed, vision, rerank)
│   ├── ingestion/            # parsers (pdf, docx, pptx, md), multimodal
│   ├── chunking/             # strategy classes behind one interface
│   ├── embeddings/
│   ├── stores/               # qdrant.py, elastic.py
│   ├── retrieval/            # similarity, mmr, bm25, hybrid, rerank
│   ├── generation/           # prompt, chain, citations, guardrails
│   ├── evaluation/           # ragas, golden set, metrics
│   ├── api/                  # FastAPI routers, schemas
│   └── config.py
├── frontend/                 # method selector UI
├── data/                     # sample technical docs
└── tests/
```

---

# 5. The Claude Code prompt sequence

Paste **Prompt 0 first**. Then one prompt at a time, in order, only when you understand the previous step.

---

### Prompt 0 — Initialize the project + set the teaching contract

```
We're going to build a multimodal technical RAG together, step by step, over many sessions. I'm doing this to LEARN and to prepare for AI Product Manager interviews, so pace matters more than speed.

First, create a file CLAUDE.md at the repo root that encodes these rules for every future step — you must follow them in all later work:

TEACHING CONTRACT
1. Before writing any code for a step, explain: the options, the trade-offs, and your recommended choice with the reason. Then STOP and wait for my "go".
2. Implement in small commits, one concept at a time. Never jump ahead to a later phase.
3. After implementing a step, explain what you built, the key decisions, and the failure modes. Then ask me 3 interview-style questions about it and wait for my answers before we continue.
4. Prefer clarity over cleverness. Add short comments explaining WHY, not just what.
5. If I seem to misunderstand, slow down and use an analogy.

Then, WITHOUT building any RAG logic yet, set up only the skeleton:
- A Python project with pyproject.toml (ruff + pytest + mypy).
- The folder structure for: providers, ingestion, chunking, embeddings, stores, retrieval, generation, evaluation, api, plus tests/, data/, frontend/.
- A README explaining the architecture at a high level.
- Initialize a git repo with a sensible .gitignore.

Also explain briefly which Claude Code features (plan mode, subagents, skills, CLAUDE.md) we'll use and when. Do not install heavy dependencies yet. Follow the teaching contract starting now.
```
*Learn: project hygiene, secrets separation, how Claude Code's own workflow features work.*

---

### Prompt 1 — Environment portability layer (Mac ↔ server)

```
Before any RAG logic, set up the environment-portability layer, because this project runs in TWO places: my Mac for development/testing (external API key or local models, dummy docs) and an air-gapped company GPU server for production (internal LLM endpoint, real sensitive docs, no internet).

First explain the design to me: provider interfaces + config profiles + factories, and why an OpenAI-compatible interface (via LiteLLM or equivalent) lets me switch the LLM by config alone. Wait for my go.

Then implement:
1. config.py using pydantic-settings, profile selected by RAG_ENV (local|server), loading .env.local or .env.server. Give me both .env.example files. Include: llm/embed/vision provider + model + base_url + api_key, qdrant_url, elastic_url, device, allow_external.
2. A device resolver (auto -> cuda|mps|cpu).
3. Abstract interfaces: LLMProvider, EmbeddingProvider, VisionProvider, Reranker.
4. Concrete providers: for LLM, one LiteLLM-based provider covering OpenAI (my key), a local option (Ollama or HF), AND an internal OpenAI-compatible endpoint via base_url; plus a stub InternalServerLLM for a non-compatible endpoint. For embeddings, a sentence-transformers provider (device-aware) plus an API stub.
5. Factories get_llm/get_embedder/get_vision/get_reranker for the rest of the app to use.
6. A PRIVACY GUARD: when allow_external is False (server profile), it must be impossible to call an external endpoint — assert the resolved host is local/internal, and set HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE.
7. A script `python -m app.env_check` printing the active profile, resolved device, and wired providers — so I can verify local vs server instantly.

Follow the teaching contract. Then quiz me on why the app must never import providers directly.
```
*Learn: provider abstraction, config profiles, dependency injection, air-gapped/privacy design — a strong senior talking point.*

---

### Prompt 2 — Ingestion & multimodal extraction

```
Step: document ingestion. Goal — parse PDF, Markdown, DOCX and PPTX into a common internal representation where each document becomes a list of ELEMENTS, each tagged with type (title / paragraph / table / image / chart) and metadata (source file, page/slide, position).

Before coding, compare the options for parsing each format (unstructured vs PyMuPDF vs python-docx/pptx) and how we'll handle images, charts and tables specifically for TECHNICAL documents. Recommend an approach and wait for my go.

Then implement:
- A parser per format returning our common Element schema (pydantic).
- For images and charts: extract the image bytes AND generate a text description using get_vision() from the provider factory, storing both (we embed the description, keep the image for display) — the "multi-vector" idea; explain it. If no vision provider is available (e.g. locally), degrade gracefully: store a placeholder and flag the element, so ingestion still works on my Mac and full multimodal runs on the server.
- For tables: convert to markdown text plus an optional short summary.
- A couple of sample technical files in data/ to test on.

Show me the extracted elements for one sample of each format so I can see where parsing struggles. Follow the teaching contract.
```
*Learn: why technical docs break naïve extraction; the multi-vector multimodal pattern; graceful degradation across environments.*

---

### Prompt 3 — Chunking strategies (pluggable + comparable)

```
Step: chunking. I want to LEARN chunking by comparing methods, so build them behind one interface (a base Chunker with a .chunk(elements) method).

Before coding, explain each strategy, when it wins, and its downside: (1) fixed-size with overlap, (2) recursive character splitting, (3) structure-aware (split on titles/sections using our element types), (4) semantic chunking (embedding-similarity boundaries), (5) parent-child / small-to-big. Recommend defaults for technical docs and wait for my go.

Then implement all five as swappable strategies (LangChain splitters where they fit; hand-write structure-aware and parent-child). Add a script that runs the SAME document through each strategy and prints chunk counts, sizes, and example chunks. Follow the teaching contract.
```
*Learn: the biggest lever on RAG quality; chunk-size vs context-cost; parent-child retrieval.*

---

### Prompt 4 — Embeddings

```
Step: embeddings. Before coding, explain what an embedding is, dimensionality, and the trade-offs between model options (hosted general vs technical/code-tuned vs local open-source), including cost, latency, and quality on technical vocabulary. Recommend one plus a cheaper local fallback. Wait for my go.

Then implement, using get_embedder() from the factory: at least two backends (one hosted, one local, device-aware), batching, and a stored model_id + dimension on every vector. Make explicit that local and server may use DIFFERENT embedding models, that vectors from different models must never be mixed, and that model_id lets us detect and re-embed on the server. Add a test embedding two similar and two dissimilar technical sentences, printing cosine similarities. Follow the teaching contract.
```
*Learn: embeddings intuition, domain fit, embedding versioning (sets up the changing-DB and portability steps).*

---

### Prompt 5 — Vector store + the meaning of "indexing" (Qdrant / HNSW)

```
Step: dense vector store with Qdrant. Before coding, explain what INDEXING means for vectors: what HNSW is, and how M / ef_construction / ef_search trade recall against latency and memory. Contrast with brute-force. Wait for my go.

Then implement a Qdrant store (local via Docker), taking the connection URL from config (no hardcoded localhost, so it works on my Mac and against server hostnames): create-collection (correct size + distance), upsert chunks with payload (source, element_type, model_id, doc_id, chunk_id), and similarity search. Expose the HNSW/search params so I can experiment; show results on our samples and let me change ef_search to see the effect. Follow the teaching contract.
```
*Learn: ANN indexing, HNSW parameters — a very common interview question.*

---

### Prompt 6 — Keyword search + inverted index (Elasticsearch / BM25)

```
Step: keyword search with Elasticsearch. Before coding, explain the inverted index and BM25: why keyword/lexical search still matters in technical RAG (exact part numbers, acronyms, error codes that embeddings blur), and what analyzers/tokenizers do. Wait for my go.

Then implement: Elasticsearch via Docker, connection from config; an index with a sensible analyzer; index the same chunks; a BM25 search returning ranked results with scores. Show me a query where BM25 beats vector search and one where it loses. Follow the teaching contract.
```
*Learn: lexical vs semantic retrieval, inverted index, analyzers — the "why hybrid" foundation.*

---

### Prompt 7 — Retrieval methods, user-selectable (similarity · MMR · BM25 · hybrid RRF · rerank)

```
Step: the retrieval layer — the core learning goal. Build a Retriever interface where the METHOD is selectable at query time (later a frontend dropdown).

Before coding, explain each and when to prefer it: (1) basic cosine similarity, (2) MMR — how lambda trades relevance vs diversity, (3) BM25 keyword, (4) hybrid via Reciprocal Rank Fusion — write out the RRF formula and explain the k constant, (5) optional reranking with a cross-encoder (via get_reranker()) — precision vs cost. Wait for my go.

Then implement all behind one `retrieve(query, method, top_k, params)` call, reusing Qdrant and Elasticsearch. Add a comparison script that runs one technical query through every method and prints the top chunks side by side. Follow the teaching contract.
```
*Learn: the full retrieval toolbox, RRF, MMR lambda, reranking — the densest interview area.*

---

### Prompt 8 — Generation: LLM + context assembly + citations + guardrail

```
Step: answer generation using get_llm() from the factory and LangChain LCEL for the chain. The SAME chain must run against my OpenAI key locally and the internal company LLM on the server, with no code change — only config differs.

Before coding, explain: assembling retrieved chunks into a prompt within a token/latency budget, forcing citations back to source chunks, and an "only answer from context, otherwise say you don't know" guardrail. Mention prompt-injection risk from ingested documents. Wait for my go.

Then implement the RAG chain: retrieve → grounded prompt → LLM → answer WITH citations (source file + page/slide). Show one question it should refuse because the answer isn't in the corpus, and show it refusing. Follow the teaching contract.
```
*Learn: grounded generation, citations, hallucination guardrail, prompt-injection awareness, provider-agnostic generation.*

---

### Prompt 9 — FastAPI backend

```
Step: wrap everything in a FastAPI (async) service. Before coding, propose the endpoint design and pydantic request/response schemas, and wait for my go.

Then implement: POST /ingest (file → parse → chunk → embed → index in both stores), GET /documents, POST /query (params: question, retrieval_method, top_k, filters), POST /feedback (thumbs up/down + comment), GET /metrics, GET /health. Keep ingestion idempotent-ready (change-handling comes next). Add tests with pytest + httpx. Follow the teaching contract.
```
*Learn: API design, async, typed schemas, the shape of a real product backend.*

---

### Prompt 10 — Handling a corpus that changes over time

```
Step: make the corpus updatable safely — a big interview topic, and also how we re-index on the server. Before coding, explain the problems: detecting changed vs unchanged documents, avoiding duplicate chunks, deleting removed content, and what happens when we switch embedding models. Recommend an approach and wait for my go.

Then implement:
- Content hashing per document and per chunk; ingestion becomes an idempotent UPSERT (unchanged docs skipped).
- Delete-by-doc_id across BOTH stores (removed-document / tombstone case).
- Re-index a single changed document without rebuilding everything.
- An embedding-model migration path: because vectors carry model_id, implement how we'd re-embed under a new model version without downtime — this is exactly what we'll run on the server, whose embedding model differs from my Mac's.
Add a test that ingests a doc, edits it, re-ingests, and proves no duplicates and correct updates. Follow the teaching contract.
```
*Learn: idempotent pipelines, deletes/versioning, embedding migration — and the mechanism that makes server re-indexing safe.*

---

### Prompt 11 — Frontend with retrieval-method selector

```
Step: a simple frontend to demo it. Keep it lightweight (single-page app or Streamlit — recommend one for speed, explain the trade-off, wait for my go).

Then build a UI with: file-upload to ingest, a question box, a DROPDOWN to pick the retrieval method (similarity / MMR / BM25 / hybrid / hybrid+rerank) and top_k, an answer panel with citations, the retrieved chunks (so I see what each method returned), and thumbs up/down wired to /feedback. Read the API URL from config so it works locally and on the server. Follow the teaching contract.
```
*Learn: turning the retrieval choice into a visible product decision — great demo material.*

---

### Prompt 12 — Docker & docker-compose (local + server profiles)

```
Step: containerize for BOTH environments. Before coding, explain images vs containers, why compose here, volumes for persistence, service networking, and GPU vs CPU runtimes. Wait for my go.

Then write: a small multi-stage Dockerfile for the FastAPI app, and a base docker-compose.yml plus two overrides:
- docker-compose.local.yml — CPU/MPS, external providers allowed, sample data, Qdrant + Elasticsearch on localhost.
- docker-compose.server.yml — NVIDIA GPU runtime (`--gpus all`), allow_external=false, offline env vars (HF_HUB_OFFLINE etc.), points at the internal LLM endpoint, model weights MOUNTED or pre-baked (no downloads), volumes persisting the indexes.
Explain how to pre-bake or mount weights so the air-gapped server needs no internet. Verify the local stack runs from clean with one command. Follow the teaching contract.
```
*Learn: Docker fundamentals, compose overrides, GPU runtime, persistence, air-gapped packaging.*

---

### Prompt 13 — KPIs, evaluation set & observability

```
Step: measurement — the most PM-relevant part, and directly closes my "evals" interview gap. Before coding, explain two layers: RETRIEVAL metrics (recall@k, MRR, nDCG) and GENERATION metrics (faithfulness/groundedness, answer relevance, hallucination rate), plus product KPIs (p50/p95 latency, cost per query, tokens, thumbs-up rate, refusal rate). Wait for my go.

Then implement:
- A small "golden set" of question→expected-source pairs over our sample docs.
- A RAGAS-based (or LLM-as-judge) evaluation script scoring each retrieval method and printing a comparison table (so I can say "hybrid+rerank beat similarity by X on faithfulness").
- Langfuse tracing through the query path to capture latency and cost per query.
- A /metrics summary (feedback rate, avg latency, refusal rate).
Follow the teaching contract.
```
*Learn: RAG evaluation, LLM-as-judge, tracing, cost/latency budgets — pure AI-PM signal.*

---

### Prompt 14 — CI/CD with GitHub Actions

```
Step: CI/CD. Before coding, explain what belongs in CI vs CD and why. Wait for my go.

Then create .github/workflows/ci.yml that on every push/PR: installs deps, runs ruff (lint+format check), mypy, and pytest, then builds the Docker image. Explain how I'd extend it to push the image to a registry and deploy to the server. Make the pipeline actually pass. Follow the teaching contract.
```
*Learn: pipelines, quality gates, the build→test→ship flow.*

---

### Prompt 15 — Server deployment & portability hardening

```
Step: make the project deploy cleanly on the air-gapped company GPU server and prove portability.

Before coding, explain the risks of moving from my Mac to an offline GPU server: no internet for model downloads, different device (cuda), the internal LLM endpoint, and the rule that real documents must never leave the server. Wait for my go.

Then:
1. Produce a PORTABILITY.md checklist: setting RAG_ENV=server, filling .env.server (internal LLM base_url + model, embedding model, GPU device), pre-baking/mounting model weights, forcing offline mode.
2. Add scripts / make targets: serve-local and serve-server (right compose profile each).
3. Add "contract tests": the SAME tests run against a fake in-memory provider (pass on my Mac / CI with no GPU or keys) and, when RAG_ENV=server, against the real internal LLM — so I trust the swap.
4. Add a server SMOKE TEST: ingest a couple of real documents, run one query per retrieval method, print latency + which providers were used, confirming allow_external is false.
5. Explain how I re-embed and re-index on the server (reusing the Prompt 10 migration path) because the server's embedding model differs from my Mac's.
Follow the teaching contract, and finish with 5 interview questions about deploying an LLM app from a laptop to an air-gapped GPU server.
```
*Learn: air-gapped deployment, device detection, contract testing, the privacy guarantee — senior-level portability.*

---

### Prompt 16 — Polish, demo script & interview recap

```
Step: finalize for demo and interviews.
1. Write DEMO.md: how to spin up the stack and a 3-minute demo script showing ingestion, switching retrieval methods live, citations, and the metrics comparison.
2. Write ARCHITECTURE.md with the diagram and every key decision + its rationale (my interview crib sheet), including the local↔server portability design.
3. Generate 20 likely interview questions about THIS project (chunking, indexing, hybrid/RRF, MMR, changing corpus, evaluation, guardrails, Docker, CI/CD, air-gapped deployment) with concise model answers grounded in what we actually built.
Follow the teaching contract.
```
*Learn: consolidation — turns the codebase into a story you can tell fluently.*

---

## 6. Extra skills to fold in (that make you a stronger AI PM)

Not separate phases — ask Claude Code to touch them as you go:

- **Prompt versioning** — keep prompts in files, version them; a prompt change is a product change (A/B it).
- **Cost & latency budgeting** — quote p95 latency and cost-per-query per retrieval method.
- **A/B testing retrieval** — selectable methods + eval harness = an experiment framework. Frame it that way.
- **Security** — API-key hygiene (via config), prompt-injection from ingested docs (instruction isolation), and the `allow_external=false` privacy guarantee on the server.
- **Data privacy / PII** — where you'd add redaction if the corpus were sensitive.
- **Scalability story** — sketch how it scales (batch ingestion, async workers, sharded indices) even if you don't build it.
- **Observability** — Langfuse traces answer "how do you know it works in production?" — a question juniors fail.

---

## 7. Your two-step workflow & pace

- **On the Mac:** `RAG_ENV=local`. Cheap/external LLM (your ChatGPT key) or a local Ollama model, small local embeddings on MPS, sample docs. Build, unit-test, iterate fast. External calls allowed (no sensitive data here).
- **On the server:** `RAG_ENV=server`. Internal LLM via `base_url`, GPU embeddings, `allow_external=false`, offline mode, real docs. Re-embed + re-index against the real corpus, then run the smoke test and the evaluation set (Prompt 13) on real technical documents — this is where your true KPIs come from.

Because every difference lives in `.env.*` + the factories, moving between environments is flipping one variable — not editing code. That's the thing to demo and to say in interviews.

**Pace:** roughly one prompt per sitting (3, 7, 10, 13, 15 may take two). At 6–8h/week you have a demoable v1 in about **3–4 weeks**. For interviews **next week**, prioritize understanding Prompts 3, 5, 6, 7, 10, 13 and 15 conceptually even before the code is finished — those are the questions you're most likely to be asked.
```
