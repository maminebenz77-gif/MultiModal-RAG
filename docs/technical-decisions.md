# Technical decisions and trade-offs

A retrospective, not a plan — this documents what was actually built, why it was built that way, and what it cost. See [`Multimodal_RAG_Build_Plan.md`](../Multimodal_RAG_Build_Plan.md) for the original forward-looking outline; this file is the after-the-fact record of the real decisions, including the ones the plan didn't anticipate.

Written for interview prep: each section pairs a **choice**, the **trade-off**, and — where it matters — **how the underlying mechanism actually works**, not just what library call invokes it.

## Contents

1. [Overall shape: ports & adapters](#1-overall-shape-ports--adapters)
2. [Environment portability](#2-environment-portability)
3. [Providers layer](#3-providers-layer)
4. [Ingestion](#4-ingestion)
5. [Chunking](#5-chunking)
6. [Embeddings](#6-embeddings)
7. [Vector store — Qdrant](#7-vector-store--qdrant)
8. [Keyword store — Elasticsearch](#8-keyword-store--elasticsearch)
9. [Hybrid indexing & consistency](#9-hybrid-indexing--consistency)
10. [Retrieval](#10-retrieval)
11. [Generation](#11-generation)
12. [Document identity & incremental re-ingestion](#12-document-identity--incremental-re-ingestion)
13. [API layer](#13-api-layer)
14. [Frontend](#14-frontend)
15. [Testing philosophy](#15-testing-philosophy)
16. [Known limitations, named on purpose](#16-known-limitations-named-on-purpose)

---

## 1. Overall shape: ports & adapters

Every swappable concern (LLM/embedding/vision/reranker providers, vector store, keyword store) is split into an **abstract interface** (`*/base.py`) and **concrete implementations**, constructed only by a single `factory.py` per module. Calling code depends on the interface, never on a concrete class (`QdrantStore`, `LiteLLMProvider`) directly.

**Why**: the entire premise of the project is that pieces get swapped — chunking strategy, embedding model, vector backend — while iterating. Ports & adapters makes "swap the vector store" a one-file change (the factory), not a grep-and-replace across the codebase. `Retriever` is the one deliberate exception: it's *not* an ABC, because retrieval *method* is chosen per-query by the caller (a UI dropdown), not per-deployment-environment the way providers/stores are — a different axis of variation, so it doesn't fit the same pattern.

**Cost**: an extra layer of indirection for every module, and the discipline of never letting a concrete class leak into calling code (even for something as small as "just check `isinstance(store, QdrantStore)`"). Worth it here because the swapping is real and expected, not hypothetical.

## 2. Environment portability

`RAG_ENV` (an OS environment variable, read *before* pydantic loads anything, since it decides *which* file to load) selects `.env.local` or `.env.server`. Both files populate the same `Settings` model (`pydantic-settings`), so the rest of the app never branches on environment — it just reads `Settings` fields that happen to differ.

**The privacy guard**: on the server profile (`ALLOW_EXTERNAL=false`), `enforce_privacy_guard()` checks every provider's `base_url` against loopback/private IP ranges before construction — an external URL raises immediately, not "might silently phone home." Also forces `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` so the moment a `sentence-transformers` model isn't already cached, it fails loudly instead of hanging on a DNS lookup that never resolves on an air-gapped box.

**Trade-off**: this is a *structural* guarantee, not a network-level one — it stops the *application* from calling out, not a compromised dependency from doing so. A real air-gapped deployment still wants network-level isolation (firewall rules) as the actual boundary; the privacy guard is defense sitting inside that boundary, not a replacement for it.

## 3. Providers layer

`LiteLLM` is the mechanism that makes "same code, different backend" possible for LLM/embedding/vision calls — any OpenAI-compatible endpoint (real OpenAI, an internal company gateway, a local vLLM server) is reachable through the same `litellm.completion()`/`litellm.embedding()` call, differing only by `model`/`base_url`/`api_key`. LangChain's own model-wrapper classes were deliberately *not* used for this, even where LangChain is used elsewhere (the generation chain) — those wrappers would reintroduce vendor coupling at exactly the point the whole providers layer exists to avoid.

**Local embeddings** (`SentenceTransformerEmbeddingProvider`) run entirely offline once the model is cached — no network call at all, which is what makes them safe on the air-gapped profile. **Hosted embeddings** (`LiteLLMEmbeddingProvider`) manually batch input (LiteLLM doesn't batch for you, and real APIs cap request size) and retry each batch with exponential backoff; a persistently-failing batch doesn't abort the whole call — already-embedded batches are kept and returned alongside which texts failed, so a caller can decide to use the partial result or retry just the gap. Constructing the underlying `openai.OpenAI` client explicitly (rather than letting LiteLLM build its own internally) turned out to matter for one specific reason: on a corporate network requiring `truststore`'s SSL patch (see below), LiteLLM's own internally-constructed client didn't reliably pick that patch up.

**Corporate proxy support** (`TRUST_SYSTEM_CERTS`): a TLS-inspecting corporate proxy presents certificates signed by an internal CA that the OS trusts but Python's bundled `certifi` CA list doesn't. `truststore.inject_into_ssl()` patches the `ssl` module to defer to the OS trust store instead. Gated behind a settings flag, default off — inert everywhere the flag isn't set, so it never changes behavior on Linux, CI, or a Mac.

## 4. Ingestion

File type is detected from actual file **content** via `libmagic`, not the extension — a renamed or extensionless PDF still routes to the PDF parser. Markdown is the one exception: plain text has no distinguishing magic bytes, so `libmagic` can only ever report `text/plain`, and the dispatcher falls back to the file extension for that one case.

PDF parsing uses `unstructured[pdf]` (poppler + tesseract) rather than `PyMuPDF` — a deliberate choice after flagging that PyMuPDF is AGPL-licensed, which isn't a "we'll deal with it later" concern for anything meant to leave a personal sandbox. DOCX/PPTX/Markdown keep their native-library parsers (`python-docx`, `python-pptx`, `markdown-it-py`) rather than being funneled through `unstructured` too — `unstructured` earns its dependency weight specifically for PDF's harder extraction problem (layout analysis, OCR fallback), not for formats with a straightforward native parser already available.

Every format-specific parser converges on one common `Element` schema (type, text, metadata: source file, position, page/slide number) — everything downstream (chunking, citations) works against that one shape, never against a format-specific structure.

## 5. Chunking

Five strategies live behind one `Chunker` interface: fixed-size, recursive character, semantic (embedding-similarity boundary detection), structure-aware (splits on document structure — headings, sections), and parent-child (structure-aware "parent" sections, each further split into smaller "child" pieces via `RecursiveCharacterTextSplitter`).

### Content-addressed chunk IDs

```
chunk_id = f"{doc_id}::{strategy}::{index}::{sha256(chunk_text)[:10]}"
```

Not purely positional (`doc_id::strategy::index`) — a hash of the chunk's *own text* is baked in. This single design choice is what makes several other things possible:

- **Idempotent re-ingestion**: unchanged text at the same position re-hashes to the same ID, so re-upserting is a no-op at the storage level, not a duplicate.
- **Chunk-level diffing on edits** (see [§12](#12-document-identity--incremental-re-ingestion)): a changed chunk gets a *new* ID; an unchanged one doesn't — the ID itself carries the information needed to tell "this changed" from "this is stale but identical," without diffing raw text at ingest time.

**Why `doc_id` has to be part of the hash, not just the text**: if `chunk_id` were `hash(text)` alone, two *different* documents that happen to share identical text (a boilerplate disclaimer, a repeated section header, common code) would collide into the *same* stored point. Whichever document was ingested most recently would silently win the citation metadata for that chunk, and deleting one document's stale chunks (an edit's cleanup step) could delete a completely unrelated document's copy of that same text. Scoping the ID to `doc_id` avoids this at the cost of never deduplicating storage across documents that happen to share content — a trade-off judged appropriate at this project's scale (see [§16](#16-known-limitations-named-on-purpose) for the fuller architecture that *would* dedupe across documents, and why it wasn't built).

### Parent-child retrieval

Children (small, ~200-character pieces) are what's actually searched — precise matching. Parents (the full section) are marked `is_parent=True` and excluded from search **natively at the store level** (a Qdrant payload filter, an Elasticsearch `bool.must_not`) — not a post-hoc application-level filter, so there's no way for a parent to leak into results by a caller forgetting to filter. On a matched child, `resolve_parent_context()` swaps the child's *text* for its parent's full text (more context for the LLM) while every other field — `chunk_id`, `source`, `pages` — keeps pointing at the child, so citations stay precise even though the model saw the fuller paragraph. If two different children of the *same* parent both rank in the top-k, only the higher-ranked one survives resolution — otherwise the LLM would see that parent's text twice for zero new information.

## 6. Embeddings

Multi-vector pattern: the raw content (text, image bytes) is kept for display; a *text projection* is what actually gets embedded and searched (a vision caption stands in for an image, a markdown rendering stands in for a table). Same embedding pipeline for every modality — nothing downstream needs to know a chunk originated from an image.

`ModelMismatchError`: `Qdrant.search()` compares the query vector's `model_id` against what's actually stored before searching. Dimension matching alone (which Qdrant already enforces) isn't sufficient — two different models can share a dimension count while encoding meaning in incompatible vector spaces, which would return confident-looking, *meaningless* results with no error at all otherwise.

## 7. Vector store — Qdrant

### How HNSW indexing actually works

Qdrant's approximate-nearest-neighbor index is a **Hierarchical Navigable Small World** graph — points connected into a multi-layer graph, top layers sparse with long-range links (like an express lane), the bottom layer dense with short-range links to every point. A search starts at the top layer, greedily walks toward the query vector until no closer neighbor exists at that layer, drops down one layer, and repeats — arriving at layer 0 already close to the true answer, then does a final local search there. This avoids a full linear scan (`O(n)`) in favor of something closer to `O(log n)`, at the cost of being *approximate* — it can miss the true nearest neighbor, traded for large speedups at scale.

Three parameters control the recall/speed/memory trade-off directly:

| Parameter | What it controls | Effect |
|---|---|---|
| `m` | max connections per node | higher → better recall, more memory |
| `ef_construct` | candidates considered while *building* the graph | higher → better graph quality, slower build |
| `ef_search` | candidates considered at *query* time | higher → better recall, slower query, exposed **per-call** (no rebuild needed to tune it) |

### Blue-green collection versioning

`collection_name` is a Qdrant **alias**, not a physical collection — `search()` always reads through it. `create_collection()` builds a new, uniquely-named physical collection *without* touching the alias; `publish()` atomically repoints the alias at the new collection and removes the old one. A production pipeline that instead deleted-and-rebuilt the live collection in place would have no rollback if the new version turned out broken, and no way to diff what changed to debug it — this makes that structurally impossible, since the previous version stays fully intact and queryable right up until the atomic cutover. `ensure_ready()` is the idempotent version used at service startup: create + publish an empty collection only if none exists yet, never touching an already-live one — safe to call on every boot.

Point IDs are a UUID5 hash of the human-readable `chunk_id` (Qdrant requires an int or UUID, not an arbitrary string) — deterministic, so re-upserting the same `chunk_id` updates the same point rather than creating a duplicate. The original readable `chunk_id` is kept in the payload for citation/debugging.

## 8. Keyword store — Elasticsearch

### How the inverted index + BM25 actually work

A forward index maps document → terms; an **inverted index** maps *term* → the list of documents containing it (a "postings list"), which is what makes a keyword search fast — looking up "GPU" jumps straight to the documents containing it instead of scanning the whole corpus.

**BM25** ranks the matching documents by combining three signals:
- **Term frequency**, saturating (not linear) — the 10th occurrence of a term in a document adds much less score than the 2nd, controlled by a `k1` parameter.
- **Inverse document frequency** — a term that appears in *most* documents (like "the") is a weak signal; a term appearing in only a few documents is a strong, discriminating one.
- **Length normalization** — a term appearing once in a 20-word chunk is more significant than once in a 2,000-word one, controlled by a `b` parameter.

**Why keyword search still matters next to embeddings**: exact identifiers, part numbers, and error codes are exactly what embedding similarity blurs (an embedding model represents *meaning*, and "ERR-4471" doesn't have meaning to blur toward) — BM25 matches the literal token.

**Analyzer choice**: the `standard` analyzer (lowercasing + whitespace/punctuation tokenization, *no stemming*) was chosen explicitly over Elasticsearch's fancier options — stemming risks mangling exact codes/identifiers ("CVE-2024-12345") in exchange for generalizing prose forms ("test"/"tests"/"testing") this corpus doesn't especially need. Known, documented, *not-yet-fixed* gap: the standard tokenizer splits on hyphens, so `"internal-llama-70b"` indexes as three separate tokens — still matches on any of them, but blurs exact-phrase precision. The real fix (a second, un-analyzed `keyword`-typed field for exact matching) was named and deliberately deferred, not missed.

`chunk.id` is used directly as the Elasticsearch `_id` — no UUID conversion needed here, unlike Qdrant, since ES document IDs accept arbitrary strings natively.

## 9. Hybrid indexing & consistency

`HybridIndexer` coordinates writes to both stores so "upsert into Qdrant, then index into Elasticsearch" isn't two separate, uncoordinated calls scattered through the codebase. **Not** true cross-database atomicity — Qdrant and Elasticsearch share no transaction mechanism, and building one (a single source-of-truth store both stores sync from independently, the way real search infrastructure eventually solves this) was judged disproportionate for this project's scale. What it *does* give: a persistent failure on the second write raises `IndexConsistencyError` carrying exactly which `chunk_id`s are now inconsistent, rather than a silent gap discovered later via a confusing search result — plus `check_consistency()`, which compares `list_chunk_ids()` between both stores independently to catch drift from *any* cause (a failed write, a deletion that only touched one store, manual intervention).

`delete()` mirrors `index()`'s ordering and error-handling exactly, for the same reason: vector store first, keyword store second, so a persistent failure on the second step is reported precisely instead of silently leaving one store cleaned and the other not.

## 10. Retrieval

One `Retriever.retrieve(query, method, top_k, ...)` call, method selected per-query:

- **Cosine** — a thin wrapper over the vector store's own similarity search.
- **BM25** — a thin wrapper over the keyword store.
- **MMR (Maximal Marginal Relevance)** — trades relevance against redundancy:

  ```
  MMR = argmax_{d ∈ R∖S} [ λ·Sim(d, query) − (1−λ)·max_{s∈S} Sim(d, s) ]
  ```

  Greedy iterative selection: repeatedly pick the candidate maximizing relevance minus similarity-to-already-selected. `λ=1` degenerates to pure relevance ranking (identical to plain cosine); low `λ` actively prefers a diverse set over a cluster of near-duplicates, even at some cost to raw relevance. Requires candidate vectors (`with_vectors=True`) to compute the redundancy term.

- **Hybrid (Reciprocal Rank Fusion)** — fuses cosine and BM25 rankings:

  ```
  RRF_score(d) = Σ_r 1 / (k + rank_r(d))
  ```

  summed over each ranking method `r` that returned `d`, `k=60` a standard damping constant. Fused by **rank position**, not raw score — a BM25 score (unbounded, corpus-dependent) and a cosine score (bounded, model-dependent) aren't measuring comparable things and can't be meaningfully rescaled onto each other, but "ranked #1" means the same thing regardless of which method produced that ranking. This is the real, practical answer to "how do you combine two different scoring systems."

- **Reranking** — an *orthogonal* `rerank: bool` flag applicable to any method above, not a fifth method: retrieve a broader candidate pool, then re-score it with a cross-encoder (`CrossEncoderReranker`, local `sentence-transformers`) before cutting to `top_k`. Cross-encoders score a (query, document) pair *jointly* — much more accurate than comparing two independently-computed embeddings, but too expensive to run over an entire corpus, hence "retrieve broad, rerank down" rather than "rerank everything." Only the local cross-encoder path was built; a hosted reranker (Cohere/Voyage-style) was explicitly *not*, for lack of credentials to test against — a scoped decision, not an oversight.

Since `is_parent=True` chunks are excluded from search **natively at the store level** (§5), reranking *always* operates on precise child text, never parent text — this became true unconditionally once that store-level filter was in place, not just "usually true because children tend to rank higher."

## 11. Generation

### The chain

Built with LangChain's LCEL (`RunnableLambda` steps composed via `|`), used purely as an **orchestration layer** — the actual model call still goes through `get_llm()` from the providers factory, so "same chain runs against a local OpenAI key and an internal company LLM, no code change" stays exactly as true in generation as everywhere else. LangChain's own model-wrapper classes (`ChatOpenAI` etc.) were deliberately not used, for the same vendor-coupling reason as [§3](#3-providers-layer). State threads through the chain as a dict, each step adding a key — the standard LCEL pattern for carrying auxiliary data (the retrieved context, needed again after the LLM call to map citation markers back to real metadata) alongside the main value.

### Context assembly

Token-budgeted, **greedy fill by rank**: include results in ranked order until the next one would exceed the budget, then stop — dropping the *tail* of the ranking (lower-relevance results), never truncating an individual chunk's text (a half-sentence of context is worse than one fewer whole chunk). Always includes the top result even if it alone exceeds budget — an empty context defeats the point of the guardrail below.

### Citations and the guardrail

The system prompt instructs the model to answer *only* from the numbered context blocks, to refuse with an exact, matchable sentence if the answer isn't present, and to cite claims inline using a distinctive marker: `⟦1⟧`, not `[1]`. That choice mattered in practice, not just in theory — plain `[N]` collides with bracketed numbers that occur naturally in source documents (footnotes, step lists, array indices) *and* in a model's own enumerated-list writing style, and either would be misread as a citation by a naive `\[(\d+)\]` regex. `⟦…⟧` is distinctive enough that it only appears when the prompt put it there.

Citations are never trusted from the model's own recall of filenames or page numbers — `parse_answer()` only extracts *which numbered marker* was used, then looks up the real `chunk_id`/`source`/`pages` from the context **we** already built. The model's job is picking which block is relevant; the mapping back to real metadata is entirely programmatic.

**Prompt injection**: context blocks are framed explicitly as *data to read*, not instructions to obey — a mitigation, not an elimination (an LLM reliably distinguishing "instructions" from "data that looks like instructions" isn't a solved problem). The real limiter on blast radius is architectural: `LLMProvider.generate()` has no tool-use wired up at all, so even a fully successful injection can only manipulate the *text* of the answer, never trigger a real action.

### Conversational rewriting

A follow-up question ("What about the P95 numbers instead?") retrieves badly on its own — it lacks the referent. Before retrieval, if conversation history is non-empty, a separate LLM call rewrites the latest turn into a standalone question using the prior turns as context; the rewritten (not the original) query is what's embedded and searched. Skipped entirely when there's no history — single-turn use pays zero extra latency or cost for a rewrite it doesn't need.

## 12. Document identity & incremental re-ingestion

This is the area with the most iteration, because it surfaced a real tension worth stating precisely: a *single* identity value cannot simultaneously be (a) stable across a file being renamed, and (b) stable across a file's content being edited — because "derived from content" (needed for (a) without relying on the filename) definitionally changes the moment content changes (breaking (b)), and "not derived from content" (needed for (b)) has nothing to anchor renaming-invariance to.

The resolution actually shipped:

- **`doc_id = sha256(filename)`** — deliberately *not* a content hash. This is what makes chunk-level diffing possible: `chunk_id` embeds `doc_id` (§5), so a stable `doc_id` across an edit means only the chunks whose *text* actually changed get new IDs — the rest are recognized as already correctly stored and skipped, not blindly re-embedded. Editing one word in a 34-chunk sample document, verified directly: 29 chunks untouched, 5 re-embedded, 5 stale ones actually deleted — not 34 orphaned and 34 redone (an earlier, content-hash-based `doc_id` design had exactly that failure mode, caught and fixed).
- **`content_hash = sha256(bytes)`**, tracked separately in the `documents` table, purely to detect "nothing changed" (an exact re-upload under the *same* filename) without redoing any work to find out.
- **A global content-hash check**, independent of `doc_id`: before parsing anything, check whether this exact content already exists *anywhere* in the corpus, under *any* filename. This is what actually solved a real bug — the same file uploaded once via a single-file picker (reports a bare filename) and once via a native folder picker (reports a path-prefixed name, via the browser's `webkitRelativePath`) wasn't recognized as the same document, because the two upload paths produced two different filename strings and therefore two different `doc_id`s. Rather than normalizing filenames (guessing which of several possible naming quirks — a path prefix, Unicode normalization — was responsible in a given case), checking content globally sidesteps the question entirely.

**Named trade-off, not a bug**: renaming a file with *unchanged* content is no longer deduplicated as "the same document" the way it briefly was in an earlier iteration — a rename is a new `doc_id`. Two genuinely different files that happen to share a leaf filename (`notes.md` in two unrelated folders) will collide into one `doc_id` if both are ever ingested. Deletion of orphaned chunks on an edit (`Chunk_ids present before, absent in the new chunking`) is computed by filtering `list_chunk_ids()` for the `doc_id::` prefix every chunk_id already carries — no separate tracking table needed, since the scoping is already baked into the ID itself.

The fully general fix — content-addressed chunk *storage* decoupled from document *membership*, the way git separates content-addressed blobs from the tree/commit structure that tracks which blob belongs to which path — would get every property at once (including deduplicating identical content *across* unrelated documents), at the cost of a real architecture change: a many-to-many storage model instead of one-point-per-chunk, reference-counted deletion, and citation logic that can attribute a chunk to more than one document. Named explicitly as the "textbook correct" answer and explicitly not built, as disproportionate to this project's single-user scale.

## 13. API layer

FastAPI, `async def` route handlers wrapping every blocking call (Qdrant/ES clients, `litellm`, local model inference) in `starlette.concurrency.run_in_threadpool` — without that, a single slow embedding call would block the entire event loop and serialize every other request behind it, silently defeating the point of declaring the endpoint `async` at all.

**Idempotent-ready ingestion** short-circuits on a known, unchanged `doc_id`+`content_hash` *before* touching the file — parsing, chunking, and embedding are skipped entirely, not merely deduplicated after the fact. `IngestResponse.status` distinguishes `"ingested"` / `"already_ingested"` / `"duplicate_content"` so a caller (and the frontend) can tell those apart instead of everything looking like a fresh ingest.

**`ensure_ready()` vs. blind recreation at boot**: Elasticsearch's `create_index()` is destructive (drops and recreates), so calling it unconditionally on every service startup would silently wipe a previously-indexed corpus on every restart. `ensure_ready()` checks existence first — create only if missing, otherwise a no-op — and is what's actually safe to call on every boot.

**Self-healing sqlite schema**: `CREATE TABLE IF NOT EXISTS` runs on *every* connection now, not once at `Database.__init__`. Found the hard way — `tests/live/wipe_db.py` is explicitly designed to be safe to run against an already-running server, but schema-once-at-construction meant deleting the underlying file out from under a live process left every later query hitting "no such table" until that process restarted. Cheap, idempotent, and closes exactly that gap.

**`DELETE /documents`** wipes every chunk in both stores (unioning both stores' `list_chunk_ids()` rather than trusting either alone, so it also repairs any pre-existing drift) and every row in `documents` — deliberately leaves query/feedback history alone, since that's a log of past activity, not corpus state, and stays meaningful after the corpus itself is reset.

## 14. Frontend

Streamlit, talking to the API **only over HTTP** — never importing `multimodal_rag` directly, which keeps the API a real architectural boundary rather than just an internal layering convention on paper.

Folder ingestion uses `st.file_uploader(accept_multiple_files="directory")` — a genuine native-OS folder picker (not a hand-rolled in-app directory browser, which was tried first and turned out unreliable for selecting subfolders in real browser use). Junk filtering (Office `~$` lock files, dotfiles, OS metadata) happens client-side before anything reaches `/ingest`, since the native folder picker has no per-file filtering of its own once a folder is chosen — and it has to live *outside* the surrounding form, since a form only exposes selected files to the script after submission, which would make it impossible to preview what's about to be skipped before committing to the click.

**Metrics staleness, and why it happened**: the Metrics/Documents panels sit earlier in the script's top-to-bottom render order than the query form and feedback buttons. Streamlit reruns the whole script on every interaction — but *within one run*, code executes in the order it's written, so the metrics fetch (reading current state) happened *before* that same run's query or feedback mutation. Fixed with `st.rerun()` right after those specific mutations; `st.toast()` (not `st.success()`) for the resulting confirmation, since toasts are specifically designed to survive an immediate rerun where a normal message would be cut off mid-render.

## 15. Testing philosophy

**Real services over mocks, at the store layer.** `QdrantStore`/`ElasticsearchStore` tests run against actual local Qdrant/Elasticsearch instances (via `docker-compose.yml`), not mocked clients — a store class is mostly a thin wrapper around real network calls, and mocking the client would mostly test the mock's fidelity to the real API, not the store's actual behavior. Provider-layer and chain-layer tests use fake/duck-typed implementations instead (`FakeEmbedder`, `FakeLLM`) — deterministic, free, fast, and appropriate once the layer under test doesn't depend on a *specific* backend's real behavior, just on the interface contract.

**`retry_with_backoff` tests mock `time.sleep`**, not the retried operation — the point under test is the retry *logic* (does it retry the right number of times, does it eventually raise), and real sleeps would make the suite slow for no benefit.

**Live scripts, separate from the automated suite** (`tests/live/`): `start_all.py`, `wipe_db.py`, `stop_stores.py` — manual convenience tooling for actually clicking around the running app, explicitly *not* named `test_*.py` so pytest never collects them as tests. `start_all.py`'s shutdown handling specifically survives a second, impatient Ctrl+C during cleanup (escalates to a hard kill instead of crashing with a raw traceback) — caught via an actual live double-interrupt test while building it, not assumed correct.

## 16. Known limitations, named on purpose

Each of these was a real judgment call, not an oversight — named here so they read as decisions, not gaps discovered later:

- **No cross-database transactions** between Qdrant and Elasticsearch (§9) — a single source-of-truth store both stores sync from independently would remove this, at real architectural cost.
- **No content deduplication across different documents** that happen to share text (§5, §12) — the fully general fix is a git-blob-style content-addressed storage layer decoupled from document membership; not built, named explicitly as the "textbook correct, more capable" answer.
- **No collection-name-scoped sqlite state** — `documents`/`queries`/`feedback` are one global table each, not partitioned per Qdrant collection. Only matters once more than one corpus is meant to coexist behind one API process, which hasn't come up yet.
- **Elasticsearch's standard analyzer splits on hyphens** (§8) — blurs exact-phrase precision for hyphenated identifiers; the real fix (a second `keyword`-typed field) is named, not built.
- **No schema migration framework** for the sqlite state — a schema change requires a full `wipe_db.py` reset, not an in-place migration. Proportionate for a single-user learning project; would not be for anything with real data at stake.
- **Local, containerized dev has no MPS acceleration** — Docker on macOS runs inside a Linux VM with no GPU passthrough to Apple Silicon, so a containerized local stack falls back to CPU even though running the app natively on the same Mac gets MPS.
