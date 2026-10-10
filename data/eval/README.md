# Expert-authored evaluation sets

Real documents plus real expert-written answers, evaluated against the
actual production agent (`AgentChain`, the same construction
`routers/query.py` uses for real `/query` requests). Different purpose
from `data/golden_set.json`'s synthetic comparison (`run_eval.py`,
`langfuse_experiment.py`), which asks "which retrieval method is best?" —
this asks "is the agent a real user actually gets correct, on real domain
content?"

## Adding your own expertise

No code changes needed — just add files:

```
data/eval/<expertise-name>/
  documents/               # real source documents (PDF, DOCX, PPTX, Markdown, CSV, or Excel) --
                            # may itself contain real subfolders (e.g. documents/runbooks/),
                            # discovered recursively, not just at the top level
  documents_metadata.json  # optional: filename -> DocumentMetadata fields
  qa.json                  # [{"id": "...", "question": "...", "expert_answer": "...",
                            #   "expect_refusal": false}, ...]
```

- `documents_metadata.json` -- optional. Without it, every document is ingested
  with a bare `{"classification": "public"}` and no lineage, same as before this
  file existed. Add an entry for a filename to give it real `DocumentMetadata`
  fields -- most commonly `doc_family_id`/`version`/`status`/`effective_from`, to
  make two of your documents genuinely supersede each other (so you can write a
  question whose correct answer depends on Phase 5/6 actually working, not just
  on retrieval quality). See `company-spreadsheets/documents_metadata.json` for
  two real examples: a **declared** supersession (same `doc_family_id`, versions
  1 and 2, old one `status: "superseded"`) and an **undeclared** conflict (two
  unrelated, independently dated documents that simply disagree -- no
  `doc_family_id` at all, so nothing hides either one; the correct answer
  depends on the agent reading both dates itself).

- `demo/` -- also a live-presentation script, not just a regression set. Its
  `SCENARIO.md` is a step-by-step walkthrough (bulk folder ingest with tags, version
  lineage, author filtering, and an undeclared-conflict "hard case," each with the exact
  live Filters-panel interaction to run) for presenting every metadata/access-control/
  filtering feature to an audience -- the same corpus and core questions this file's own
  convention already runs automatically, so the live demo can never silently drift from
  what the automated eval actually verifies.

- `id` — a short, stable identifier for the question (for your own reference only).
- `question` — exactly what you'd ask the agent.
- `expert_answer` — what a real expert would actually say. For a normal
  question, the agent's real answer is judged against this for
  *substantive* agreement, not exact wording (see `evaluation/judge.py`'s
  `score_answer_correctness`). For an `expect_refusal` question (below),
  this should explain *why* there's no answer — it's shown to you in the
  results table but isn't run through that judge.
- `expect_refusal` — optional, defaults to `false`. Set `true` for a
  question your documents genuinely don't answer. These are scored
  differently: correctness is *did the agent decline to answer*, not
  *does its answer match `expert_answer` word-for-word* — a judge
  comparing free text would unfairly mark a terse, correct "I don't know"
  wrong against a longer reference explanation, so refusal questions get
  their own `RefusalAcc` column instead of feeding into the `Correctness`
  average. Mirrors `data/golden_set.json`'s `expect_refusal` field.

Adding a new question later is a one-line edit to an existing `qa.json` —
no re-registration anywhere.

## Running it

One command:

```
uv run python -m multimodal_rag.evaluation.run_expert_eval
```

It always prints the results to the console — one row per expertise
(average correctness over answerable questions, refusal accuracy over
`expect_refusal` questions) plus an overall average. Each expertise
folder's documents are ingested into their own isolated collection, so a
question about one expertise can't accidentally retrieve another
expertise's content.

It also tries to reach Langfuse first, and tells you which case you're in
(printed up front, and again under the table, since a run's log noise
scrolls the first one away):

- **Connected** — each expertise's `qa.json` is also synced to its own
  Langfuse Dataset (`expert-eval-<expertise-name>`) and run as a real
  Experiment, which is what shows up under **Datasets** in the Langfuse
  UI. Langfuse's own `run_experiment()` opens a trace context per
  question, so every call one question triggers (embed, retrieve,
  generate, the correctness/refusal judge) lands nested under one shared
  trace instead of scattered as unrelated root traces. Each expertise's
  Dataset Run URL is printed under the table. Every run is timestamped in
  its own run name (`<expertise-name>-<UTC timestamp>`), so re-running
  after editing a `qa.json` shows up as a new, distinguishable run rather
  than silently overwriting the last one.
- **Not connected** — Langfuse isn't configured, is blocked by the
  privacy guard (see `tracing.py`), or fails its auth check. The script
  says so and carries on console-only; it never fails because Langfuse is
  missing. In this mode the agent's calls are still individually traced if
  Langfuse happens to be configured (that tracing lives at the
  provider/retriever layer), but with no Experiment there's nothing tying a
  `retrieve`/`generate_with_tools` pair to the question that produced it.

Either way the scoring is identical — the same evaluator functions feed the
same aggregation — so the console table means the same thing in both
modes; the only difference is whether Langfuse's `run_experiment()` or a
plain local loop drives the questions.

## Docgen eval sets

A second, parallel convention for `src/multimodal_rag/docgen/` — the LangGraph
document-generation workflow, not `AgentChain` — lives in this same
`data/eval/` root, distinguished by filename/shape so the two never collide:

```
data/eval/<name>/
  documents/
    task/        # required -- the task_docs source
    reference/   # optional -- the reference_kb source, for comparison questions
  docgen_qa.json  # [{"id", "question", "sources_required", "expert_answer",
                   #   "expect_refusal"}, ...]
  SCENARIO.md     # optional -- a live-presentation script, same convention as demo/
```

Why a second shape instead of reusing the plain `qa.json`/`documents/` one: docgen answers a
whole LIST of questions in one run, from up to TWO tagged document sets (`task_docs` always,
`reference_kb` only for questions that need to compare against it) — `sources_required`
says which role(s) a given question needs (`["task_docs"]` by default, or
`["task_docs", "reference_kb"]` for a genuine comparison question).

See `data/eval/docgen-compliance-report/` for a real example: a fictional company's own
security practices (`task_docs`) compared against a fictional external standard's
requirements (`reference_kb`), with one easy question, three genuine two-source comparisons,
two more task-only questions, and one `expect_refusal` question (a plausible-sounding fact
that's genuinely absent from both document sets).

**Scoring works differently from the plain `qa.json` convention above, on purpose.** Docgen has
no `AgentChain`-style `refused: bool` — an honest "that isn't covered by the documents" answer
is just a normal, validated answer that happens to admit absence (confirmed empirically: the
per-question validator judges an honest admission of absence as perfectly grounded, same as any
other accurate claim). So an `expect_refusal` item is scored by *substance* against the
expert's own explanation of why there's no answer, via the same correctness judge every other
question uses — a fabricated answer won't substantively match an "it isn't stated" reference,
which is what actually matters. Only a question that never got a validated answer at all
(escalated to a human, or skipped — there is no human in an automated run, so these auto-resolve
to "skip") skips the judge outright: a correct refusal for an `expect_refusal` item, a real
failure (`false_refusal`) for one that was supposed to be answerable. See
`run_docgen_eval.py`'s own module docstring for the full reasoning, including why this was
caught by a *live run* against real content, not worked out in the abstract.

### Running it

```
uv run python -m multimodal_rag.evaluation.run_docgen_eval
```

Same two-output contract as `run_expert_eval.py` above (console table always; Langfuse Datasets
named `docgen-eval-<name>` when reachable) — but it actually drives TWO different things per eval
set:

1. **The real, full graph, once, locally.** Builds an already-confirmed question list directly
   from `docgen_qa.json` (skipping the free-text interpretation step — this eval is about
   per-question answer quality and the harmonize/review/escalation machinery, not about whether
   free text parses into the right questions), then drives the compiled graph to completion
   unattended: every escalation gets "skip" (no human exists), every review gets "approve"
   (measuring first-pass quality). **This is what the printed table and the CI pass/fail signal
   are based on.**
2. **One question at a time, via Langfuse's `run_experiment()`**, only when Langfuse is
   reachable — a narrower, visibility-only view for per-question tracing in the Langfuse UI
   (one retrieve → generate → validate cycle per question, no retry, since `harmonize_answers`/
   `human_review` are necessarily whole-run concepts). Uses the exact same scoring logic as run 1,
   so "correct" means the same thing in both places; it never overrides the console table's
   numbers.

### Running it at every push

`.github/workflows/docgen-eval.yml` runs this on every push — the first CI workflow in this
repo. It needs real LLM credentials reachable from a GitHub-hosted runner, supplied as repository
secrets (`DOCGEN_EVAL_LLM_BASE_URL`/`DOCGEN_EVAL_LLM_MODEL`/`DOCGEN_EVAL_LLM_API_KEY`) — **not**
this project's own `.env.local.example` default, which points at an internal-only company
endpoint a public runner can't reach. Embeddings default to the free local
`sentence-transformers` model specifically so only the LLM secret family is required. Without
those secrets configured, the workflow logs a warning and skips the eval step rather than
failing in a way that looks like a real regression — see the workflow file's own header comment
for the exact secret names and reasoning. Like `run_expert_eval.py`, this never hard-fails CI on
a numeric score threshold (LLM-judge scores are inherently a little noisy run to run) — it fails
only on a genuine crash (a missing eval set, an exception in the pipeline itself), and leaves
score *trends* to a human reading the printed table or the Langfuse Experiment history over
time, the same philosophy this whole file's eval sets already follow.

### Running it manually through the UI

`data/eval/docgen-compliance-report/SCENARIO.md` is a step-by-step walkthrough of the exact
same corpus and questions, run live through the Streamlit **Docgen** page (`frontend/pages/
1_Docgen.py`) instead of the automated harness — useful for demoing the feature, or for sanity-
checking by hand that what the automated eval measures matches what a real user actually sees.

### Adding your own docgen eval set (including from an expert in a different domain)

Same promise as the plain convention above — no code changes, just files:

```
data/eval/<your-name>/
  documents/task/        # your own documents
  documents/reference/   # optional -- a second set to compare against
  docgen_qa.json          # your questions, same shape as docgen-compliance-report's
```

`run_docgen_eval.py` discovers it automatically on the next run (local or CI) — there is no
registration step. Give at least one `expect_refusal` question a genuinely absent fact (not
just a hard one) so the eval set actually exercises the honest-refusal path, not only
substantive answers.
