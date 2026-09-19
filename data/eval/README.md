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
  documents/   # real source documents (PDF, DOCX, PPTX, Markdown, CSV, or Excel)
  qa.json       # [{"id": "...", "question": "...", "expert_answer": "...",
                #   "expect_refusal": false}, ...]
```

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

Two scripts, for two different things — easy to mix up, so the
distinction is worth being explicit about:

**Console only, no Langfuse involved:**

```
uv run python -m multimodal_rag.evaluation.run_expert_eval
```

Each expertise folder's documents are ingested into their own isolated
collection, so a question about one expertise can't accidentally retrieve
another expertise's content. Prints one row per expertise (average
correctness over answerable questions, refusal accuracy over
`expect_refusal` questions) plus an overall average. This never talks to
Langfuse at all — no dataset, no experiment. The real `AgentChain` calls
it makes are still individually traced if Langfuse happens to be
configured (that tracing is unconditional, at the provider/retriever
layer), but each trace lands disconnected from the others, with nothing
tying a `retrieve`/`generate_with_tools` pair to the question that
produced it or to this run.

**Also sent to Langfuse as a real Experiment, with everything properly grouped:**

```
uv run python -m multimodal_rag.evaluation.langfuse_expert_eval
```

Syncs each expertise's `qa.json` to its own Langfuse Dataset
(`expert-eval-<expertise-name>`) and runs it as a Dataset Experiment —
this is the one that actually shows up under **Datasets** in the
Langfuse UI. It also solves the trace-disconnection problem above:
Langfuse's own `run_experiment()` opens a real trace context per
question, so every call that one question triggers (embed, retrieve,
generate, the correctness/refusal judge) lands nested under one shared
trace instead of scattered as unrelated root traces. Each run is
timestamped in its own run name (`<expertise-name>-<UTC timestamp>`), so
re-running after editing a `qa.json` — or just to re-check after a code
change — shows up as a new, distinguishable run rather than silently
overwriting the previous run's identity.
