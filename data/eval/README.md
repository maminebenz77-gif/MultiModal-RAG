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
