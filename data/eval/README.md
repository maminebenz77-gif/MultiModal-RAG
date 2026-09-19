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
  documents/   # real source documents (PDF, DOCX, PPTX, or Markdown)
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

```
uv run python -m multimodal_rag.evaluation.run_expert_eval
```

Each expertise folder's documents are ingested into their own isolated
collection, so a question about one expertise can't accidentally retrieve
another expertise's content. Prints one row per expertise (average
correctness over answerable questions, refusal accuracy over
`expect_refusal` questions) plus an overall average.
