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
  qa.json       # [{"id": "...", "question": "...", "expert_answer": "..."}, ...]
```

- `id` — a short, stable identifier for the question (for your own reference only).
- `question` — exactly what you'd ask the agent.
- `expert_answer` — what a real expert would actually say. The agent's real
  answer is judged against this for *substantive* agreement, not exact
  wording (see `evaluation/judge.py`'s `score_answer_correctness`).

Adding a new question later is a one-line edit to an existing `qa.json` —
no re-registration anywhere.

## Running it

```
uv run python -m multimodal_rag.evaluation.run_expert_eval
```

Each expertise folder's documents are ingested into their own isolated
collection, so a question about one expertise can't accidentally retrieve
another expertise's content. Prints one row per expertise (average
correctness + question count) plus an overall average.
