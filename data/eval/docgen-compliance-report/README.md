# docgen-compliance-report

An eval set for the `docgen` workflow (`src/multimodal_rag/docgen/`), which
needs two document sets per run instead of one. The narrative: the
fictional company **Meridian Cloud Labs** is producing a compliance
gap-analysis report comparing its own current practices
(`documents/task/`, the `task_docs` role) against the requirements of a
fictional external standard, the **Starling Trust Framework**, imposed by
a fictional partner, **Starling Cloud Partners** (`documents/reference/`,
the `reference_kb` role).

`docgen_qa.json` follows the same shape as the `qa.json` files described
in `data/eval/README.md`, plus a `sources_required` field (`["task_docs"]`
or `["task_docs", "reference_kb"]`) saying which document set(s) each
question genuinely needs.

Run it: `uv run python -m multimodal_rag.evaluation.run_docgen_eval` (see
`data/eval/README.md`'s "Docgen eval sets" section for the full
convention and how scoring works). `SCENARIO.md` in this folder is the
live-presentation script for the same corpus and questions.
