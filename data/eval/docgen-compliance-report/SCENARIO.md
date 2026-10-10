# Live demo script: docgen, end to end through the UI

Audience-facing walkthrough of the docgen workflow (Phases 1-12), using the same fictional
company ("Meridian Cloud Labs") and fictional external standard ("the Starling Trust
Framework") as this folder's own automated eval -- `docgen_qa.json` -- so the live demo can
never silently drift from what `run_docgen_eval.py` actually verifies on every push.

## Before people arrive

1. Clean slate (optional -- docgen works fine layered on an existing corpus too):
   `uv run python tests/live/wipe_db.py`.
2. `uv run python tests/live/start_all.py` -- wait for the backend and the browser tab at
   `http://127.0.0.1:8501`.
3. In the left sidebar nav, click **Docgen** (above the main chat page -- this is a separate
   Streamlit page, added in Phase 12b).
4. Have `data/eval/docgen-compliance-report/documents/task/` and `.../documents/reference/`
   open in a Finder/Explorer window so the file pickers below don't require hunting for the
   path live.

**One honest caveat, same as the main chat demo:** this runs with `auth_mode=disabled`, one
shared identity for every request -- there's no second person in the room to demonstrate being
denied access to someone else's run.

---

## Act 1 -- Starting a run: two document sets, not one

**Story beat:** Meridian's compliance team needs a gap-analysis report comparing their own
practices against a partner's security standard -- two different document sets, not one.

1. In the sidebar's **Start a new run**, type the request:
   *"Compare Meridian Cloud Labs' current security practices against the Starling Trust
   Framework's requirements, and call out where we fall short."*
2. Under **Task documents (required)**, choose **Ingest a new folder**, click the uploader and
   pick the whole `documents/task/` folder (your OS's native folder picker opens -- every file
   inside it gets listed at once), label it `meridian-practices`, classification `public`, then
   click **Ingest & use as source**.
3. Check **Include this source** under **Reference/comparison (optional)**, choose **Ingest a
   new folder**, and pick the whole `documents/reference/` folder the same way, under a
   `starling-framework` label.
4. Click **Start run**. *(Talking point: this is the exact same `/ingest` pipeline the main
   chat page's own sidebar uses -- docgen doesn't have a separate upload path, it just adds a
   `docgen:<role>:<label>` tag convention on top.)*

---

## Act 2 -- Confirming the question list

**Story beat:** free text becomes a structured plan, and a human signs off before anything
runs.

1. The page pauses almost immediately on **confirm_configuration** -- read the summary: it
   should list several questions and note the output format (pptx or docx).
2. Click **Confirm**. *(Talking point: if the list looked wrong, "Revise instead" sends a
   correction back through the SAME interpretation step, with the correction added to context --
   never discarded, never silently dropped.)*

---

## Act 3 -- Watching it work, and the honest escalation

**Story beat:** the system answers each question with real retrieval, and is honest when it
can't.

1. Click **Refresh** every so often -- the page shows live question status and the running LLM
   call count while it works through the list.
2. When it pauses again on **human_review**, open the questions expander: point out which
   answers needed BOTH document sets (the comparison questions) versus which came from
   Meridian's own documents alone.
3. *(If a question escalated instead of answering -- visible as `[escalated]` in the question
   list -- this is `docgen_qa.json`'s own "access-control-mfa-partial-gap" item in the automated
   eval, a real, known-hard comparison question. Talking point: this is the SAME safety net
   Phase 7 built -- after exhausting its retry budget without a validator-approved answer, it
   stops rather than guessing, exactly the behavior `run_docgen_eval.py`'s `false_refusal`
   scoring is watching for.)*

---

## Act 4 -- Review, and the one trap question

**Story beat:** a human reviews the harmonized answers before anything is exported -- and one
question tests honesty, not just accuracy.

1. Read through the harmonized answers in the **human_review** pause.
2. Ask the room: *"What would happen if I asked it for the financial penalty Starling imposes
   for non-compliance?"* -- this is deliberately not in either document set
   (`docgen_qa.json`'s `starling-penalty-amount`, `expect_refusal: true`). The correct behavior
   is an honest "that isn't stated," not a fabricated number -- point out that this is scored
   automatically on every push by the same judge that scores the substantive questions, not a
   separate mechanism.
3. Click **Approve**.

---

## Act 5 -- The deliverable

1. Once the run shows **done**, click **Prepare download**, then **Download document** -- open
   the resulting file and show the real compliance findings inside it, word for word what the
   harmonized answers already showed on screen.
2. *(Closing talking point: everything just demonstrated live -- the two-source comparison, the
   honest escalation, the refusal trap question -- runs unattended and is scored automatically
   by `uv run python -m multimodal_rag.evaluation.run_docgen_eval` on every push to this repo,
   using this exact corpus. The demo never drifts from the regression net because they're the
   same files.)*
