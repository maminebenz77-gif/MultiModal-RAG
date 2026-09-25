# Live demo script: metadata, access control, lineage, and filtering

Audience-facing walkthrough of every feature built in the metadata/access-control/filtering
work, using one coherent fictional company ("Solstice Systems") so the story holds together
instead of jumping between unrelated toy examples. Four acts, each demonstrating one
capability end to end -- ingest, then query, then (where relevant) filter -- plus a closing
honesty check.

**Everything below is also `data/eval/demo/qa.json` + `documents_metadata.json`** -- the same
corpus and the same core questions run automatically every time
`uv run python -m multimodal_rag.evaluation.run_expert_eval` runs, as a regression net. This
script adds the live *filtering* interactions (tags/author/date-range) that the automated
harness doesn't exercise, plus the presenter narration.

## Before people arrive

1. Clean slate: `uv run python tests/live/wipe_db.py` (skip this if you want to keep whatever
   is already in the corpus -- the demo works fine layered on top of existing documents too,
   it just won't be as tidy on screen).
2. `uv run python tests/live/start_all.py` -- wait for both "backend ready" and the browser
   tab to open at `http://127.0.0.1:8501`.
3. Have `data/eval/demo/documents/` open in a Finder/Explorer window so the folder picker in
   Act 1 doesn't require hunting for the path live.

**One honest caveat to state up front, briefly, before Act 3:** this local demo runs with
`auth_mode=disabled`, meaning every request shares one identity. "Private" and classification
are real, enforced code paths (there are tests that prove it), but *in this room* there's no
second person to demonstrate being denied access -- say so plainly rather than letting someone
assume more than what's actually being shown.

---

## Act 1 -- Bulk folder ingest, tags, and why filtering isn't decoration

**Story beat:** a new engineer joins and needs to bulk-load the team's existing runbooks.

1. In the sidebar, under **Bulk ingest a folder**, click **Choose a folder** and select
   `data/eval/demo/documents/runbooks/` (four files: a database failover runbook, a network
   outage runbook, a deploy rollback runbook, an on-call escalation runbook).
2. Set **Classification** to `c1`, type `runbook` into **Tags for every file in this batch**,
   leave author/date blank (the whole batch gets the same tag; individual tags get refined
   next). Click **Ingest files**.
3. Open **Documents in the corpus**, click into one row (e.g. the database failover runbook)
   to show the detail view -- classification, tags, the editable Tags field. Add `database` to
   its tags and **Save tags**. Repeat quickly for the network outage runbook, adding
   `networking`. *(Talking point: a tag added after the fact is a metadata-only patch --
   nothing gets re-embedded or re-chunked.)*
4. Ask, with no filter: **"What command restarts the service, and when should I run it?"**
   Expect an honest, correctly-hedged answer that surfaces *both* `postgres-pooler` (database
   failover) and `nginx-edge` (load balancer outage) and says it depends on context -- this is
   the system being right about a genuinely ambiguous question, not a bug.
5. Click **🔍 Filters**, select tag `database`, Apply. Ask the same question again -- now a
   single clean answer, `systemctl restart postgres-pooler` only.
6. Clear filters, select tag `networking` instead, ask again -- `systemctl restart nginx-edge`
   only. *(Talking point: two runbooks describe "restarting the service" for two completely
   different systems -- without the tag filter the answer is honest but unresolved; with it,
   the ambiguity is gone because the wrong document was never in the retrieval pool at all.)*

---

## Act 2 -- Version lineage: replacing a document without deleting it

**Story beat:** the deploy-approval policy changes after a real incident.

1. Single-file ingest `data/eval/demo/documents/deploy-approval-policy-v1.md`, classification
   `c2`, author `Dana Fitzgerald`, tags `policy, deploys`. Leave **This replaces** empty (it's
   the first version).
2. Ask: **"How many approvers does a production deploy need at Solstice Systems?"** → "one".
3. Ingest `deploy-approval-policy-v2.md`, same classification/author/tags, and this time pick
   `deploy-approval-policy-v1.md` in **This replaces**.
4. Ask the identical question again → "two", explaining the January 2026 incident that
   prompted the change.
5. Open **Documents in the corpus** -- the v1 row now shows the superseded marker. *(Talking
   point: v1 was never deleted -- it's still there, still fully readable if you explicitly ask
   for history, but it structurally can't be cited as the current answer anymore. Nothing was
   re-embedded: replacing a document is a metadata flip, not a re-ingest.)*

---

## Act 3 -- Two authors, and filtering by who wrote something

**Story beat:** two engineers each own a different architecture decision.

1. Single-file ingest `adr-002-database-choice.md` -- **before submitting**, pause on the Tags
   field: it's already pre-filled by the AI suggestion endpoint the moment the file was
   chosen. *(Talking point: suggested, never silently applied -- edit or clear it, nothing is
   written until Ingest is actually clicked.)* Set classification `c1`, author `Priya Shah`.
2. Single-file ingest `adr-005-caching-layer.md` the same way, author `Marcus Webb`.
3. Ask: **"What primary datastore did Solstice Systems choose for the core application?"**
   with no filter -- answers Postgres, citing Priya's ADR.
4. Open **Filters**, set Author to `Marcus Webb`, Apply, ask the identical question again →
   the system correctly says it doesn't know, because Marcus's document is about Redis, not
   the primary datastore. *(Talking point: this is the filter actually narrowing the
   retrieval pool, not just re-ranking -- the right document was never in play.)*
5. Clear the author filter, set it to `Priya Shah` instead, ask again → the Postgres answer
   comes back.

---

## Act 4 -- The hard case: two documents that disagree and were never told to

**Story beat:** a compliance policy changed, but nobody linked the two documents together --
the RAG system has to work this out from the content and dates alone, not from a data
structure telling it which one wins.

1. Ingest `log-retention-policy-2025.md` (classification `c2`, author `Dana Fitzgerald`, tags
   `policy, compliance`, doc date `2025-03-01`) and `log-retention-policy-2026-update.md`
   (same classification/author/tags, doc date `2026-06-01`) as **two completely independent
   documents** -- do NOT use "This replaces" for these. This is the point: nothing declares
   that the second supersedes the first.
2. Ask: **"How long does Solstice Systems retain customer-facing logs for?"** with no filter.
   Both documents are retrieved and both are visible to the model -- it has to read the dates
   itself. Expect: **90 days**, citing the 2026 document, explicitly noting the 30-day figure
   is the outdated one. *(This is the genuinely hard case -- two real documents, same topic,
   different numbers, no supersession link between them. Verified live before this demo: 3/3
   runs answer correctly by reading the "effective June 1, 2026" language and the SOC 2
   rationale directly in the text, not just the metadata date.)*
3. Open **Filters**, set the date range to end **before June 2026** (e.g. up to
   `2025-12-31`), Apply, ask the same question again → now answers **30 days** -- the correct
   answer *as of that point in time*, because the 2026 document is outside the requested
   range. *(Talking point: this is a different capability from the conflict-resolution in
   step 2 -- that was "what's true now," this is "what was true then," and both come from the
   same two documents.)*

**Honest note for Q&A, not necessarily demoed live:** this kind of undeclared-conflict
resolution is reliable when the documents state their own dates in plain language (as these
do), but it is not perfect in general -- a terser pair of documents (bare facts in a table,
no "effective as of" phrasing in the text itself) has been observed to fail this same kind of
question. That's a known, named limitation (see `docs/technical-decisions.md`), not something
papered over for this demo.

---

## Closing -- refusal, not hallucination

Ask: **"What was Solstice Systems' quarterly revenue?"** Expect a plain refusal -- none of
this corpus is financial data, and the system says so instead of inventing a number.
*(Talking point: everything shown so far depended on the system being willing to say "I don't
know" when that's the honest answer -- Act 1's ambiguous-question answer and this refusal are
the same underlying discipline, just at different confidence levels.)*
