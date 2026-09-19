# Incident Postmortem: Embedding Service Outage

## Summary

The embedding service was unavailable for 40 minutes. Ingestion of new
documents failed entirely during the outage; search over already-indexed
content was only partially affected.

## Root cause

An internal TLS certificate on the model-serving endpoint expired without
triggering a renewal. Every embedding request failed closed -- the client
treated the TLS handshake failure as a hard error rather than retrying
against a fallback -- so no new vectors could be produced for the duration
of the outage.

## Impact

100% of ingestion requests failed during the 40-minute window, since
ingestion always needs a fresh embedding for each new chunk. Search
traffic was affected differently depending on retrieval method: keyword
(BM25) search was completely unaffected, since it doesn't call the
embedding service at all. Vector search failed for the same reason
ingestion did -- it needs to embed the incoming query -- so hybrid
retrieval degraded to keyword-only results for the whole 40 minutes
rather than failing outright.

## Remediation

Certificate rotation is now automated, with an alert firing 30 days before
any certificate on the model-serving path is due to expire, rather than
relying on manual tracking. A follow-up action was also opened to make the
embedding client retry against a secondary endpoint on a TLS failure,
instead of failing closed immediately.
