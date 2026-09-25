# ADR-005: Caching Layer Choice

## Status

Accepted.

## Context

Read traffic on session and rate-limit lookups had grown to the point
where it was a measurable fraction of primary database load, for data
that tolerates being slightly stale and doesn't need Postgres's
durability guarantees.

## Decision

We chose Redis, deployed as a single cluster with 3 shards, for session
storage and rate-limit counters. Nothing that requires durable,
transactional writes is allowed to live in this layer -- it is explicitly
scoped to data that is acceptable to lose on a cache-layer restart.

## Consequences

Session storage moving to Redis reduced primary database read load by
roughly a third. The tradeoff is an explicit one: a Redis cluster
restart invalidates all active sessions, which we accept because a forced
re-login is a minor inconvenience, not a correctness problem.

## Alternatives considered

An in-process cache per application instance was considered and rejected
-- it would have made rate-limit counters inconsistent across instances,
defeating the purpose of a rate limit.
