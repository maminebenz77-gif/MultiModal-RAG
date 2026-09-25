# ADR-002: Primary Datastore Choice

## Status

Accepted.

## Context

The core application needed a primary datastore with strong consistency
guarantees for financial-adjacent data (billing records, usage metering)
and mature support for complex, multi-table transactional writes.

## Decision

We chose Postgres as the primary datastore, run as a single-primary /
standby pair with synchronous replication for the billing tables and
asynchronous replication for everything else.

## Consequences

Synchronous replication on the billing tables adds roughly 3-5ms of write
latency compared to async-only, which is an acceptable cost for not being
able to lose a committed billing write during a failover. Everything
outside the billing tables accepts the small window of possible data loss
that async replication implies, in exchange for lower write latency.

## Alternatives considered

A multi-primary setup was considered and rejected: the added operational
complexity of conflict resolution wasn't justified given the write volume
this system actually sees.
