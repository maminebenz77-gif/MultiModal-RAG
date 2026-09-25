# Runbook: Primary Database Failover

## When to use this

Use this runbook when the primary Postgres instance (`prod-pg-primary`) is
unreachable for more than 60 seconds, or when replication lag on the
standby exceeds 5 minutes and the primary is showing disk I/O errors.

## Steps

1. Confirm the primary is actually down: `pg_isready -h prod-pg-primary`.
   A single failed check can be a transient network blip -- wait for two
   consecutive failures 15 seconds apart before proceeding.
2. Promote the standby: `pg_ctl promote -D /var/lib/postgresql/standby`.
   This is a one-way operation -- the old primary cannot rejoin as primary
   without a full re-clone afterward.
3. Update the connection string in the app config service to point at the
   newly-promoted instance. The application layer picks this up within 30
   seconds without a redeploy.
4. Restart the service (`systemctl restart postgres-pooler`) so pooled
   connections drop their stale routes and reconnect against the new
   primary. Do not restart the application servers themselves -- the pool
   restart alone is sufficient and avoids an unnecessary traffic blip.
5. Verify writes are succeeding again with a canary insert against the
   `health_check` table.

## Rollback

If the promoted standby is also unhealthy, do not attempt a second
promotion -- escalate immediately per the on-call escalation runbook
instead of guessing at a third failover target.

## Owner

Maintained by the database reliability team.
