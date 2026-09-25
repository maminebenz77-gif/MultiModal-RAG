# Runbook: Production Deploy Rollback

## When to use this

Use this runbook when a deploy to production has caused an error-rate
spike, a latency regression above the SLO, or a failed canary check that
wasn't caught before full rollout.

## Steps

1. Identify the last known-good release tag from the deploy history:
   `deployctl history --service <service-name> --limit 5`.
2. Freeze further deploys to the affected service so a second, unrelated
   change doesn't land mid-rollback and muddy the signal:
   `deployctl freeze <service-name>`.
3. Roll back: `deployctl rollback <service-name> --to <known-good-tag>`.
   This redeploys the previous artifact; it does not revert any database
   migration that shipped alongside the bad release -- check the release
   notes for a migration before assuming the rollback is complete.
4. Watch the error rate and latency dashboards for 10 minutes after the
   rollback completes before declaring the incident resolved.
5. Unfreeze the service once the rollback is confirmed stable:
   `deployctl unfreeze <service-name>`.

## Rollback of the rollback

If the "known-good" tag turns out to have its own unrelated issue, do not
chain a second automatic rollback -- pause and get a second engineer to
confirm the actual last-good tag before trying again.

## Owner

Maintained by the platform engineering team.
