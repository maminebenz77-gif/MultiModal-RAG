# Production Deploy Approval Policy

## Policy

Every production deploy requires sign-off from **one** approver who is
not the author of the change, recorded as an approving review on the
deploy request before the pipeline is allowed to run.

## Scope

Applies to all services deployed through the standard `deployctl`
pipeline. Hotfixes follow the same rule -- there is no expedited path
that skips review, even for a single-line change.

## Rationale

A single independent reviewer catches the large majority of obvious
mistakes (a wrong config value, a missing migration, a leftover debug
flag) at negligible cost to deploy velocity.

## Owner

Maintained by the platform engineering team.
