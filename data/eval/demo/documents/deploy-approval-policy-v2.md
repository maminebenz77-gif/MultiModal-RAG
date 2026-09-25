# Production Deploy Approval Policy

## Policy

Every production deploy requires sign-off from **two** independent
approvers, neither of whom is the author of the change, recorded as
approving reviews on the deploy request before the pipeline is allowed to
run.

## What changed from the previous policy

The single-approver rule was raised to two approvers after a January 2026
incident in which one reviewer approved a deploy without noticing a
missing database migration, causing a 40-minute outage. A second
independent reviewer is required specifically to catch exactly this class
of miss -- one reviewer being wrong is a person problem; two reviewers
being wrong the same way, independently, is much rarer.

## Scope

Applies to all services deployed through the standard `deployctl`
pipeline. Hotfixes follow the same rule -- there is no expedited path
that skips review, even for a single-line change.

## Owner

Maintained by the platform engineering team.
