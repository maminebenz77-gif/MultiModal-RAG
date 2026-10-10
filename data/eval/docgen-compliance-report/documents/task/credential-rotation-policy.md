# Credential Rotation Policy

## Scope

Covers all workforce login credentials (SSO-backed accounts used by
Meridian Cloud Labs employees and contractors) and all service account /
API credentials used by automated systems in production.

## Workforce credential rotation

Workforce passwords are rotated every **180 days**. Rotation is enforced
automatically by the identity provider (Okta): any account whose password
has not been changed in 180 days is locked out of SSO until the user sets
a new one. There is no manual exception process -- the lockout cannot be
deferred by a manager or admin.

## Service account and API credential rotation

Service account API keys and automated-system credentials are rotated
every **365 days** by the platform team, tracked in the internal
credential inventory. Each key is scoped to a single service and cannot
be shared across services.

## Compromise handling

If a credential (workforce or service account) is suspected to be
compromised, it is rotated immediately, and in all cases **within 24
hours** of the suspicion being reported to the security team. This is a
hard deadline tracked in the incident ticket for the compromise.

## Owner

Maintained by the security engineering team. Reviewed annually, or
sooner if a credential-related incident reveals a gap in the policy.
