# Access Control and Least-Privilege Policy

## Role-based access

All access to internal systems is granted through role-based groups
managed in the SSO identity provider (Okta), not through direct per-user
grants. Requesting access to a new role requires sign-off from both the
requester's manager and the system owner before the group membership is
applied.

## Human access to production

Any human interactive login to a production system (shell access, admin
console, or database console) requires **multi-factor authentication**,
enforced as a conditional-access rule in Okta. The accepted MFA methods
are a hardware security key or a TOTP authenticator app -- SMS-based codes
are explicitly not accepted as a second factor.

## Automated and service-account access

Automated systems and internal services that need to call production
APIs authenticate using long-lived static API keys, issued and tracked
by the platform team. Each key is scoped to exactly one service. Because
these are machine-to-machine calls rather than interactive logins, they
are not subject to the MFA requirement above -- the static key is the
sole credential presented.

## Access review

All access entitlements (both human role memberships and service account
key assignments) are reviewed **quarterly** by the relevant team lead.
Any entitlement that hasn't been used in the prior quarter is revoked as
part of this review.

## Owner

Maintained by the security engineering team.
