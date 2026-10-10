# Starling Trust Framework -- Access Control Requirements

## Requirement 5.1 -- Strong authentication for all production access

All access to a partner's production systems or to customer data must be
protected by multi-factor authentication (MFA) or an equivalent strong
authentication mechanism. This requirement applies to **both human and
automated access** -- there is no carve-out for machine-to-machine calls.
Acceptable equivalents to MFA for automated access include mutually
authenticated TLS certificates and hardware-bound keys that require a
second factor to use.

A static, long-lived shared secret (such as a plain API key or password)
used by itself, with nothing else required to authenticate, **does not
satisfy this requirement** -- whether the party presenting it is a human
or an automated system.

## Requirement 5.2 -- Periodic access review

Access entitlements (both human role assignments and service account or
API key assignments) must be reviewed for continued necessity **at least
once per quarter**. Entitlements found to be unused or unnecessary must
be revoked as part of the review, not left for a later cycle.

## Requirement 5.3 -- Timely revocation

Access must be revoked within 24 hours of a role change or termination
that makes the access no longer appropriate.
