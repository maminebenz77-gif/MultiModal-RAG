# Starling Trust Framework -- Credential Management Requirements

The Starling Trust Framework is the security baseline that Starling
Cloud Partners imposes on every company distributing through its partner
marketplace. A partner must demonstrate it meets the Framework's
requirements to keep its marketplace listing active.

## Requirement 4.1 -- Workforce credential rotation

All workforce credentials (any login credential belonging to an
employee or contractor) that grant access to customer data or to
production systems must be rotated **at least every 90 days**. A
rotation interval longer than 90 days does not satisfy this requirement,
regardless of how the rotation is enforced.

## Requirement 4.2 -- Service account and API credential rotation

All service account credentials and API keys used by automated systems
must be rotated **at least every 180 days**. As with workforce
credentials, any rotation interval longer than 180 days fails this
requirement.

## Requirement 4.3 -- Compromise response

Any credential known or suspected to be compromised must be rotated
**within 24 hours** of the compromise being identified, independent of
its normal rotation schedule.

## Scope note

These requirements apply uniformly across all environments the partner
operates (production, staging, and any environment that can reach
customer data), not only to the systems that directly serve customer
traffic.
