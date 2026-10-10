# Logging and Monitoring Practices

## What gets logged

A centralized internal SIEM ingests security-relevant events from across
the platform: authentication successes and failures, privilege and role
changes, admin console actions, and infrastructure configuration changes.
Ordinary application request logs are handled separately and are not
covered by this document.

## Retention

Security-relevant logs in the SIEM are retained for **180 days**, after
which they are deleted. This is a longer window than ordinary application
logs because security logs are the primary evidence used during an
incident investigation, which can take weeks to fully resolve.

## Automated alerting

The SIEM automatically generates an alert for anomalous login patterns,
including logins from an impossible-travel distance from the user's last
known login and repeated MFA failures on the same account within a short
window. Alerts page the on-call security engineer, who is expected to
triage within **1 business hour** during business hours.

## Manual review

The security team performs a manual review of privileged-access logs
(admin console actions and role changes) once a month, independent of
the automated alerting, specifically to catch patterns that wouldn't
trigger an automated rule on their own.

## Owner

Maintained by the security engineering team.
