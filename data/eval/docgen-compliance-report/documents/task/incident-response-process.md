# Incident Response Process

## Severity levels

- **SEV1** -- confirmed or strongly suspected exposure of customer data,
  or a production outage affecting all customers.
- **SEV2** -- significant degradation affecting a subset of customers, or
  a failed safeguard (such as a failed backup restore test) that has not
  yet caused customer-visible impact.
- **SEV3** -- minor issue with no meaningful customer impact.

## Detection and escalation

Any employee who suspects a SEV1 or SEV2 event pages the on-call engineer
through PagerDuty. The on-call engineer opens an incident bridge on the
`#incident-bridge` Slack channel within 15 minutes of being paged, and
pulls in the security team directly for any incident involving suspected
data exposure.

## Customer notification

For a SEV1 incident involving actual or suspected exposure of customer
data, affected customers are notified **within 72 hours** of the
exposure being confirmed. The notification is sent by the customer
success team using a pre-approved template, after the security team signs
off on the facts being accurate.

## Postmortems

Every SEV1 incident gets a written postmortem within **5 business days**
of resolution, covering root cause, timeline, and follow-up action items
with owners and due dates. SEV2 postmortems are written within 10
business days. Postmortems are reviewed at the monthly engineering review
meeting, and open action items are tracked until closed.

## Owner

Maintained by the security engineering team.
