# Runbook: On-Call Escalation

## When to use this

Use this runbook whenever a runbook's own steps don't resolve an incident
within their stated time window, or when an incident spans more than one
team's system (e.g. a database failover that also requires a load
balancer change).

## Escalation path

1. Page the secondary on-call for the affected service if the primary
   hasn't acknowledged within 5 minutes.
2. If the incident is still unresolved 15 minutes after paging the
   secondary, escalate to the on-call engineering lead directly rather
   than continuing to wait -- do not page a third individual contributor
   before looping in a lead.
3. Any incident that is customer-visible for more than 30 minutes must be
   declared a formal incident (not just paged) in the incident channel,
   with a named incident commander.
4. The incident commander's job is coordination, not hands-on-keyboard
   fixing -- if the commander is also the only person who can fix the
   issue, page a second engineer immediately so the two roles can split.

## After the incident

Every formal incident gets a postmortem within 3 business days. A
postmortem without a concrete remediation action is considered incomplete
and will be sent back for revision.

## Owner

Maintained by the on-call program lead.
