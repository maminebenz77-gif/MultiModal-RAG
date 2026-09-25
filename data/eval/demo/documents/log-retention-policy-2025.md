# Customer Log Retention Policy

Solstice Systems retains customer-facing application logs (request logs,
access logs, and audit logs) for **30 days** from the time they are
written, after which they are permanently deleted from all storage
tiers, including backups.

This retention window was set to balance debugging usefulness against
storage cost: in practice, the large majority of support investigations
reference logs from within the past two weeks, so 30 days provides a
comfortable margin without keeping data indefinitely.

Logs older than 30 days cannot be recovered under any circumstances,
including for a customer support request or a legal hold -- if a longer
retention window is ever needed for a specific case, it must be arranged
in advance, before the data ages out.

This policy applies uniformly across all regions and all customer tiers.
