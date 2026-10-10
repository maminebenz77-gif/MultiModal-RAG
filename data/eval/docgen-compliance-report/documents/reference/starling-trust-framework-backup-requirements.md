# Starling Trust Framework -- Backup Requirements

## Requirement 7.1 -- Backups required

A partner must maintain backups of every production system that stores
customer data, taken on a regular schedule.

## Requirement 7.2 -- Restorability must be verified, not assumed

A backup is only credited toward this requirement if its restorability
is actually verified by performing a real restore into a test
environment, **at least once per fiscal quarter**, with the result
(pass or fail, and what was found) documented. Confirming that a backup
file exists in storage, without a restore test, does not satisfy this
requirement.

## Requirement 7.3 -- Minimum retention

Backup copies must be retained for a **minimum of 30 days** before they
may be deleted or allowed to expire.
