# Backup and Retention Policy

## Backup schedule

All production databases (the primary customer-data cluster and the
internal billing database) are backed up in full every night. Backups
are written to a separate storage account from the production data they
back up, and are immutable for the duration of their retention window --
no process, including an admin account, can delete or modify a backup
before it ages out.

## Retention window

Backups are retained for **35 days** from the night they are taken.
After 35 days, a backup is automatically and permanently deleted. There
is no mechanism to extend an individual backup's retention past 35 days.

## Restorability testing

A full restore of the most recent nightly backup is performed into an
isolated test environment on the **first Monday of every month**, to
confirm the backup is actually restorable and not merely present in
storage. Results (pass/fail, restore duration, any data discrepancies
found) are logged in the internal backup-restore-test log.

If a monthly restore test fails, it is treated as a SEV2 incident and
triggers the standard incident response process until root-caused and a
successful restore is confirmed.

## Owner

Maintained by the infrastructure team.
