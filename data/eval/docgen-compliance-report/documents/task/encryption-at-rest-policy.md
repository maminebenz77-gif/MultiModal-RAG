# Encryption Practices

## Data at rest

All production data stores -- the primary customer-data database cluster,
the internal billing database, object storage (customer-uploaded files),
and every nightly backup described in the backup and retention policy --
are encrypted at rest using **AES-256**.

## Key management

Encryption keys are managed by an internal key management service (KMS)
that is architecturally separate from the data stores it protects: a key
is never stored in the same database, bucket, or backup as the data it
encrypts. Keys are rotated **once per year**. A key rotation re-wraps the
affected data without requiring downtime.

## Data in transit

All traffic between services, and between customers and the platform,
uses TLS 1.2 or higher. TLS 1.0 and 1.1 are disabled at the load balancer
level.

## Owner

Maintained by the infrastructure team, with the key management service
itself operated by the security engineering team.
