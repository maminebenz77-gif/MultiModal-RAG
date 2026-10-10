# Starling Trust Framework -- Data Protection Requirements

## Requirement 6.1 -- Encryption at rest

All customer data stored at rest in a partner's production environment,
**including backup copies**, must be encrypted using an industry-standard
algorithm with a key length of at least 256 bits. Algorithms or key
lengths weaker than this do not satisfy the requirement, even if the
data is also protected by other controls.

## Requirement 6.2 -- Key management

Encryption keys must be rotated **at least once every 24 months**.
Rotating more frequently than every 24 months is acceptable and does not
create a finding by itself. Keys must never be stored alongside the data
they protect -- for example, a key must not live in the same database,
bucket, or backup archive as the ciphertext it decrypts.

## Requirement 6.3 -- Encryption in transit

All data in transit between the partner's systems, and between the
partner and its customers, must use TLS 1.2 or higher.
