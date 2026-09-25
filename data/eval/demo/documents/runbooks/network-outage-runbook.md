# Runbook: Edge Load Balancer Outage

## When to use this

Use this runbook when the edge load balancer fleet (`prod-lb-edge`) is
returning elevated 502/504 rates (above 2% of traffic over a 5-minute
window), or when the load balancer health-check dashboard shows more than
one AZ fully unhealthy at once.

## Steps

1. Check the load balancer's own health-check page first
   (`https://prod-lb-edge.internal/status`) -- if it reports healthy but
   traffic is still failing, the problem is more likely a downstream
   service than the balancer itself; stop and check the affected
   service's own runbook instead.
2. Pull the current upstream pool membership: `lbctl pool show edge`. A
   pool with fewer than 3 healthy upstreams in an AZ is under-provisioned
   for failover and should be treated as an active incident, not just a
   warning.
3. If a specific upstream is flapping, drain it: `lbctl drain <upstream-id>
   --timeout=30s`. Draining is graceful -- in-flight requests complete
   before the upstream is removed from rotation.
4. Restart the service (`systemctl restart nginx-edge`) on any upstream
   that fails its health check three times in a row after being drained
   and re-added. A flapping health check that survives a drain usually
   means the process itself is wedged, not the network path to it.
5. If restarting does not clear the flapping state within 5 minutes,
   replace the instance rather than continuing to restart it.

## Rollback

Re-adding a drained upstream before its health check passes cleanly will
just reintroduce the same error rate -- always wait for two consecutive
green health checks before returning an upstream to the pool.

## Owner

Maintained by the network infrastructure team.
