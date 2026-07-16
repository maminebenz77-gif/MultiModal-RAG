# Retrieval Latency Benchmark: Provider Comparison

This report documents a benchmark comparing response latency across the
three LLM provider configurations supported by our environment-portability
layer: a hosted external API, an internal company gateway, and a local
model running entirely on-device. The goal was to quantify the real-world
cost, in milliseconds, of the abstraction that lets us switch providers
through configuration alone.

## Methodology

We issued a fixed prompt requesting a 200-token completion, repeated ten
times per provider, and recorded wall-clock latency from request
submission to final token received. All requests originated from the same
machine to control for network path variance between the client and each
provider's endpoint.

The hosted API configuration used a standard external key with no rate
limiting applied during the test window. The internal gateway
configuration routed through the company's OpenAI-compatible proxy,
which itself forwards to a self-hosted inference cluster inside the
corporate network. The local configuration ran a quantized model directly
on the test machine's GPU, with no network round trip involved at all.

We deliberately excluded the first request to each provider from the
average, since cold-start effects (connection setup, model loading into
GPU memory for the local case) would have skewed the numbers in a way
that doesn't reflect steady-state usage.

## Results

The table below summarizes average latency per provider across the nine
warm requests measured for each.

| Provider | Model | Avg Latency (ms) | P95 Latency (ms) |
| --- | --- | --- | --- |
| OpenAI (hosted) | gpt-4o-mini | 220 | 310 |
| Internal gateway | internal-llama-70b | 340 | 490 |
| Local (on-device) | quantized-7b | 180 | 240 |

The local configuration was fastest on average, which is expected given
it avoids any network round trip entirely. The internal gateway was the
slowest of the three, and also had the widest gap between average and P95
latency, suggesting more variance in how consistently the internal
inference cluster responds under load compared to the hosted API.

It's worth noting that the hosted API's latency includes whatever queueing
or rate-limiting behavior the provider applies on their end, which we
have no visibility into and cannot control for directly.

## Discussion and Caveats

These numbers should not be read as a general verdict on any provider's
performance. They reflect one machine, one network path, one prompt
shape, and one point in time. A production deployment making many
concurrent requests would likely see different relative rankings,
particularly for the hosted API, where batching and queuing behavior
under real load differs substantially from a single sequential benchmark
like this one.

The local configuration's advantage also comes with a real cost that this
latency-only comparison doesn't capture: it requires a GPU-equipped
machine, consumes local compute resources that could otherwise be used
for other work, and is constrained by whatever model can actually fit
and run acceptably on that hardware. The hosted and internal gateway
options both offload that cost elsewhere, at the price of a network
round trip and, in the hosted case, a dependency on a provider outside
our control.

Future benchmarking work should measure latency under concurrent load
rather than sequential single requests, since that's a much closer match
to how the system will actually be used once it's serving real traffic.
