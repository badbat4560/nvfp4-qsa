# Evidence report

This report distinguishes the paired serving evaluation from earlier numerical validation.

## Scope

The work concerns **runtime cache storage**, not a new quantized weight release.
The inspected model configuration repeats three linear-attention blocks and one
full-attention block, for 36 and 12 respectively. Main QSA K/V arrays are packed
into NVFP4; recurrent state remains float32 in the inspected launch arguments.
Indexer state and the rest of the runtime must be accounted for separately.

## Accounting and the failure

For head dimension D=256, a K or V vector occupies `D/2 + D/16 = 144` bytes.
With K and V and H=2 heads the main-K/V cost is 576 bytes/token/layer.
The comparable FP8 and BF16 costs are 1,024 and 2,048 bytes. Across the twelve
full-attention layers that is 6,912, 12,288 and 24,576 bytes per token before
padding, indexer storage, metadata and other model state.

The hybrid planner was supplied a logical width that implied 1,024 bytes, then
received a physical specification using 576. A 3,136-token attention page was
too small to pair with the 3,207,168-byte recurrent-state page. Making physical
content bytes available before platform adjustment gives an exact 5,568-token
match. The finalized physical uint8 specification uses quantization mode NONE,
preventing the packed width from being transformed a second time.

The CPU test executes the actual extracted `customize_spec()` method against a
minimal specification, checking the NVFP4 transformation, repeated application,
and unchanged FP8/NONE paths. It is not a test of the complete vLLM planner.

## Numerical evidence

The archived read suite compares an independently implemented FP4 codec to both
a Python quantizer and a CUDA quantizer. Byte matches differ: 0.932129 vs the
Python reference and 0.999634 vs CUDA. Only 123 of 567 nibble differences against
the Python reference are exact ties; the rest are near-boundary cases associated
with floating-point scaling. All differences must remain visible in the report.

Decoded-value cosine against the Python reference is 0.99999726. Fused read vs
dense read of the same quantized data is 0.99999702. Comparing the fused NVFP4
result with the unquantized BF16 oracle gives about 0.99079. The first comparison
tests arithmetic/layout; the latter includes quantization error. Neither is task
accuracy. The saved read run reports 27/27 boundary cases and clean valid outputs;
the write suite reports 28/28 and tests slot reuse. Sanitizer results are historical,
restricted to tested shapes and do not certify all long-context server paths.

The standalone codec chooses a global scale from tensor amax. In the integrated
cache the K/V global scales are initialized to 1.0 and remain fixed; per-group
scales are written with each token. Recomputing the global scale on append would
invalidate older entries unless their decoding scale were preserved. Task-quality
evaluation must therefore exercise the **actual fixed-scale serving path**.

One older write-suite README characterizes discrepancies as tie-only. This is
too broad compared with the read-suite raw data; that wording is not carried
forward. Similarly, computed load-address counts do not measure DRAM traffic:
cache locality and physical traffic require a profiler.

## Historical server measurements

The saved streaming client uses temperature 0 and ignore_eos, requests 256 output
tokens for short decode and one output token for long prefill, and records usage
from the server. Each request has a timestamp/index nonce before the common body.
The long prompt repeats a short synthetic phrase. No prompt/response task score
is recorded. Some short outputs contain role/thinking delimiters; this is a
raw completion throughput workload, not a validated conversational benchmark.

| Scenario | Requests | Total input tokens | Median TTFT | Max TTFT | Wall time |
|---|---:|---:|---:|---:|---:|
| Short decode, single | 1 | 73 | 0.234 s | 0.234 s | 4.791 s |
| Short decode, seven | 7 | 511 | 0.434 s | 0.436 s | 5.746 s |
| Long prefill, single | 1 | 120,064 | 14.569 s | 14.569 s | 14.573 s |
| Long prefill, seven | 7 | 840,448 | 58.359 s | 101.939 s | 101.953 s |

Single decode is 55.99 output tokens/s after the first text chunk. The seven-way
median is 52.48, with a range approximately 46.09-96.14. Aggregate E2E output is
311.89 tokens/s. Prefill wall-time input rates are approximately 8,239 and 8,243
tokens/s. The historical ~8,242 figure for the single prompt uses TTFT as the
denominator instead of full client wall time; both definitions are explicit here.

The client defines decode rate as `(output_tokens - 1)/(end - first_text_chunk)`.
With speculative decoding, a first chunk need not contain exactly one token;
label the metric as this client estimate, not an exact inter-token GPU timing.
The helper's p95 is a nearest selected order statistic: for n=7 it is the maximum.
No repeated trials, confidence intervals, matched baseline or warmed/cold prefix
cache counters are in the saved record.

## Runtime identity vs runtime performance

| Setting | Historical benchmark | Observed live arguments, Sep 25 |
|---|---:|---:|
| max model length | 169,984 | 169,984 |
| MTP speculative tokens | 2 | Cmd says 1; persisted wrapper selection is 3 |
| max sequences | 7 | 12 |
| scheduler token budget | 2,048 | 8,192 |
| GPU memory utilization | 0.99 | 0.94 |
| recurrent-state dtype | Historical narrative: float32 | float32 |

The launch wrapper can override the command-line speculative count; its persisted last-good value was 3 before the maintenance window.

The five inspected implementation files matched the archived SHA-256 values.
Identical source does not make the new configuration's latency identical.
The historical ~1,002,197-token pool and 5.90 full-window concurrency are recorded
in the integration narrative, not in the streaming JSON; treat them as historical
reported configuration values pending independent allocator evidence. Seven
120K inputs do not demonstrate seven simultaneous full 169,984-token windows.

## Fresh evaluation and remaining gates

See [Paired serving evaluation](current-evaluation.md) for the fresh serving results,
failed startup configurations, exact protocol and limitations. The installable
portable wheel passed nine CPU and seven real-GPU tests on an RTX 4050 Laptop.
A public vLLM port is pinned and passes patch/spec checks, but has not been built
and served end to end. Extended quality, MTP-disabled startup, concurrent streams,
cache lifecycle stress and a full public build remain separate validation gates.
