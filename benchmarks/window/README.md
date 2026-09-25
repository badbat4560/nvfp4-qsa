# Matched cache-mode evaluation

`manifest.json` pins public dataset revisions, source hashes, exact case IDs,
label ordering and synthetic prompts. `prepare.py` downloads those exact source
files and verifies their hashes. External text is not relicensed by this package.
The 16 examples per public dataset were selected before seeing model outputs.
This is a small regression/smoke evaluation, not a full AG News or SIB-200 score.

The runner expects an **isolated** OpenAI-compatible server at
`http://127.0.0.1:8000` serving the alias `audit-model`:

```bash
cd benchmarks/window
python prepare.py
python run.py bfloat16
# Restart the same model with only the cache mode changed, then:
python run.py fp8
python run.py nvfp4
```

Run each command only once into an empty output directory. The script appends
individual observations so interrupted attempts remain inspectable. It performs
two warmup requests, 48 public classification questions and nine synthetic
retrieval questions, then 12 sequential and 12 four-way concurrent speed requests.
Quality uses greedy decoding and exact stripped-answer matching, including format
compliance. Retrieval filler labels in IDs are character-derived size parameters,
**not measured token counts**; actual token usage is saved by the server.

Speed requests have unique user-message prefixes and request 128 output tokens
with `ignore_eos`. TTFT is time to the first nonempty content chunk. Batch output
throughput includes request overhead and queueing. Shared chat-template prefixes
can still be cached. These are warmed application timings, not cold-cache kernel
timings. The sample is too small for a stable latency-tail claim.

Record launch flags and the actual effective speculative configuration alongside
every result. Keep the same weight files, GPU, runner, scheduler budget, sequence
limit, eager/compiled mode and memory fraction across comparisons. A startup
failure is a failed configuration, not a zero-throughput measurement.
