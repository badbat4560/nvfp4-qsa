# Benchmark clients

`historical_bench_stream.py` preserves the earlier workload and metric definitions;
only its private model-alias default was removed. It sends requests immediately
when invoked. It is retained to interpret the historical record, not as the
recommended new measurement protocol.

`run_streaming.py` is a new client prepared during the audit. It is dry-run by
default, requires actual server token-usage fields, retains failures and refuses
to overwrite an existing result. It never saves auth credentials, endpoints or
prompt text. It adds unique prefixes; this makes it inappropriate for measuring
warm-prefix reuse or scoring a prompt that must be byte-for-byte unchanged.

Example, from the repository root (no network requests without `--execute`):

```bash
python benchmarks/run_streaming.py --url http://127.0.0.1:8000 \
  --model YOUR_TEST_MODEL --cases benchmarks/example.jsonl \
  --output runs/new-benchmark.jsonl --repeats 30 --concurrency 1
```

Only add `--execute` on a dedicated test endpoint or during an agreed window.
`BENCH_API_KEY` is optional. This client has been checked for syntax and dry-run
behavior only; no server end-to-end validation was performed in the audit.

A repeat submits each input row once. Concurrency is the maximum outstanding
requests, not a duplication count: provide seven rows for a seven-request group.
Output includes every request and group wall time. No warmup is implicit: conduct
and record a separate warmup under the agreed protocol. See
`docs/validation-plan.md` for baseline controls and quality evaluation.
