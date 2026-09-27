# Remaining validation protocol

The paired evaluation completed a portable wheel build, nine CPU and seven GPU tests,
public-patch application/spec checks and a maintenance-window serving evaluation.
See `current-evaluation.md` for the exact completed subset and raw results.
MTP-disabled startup failed twice; a separate GDN warmup passed and did not locate
the earlier fault. Treat that integration path as unresolved.

The extended protocol below is still required before broad compatibility,
quality-preservation, long-context reliability or stable latency-tail claims.

## 1. Resolve the build

Identify a public model-enabled vLLM revision, save its full commit, source/license
provenance, container digest, installed package versions and dependency lock.
Apply small preimage-checked patches in a new checkout. Do not push private Git
history or build against a moving latest tag. Verify that QSA indexer and model
support exist before attempting to start the server. Record model repository,
revision, architecture and weight manifest hash without redistributing weights.

## 2. Correctness before performance

Re-run independent codec, dense/fused parity, slot overwrite, fragmented block
tables, invalid slots, zero/outlier inputs and non-finite gates. Cover fixed global
scales used in serving, especially small-value groups whose FP8 scales underflow
and outlier groups whose scales clip. Verify append continuity and prefix reuse.
Reject duplicate valid slots or demonstrate scheduler uniqueness on every path.
Run all four sanitizer modes at agreed shapes. Preserve raw per-case results.

Test B12x independently: invalid IDs, valid route invariance, differing wrapper
configs, serialized scratch reuse, concurrent stream ownership and graph replay.
Test draft/target backend separation and buffer aliases under MTP=0/1/2 and
the intended parallelism. A CPU mock test is not a GPU safety proof.

## 3. Task quality on the actual serving path

Keep weights, tokenizer, prompts, sampling and indexer behavior identical. Compare
BF16 KV / FP8 KV / NVFP4 KV where supported. Begin with MTP off, then test it
separately. Save exact dataset revisions, split, seed and prompt formatting.

- Use representative classification/extraction and short reasoning tasks with
  answer scoring; predeclare acceptable per-task regression before viewing results.
- Long-context retrieval: several unique facts, 8K/32K/64K/120K lengths, needles
  at 10/50/90 percent depth, at least 20 randomized cases per cell. Save tokenized
  lengths and score exact normalized answers; avoid repeated-phrase-only inputs.
- Repeat selected long tests across append/chunk boundaries and reused physical
  slots. Distinguish quantization error from sparse selection/indexer behavior.
- Log saturation/underflow by layer and compare full score/logit distributions
  where feasible. Attention cosine alone is insufficient.

## 4. Paired performance protocol

Freeze model, MTP, scheduler budget, max sequences, memory utilization, graph mode,
offload policy and workload. Record utilization and co-tenancy, GPU clocks/power,
driver and thermal conditions. Warm each configuration consistently and report
cold start separately. Randomize/interleave baseline and candidate runs when safe.

Use unique-prefix and warm-prefix scenarios separately. Preserve measured prefix
cache hits, preemptions, queue delay, request errors and actual usage counts.
Use at least 30 successful requests per short scenario and several independent
groups; choose a feasible long-input trial count before running. Report raw points,
median, p95 with its estimator and confidence intervals across independent groups.
Do not silently discard failures or estimate missing usage with requested tokens.

Record TTFT to first nonempty text chunk, E2E latency, aggregate output throughput,
and decode estimate with its explicit formula. If exact token timing is required,
instrument token emissions; SSE chunks can bundle speculative tokens.

## 5. Memory and release

Record model/draft memory, recurrent state, main K/V, indexer storage, graph buffers,
allocator reserved vs allocated, page padding and prefill peak separately. Test
the longest claimed windows and sustained concurrency, including preemption and
cancellation. A calculated token pool is not proof of successful workload capacity.

Release only after a clean clone can recreate the environment, run tests and
regenerate figures. Keep negative findings and limits next to headline numbers.
