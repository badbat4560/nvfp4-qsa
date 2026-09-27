# Integration status: historical snapshot and public port candidate

The read-only audit compared SHA-256 for five files in the running
container to the archived sources. All five matched. `SOURCE_MANIFEST.json` holds
the exact source hashes. This establishes source identity at that time, not
fresh validation of the runtime's behavior.

## Source destinations in the historical build

| Distributed file | Historical package path |
|---|---|
| snapshot/qsa.py | vllm/models/qwen3_8_flash_next/nvidia/qsa.py |
| snapshot/nvfp4_qsa.py | vllm/models/qwen3_8_flash_next/nvidia/ops/nvfp4_qsa.py |
| snapshot/interface.py | vllm/platforms/interface.py |
| ../deployment/vllm/native_nvfp4/eagle_utils.py | vllm/v1/worker/gpu/spec_decode/eagle/utils.py |
| ../deployment/vllm/native_nvfp4/b12x_sanitize.py | b12x_sanitize.py at the package root |

The diff is relative to a **local, already FP8-enabled QSA source**, not vanilla
upstream. Its baseline hash is in `baseline.json`. It neither contains the entire
model integration nor establishes compatibility with a stock wheel.

The historical Dockerfiles used local images, so they are deliberately omitted.
A replacement Dockerfile that merely says `FROM vllm:latest` would be misleading.
The public base revision must be identified, required model support included,
patches applied against exact preimage hashes, and a clean build tested before
an installation command can be promoted as supported.

## Separate runtime corrections

`b12x_sanitize.py` maps negative/out-of-range expert IDs to zero and assigns those
routes zero weight. It also reuses buffers between wrappers with identical
configuration. The CPU tests cover route masking and shared-object identity.
They do **not** demonstrate safety under concurrent streams, overlapping wrapper
execution or CUDA Graph capture. Shared scratch buffers require serialized use
or an explicit ownership/synchronization design. Do not generalize the historical
sequential-layer deployment to arbitrary concurrent wrappers.

`eagle_utils.py` keeps the target's B12x backend while allowing the BF16 draft to
choose a backend, and adjusts embedding/head/index-buffer sharing. These changes
are separate from the packed KV format and need separate integration gates.

The integrated writer relies on unique valid physical slots per call. The
standalone writer validates duplicates on the host, but that guard is not the
same as proof that every server scheduler path satisfies the contract.

## Deferred GPU checks

Only on an isolated test worker with a validated matching installation:

```bash
python src/run_validation.py --out runs/read/comparisons.json
python src/boundary_matrix.py --out runs/read/boundary_matrix.json
python slice5/src/w_validate.py --out runs/write/w_validate.json
python slice5/src/cache_spec.py --out runs/write/cache_spec.json
python integration/test_slice6.py
```

These commands are not a fresh-install recipe. They allocate GPU memory and are
not part of the CPU-only default test suite. Compute Sanitizer must be installed
separately from NVIDIA and repeated on agreed shapes in the same isolated window.

## New public port candidate

`public/qsa-nvfp4.patch` and `public/qsa.py` target the exact revision recorded in
`public/baseline.json`. The patch applies cleanly and physical-spec checks pass.
This is a separate port, not the historical serving build. Its complete server
startup remains unverified. The independently installable `nvfp4_qsa` package
and its GPU tests do not require either full vLLM integration.
