# Install and reproduce the portable cache

The portable package does not import vLLM. It exposes the archived Triton
writer/reader through a checked Python API and an independent CPU codec.
It is an experimental cache component, not a model checkpoint or a drop-in
replacement for every vLLM attention backend.

Install a CUDA-enabled PyTorch build for your system first. Then:

```bash
python -m pip install ".[test,gpu-linux]"
python -m pytest tests -q
```

On Windows use `gpu-windows` instead of `gpu-linux`. The isolated wheel was
installed and tested on Windows, Python 3.12.9, PyTorch 2.11.0+cu128,
Triton 3.6.0 and an RTX 4050 Laptop GPU. All 16 tests passed: nine CPU tests
and seven GPU cases. Other combinations require their own validation.

```python
import torch
from nvfp4_qsa import PackedCache

cache = PackedCache(blocks=4, page_size=16, device="cuda")
k = torch.randn(16, 2, 256, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
cache.write(k, v, torch.arange(16, device="cuda", dtype=torch.int64))
q = torch.randn(1, 24, 256, device="cuda", dtype=torch.bfloat16)
selected = torch.arange(16, device="cuda", dtype=torch.int32)[None]
block_table = torch.tensor([[0]], device="cuda", dtype=torch.int32)
requests = torch.zeros(1, device="cuda", dtype=torch.int32)
result = cache.attend(q, selected, block_table, requests)
```

The wrapper validates duplicate write slots, non-finite input, dimensions,
devices and metadata dtypes. Its reductions synchronize with the host; it is
not a timed production fast path. It intentionally rejects CUDA graph capture.
Callers must order streams and manage cache lifetime. Global scales are fixed
at construction. Do not change them while stored tokens remain in the cache.
The checked attention API supports 24 query heads, two KV heads and width 256.
Invalid write slots are ignored by the kernel; duplicate valid slots raise.

The GPU tests compare decoded writer output against the independent CPU codec
for zeros, small values, ordinary values and saturation. Attention is compared
against a CPU dense calculation using decoded cache values and a fragmented
block table. These tests demonstrate implementation agreement, not model quality.

## Public vLLM port candidate

`integration/public/qsa-nvfp4.patch` targets exactly upstream commit
`e126687a9a828d513c01a07cd69f025f27d63280` from
[vLLM PR 53896](https://github.com/vllm-project/vllm/pull/53896).
That public tree names the model module `qwen4_exp`.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout e126687a9a828d513c01a07cd69f025f27d63280
git apply --check /path/to/qsa-nvfp4.patch
git apply /path/to/qsa-nvfp4.patch
```

The patch adds the owned packed cache reader/writer and physical-byte planner
accounting without replacing the entire platform module. Patch application and
Python syntax were checked. **A full server build/run of this public port has
not been validated.** The production snapshot and server measurements use a
different historical base; their success does not establish this port's success.
Do not overwrite an active deployment with this candidate.
