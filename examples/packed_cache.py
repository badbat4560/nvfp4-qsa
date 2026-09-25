"""Minimal CUDA example; install the local package before running."""
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
assert torch.isfinite(result).all().item()
print(f"Output: {tuple(result.shape)}; cache storage: {cache.bytes} bytes")
