#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Slice 5 cache-spec proof (standalone; NOT integrated into vLLM).

Computes the NVFP4 split-layout allocation formula, compares it to the ACTUAL
torch.cuda allocation, and reports per-token QSA-KV capacity ratios. Explicitly
separates QSA-only theoretical from actual allocation and the (unchanged) fixed
overhead — it does NOT claim a full-server capacity number.
"""
from __future__ import annotations
import argparse, json, os, sys
import torch

DEV = "cuda"


def formula(blocks, page, kv_heads, head_dim):
    dpg = head_dim // 2          # packed data bytes per (K|V) per token per kv_head
    spg = head_dim // 16         # fp8 group-scale bytes per (K|V) per token per kv_head
    per_tok_kv = (dpg + spg) * 2 * kv_heads     # both K,V and all kv_heads
    total = blocks * page * per_tok_kv          # whole paged cache (K data+scale + V data+scale)
    return {
        "data_bytes_per_KorV_per_token_per_head": dpg,
        "scale_bytes_per_KorV_per_token_per_head": spg,
        "per_token_nvfp4_bytes(K+V, all heads)": per_tok_kv,
        "k_data_bytes": blocks * page * kv_heads * dpg,
        "k_scale_bytes": blocks * page * kv_heads * spg,
        "v_data_bytes": blocks * page * kv_heads * dpg,
        "v_scale_bytes": blocks * page * kv_heads * spg,
        "total_bytes": total,
    }


def actual_alloc(blocks, page, kv_heads, head_dim):
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    kd = torch.zeros(blocks, page, kv_heads, head_dim // 2, device=DEV, dtype=torch.uint8)
    ks = torch.zeros(blocks, page, kv_heads, head_dim // 16, device=DEV, dtype=torch.uint8)
    vd = torch.zeros(blocks, page, kv_heads, head_dim // 2, device=DEV, dtype=torch.uint8)
    vs = torch.zeros(blocks, page, kv_heads, head_dim // 16, device=DEV, dtype=torch.uint8)
    torch.cuda.synchronize()
    used = torch.cuda.memory_allocated() - base
    numel_bytes = sum(t.numel() * t.element_size() for t in (kd, ks, vd, vs))
    del kd, ks, vd, vs
    return {"sum_numel_bytes": numel_bytes, "torch_allocated_bytes": used,
            "alloc_overhead_bytes": used - numel_bytes,
            "note": "torch rounds each block up to its allocator granularity (>=512B); numel bytes is the logical footprint"}


def capacity_ratios(kv_heads, head_dim):
    bf16 = 2 * head_dim * 2 * kv_heads                       # 2 bytes/elem, K+V, heads
    fp8 = 1 * head_dim * 2 * kv_heads                        # 1 byte/elem
    nvfp4 = (head_dim // 2 + head_dim // 16) * 2 * kv_heads  # packed + scale
    return {"per_token_bytes": {"bf16": bf16, "fp8": fp8, "nvfp4": nvfp4},
            "nvfp4_vs_bf16": round(bf16 / nvfp4, 4),
            "nvfp4_vs_fp8": round(fp8 / nvfp4, 4),
            "fp8_vs_bf16": round(bf16 / fp8, 4)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, default=2048); ap.add_argument("--page", type=int, default=16)
    a = ap.parse_args()
    kv_heads, head_dim = 2, 256
    f = formula(a.blocks, a.page, kv_heads, head_dim)
    act = actual_alloc(a.blocks, a.page, kv_heads, head_dim)
    ratios = capacity_ratios(kv_heads, head_dim)
    match = abs(act["sum_numel_bytes"] - f["total_bytes"]) == 0
    R = {
        "config": {"blocks": a.blocks, "page": a.page, "kv_heads": kv_heads, "head_dim": head_dim},
        "formula": f, "actual_allocation": act,
        "formula_matches_numel_bytes": bool(match),
        "capacity_ratios_QSA_KV_only": ratios,
        "scope_and_honesty": {
            "this_is": "QSA main-KV split-buffer allocation ONLY (K/V data + fp8 group scales)",
            "qsa_kv_ratio_vs_fp8": ratios["nvfp4_vs_fp8"],
            "NOT_claimed": "a full-server capacity gain (e.g. +70-80%) — that would require the "
                           "actual KV pool measured with the write path integrated (Slice 6, needs a window)",
            "fixed_overhead_unchanged_by_kv_dtype": [
                "GDN/Mamba SSM cache (FP32)", "MTP=3 draft model + its KV",
                "model weights (NVFP4)", "PLE per-layer embeddings", "activation/graph buffers",
            ],
            "implication": "server-level capacity gain < the QSA-KV-only ratio because fixed overhead "
                           "does not shrink; the real number is a measured server quantity, not asserted here",
        },
        "gate": "PASS" if match else "FAIL",
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=2)
    print(f"formula total_bytes={f['total_bytes']} numel_bytes={act['sum_numel_bytes']} "
          f"torch_alloc={act['torch_allocated_bytes']} match={match}")
    print(f"QSA-KV capacity: nvfp4 vs bf16 x{ratios['nvfp4_vs_bf16']}, vs fp8 x{ratios['nvfp4_vs_fp8']}")
    print("CACHE_SPEC_PASS" if match else "CACHE_SPEC_FAIL")
    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()
