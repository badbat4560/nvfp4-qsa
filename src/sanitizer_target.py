#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal fused-NVFP4-kernel run for compute-sanitizer. Small shapes (sanitizer
is ~10-50x slow). Exercises packed FP4 loads, group-scale loads, block-table
indirection, contiguous + fragmented tables, and adversarial (invalid) indices."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_validation import build_case, nvfp4_split_from_half, DEV
from nvfp4_kernel import qsa_sparse_paged_attention_nvfp4

QH, KVH, HD, PAGE = 24, 2, 256, 16


def run(ctx, rows, topk, dist, frag, adversarial=False):
    kv, q, idx, bt, ttr, nb = build_case(ctx, rows, topk, QH, KVH, HD, PAGE, dist, 7, frag=frag)
    if adversarial:  # inject out-of-range / negative / huge indices
        idx[0, 0] = -1
        idx[0, 1] = ctx + 999
        idx[0, 2] = 2**31 - 1
        if rows > 1:
            idx[1, :] = -1
    kd, ks, gk, _, _ = nvfp4_split_from_half(kv[..., :HD], PAGE, KVH, HD)
    vd, vs, gv, _, _ = nvfp4_split_from_half(kv[..., HD:], PAGE, KVH, HD)
    out = qsa_sparse_paged_attention_nvfp4(q.contiguous(), kd, ks, vd, vs, gk, gv, idx, bt, ttr)
    torch.cuda.synchronize()
    return out


if __name__ == "__main__":
    run(256, 2, 64, "normal", frag=False)
    run(256, 2, 64, "normal", frag=True)
    run(64, 2, 2051, "heavytail", frag=False)     # topk > ctx (heavy masking)
    run(256, 2, 64, "normal", frag=False, adversarial=True)
    torch.cuda.synchronize()
    print("SANITIZER_TARGET_DONE")
