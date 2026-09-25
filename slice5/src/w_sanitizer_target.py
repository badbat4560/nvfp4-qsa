#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal write-kernel run for compute-sanitizer. Small shapes; exercises valid
unique slots incl. -1(skip)/0/15/16/17/last, fragmented physical blocks, K & V,
partial pages, and a duplicate-slot REJECTION (contract validated on host)."""
import os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../src")))
from write_kernel import nvfp4_write, DuplicateSlotError
from nvfp4_oracle import global_scale_for

DEV = "cuda"; KVH, HD, PAGE = 2, 256, 16


def run_valid(tokens, blocks, slots):
    src = torch.randn(tokens, KVH, HD, device=DEV, dtype=torch.bfloat16)
    g = global_scale_for(src.reshape(-1, HD))
    data = torch.zeros(blocks, PAGE, KVH, HD // 2, device=DEV, dtype=torch.uint8)
    scale = torch.zeros(blocks, PAGE, KVH, HD // 16, device=DEV, dtype=torch.uint8)
    nvfp4_write(src, data, scale, torch.tensor(slots, dtype=torch.int64, device=DEV), g)
    torch.cuda.synchronize()


if __name__ == "__main__":
    run_valid(6, 4, [-1, 0, 15, 16, 17, 63])       # boundary slots, one skip
    run_valid(8, 3, [18, 5, 47, 0, 33, 16, 2, 40]) # unordered, disjoint, partial pages
    run_valid(1, 2, [0])                            # single token
    run_valid(2, 2, [-1, 31])                       # skip + last
    # duplicate-slot must be rejected BEFORE any kernel launch (no race reaches the GPU)
    rej = False
    try:
        run_valid(3, 2, [4, 4, 9])
    except DuplicateSlotError:
        rej = True
    assert rej, "duplicate slots were not rejected"
    torch.cuda.synchronize()
    print("W_SANITIZER_TARGET_DONE")
