#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""fp32 reference sparse QSA attention returning QK scores, softmax probs, and
output SEPARATELY (plan §4). Runs on CPU by default to keep the GPU footprint to
just the kernels under test. Semantics mirror the shipped QSA kernel:
scale = head_dim**-0.5, GQA (kv_head = q_head // group), invalid index -> masked.
"""
from __future__ import annotations
import torch


def qsa_reference(q, key_cache, value_cache, logical_indices, block_table,
                  token_to_req, page_size):
    """q:[rows,qh,hd]; caches:[blocks,page,kvh,hd]; indices:[rows,W] int.
    Returns dict of fp32 tensors: out[rows,qh,hd], scores[rows,qh,W],
    probs[rows,qh,W], valid[rows,W] bool."""
    q = q.float(); kc = key_cache.float(); vc = value_cache.float()
    rows, qh, hd = q.shape
    num_blocks, page, kvh, _ = kc.shape
    W = logical_indices.shape[1]
    group = qh // kvh
    ptw = block_table.shape[1]
    scale = hd ** -0.5
    out = torch.zeros(rows, qh, hd, dtype=torch.float64)
    scores_all = torch.full((rows, qh, W), float("nan"), dtype=torch.float64)
    probs_all = torch.zeros((rows, qh, W), dtype=torch.float64)
    valid_all = torch.zeros((rows, W), dtype=torch.bool)
    for r in range(rows):
        req = int(token_to_req[r]); idx = logical_indices[r].long()
        lpage = torch.clamp(idx // page_size, min=0)
        poff = idx % page_size
        valid = (idx >= 0) & (lpage < ptw) & (req >= 0)
        phys = torch.where(valid, block_table[req, torch.clamp(lpage, max=ptw - 1)].long(),
                           torch.full_like(idx, -1))
        valid = valid & (phys >= 0) & (phys < num_blocks)
        valid_all[r] = valid
        sp = torch.clamp(phys, min=0)
        Ksel = kc[sp, poff]           # [W, kvh, hd]
        Vsel = vc[sp, poff]
        for h in range(qh):
            kv = h // group
            k = Ksel[:, kv, :].double(); v = Vsel[:, kv, :].double()
            s = (q[r, h].double() @ k.T) * scale
            s = torch.where(valid, s, torch.full_like(s, float("-inf")))
            scores_all[r, h] = torch.where(valid, s, torch.full_like(s, float("nan")))
            if not bool(valid.any()):
                continue
            p = torch.softmax(s, dim=0)
            probs_all[r, h] = p
            out[r, h] = p @ v
    return {"out": out.float(), "scores": scores_all.float(),
            "probs": probs_all.float(), "valid": valid_all}
