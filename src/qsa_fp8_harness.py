#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated BF16-oracle vs FP8-QSA kernel harness for Qwen3.8-Flash-Next.

Requires the matching model-enabled vLLM build to import the
vendored QSA Triton kernels. It does NOT touch the running server: it builds
its own tiny paged K/V cache, runs the shipped BF16 sparse-attention kernel as
an oracle, a pure-fp32 PyTorch reference as ground truth, and an FP8-E4M3
variant of the sparse-attention kernel (dequant-on-read) as the candidate.

Three questions are answered separately (§4.2 / §7 of the plan):
  A. kernel/ref agreement      : bf16_kernel  vs fp32_ref(bf16_cache)   -> layout+semantics correct
  B. fp8 kernel arithmetic     : fp8_kernel   vs fp32_ref(dequant_fp8)  -> my dequant kernel is correct
  C. fp8 quantization quality  : fp32_ref(deq) vs fp32_ref(bf16)        -> the real fp8 storage cost
  D. end-to-end A/B            : fp8_kernel   vs bf16_kernel            -> what production would see
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import (
    qsa_sparse_paged_attention,          # BF16 oracle (shipped)
    _qsa_merge_splitk_kernel,            # dtype-agnostic fp32 split merge (reused)
)
from vllm.triton_utils import HAS_TRITON, tl, triton

E4M3_MAX = 448.0
LOG2E = 1.4426950408889634


# --------------------------------------------------------------------------- #
# FP8 variant of _qsa_sparse_paged_gqa_splitk_kernel: dequant-on-read.
# Identical arithmetic to the shipped kernel except K/V are fp8_e4m3 in the
# cache and are upcast to bf16 after load; K scale folds into the score scale,
# V scale multiplies the normalized output (both per-tensor scalars here).
# --------------------------------------------------------------------------- #
@triton.jit
def _qsa_sparse_paged_gqa_splitk_fp8_kernel(
    q_ptr,
    k_cache_ptr,          # fp8_e4m3
    v_cache_ptr,          # fp8_e4m3
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    k_scale,              # runtime scalar
    v_scale,              # runtime scalar
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    # Fold the per-tensor K dequant scale straight into the softmax scale.
    softmax_scale_log2 = (HEAD_DIM**-0.5) * 1.4426950408889634 * k_scale

    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        ).to(tl.bfloat16)          # fp8 -> bf16 dequant (scale folded into softmax_scale_log2)
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.bfloat16)          # fp8 -> bf16 dequant (v_scale applied to output)
        scores = tl.dot(query, keys)
        scores *= softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    normalized_output *= v_scale     # per-tensor V dequant (linear, commutes with merge)
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        tl.store(
            partial_output_ptr
            + (
                (split_id * num_rows + row) * NUM_QUERY_HEADS
                + first_head
                + head_offsets[:, None]
            )
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr
            + (split_id * num_rows + row) * NUM_QUERY_HEADS
            + first_head
            + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


def qsa_sparse_paged_attention_fp8(
    q: torch.Tensor,           # bf16 [rows, q_heads, head_dim]
    k_cache: torch.Tensor,     # fp8_e4m3 [blocks, page, kv_heads, head_dim]
    v_cache: torch.Tensor,     # fp8_e4m3 same shape
    k_scale: float,
    v_scale: float,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP8 twin of the shipped qsa_sparse_paged_attention (same profile logic)."""
    assert q.dtype == torch.bfloat16
    assert k_cache.dtype == v_cache.dtype == torch.float8_e4m3fn
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    if out is None:
        out = torch.empty_like(q)
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    base_programs = q.shape[0] * k_cache.shape[2]
    small_profile_limit = 8 if block_m <= 8 else 4
    if base_programs <= small_profile_limit:
        block_n, target_splits, partial_warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, partial_warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, partial_warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, partial_warps = 64, 4, 2
    else:
        block_n, target_splits, partial_warps = 64, 1, 2

    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    num_splits = min(max_useful_splits, target_splits)

    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device
        )

    partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
    _qsa_sparse_paged_gqa_splitk_fp8_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        float(k_scale),
        float(v_scale),
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        logical_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=partial_warps,
        num_stages=2,
    )
    if num_splits == 1:
        return out
    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out,
        out.stride(0),
        out.stride(1),
        q.shape[0],
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=2,
        num_stages=1,
    )
    return out


# --------------------------------------------------------------------------- #
# Pure fp32 PyTorch reference (ground truth), matching the kernel's semantics.
# --------------------------------------------------------------------------- #
def ref_sparse_attention(
    q: torch.Tensor,               # bf16/f32 [rows, q_heads, head_dim]
    key_cache: torch.Tensor,       # [blocks, page, kv_heads, head_dim] (any float dtype)
    value_cache: torch.Tensor,
    logical_indices: torch.Tensor, # int32 [rows, W]
    block_table: torch.Tensor,     # int32 [reqs, pages]
    token_to_req: torch.Tensor,    # int32 [rows]
    page_size: int,
) -> torch.Tensor:
    rows, q_heads, head_dim = q.shape
    num_blocks, _, kv_heads, _ = key_cache.shape
    W = logical_indices.shape[1]
    group = q_heads // kv_heads
    page_table_width = block_table.shape[1]
    qf = q.float()
    kc = key_cache.float()
    vc = value_cache.float()
    out = torch.zeros(rows, q_heads, head_dim, device=q.device, dtype=torch.float32)
    scale = head_dim**-0.5
    for r in range(rows):
        req = int(token_to_req[r].item())
        idx = logical_indices[r].long()               # [W]
        lpage = torch.clamp(idx // page_size, min=0)
        poff = idx % page_size
        valid = (idx >= 0) & (lpage < page_table_width) & (req >= 0)
        phys = torch.where(
            valid, block_table[req, torch.clamp(lpage, max=page_table_width - 1)].long(),
            torch.full_like(idx, -1),
        )
        valid = valid & (phys >= 0) & (phys < num_blocks)
        safe_phys = torch.clamp(phys, min=0)
        # gather [W, kv_heads, head_dim]
        Ksel = kc[safe_phys, poff]                     # [W, kv_heads, head_dim]
        Vsel = vc[safe_phys, poff]
        for h in range(q_heads):
            kv = h // group
            k = Ksel[:, kv, :]                         # [W, head_dim]
            v = Vsel[:, kv, :]
            s = (qf[r, h] @ k.T) * scale               # [W]
            s = torch.where(valid, s, torch.full_like(s, float("-inf")))
            if not bool(valid.any()):
                continue
            p = torch.softmax(s, dim=0)
            out[r, h] = p @ v
    return out


# --------------------------------------------------------------------------- #
def metrics(name: str, a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.float().flatten()
    b = b.float().flatten()
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-4)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    d = {
        "cmp": name,
        "cosine": round(cos, 6),
        "max_abs": round(diff.max().item(), 5),
        "mean_abs": round(diff.mean().item(), 6),
        "rel_mean": round((diff / denom).mean().item(), 5),
        "ref_rms": round(b.pow(2).mean().sqrt().item(), 5),
    }
    return d


def quantize_kv(kv_cache_bf16: torch.Tensor, head_dim: int, scale_mode: str):
    """Split [.. , 2D] into K|V halves, quantize each to fp8 with its own scale."""
    k_half = kv_cache_bf16[..., :head_dim].float()
    v_half = kv_cache_bf16[..., head_dim:].float()

    def make_scale(t):
        if scale_mode == "unit":
            return 1.0
        if scale_mode == "absmax":
            return max(t.abs().amax().item() / E4M3_MAX, 1e-8)
        if scale_mode == "p999":
            return max(torch.quantile(t.abs().flatten().float(), 0.999).item() / E4M3_MAX, 1e-8)
        raise ValueError(scale_mode)

    ks = make_scale(k_half)
    vs = make_scale(v_half)
    kq = (k_half / ks).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    vq = (v_half / vs).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    # dequantized bf16 caches for the "quality" reference
    k_deq = (kq.to(torch.float32) * ks).to(torch.bfloat16)
    v_deq = (vq.to(torch.float32) * vs).to(torch.bfloat16)
    kv_fp8 = torch.empty_like(kv_cache_bf16, dtype=torch.float8_e4m3fn)
    kv_fp8[..., :head_dim] = kq
    kv_fp8[..., head_dim:] = vq
    kv_deq = torch.empty_like(kv_cache_bf16)
    kv_deq[..., :head_dim] = k_deq
    kv_deq[..., head_dim:] = v_deq
    return kv_fp8, kv_deq, ks, vs


def run_case(cfg: dict) -> dict:
    dev = "cuda"
    torch.manual_seed(cfg["seed"])
    q_heads = cfg["q_heads"]; kv_heads = cfg["kv_heads"]; head_dim = cfg["head_dim"]
    page = cfg["page_size"]; ctx = cfg["ctx"]; rows = cfg["rows"]; W = cfg["topk"]
    dist = cfg["dist"]; scale_mode = cfg["scale_mode"]

    num_blocks = (ctx + page - 1) // page
    # identity page mapping for a single request (exercises page_offset math + boundaries)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=dev).view(1, num_blocks)
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=dev)

    def gen(shape):
        if dist == "normal":
            return torch.randn(shape, device=dev, dtype=torch.bfloat16)
        if dist == "heavytail":
            base = torch.randn(shape, device=dev)
            spike = (torch.rand(shape, device=dev) < 0.02) * torch.randn(shape, device=dev) * 8.0
            return (base + spike).to(torch.bfloat16)
        raise ValueError(dist)

    # storage layout: (num_blocks, kv_heads, page, 2*head_dim), K|V packed in last dim
    kv_cache = gen((num_blocks, kv_heads, page, 2 * head_dim))
    q = gen((rows, q_heads, head_dim))

    # query positions: spread across context; each row's selection is causal (< pos)
    positions = torch.linspace(max(1, ctx // rows), ctx - 1, rows).long().clamp(1, ctx - 1)
    logical_indices = torch.full((rows, W), -1, dtype=torch.int32, device=dev)
    for r in range(rows):
        pos = int(positions[r].item())
        avail = pos  # tokens [0, pos)
        if avail <= W:
            sel = torch.arange(avail, device=dev)
        else:
            # keep boundary tokens + random middle (representative sparse selection)
            forced = torch.tensor([0, 1, page - 1, page, page + 1, pos - 1], device=dev)
            forced = forced[(forced >= 0) & (forced < avail)].unique()
            need = W - forced.numel()
            rand = torch.randperm(avail, device=dev)[:need]
            sel = torch.cat([forced, rand]).unique()[:W]
        logical_indices[r, : sel.numel()] = sel.to(torch.int32)

    def views(kv):
        kc, vc = kv.transpose(1, 2).split(head_dim, dim=-1)
        return kc, vc

    # ---- oracle (shipped BF16 kernel) ----
    kc_bf16, vc_bf16 = views(kv_cache)
    out_bf16 = qsa_sparse_paged_attention(
        q.contiguous(), kc_bf16, vc_bf16, logical_indices, block_table, token_to_req
    )

    # ---- fp32 reference on the bf16 cache ----
    ref_bf16 = ref_sparse_attention(
        q, kc_bf16, vc_bf16, logical_indices, block_table, token_to_req, page
    )

    # ---- fp8 quantize ----
    kv_fp8, kv_deq, ks, vs = quantize_kv(kv_cache, head_dim, scale_mode)
    kc_fp8, vc_fp8 = views(kv_fp8)
    kc_deq, vc_deq = views(kv_deq)

    out_fp8 = qsa_sparse_paged_attention_fp8(
        q.contiguous(), kc_fp8.contiguous(), vc_fp8.contiguous(), ks, vs,
        logical_indices, block_table, token_to_req,
    )
    ref_deq = ref_sparse_attention(
        q, kc_deq, vc_deq, logical_indices, block_table, token_to_req, page
    )

    torch.cuda.synchronize()
    res = {
        "cfg": {k: cfg[k] for k in ("ctx", "rows", "topk", "dist", "scale_mode", "seed")},
        "k_scale": round(ks, 8) if isinstance(ks, float) else ks,
        "v_scale": round(vs, 8) if isinstance(vs, float) else vs,
        "num_blocks": num_blocks,
        "nan_fp8": bool(torch.isnan(out_fp8).any().item()),
        "A_kernel_vs_ref_bf16": metrics("bf16_kernel~ref", out_bf16, ref_bf16),
        "B_fp8_kernel_vs_ref_deq": metrics("fp8_kernel~ref_deq", out_fp8, ref_deq),
        "C_quant_quality": metrics("ref_deq~ref_bf16", ref_deq, ref_bf16),
        "D_end2end_ab": metrics("fp8_kernel~bf16_kernel", out_fp8, out_bf16),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[1024, 8192, 32768])
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--topk", type=int, default=2051)
    ap.add_argument("--q-heads", type=int, default=24)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--dist", choices=["normal", "heavytail"], nargs="+", default=["normal", "heavytail"])
    ap.add_argument("--scale-mode", choices=["unit", "absmax", "p999"], nargs="+", default=["absmax"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    assert HAS_TRITON
    free0, total = torch.cuda.mem_get_info()
    print(f"# device {torch.cuda.get_device_name(0)} cap {torch.cuda.get_device_capability(0)} "
          f"free {free0/2**20:.0f}/{total/2**20:.0f} MiB")
    print(f"# q_heads={args.q_heads} kv_heads={args.kv_heads} head_dim={args.head_dim} "
          f"page={args.page_size} topk={args.topk} rows={args.rows}")

    all_res = []
    for ctx in args.ctx:
        for dist in args.dist:
            for sm in args.scale_mode:
                cfg = dict(
                    q_heads=args.q_heads, kv_heads=args.kv_heads, head_dim=args.head_dim,
                    page_size=args.page_size, ctx=ctx, rows=args.rows, topk=args.topk,
                    dist=dist, scale_mode=sm, seed=args.seed,
                )
                r = run_case(cfg)
                all_res.append(r)
                A = r["A_kernel_vs_ref_bf16"]; B = r["B_fp8_kernel_vs_ref_deq"]
                C = r["C_quant_quality"]; D = r["D_end2end_ab"]
                print(f"\n## ctx={ctx:>7} dist={dist:<9} scale={sm:<7} "
                      f"blocks={r['num_blocks']} kscale={r['k_scale']} vscale={r['v_scale']} "
                      f"nan={r['nan_fp8']}")
                print(f"   A kernel~ref_bf16 : cos={A['cosine']:.6f} max={A['max_abs']:.4f} "
                      f"mean={A['mean_abs']:.5f} rms={A['ref_rms']:.4f}")
                print(f"   B fp8ker~ref_deq  : cos={B['cosine']:.6f} max={B['max_abs']:.4f} "
                      f"mean={B['mean_abs']:.5f}")
                print(f"   C quant quality   : cos={C['cosine']:.6f} max={C['max_abs']:.4f} "
                      f"mean={C['mean_abs']:.5f}  <-- real fp8 cost")
                print(f"   D fp8~bf16 (A/B)  : cos={D['cosine']:.6f} max={D['max_abs']:.4f} "
                      f"mean={D['mean_abs']:.5f}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_res, f, indent=2)
        print(f"\n# wrote {args.json_out}")


if __name__ == "__main__":
    main()
