#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""NVFP4-QSA fused sparse read kernel (plan Part II, Slice 4).

Split-layout paged cache (§21.3):
  k_data [blocks, page, kv_heads, head_dim/2]  uint8  (2 FP4 E2M1 / byte)
  k_scale[blocks, page, kv_heads, head_dim/16] uint8  (fp8_e4m3 group scale)
  + per-tensor global scale (folded into the softmax/output scale).

The kernel loads packed FP4 + group scales for the selected top-k tokens only,
decodes E2M1 in-register (reusing vLLM's _e2m1_inline), applies the group scale,
runs QK/PV in bf16 with fp32 softmax. Validated vs the dense dequant oracle.
"""
from __future__ import annotations
import sys
import torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
from qsa_fp8_harness import ref_sparse_attention, qsa_sparse_paged_attention, metrics
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import ref_nvfp4_quant
from vllm.triton_utils import tl, triton

E4M3_MAX, E2M1_MAX, GROUP = 448.0, 6.0, 16
_KE2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


# ---- host E2M1 pack (matches device _e2m1_inline decode) ------------------ #
def pack_fp4(fp4_floats: torch.Tensor) -> torch.Tensor:
    """[..., D] float E2M1-grid values -> [..., D/2] uint8 (low=2j, high=2j+1)."""
    grid = torch.tensor(_KE2M1, device=fp4_floats.device)
    absv = fp4_floats.abs().unsqueeze(-1)
    mag = (absv - grid).abs().argmin(dim=-1).to(torch.uint8)      # 0..7
    sign = (fp4_floats < 0).to(torch.uint8)
    nib = (sign << 3) | mag                                        # [..., D]
    low = nib[..., 0::2]; high = nib[..., 1::2]
    return (low | (high << 4)).contiguous()


@triton.jit
def _e2m1(nibble):
    magnitude = nibble & 0x07
    sign = (nibble >> 3) & 1
    val = (0x3F000000 + (magnitude.to(tl.int32) << 22)).to(tl.float32, bitcast=True)
    val = tl.where(magnitude == 0, 0.0, val)
    val = tl.where(magnitude == 1, 0.5, val)
    return tl.where(sign == 1, -val, val)


@triton.jit
def _qsa_nvfp4_kernel(
    q_ptr, kd_ptr, ks_ptr, vd_ptr, vs_ptr, indices_ptr, block_table_ptr,
    token_to_req_ptr, output_ptr,
    inv_gk, inv_gv,
    stride_q_row, stride_q_head,
    stride_kd_block, stride_kd_token, stride_kd_head,
    stride_ks_block, stride_ks_token, stride_ks_head,
    stride_out_row, stride_out_head,
    num_cache_blocks, num_requests,
    TOPK: tl.constexpr, PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, PACKED: tl.constexpr,
    NGROUPS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_off = tl.arange(0, BLOCK_M)
    dim = tl.arange(0, HEAD_DIM)
    col = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr + row * stride_q_row + (first_head + head_off[:, None]) * stride_q_head + dim[None, :],
        mask=head_off[:, None] < GROUP_SIZE, other=0.0,
    )  # [BLOCK_M, HEAD_DIM] bf16
    softmax_scale_log2 = (HEAD_DIM ** -0.5) * 1.4426950408889634 * inv_gk

    max_v = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    norm = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    byte_off = dim // 2          # [HEAD_DIM] which packed byte holds element d
    is_high = (dim % 2) == 1
    grp = dim // 16              # [HEAD_DIM] which group scale applies (NVFP4 block=16)

    num_tiles = tl.cdiv(TOPK, BLOCK_N)
    for tile in range(0, num_tiles):
        cols = tile * BLOCK_N + col
        logical = tl.load(indices_ptr + row * TOPK + cols, mask=cols < TOPK, other=-1)
        safe_tok = tl.maximum(logical, 0)
        lpage = safe_tok // PAGE_SIZE
        poff = safe_tok % PAGE_SIZE
        valid = (request >= 0) & (request < num_requests) & (logical >= 0) & (lpage < PAGE_TABLE_WIDTH)
        phys = tl.load(block_table_ptr + safe_request * PAGE_TABLE_WIDTH
                       + tl.minimum(lpage, PAGE_TABLE_WIDTH - 1), mask=valid, other=-1)
        valid &= (phys >= 0) & (phys < num_cache_blocks)
        safe_page = tl.maximum(phys, 0).to(tl.int64)

        # --- unpack K for the tile: [BLOCK_N, HEAD_DIM] ---
        kbase = safe_page[:, None] * stride_kd_block + poff[:, None] * stride_kd_token + kv_head * stride_kd_head
        kraw = tl.load(kd_ptr + kbase + byte_off[None, :], mask=valid[:, None], other=0)  # [BLOCK_N, HEAD_DIM] u8
        knib = tl.where(is_high[None, :], (kraw >> 4) & 0x0F, kraw & 0x0F)
        ksb = safe_page[:, None] * stride_ks_block + poff[:, None] * stride_ks_token + kv_head * stride_ks_head
        ksc_raw = tl.load(ks_ptr + ksb + grp[None, :], mask=valid[:, None], other=0)      # [BLOCK_N, HEAD_DIM] fp8 byte
        ksc = tl.cast(ksc_raw, tl.float8e4nv, bitcast=True).to(tl.float32)
        keys = (_e2m1(knib) * ksc).to(tl.bfloat16)                                        # [BLOCK_N, HEAD_DIM]

        # --- unpack V for the tile: [BLOCK_N, HEAD_DIM] ---
        vbase = safe_page[:, None] * stride_kd_block + poff[:, None] * stride_kd_token + kv_head * stride_kd_head
        vraw = tl.load(vd_ptr + vbase + byte_off[None, :], mask=valid[:, None], other=0)
        vnib = tl.where(is_high[None, :], (vraw >> 4) & 0x0F, vraw & 0x0F)
        vsb = safe_page[:, None] * stride_ks_block + poff[:, None] * stride_ks_token + kv_head * stride_ks_head
        vsc_raw = tl.load(vs_ptr + vsb + grp[None, :], mask=valid[:, None], other=0)
        vsc = tl.cast(vsc_raw, tl.float8e4nv, bitcast=True).to(tl.float32)
        values = (_e2m1(vnib) * vsc).to(tl.bfloat16)                                      # [BLOCK_N, HEAD_DIM]

        scores = tl.dot(query, tl.trans(keys))                                            # [BLOCK_M, BLOCK_N]
        scores *= softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_v, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_v - next_max)
        p = tl.where(valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0)
        acc = tl.dot(p.to(tl.bfloat16), values, acc=acc * alpha[:, None])
        norm = norm * alpha + tl.sum(p, axis=1)
        max_v = next_max

    has = norm > 0
    out = tl.where(has[:, None], acc / tl.maximum(norm[:, None], 1e-20), 0.0) * inv_gv
    tl.store(output_ptr + row * stride_out_row + (first_head + head_off[:, None]) * stride_out_head + dim[None, :],
             out, mask=head_off[:, None] < GROUP_SIZE)


def qsa_sparse_paged_attention_nvfp4(q, kd, ks, vd, vs, gk, gv, idx, bt, ttr, out=None):
    rows, qh, hd = q.shape
    kvh = kd.shape[2]
    group = qh // kvh
    if out is None:
        out = torch.empty_like(q)
    grid = (rows, kvh)
    _qsa_nvfp4_kernel[grid](
        q, kd, ks, vd, vs, idx, bt, ttr, out,
        float(1.0 / gk), float(1.0 / gv),
        q.stride(0), q.stride(1),
        kd.stride(0), kd.stride(1), kd.stride(2),
        ks.stride(0), ks.stride(1), ks.stride(2),
        out.stride(0), out.stride(1),
        kd.shape[0], bt.shape[0],
        TOPK=idx.shape[1], PAGE_SIZE=kd.shape[1], PAGE_TABLE_WIDTH=bt.shape[1],
        GROUP_SIZE=group, HEAD_DIM=hd, PACKED=hd // 2, NGROUPS=hd // GROUP,
        BLOCK_M=triton.next_power_of_2(group), BLOCK_N=32,
        num_warps=4, num_stages=2,
    )
    return out


# --------------------------------------------------------------------------- #
def quant_half_to_split(x_bf16, page, kvh, hd, dev):
    """[blocks, kvh, page, hd] bf16 -> packed split buffers + dequant-bf16 view."""
    blocks = x_bf16.shape[0]
    x = x_bf16.reshape(-1, hd).float()
    amax = x.abs().amax().clamp_min(1e-8)
    g = (E4M3_MAX * E2M1_MAX) / amax
    fp4, bscale = ref_nvfp4_quant(x, g.reshape(1), GROUP)          # fp4: E2M1-grid floats, bscale: fp8->f32
    # split buffers indexed [blocks, page, kvh, ...] (matches the bf16 QSA layout)
    data = pack_fp4(fp4).reshape(blocks, kvh, page, hd // 2).transpose(1, 2).contiguous()
    scale = (bscale.reshape(blocks, kvh, page, hd // GROUP).transpose(1, 2).contiguous()
             .to(torch.float8_e4m3fn).view(torch.uint8))
    # dequant reference cache stays in the input [blocks, kvh, page, hd] layout
    deq = (fp4.reshape(-1, hd // GROUP, GROUP) * (bscale.reshape(-1, hd // GROUP, 1) / g)).reshape(x_bf16.shape).to(torch.bfloat16)
    return data, scale, float(g.item()), deq

