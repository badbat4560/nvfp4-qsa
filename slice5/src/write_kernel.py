#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Slice 5: standalone NVFP4 quantize-pack WRITE kernel (Triton).

Input : BF16 K or V after projection [num_tokens, kv_heads, head_dim].
Output: split-layout paged cache
        data [blocks, page, kv_heads, head_dim/2]  uint8   (2 FP4 E2M1 / byte)
        scale[blocks, page, kv_heads, head_dim/16] uint8   (fp8_e4m3 group scale)
Semantics identical to the ACCEPTED read kernel: E2M1, group 16, low nibble = elem 2j,
high = 2j+1, per-tensor global scale, block scale fp8_e4m3(clamp(g*group_amax/6,±448)),
stored value q = round_e2m1(x * g / block_scale). Invalid slot (<0) or out-of-range
block => NO write. Not fused with projection/reshape_and_cache (correctness first).
"""
from __future__ import annotations
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _e2m1_nib(y):
    """Encode a clamped [-6,6] value to an E2M1 nibble (sign<<3 | magnitude),
    round-half-to-even (matches cast_to_fp4 and the read-kernel _e2m1 decode)."""
    ay = tl.abs(y)
    m = tl.where(ay > 5.0, 7, 0)
    m = tl.where((ay >= 3.5) & (ay <= 5.0), 6, m)
    m = tl.where((ay > 2.5) & (ay < 3.5), 5, m)
    m = tl.where((ay >= 1.75) & (ay <= 2.5), 4, m)
    m = tl.where((ay > 1.25) & (ay < 1.75), 3, m)
    m = tl.where((ay >= 0.75) & (ay <= 1.25), 2, m)
    m = tl.where((ay > 0.25) & (ay < 0.75), 1, m)
    s = tl.where(y < 0.0, 1, 0)
    return (s << 3) | m.to(tl.int32)


@triton.jit
def _nvfp4_write_kernel(
    src_ptr, data_ptr, scale_ptr, slot_ptr, global_scale,
    stride_src_tok, stride_src_head,
    stride_data_block, stride_data_tok, stride_data_head,
    stride_scale_block, stride_scale_tok, stride_scale_head,
    num_tokens, num_blocks,
    PAGE: tl.constexpr, HEAD_DIM: tl.constexpr, NGROUPS: tl.constexpr, HALF: tl.constexpr,
):
    tok = tl.program_id(0)
    kvh = tl.program_id(1)
    if tok >= num_tokens:
        return
    slot = tl.load(slot_ptr + tok)
    valid = slot >= 0
    block = tl.where(valid, slot // PAGE, 0)
    off = tl.where(valid, slot % PAGE, 0)
    valid = valid & (block < num_blocks)
    if not valid:
        return                                            # invalid/padded slot: no write

    grp = tl.arange(0, NGROUPS)
    jj = tl.arange(0, 8)
    base = tok * stride_src_tok + kvh * stride_src_head + grp[:, None] * 16
    x_lo = tl.load(src_ptr + base + (jj[None, :] * 2)).to(tl.float32)        # even elems [NG,8]
    x_hi = tl.load(src_ptr + base + (jj[None, :] * 2 + 1)).to(tl.float32)    # odd elems  [NG,8]
    amax = tl.maximum(tl.max(tl.abs(x_lo), axis=1), tl.max(tl.abs(x_hi), axis=1))  # [NG]
    bscale = tl.minimum(tl.maximum(global_scale * amax / 6.0, -448.0), 448.0)
    bscale_fp8 = bscale.to(tl.float8e4nv)                 # round-to-fp8
    bscale_f = bscale_fp8.to(tl.float32)
    out_scale = global_scale / tl.where(bscale_f != 0.0, bscale_f, 1.0e30)   # [NG]

    y_lo = tl.minimum(tl.maximum(x_lo * out_scale[:, None], -6.0), 6.0)
    y_hi = tl.minimum(tl.maximum(x_hi * out_scale[:, None], -6.0), 6.0)
    nib_lo = _e2m1_nib(y_lo)                              # [NG,8]  (elements 2j)
    nib_hi = _e2m1_nib(y_hi)                              # [NG,8]  (elements 2j+1)
    packed = (nib_lo | (nib_hi << 4)).to(tl.uint8)        # [NG,8]

    data_base = block * stride_data_block + off * stride_data_tok + kvh * stride_data_head
    tl.store(data_ptr + data_base + grp[:, None] * 8 + jj[None, :], packed)
    scale_base = block * stride_scale_block + off * stride_scale_tok + kvh * stride_scale_head
    tl.store(scale_ptr + scale_base + grp, tl.cast(bscale_fp8, tl.uint8, bitcast=True))


class DuplicateSlotError(ValueError):
    pass


def nvfp4_write(src, data, scale, slot_mapping, global_scale, check_unique=True):
    """src: [num_tokens, kv_heads, head_dim] bf16 -> data/scale paged split buffers.

    CONTRACT: valid slots (>=0) in slot_mapping MUST be unique within one call
    (parallel writes to the same physical slot are a race). check_unique=True
    validates this on the host and raises DuplicateSlotError before launch."""
    assert src.dtype == torch.bfloat16 and src.ndim == 3
    nt, kvh, hd = src.shape
    assert data.dtype == torch.uint8 and scale.dtype == torch.uint8
    assert slot_mapping.dtype == torch.int64
    if check_unique:
        valid = slot_mapping[slot_mapping >= 0]
        if valid.numel() != torch.unique(valid).numel():
            raise DuplicateSlotError(
                f"slot_mapping has duplicate valid slots ({valid.numel() - torch.unique(valid).numel()} dups); "
                "concurrent writes to one physical slot are a forbidden race")
    _nvfp4_write_kernel[(nt, kvh)](
        src, data, scale, slot_mapping, float(global_scale),
        src.stride(0), src.stride(1),
        data.stride(0), data.stride(1), data.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        nt, data.shape[0],
        PAGE=data.shape[1], HEAD_DIM=hd, NGROUPS=hd // 16, HALF=hd // 2,
        num_warps=4,
    )
