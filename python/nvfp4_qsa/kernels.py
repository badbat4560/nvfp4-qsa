# SPDX-License-Identifier: Apache-2.0
"""QSA-owned NVFP4 paged-cache writer and sparse reader.

Physical layout per K or V region:
  data  [..., head_dim / 2] uint8 (low nibble is element 2j)
  scale [..., head_dim / 16] uint8 containing float8_e4m3fn bytes

The global K/V scales are fixed for the lifetime of a cache. Dynamic group
scales are written for every token and group of 16 values.
"""

from __future__ import annotations

import torch

import triton
import triton.language as tl


@triton.jit
def _e2m1_decode(nibble):
    magnitude = nibble & 0x07
    sign = (nibble >> 3) & 1
    value = (0x3F000000 + (magnitude.to(tl.int32) << 22)).to(
        tl.float32, bitcast=True
    )
    value = tl.where(magnitude == 0, 0.0, value)
    value = tl.where(magnitude == 1, 0.5, value)
    return tl.where(sign == 1, -value, value)


@triton.jit
def _e2m1_encode(value):
    absolute = tl.abs(value)
    magnitude = tl.where(absolute > 5.0, 7, 0)
    magnitude = tl.where((absolute >= 3.5) & (absolute <= 5.0), 6, magnitude)
    magnitude = tl.where((absolute > 2.5) & (absolute < 3.5), 5, magnitude)
    magnitude = tl.where((absolute >= 1.75) & (absolute <= 2.5), 4, magnitude)
    magnitude = tl.where((absolute > 1.25) & (absolute < 1.75), 3, magnitude)
    magnitude = tl.where((absolute >= 0.75) & (absolute <= 1.25), 2, magnitude)
    magnitude = tl.where((absolute > 0.25) & (absolute < 0.75), 1, magnitude)
    sign = tl.where(value < 0.0, 1, 0)
    return (sign << 3) | magnitude.to(tl.int32)


@triton.jit
def _nvfp4_write_kernel(
    source_ptr,
    data_ptr,
    scale_ptr,
    slot_ptr,
    global_scale,
    stride_source_token,
    stride_source_head,
    stride_data_block,
    stride_data_token,
    stride_data_head,
    stride_scale_block,
    stride_scale_token,
    stride_scale_head,
    num_tokens,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    token = tl.program_id(0)
    kv_head = tl.program_id(1)
    if token >= num_tokens:
        return

    slot = tl.load(slot_ptr + token)
    valid = slot >= 0
    block = tl.where(valid, slot // PAGE_SIZE, 0)
    token_offset = tl.where(valid, slot % PAGE_SIZE, 0)
    valid = valid & (block < num_blocks)
    if not valid:
        return

    group = tl.arange(0, NUM_GROUPS)
    pair = tl.arange(0, 8)
    source_base = (
        token * stride_source_token
        + kv_head * stride_source_head
        + group[:, None] * 16
    )
    even = tl.load(source_ptr + source_base + pair[None, :] * 2).to(tl.float32)
    odd = tl.load(source_ptr + source_base + pair[None, :] * 2 + 1).to(tl.float32)
    group_amax = tl.maximum(
        tl.max(tl.abs(even), axis=1), tl.max(tl.abs(odd), axis=1)
    )
    block_scale = tl.minimum(global_scale * group_amax / 6.0, 448.0)
    block_scale_fp8 = block_scale.to(tl.float8e4nv)
    block_scale_f32 = block_scale_fp8.to(tl.float32)
    output_scale = global_scale / tl.where(
        block_scale_f32 != 0.0, block_scale_f32, 1.0e30
    )
    even = tl.minimum(tl.maximum(even * output_scale[:, None], -6.0), 6.0)
    odd = tl.minimum(tl.maximum(odd * output_scale[:, None], -6.0), 6.0)
    packed = (_e2m1_encode(even) | (_e2m1_encode(odd) << 4)).to(tl.uint8)

    data_base = (
        block * stride_data_block
        + token_offset * stride_data_token
        + kv_head * stride_data_head
    )
    tl.store(data_ptr + data_base + group[:, None] * 8 + pair[None, :], packed)
    scale_base = (
        block * stride_scale_block
        + token_offset * stride_scale_token
        + kv_head * stride_scale_head
    )
    tl.store(
        scale_ptr + scale_base + group,
        tl.cast(block_scale_fp8, tl.uint8, bitcast=True),
    )


def _write_one(source, data, scale, slot_mapping, global_scale):
    num_tokens, num_kv_heads, head_dim = source.shape
    if source.dtype != torch.bfloat16:
        raise TypeError("NVFP4 QSA writer requires BF16 K/V")
    if data.dtype != torch.uint8 or scale.dtype != torch.uint8:
        raise TypeError("NVFP4 QSA data and scale caches must use uint8 storage")
    if slot_mapping.dtype != torch.int64:
        raise TypeError("NVFP4 QSA slot_mapping must be int64")
    if head_dim % 16 or data.shape[-1] != head_dim // 2:
        raise ValueError("invalid NVFP4 QSA packed-data shape")
    if scale.shape[-1] != head_dim // 16:
        raise ValueError("invalid NVFP4 QSA group-scale shape")
    _nvfp4_write_kernel[(num_tokens, num_kv_heads)](
        source,
        data,
        scale,
        slot_mapping,
        float(global_scale),
        source.stride(0),
        source.stride(1),
        data.stride(0),
        data.stride(1),
        data.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        num_tokens,
        data.shape[0],
        PAGE_SIZE=data.shape[1],
        NUM_GROUPS=head_dim // 16,
        num_warps=4,
    )


def qsa_write_cache_nvfp4(
    key,
    value,
    key_data,
    key_scale,
    value_data,
    value_scale,
    slot_mapping,
    key_global_scale,
    value_global_scale,
):
    """Quantize and write K/V without synchronizing the serving hot path."""
    _write_one(key, key_data, key_scale, slot_mapping, key_global_scale)
    _write_one(value, value_data, value_scale, slot_mapping, value_global_scale)


@triton.jit
def _qsa_nvfp4_kernel(
    query_ptr,
    key_data_ptr,
    key_scale_ptr,
    value_data_ptr,
    value_scale_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_request_ptr,
    output_ptr,
    inverse_key_global,
    inverse_value_global,
    stride_query_row,
    stride_query_head,
    stride_key_data_block,
    stride_key_data_token,
    stride_key_data_head,
    stride_key_scale_block,
    stride_key_scale_token,
    stride_key_scale_head,
    stride_value_data_block,
    stride_value_data_token,
    stride_value_data_head,
    stride_value_scale_block,
    stride_value_scale_token,
    stride_value_scale_head,
    stride_output_row,
    stride_output_head,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    request = tl.load(token_to_request_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offset = tl.arange(0, BLOCK_M)
    dimension = tl.arange(0, HEAD_DIM)
    column = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        query_ptr
        + row * stride_query_row
        + (first_head + head_offset[:, None]) * stride_query_head
        + dimension[None, :],
        mask=head_offset[:, None] < GROUP_SIZE,
        other=0.0,
    )
    softmax_scale_log2 = (
        (HEAD_DIM**-0.5) * 1.4426950408889634 * inverse_key_global
    )
    maximum = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    byte_offset = dimension // 2
    high_nibble = (dimension % 2) == 1
    scale_group = dimension // 16

    for tile in range(0, tl.cdiv(TOPK, BLOCK_N)):
        columns = tile * BLOCK_N + column
        logical = tl.load(
            indices_ptr + row * TOPK + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * PAGE_TABLE_WIDTH
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)

        key_base = (
            safe_page[:, None] * stride_key_data_block
            + page_offset[:, None] * stride_key_data_token
            + kv_head * stride_key_data_head
        )
        key_raw = tl.load(
            key_data_ptr + key_base + byte_offset[None, :],
            mask=valid[:, None],
            other=0,
        )
        key_nibble = tl.where(
            high_nibble[None, :], (key_raw >> 4) & 0x0F, key_raw & 0x0F
        )
        key_scale_base = (
            safe_page[:, None] * stride_key_scale_block
            + page_offset[:, None] * stride_key_scale_token
            + kv_head * stride_key_scale_head
        )
        key_scale_raw = tl.load(
            key_scale_ptr + key_scale_base + scale_group[None, :],
            mask=valid[:, None],
            other=0,
        )
        key_group_scale = tl.cast(
            key_scale_raw, tl.float8e4nv, bitcast=True
        ).to(tl.float32)
        keys = (_e2m1_decode(key_nibble) * key_group_scale).to(tl.bfloat16)

        value_base = (
            safe_page[:, None] * stride_value_data_block
            + page_offset[:, None] * stride_value_data_token
            + kv_head * stride_value_data_head
        )
        value_raw = tl.load(
            value_data_ptr + value_base + byte_offset[None, :],
            mask=valid[:, None],
            other=0,
        )
        value_nibble = tl.where(
            high_nibble[None, :], (value_raw >> 4) & 0x0F, value_raw & 0x0F
        )
        value_scale_base = (
            safe_page[:, None] * stride_value_scale_block
            + page_offset[:, None] * stride_value_scale_token
            + kv_head * stride_value_scale_head
        )
        value_scale_raw = tl.load(
            value_scale_ptr + value_scale_base + scale_group[None, :],
            mask=valid[:, None],
            other=0,
        )
        value_group_scale = tl.cast(
            value_scale_raw, tl.float8e4nv, bitcast=True
        ).to(tl.float32)
        values = (_e2m1_decode(value_nibble) * value_group_scale).to(tl.bfloat16)

        scores = tl.dot(query, tl.trans(keys)) * softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        alpha = tl.math.exp2(maximum - next_maximum)
        probability = tl.where(
            valid[None, :],
            tl.math.exp2(scores - next_maximum[:, None]),
            0.0,
        )
        accumulator = tl.dot(
            probability.to(tl.bfloat16),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probability, axis=1)
        maximum = next_maximum

    output = tl.where(
        (normalizer > 0)[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    ) * inverse_value_global
    tl.store(
        output_ptr
        + row * stride_output_row
        + (first_head + head_offset[:, None]) * stride_output_head
        + dimension[None, :],
        output,
        mask=head_offset[:, None] < GROUP_SIZE,
    )


def qsa_sparse_paged_attention_nvfp4(
    query,
    key_data,
    key_scale,
    value_data,
    value_scale,
    key_global_scale,
    value_global_scale,
    logical_indices,
    block_table,
    token_to_request,
    output,
):
    rows, query_heads, head_dim = query.shape
    kv_heads = key_data.shape[2]
    group_size = query_heads // kv_heads
    _qsa_nvfp4_kernel[(rows, kv_heads)](
        query,
        key_data,
        key_scale,
        value_data,
        value_scale,
        logical_indices,
        block_table,
        token_to_request,
        output,
        float(1.0 / key_global_scale),
        float(1.0 / value_global_scale),
        query.stride(0),
        query.stride(1),
        key_data.stride(0),
        key_data.stride(1),
        key_data.stride(2),
        key_scale.stride(0),
        key_scale.stride(1),
        key_scale.stride(2),
        value_data.stride(0),
        value_data.stride(1),
        value_data.stride(2),
        value_scale.stride(0),
        value_scale.stride(1),
        value_scale.stride(2),
        output.stride(0),
        output.stride(1),
        key_data.shape[0],
        block_table.shape[0],
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=key_data.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        BLOCK_M=triton.next_power_of_2(group_size),
        BLOCK_N=32,
        num_warps=4,
        num_stages=2,
    )


__all__ = ["qsa_sparse_paged_attention_nvfp4", "qsa_write_cache_nvfp4"]
