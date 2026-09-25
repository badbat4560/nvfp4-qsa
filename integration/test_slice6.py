#!/usr/bin/env python3
"""Focused integration check for the installed QSA NVFP4 server module."""

from __future__ import annotations

import json
import sys

import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nvfp4_oracle import decode as independent_decode
from nvfp4_oracle import encode as independent_encode
from qsa_fp8_harness import qsa_sparse_paged_attention

from vllm.models.qwen3_8_flash_next.nvidia.ops.nvfp4_qsa import (
    qsa_sparse_paged_attention_nvfp4,
    qsa_write_cache_nvfp4,
)


def metrics(actual, expected):
    actual = actual.float().flatten()
    expected = expected.float().flatten()
    return {
        "cosine": float(torch.nn.functional.cosine_similarity(actual, expected, dim=0)),
        "max_abs": float((actual - expected).abs().max()),
        "nan": int(torch.isnan(actual).sum()),
        "inf": int(torch.isinf(actual).sum()),
    }


def main():
    torch.manual_seed(20260828)
    device = "cuda"
    page, context, kv_heads, query_heads, head_dim = 16, 128, 2, 24, 256
    blocks = context // page
    packed_dim, scale_dim = head_dim // 2, head_dim // 16
    storage_dim = packed_dim + scale_dim
    # vLLM's current FlashAttention cache is exposed to the layer as HND.
    cache = torch.zeros(
        blocks,
        kv_heads,
        page,
        2 * storage_dim,
        dtype=torch.uint8,
        device=device,
    )
    packed_cache = cache.transpose(1, 2)
    key_region, value_region = packed_cache.split(storage_dim, dim=-1)
    key_data, key_scale = key_region.split((packed_dim, scale_dim), dim=-1)
    value_data, value_scale = value_region.split((packed_dim, scale_dim), dim=-1)
    key = torch.randn(context, kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)
    slots = torch.arange(context, dtype=torch.int64, device=device)
    qsa_write_cache_nvfp4(
        key,
        value,
        key_data,
        key_scale,
        value_data,
        value_scale,
        slots,
        1.0,
        1.0,
    )
    torch.cuda.synchronize()

    key_flat = key.reshape(-1, head_dim)
    value_flat = value.reshape(-1, head_dim)
    _, expected_key_data, expected_key_scale, _ = independent_encode(
        key_flat, torch.tensor(1.0, device=device)
    )
    _, expected_value_data, expected_value_scale, _ = independent_encode(
        value_flat, torch.tensor(1.0, device=device)
    )
    actual_key_data = key_data.reshape(-1, packed_dim)
    actual_key_scale = key_scale.reshape(-1, scale_dim)
    actual_value_data = value_data.reshape(-1, packed_dim)
    actual_value_scale = value_scale.reshape(-1, scale_dim)
    byte_parity = {
        "key_data": float((actual_key_data == expected_key_data).float().mean()),
        "key_scale": float((actual_key_scale == expected_key_scale).float().mean()),
        "value_data": float((actual_value_data == expected_value_data).float().mean()),
        "value_scale": float((actual_value_scale == expected_value_scale).float().mean()),
    }

    key_dequant = independent_decode(
        actual_key_data, actual_key_scale, torch.tensor(1.0, device=device), head_dim
    ).reshape(blocks, page, kv_heads, head_dim).to(torch.bfloat16)
    value_dequant = independent_decode(
        actual_value_data,
        actual_value_scale,
        torch.tensor(1.0, device=device),
        head_dim,
    ).reshape(blocks, page, kv_heads, head_dim).to(torch.bfloat16)
    rows, topk = 2, context
    query = torch.randn(rows, query_heads, head_dim, dtype=torch.bfloat16, device=device)
    indices = torch.arange(context, dtype=torch.int32, device=device).repeat(rows, 1)
    block_table = torch.arange(blocks, dtype=torch.int32, device=device).view(1, blocks)
    token_to_request = torch.zeros(rows, dtype=torch.int32, device=device)
    fused = torch.empty_like(query)
    qsa_sparse_paged_attention_nvfp4(
        query,
        key_data,
        key_scale,
        value_data,
        value_scale,
        1.0,
        1.0,
        indices,
        block_table,
        token_to_request,
        fused,
    )
    dense = qsa_sparse_paged_attention(
        query,
        key_dequant,
        value_dequant,
        indices,
        block_table,
        token_to_request,
    )
    torch.cuda.synchronize()
    result = {
        "byte_parity": byte_parity,
        "fused_vs_dense": metrics(fused, dense),
        "cache_shape": list(cache.shape),
        "cache_bytes": cache.numel(),
        "static_global_scale": 1.0,
    }
    result["PASS"] = (
        min(byte_parity.values()) == 1.0
        and result["fused_vs_dense"]["cosine"] > 0.9999
        and result["fused_vs_dense"]["nan"] == 0
        and result["fused_vs_dense"]["inf"] == 0
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["PASS"] else 1)


if __name__ == "__main__":
    main()
