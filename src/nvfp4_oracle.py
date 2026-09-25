#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""INDEPENDENT NVFP4 codec (written from the OCP/NVIDIA spec, NOT reusing vLLM).

Used as a third, independent oracle to cross-validate the NVFP4 format against
vLLM's ref_nvfp4_quant and the CUDA scaled_fp4_quant op.

Format:
  FP4 = E2M1, magnitudes grid = [0, .5, 1, 1.5, 2, 3, 4, 6], nibble = (sign<<3)|mag_idx.
  group_size = 16; per-group block scale in fp8_e4m3; per-tensor global scale (fp32).
  global_scale G  = (448 * 6) / tensor_amax
  block_scale  s  = fp8_e4m3( clamp(G * group_amax / 6, [-448, 448]) )
  encode:  q = round_e2m1( x * G / s ),  packed low=elem 2j, high=elem 2j+1
  decode:  x_hat = grid[mag]*sign * s / G
"""
from __future__ import annotations
import torch

E2M1_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_MAX = 6.0
E4M3_MAX = 448.0
GROUP = 16


def global_scale_for(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().amax().clamp_min(1e-8).float()
    return (E4M3_MAX * E2M1_MAX) / amax          # per-tensor fp32


def _round_e2m1_magnitude(absx: torch.Tensor) -> torch.Tensor:
    """Round |x| (already scaled into [0,6]) to E2M1 magnitude index 0..7 using
    round-half-to-even (OCP FP4 spec): even magnitudes {0,1,2,4} get the closed
    midpoint intervals, odd magnitudes {0.5,1.5,3,6} the open ones. This is the
    standard E2M1 rounding; it coincides with a correct vLLM cast_to_fp4."""
    def T(k): return torch.tensor(k, device=absx.device)
    m = torch.zeros_like(absx, dtype=torch.int64)                    # [0,0.25] -> 0.0
    m = torch.where((absx > 0.25) & (absx < 0.75), T(1), m)          # 0.5
    m = torch.where((absx >= 0.75) & (absx <= 1.25), T(2), m)        # 1.0
    m = torch.where((absx > 1.25) & (absx < 1.75), T(3), m)          # 1.5
    m = torch.where((absx >= 1.75) & (absx <= 2.5), T(4), m)         # 2.0
    m = torch.where((absx > 2.5) & (absx < 3.5), T(5), m)            # 3.0
    m = torch.where((absx >= 3.5) & (absx <= 5.0), T(6), m)          # 4.0
    m = torch.where(absx > 5.0, T(7), m)                            # 6.0
    return m


def encode(x2d: torch.Tensor, global_scale: torch.Tensor, block_size: int = GROUP):
    """x2d: [M, N] (N % block_size == 0). Returns (nibbles[M,N] uint8,
    packed[M,N//2] uint8, block_scale_fp8[M,N//block_size] uint8 view,
    saturation stats dict)."""
    m, n = x2d.shape
    xg = x2d.float().reshape(m, n // block_size, block_size)
    g = global_scale.reshape(1).float()
    group_amax = xg.abs().amax(dim=-1, keepdim=True)                       # [m, nb, 1]
    block_scale = torch.clamp(g * group_amax / E2M1_MAX, -E4M3_MAX, E4M3_MAX)
    bs_fp8 = block_scale.to(torch.float8_e4m3fn)
    bs_f32 = bs_fp8.float()
    out_scale = g / bs_f32.clamp_min(1e-30)                               # x * out_scale -> grid domain
    y = xg * out_scale                                                     # [m,nb,bs]
    sat_hi = (y.abs() > E2M1_MAX).float().mean().item()                   # clamped to 6 rate
    y = torch.clamp(y, -E2M1_MAX, E2M1_MAX)
    mag = _round_e2m1_magnitude(y.abs())
    sign = (y < 0).to(torch.int64)
    nib = ((sign << 3) | mag).to(torch.uint8).reshape(m, n)
    bs_sat = ((block_scale.abs() >= E4M3_MAX - 1e-3).float().mean().item())  # block-scale saturation
    low = nib[:, 0::2]; high = nib[:, 1::2]
    packed = (low | (high << 4)).contiguous()
    scale_u8 = bs_fp8.reshape(m, n // block_size).view(torch.uint8)
    return nib, packed, scale_u8, {"clip6_rate": sat_hi, "blockscale_sat_rate": bs_sat}


def decode(packed: torch.Tensor, scale_u8: torch.Tensor, global_scale: torch.Tensor,
           n: int, block_size: int = GROUP) -> torch.Tensor:
    """Inverse of encode -> float32 [M, N]."""
    grid = torch.tensor(E2M1_GRID, device=packed.device)
    m = packed.shape[0]
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nib = torch.stack([low, high], dim=-1).reshape(m, n).long()           # interleave back
    sign = torch.where((nib >> 3) & 1 == 1, -1.0, 1.0)
    mag = grid[nib & 0x07]
    fp4 = (mag * sign).reshape(m, n // block_size, block_size)
    bs = scale_u8.view(torch.float8_e4m3fn).float().reshape(m, n // block_size, 1)
    g = global_scale.reshape(1).float()
    return (fp4 * (bs / g)).reshape(m, n)


def quant_dequant(x2d: torch.Tensor, block_size: int = GROUP):
    g = global_scale_for(x2d)
    nib, packed, scale_u8, sat = encode(x2d, g, block_size)
    xhat = decode(packed, scale_u8, g, x2d.shape[1], block_size)
    return xhat, packed, scale_u8, float(g.item()), nib, sat
