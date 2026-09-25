#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Full metric set for every comparison (plan §4), with NaN/Inf split by
valid/masked and an explicit mean-abs-error."""
from __future__ import annotations
import torch


def _count(t):
    return int(torch.isnan(t).sum()), int(torch.isinf(t).sum())


def full_metrics(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
    """a = candidate, b = reference. mask (same shape as a, or broadcastable):
    True = valid position, False = masked. NaN/Inf are reported total/valid/masked;
    numeric aggregates use only finite valid pairs."""
    a = a.detach().float(); b = b.detach().float()
    nan_total, inf_total = _count(a)
    if mask is None:
        m = torch.ones_like(a, dtype=torch.bool)
    else:
        m = mask.expand_as(a) if mask.shape != a.shape else mask
    nan_valid, inf_valid = _count(a[m])
    nan_masked, inf_masked = _count(a[~m])
    av, bv = a[m].flatten(), b[m].flatten()
    fin = torch.isfinite(av) & torch.isfinite(bv)
    af, bf = av[fin], bv[fin]
    diff = af - bf
    denom = bf.norm().clamp_min(1e-30)
    cos = (torch.nn.functional.cosine_similarity(af, bf, dim=0).item()
           if af.numel() else float("nan"))
    return {
        "n_valid_finite": int(af.numel()),
        "cosine": round(cos, 8),
        "cosine_defect": round(1.0 - cos, 10) if af.numel() else float("nan"),
        "mse": round(diff.pow(2).mean().item(), 12) if af.numel() else float("nan"),
        "mean_abs_error": round(diff.abs().mean().item(), 10) if af.numel() else float("nan"),
        "rel_l2": round((diff.norm() / denom).item(), 8) if af.numel() else float("nan"),
        "max_abs": round(diff.abs().max().item(), 8) if af.numel() else float("nan"),
        "nan_total": nan_total, "nan_valid": nan_valid, "nan_masked": nan_masked,
        "inf_total": inf_total, "inf_valid": inf_valid, "inf_masked": inf_masked,
        "valid_clean": bool(nan_valid == 0 and inf_valid == 0),
    }
