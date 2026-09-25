#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Slice 5 write-kernel performance: latency/token + GB/s, cold vs warm, batch
tokens 1/4/16/64/256/1024, vs FP8 reshape_and_cache_flash. Correctness first —
this is raw timing only, no fusion. Minimal tensors; check free VRAM."""
from __future__ import annotations
import argparse, json, os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../src")))
from write_kernel import nvfp4_write
from nvfp4_oracle import global_scale_for
from vllm import _custom_ops as ops

DEV = "cuda"; KVH, HD, PAGE = 2, 256, 16


def timeit(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts) // 2]      # median ms


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=100); a = ap.parse_args()
    rows = []
    for nt in [1, 4, 16, 64, 256, 1024]:
        blocks = (nt + PAGE - 1) // PAGE + 1
        src = torch.randn(nt, KVH, HD, device=DEV, dtype=torch.bfloat16)
        val = torch.randn(nt, KVH, HD, device=DEV, dtype=torch.bfloat16)
        slot = torch.arange(nt, dtype=torch.int64, device=DEV)
        g = global_scale_for(src.reshape(-1, HD))
        kd = torch.zeros(blocks, PAGE, KVH, HD // 2, device=DEV, dtype=torch.uint8)
        ks = torch.zeros(blocks, PAGE, KVH, HD // 16, device=DEV, dtype=torch.uint8)
        # cold = first-ever call (compile + launch); warm = median steady-state
        torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); nvfp4_write(src, kd, ks, slot, g); e.record(); torch.cuda.synchronize()
        cold = s.elapsed_time(e)
        warm = timeit(lambda: nvfp4_write(src, kd, ks, slot, g, check_unique=False), 20, a.iters)
        # logical bytes written for this K launch (data + scale), both kv heads
        wbytes = nt * (HD // 2 + HD // 16) * KVH
        gbps = wbytes / (warm * 1e-3) / 1e9
        # FP8 reshape_and_cache_flash reference (does K AND V in one op)
        kc8 = torch.zeros(blocks, PAGE, KVH, HD, device=DEV, dtype=torch.uint8)
        vc8 = torch.zeros(blocks, PAGE, KVH, HD, device=DEV, dtype=torch.uint8)
        one = torch.ones((), dtype=torch.float32, device=DEV)
        gscale = torch.tensor(float(g), device=DEV)
        def fp8_write():
            ops.reshape_and_cache_flash(src, val, kc8, vc8, slot, "fp8", gscale, gscale)
        fp8_warm = timeit(fp8_write, 20, a.iters)
        rows.append({
            "tokens": nt, "nvfp4_cold_ms": round(cold, 4), "nvfp4_warm_ms": round(warm, 5),
            "nvfp4_us_per_token": round(warm * 1000 / nt, 4), "nvfp4_write_GBps": round(gbps, 2),
            "fp8_reshape_cache_warm_ms": round(fp8_warm, 5),
            "note": "nvfp4_write does ONE of K/V per launch (x2 for both); reshape_and_cache_flash does K+V in one op",
        })
    R = {"env": {"device": torch.cuda.get_device_name(0)}, "rows": rows}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=2)
    print(f"{'tok':>5} {'nvfp4_cold':>10} {'nvfp4_warm':>10} {'us/tok':>8} {'GB/s':>7} {'fp8_warm':>9}")
    for r in rows:
        print(f"{r['tokens']:>5} {r['nvfp4_cold_ms']:>10.3f} {r['nvfp4_warm_ms']:>10.5f} "
              f"{r['nvfp4_us_per_token']:>8.3f} {r['nvfp4_write_GBps']:>7.1f} {r['fp8_reshape_cache_warm_ms']:>9.5f}")
    print("WRITE_BENCH_DONE")


if __name__ == "__main__":
    main()
