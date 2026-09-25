#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Boundary matrix for the fused NVFP4 QSA READ kernel. Each cell runs fused vs
dense-dequant on the SAME quantized data (isolates kernel correctness). q_len =
query rows; batch = number of requests (token_to_req) — kept DISTINCT."""
from __future__ import annotations
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_validation import build_case, views, fused_and_dense
from metrics import full_metrics

QH, KVH, HD, PAGE = 24, 2, 256, 16


def cell(ctx, qlen, topk, dist, frag, batch=1, dup=False, seed=7):
    kv, q, idx, bt, ttr, nb = build_case(ctx, qlen, topk, QH, KVH, HD, PAGE, dist, seed,
                                         frag=frag, num_requests=batch)
    if dup and topk >= 2:
        idx[0, :] = 5  # duplicate one logical token across all selected columns
    of, od, _, _ = fused_and_dense(kv, q, idx, bt, ttr, HD, PAGE, KVH)
    m = full_metrics(of.cpu(), od.cpu())
    dense_zero = od.abs().max().item() < 1e-6
    ok = m["valid_clean"] and (m["cosine"] > 0.999 or dense_zero)
    return {"ctx": ctx, "qlen": qlen, "batch": batch, "topk": topk, "dist": dist,
            "frag": frag, "dup": dup, "blocks": nb, "cosine": m["cosine"],
            "max_abs": m["max_abs"], "nan_valid": m["nan_valid"], "inf_valid": m["inf_valid"],
            "dense_zero": dense_zero, "PASS": bool(ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cells = []
    # ctx boundaries (page=16 -> 15/16/17 = block-1/block/block+1)
    for ctx in [1, 15, 16, 17, 1024, 8192, 32768]:
        cells.append(cell(ctx, qlen=4, topk=min(2051, ctx), dist="normal", frag=False))
    # distributions
    for dist in ["zeros", "normal", "heavytail", "outliers"]:
        cells.append(cell(1024, qlen=4, topk=1024, dist=dist, frag=False))
    # q_len (query rows) — NOT batch
    for qlen in [1, 2, 3, 4]:
        cells.append(cell(2048, qlen=qlen, topk=2051, dist="normal", frag=False, batch=1))
    # batch (requests) — NOT q_len
    for batch in [1, 2, 4]:
        cells.append(cell(1024, qlen=4, topk=1024, dist="normal", frag=False, batch=batch))
    # top-k production + neighbors
    for topk in [512, 1024, 2050, 2051, 2052]:
        cells.append(cell(8192, qlen=4, topk=topk, dist="normal", frag=False))
    # fragmented block tables (+ multi-request frag)
    cells.append(cell(2048, qlen=4, topk=2051, dist="normal", frag=True))
    cells.append(cell(8192, qlen=8, topk=2051, dist="heavytail", frag=True, batch=2))
    # duplicate logical indices
    cells.append(cell(2048, qlen=4, topk=2051, dist="normal", frag=False, dup=True))
    # page-boundary selection at tiny ctx (distinct semantics: all forced boundary tokens)
    cells.append(cell(17, qlen=4, topk=17, dist="outliers", frag=False))

    npass = sum(c["PASS"] for c in cells)
    res = {"total": len(cells), "passed": npass, "failed": len(cells) - npass,
           "all_valid_clean": all(c["nan_valid"] == 0 and c["inf_valid"] == 0 for c in cells),
           "cells": cells}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"BOUNDARY: {npass}/{len(cells)} PASS  valid_clean={res['all_valid_clean']}")
    for c in cells:
        print(f"  {'OK ' if c['PASS'] else 'FAIL'} ctx={c['ctx']:>6} qlen={c['qlen']} batch={c['batch']} "
              f"topk={c['topk']:>4} dist={c['dist']:<9} frag={int(c['frag'])} dup={int(c['dup'])} "
              f"cos={c['cosine']:.5f} nanv={c['nan_valid']} zero={int(c['dense_zero'])}")
    ok = npass == len(cells) and res["all_valid_clean"]
    print("BOUNDARY_ALL_PASS" if ok else "BOUNDARY_HAS_FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
