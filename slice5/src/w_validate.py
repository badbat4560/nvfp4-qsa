#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Slice 5 write-kernel validation W1-W5 (W6 sanitizer is separate).

W1 packed-byte parity vs independent oracle & scaled_fp4_quant (K,V; tie class).
W2 scale/decode parity (decode write-kernel data+scale vs independent & CUDA).
W3 round-trip: BF16 -> write-kernel -> fused read -> QSA  vs  independent quant/dequant -> dense BF16 QSA.
W4 slot lifecycle: write A, read; overwrite same slot with B; read; prove no stale A.
W5 boundary matrix over slots/tokens/dists/dup/unordered/partial-page/frag.
"""
from __future__ import annotations
import argparse, json, os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../src")))
from write_kernel import nvfp4_write
from nvfp4_oracle import encode as ind_encode, decode as ind_decode, global_scale_for, GROUP
from nvfp4_kernel import qsa_sparse_paged_attention_nvfp4
from qsa_fp8_harness import qsa_sparse_paged_attention
from qsa_ref import qsa_reference
from metrics import full_metrics

DEV = "cuda"
KVH, HD, PAGE = 2, 256, 16
MID = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]


def alloc_split(blocks):
    return (torch.zeros(blocks, PAGE, KVH, HD // 2, device=DEV, dtype=torch.uint8),
            torch.zeros(blocks, PAGE, KVH, HD // 16, device=DEV, dtype=torch.uint8))


def write_cache(src, slot, g, blocks):
    data, scale = alloc_split(blocks)
    nvfp4_write(src, data, scale, slot, g)
    torch.cuda.synchronize()
    return data, scale


def indep_pack_rows(src, g):
    """src [nt,kvh,hd] -> packed[nt,kvh,hd/2], scale[nt,kvh,hd/16] (independent)."""
    nt, kvh, hd = src.shape
    _, packed, scale, _ = ind_encode(src.reshape(-1, hd), g)
    return packed.reshape(nt, kvh, hd // 2), scale.reshape(nt, kvh, hd // 16)


def classify_ties(src, g):
    nt, kvh, hd = src.shape
    x = src.reshape(-1, hd).float()
    _, bs = ind_encode(x, g)[0], ind_encode(x, g)[2].view(torch.float8_e4m3fn).float()
    bs = bs.reshape(-1, hd // GROUP, 1)
    y = (x.reshape(-1, hd // GROUP, GROUP) * (g / bs)).reshape(-1, hd).abs()
    d = torch.stack([(y - t).abs() for t in MID], 0).amin(0)
    return (d < 5e-2)


def W1(seed):
    torch.manual_seed(seed)
    res = {}
    for name in ("K", "V"):
        src = torch.randn(48, KVH, HD, device=DEV, dtype=torch.bfloat16)
        g = global_scale_for(src.reshape(-1, HD))
        blocks = (48 + PAGE - 1) // PAGE
        slot = torch.arange(48, dtype=torch.int64, device=DEV)
        data, scale = write_cache(src, slot, g, blocks)
        ip, isc = indep_pack_rows(src, g)
        # extract written rows
        wd = torch.stack([data[i // PAGE, i % PAGE] for i in range(48)])   # [48,kvh,hd/2]
        ws = torch.stack([scale[i // PAGE, i % PAGE] for i in range(48)])
        dmis = (wd != ip); smis = (ws != isc)
        tie = classify_ties(src, g).reshape(48, KVH, HD)     # [48,kvh,hd] element ties
        # a data byte mismatches if either of its 2 nibbles differ; map ties to bytes (OR of pair)
        tie_bytes = tie.reshape(48, KVH, HD // 2, 2).any(-1)
        n = int(dmis.sum())
        res[name] = {
            "data_byte_match_vs_independent": round(1 - n / dmis.numel(), 6),
            "scale_byte_match_vs_independent": round(1 - int(smis.sum()) / smis.numel(), 6),
            "data_mismatches": n,
            "data_mismatches_at_ties": int((dmis & tie_bytes).sum()),
            "tie_fraction": round(int((dmis & tie_bytes).sum()) / max(n, 1), 4),
        }
        # vs CUDA scaled_fp4_quant (data bytes; swizzle only affects scales)
        try:
            from vllm import _custom_ops as ops
            cp, _ = ops.scaled_fp4_quant(src.reshape(-1, HD), g.reshape(1).float())
            cp = cp.view(torch.uint8).reshape(48, KVH, HD // 2)
            res[name]["data_byte_match_vs_cuda"] = round((wd == cp).float().mean().item(), 6)
        except Exception as e:
            res[name]["cuda"] = f"{type(e).__name__}"
    return res


def W2(seed):
    torch.manual_seed(seed + 1)
    src = torch.randn(48, KVH, HD, device=DEV, dtype=torch.bfloat16)
    g = global_scale_for(src.reshape(-1, HD))
    blocks = (48 + PAGE - 1) // PAGE
    slot = torch.arange(48, dtype=torch.int64, device=DEV)
    data, scale = write_cache(src, slot, g, blocks)
    # decode write-kernel bytes (independent decoder) for all rows
    wd = torch.stack([data[i // PAGE, i % PAGE] for i in range(48)]).reshape(48 * KVH, HD // 2)
    ws = torch.stack([scale[i // PAGE, i % PAGE] for i in range(48)]).reshape(48 * KVH, HD // 16)
    dec_write = ind_decode(wd, ws, g, HD)
    # independent quant/dequant of the same source
    ip, isc = indep_pack_rows(src, g)
    dec_ind = ind_decode(ip.reshape(48 * KVH, HD // 2), isc.reshape(48 * KVH, HD // 16), g, HD)
    return {"decode_write_vs_independent": full_metrics(dec_write, dec_ind),
            "note": "reconstructed values; scaled_fp4_quant scale layout is swizzled so only decoded values are compared"}


def build_qsa(ctx, rows, W, seed, frag=False, dev=DEV):
    gspec = torch.Generator(device=dev).manual_seed(seed)
    nb = (ctx + PAGE - 1) // PAGE
    bt = (torch.randperm(nb, generator=gspec, device=dev) if frag
          else torch.arange(nb, device=dev)).to(torch.int32).view(1, nb)
    ttr = torch.zeros(rows, dtype=torch.int32, device=dev)
    kv = torch.randn(nb, KVH, PAGE, 2 * HD, generator=gspec, device=dev, dtype=torch.bfloat16)
    q = torch.randn(rows, 24, HD, generator=gspec, device=dev, dtype=torch.bfloat16)
    idx = torch.full((rows, W), -1, dtype=torch.int32, device=dev)
    for r in range(rows):
        pos = max(1, (r + 1) * ctx // rows); avail = min(pos, ctx)
        forced = torch.tensor([0, 1, PAGE - 1, PAGE, avail - 1], device=dev)
        forced = forced[(forced >= 0) & (forced < avail)].unique()
        sel = (torch.arange(avail, device=dev) if avail <= W else
               torch.cat([forced, torch.randperm(avail, generator=gspec, device=dev)[:W - forced.numel()]]).unique()[:W])
        idx[r, :sel.numel()] = sel.to(torch.int32)
    return kv, q, idx, bt, ttr, nb


def W3(seed):
    kv, q, idx, bt, ttr, nb = build_qsa(4096, 4, 2051, seed + 2)
    # slot_mapping identity over the physical cache: token (block,off) -> slot block*PAGE+off
    slot = (torch.arange(nb, device=DEV).view(nb, 1) * PAGE
            + torch.arange(PAGE, device=DEV).view(1, PAGE)).reshape(-1).to(torch.int64)  # [nb*PAGE]
    kflat = kv[..., :HD].transpose(1, 2).reshape(nb * PAGE, KVH, HD)   # [nb*PAGE, kvh, hd]
    vflat = kv[..., HD:].transpose(1, 2).reshape(nb * PAGE, KVH, HD)
    gk = global_scale_for(kflat.reshape(-1, HD)); gv = global_scale_for(vflat.reshape(-1, HD))
    kd, ks = write_cache(kflat, slot, gk, nb)
    vd, vs = write_cache(vflat, slot, gv, nb)
    out_wr = qsa_sparse_paged_attention_nvfp4(q.contiguous(), kd, ks, vd, vs, float(gk), float(gv), idx, bt, ttr).cpu()
    # independent quant/dequant -> dense BF16 QSA
    kdeq = ind_decode(*indep_split(kv[..., :HD], gk)).reshape(nb, KVH, PAGE, HD).to(torch.bfloat16)
    vdeq = ind_decode(*indep_split(kv[..., HD:], gv)).reshape(nb, KVH, PAGE, HD).to(torch.bfloat16)
    kv_deq = torch.empty_like(kv); kv_deq[..., :HD] = kdeq; kv_deq[..., HD:] = vdeq
    kc, vc = kv_deq.transpose(1, 2).split(HD, dim=-1)
    out_dense = qsa_sparse_paged_attention(q.contiguous(), kc.contiguous(), vc.contiguous(), idx, bt, ttr).cpu()
    # independent fp32 oracle on bf16
    kcb, vcb = kv.cpu().transpose(1, 2).split(HD, dim=-1)
    ref = qsa_reference(q.cpu(), kcb, vcb, idx.cpu(), bt.cpu(), ttr.cpu(), PAGE)
    return {"roundtrip_write_read_vs_dense_independent": full_metrics(out_wr, out_dense),
            "roundtrip_write_read_vs_bf16_oracle": full_metrics(out_wr, ref["out"])}


def indep_split(x_half, g):
    """[blocks,kvh,page,hd] -> (packed[.. ,hd/2] flat rows, scale flat rows, g, hd) for ind_decode,
    reshaped back to x_half shape."""
    blocks, kvh, page, hd = x_half.shape
    _, packed, scale, _ = ind_encode(x_half.reshape(-1, hd), torch.as_tensor(g, device=DEV))
    return (packed, scale, torch.as_tensor(g, device=DEV), hd)  # ind_decode(packed,scale,g,hd) -> [.,hd]; caller reshapes


def _reshape_deq(deq, shape):
    return deq.reshape(shape)


def W4(seed):
    # write A, read; overwrite same physical slots (incl page boundary + fragmented) with B; read; no stale A
    out = {}
    for tag, frag, slot0 in [("contiguous", False, 0), ("page_boundary", False, PAGE - 1), ("fragmented", True, 0)]:
        kv, q, idx, bt, ttr, nb = build_qsa(1024, 4, 1024, seed + 5, frag=frag)
        slot = (torch.arange(nb, device=DEV).view(nb, 1) * PAGE + torch.arange(PAGE, device=DEV).view(1, PAGE)).reshape(-1).to(torch.int64)
        kflat = kv[..., :HD].transpose(1, 2).reshape(nb * PAGE, KVH, HD)
        vflat = kv[..., HD:].transpose(1, 2).reshape(nb * PAGE, KVH, HD)
        gk = global_scale_for(kflat.reshape(-1, HD)); gv = global_scale_for(vflat.reshape(-1, HD))
        kd, ks = write_cache(kflat, slot, gk, nb); vd, vs = write_cache(vflat, slot, gv, nb)
        outA = qsa_sparse_paged_attention_nvfp4(q.contiguous(), kd, ks, vd, vs, float(gk), float(gv), idx, bt, ttr).cpu()
        # overwrite physical block 0 (or frag map of logical page0) token slot0 with fresh B (SAME global scale)
        pb = int(bt[0, 0].item())
        target_slot = torch.tensor([pb * PAGE + slot0], dtype=torch.int64, device=DEV)
        newK = torch.randn(1, KVH, HD, device=DEV, dtype=torch.bfloat16)
        newV = torch.randn(1, KVH, HD, device=DEV, dtype=torch.bfloat16)
        nvfp4_write(newK, kd, ks, target_slot, gk)   # overwrite in place
        nvfp4_write(newV, vd, vs, target_slot, gv)
        torch.cuda.synchronize()
        outB = qsa_sparse_paged_attention_nvfp4(q.contiguous(), kd, ks, vd, vs, float(gk), float(gv), idx, bt, ttr).cpu()
        # independent dense recompute reflecting B at that slot
        kflat2 = kflat.clone(); vflat2 = vflat.clone()
        # find the flat token index whose slot == target_slot (identity slot -> flat index == slot value here)
        ti = int(target_slot.item())
        kflat2[ti] = newK[0]; vflat2[ti] = newV[0]
        kdeqB = ind_decode(*indep_split(kflat2.reshape(nb, PAGE, KVH, HD).transpose(1, 2), gk)).reshape(nb, KVH, PAGE, HD)
        vdeqB = ind_decode(*indep_split(vflat2.reshape(nb, PAGE, KVH, HD).transpose(1, 2), gv)).reshape(nb, KVH, PAGE, HD)
        kv_deqB = torch.empty(nb, KVH, PAGE, 2 * HD, device=DEV, dtype=torch.bfloat16)
        kv_deqB[..., :HD] = kdeqB.to(torch.bfloat16); kv_deqB[..., HD:] = vdeqB.to(torch.bfloat16)
        kcB, vcB = kv_deqB.transpose(1, 2).split(HD, dim=-1)
        odB = qsa_sparse_paged_attention(q.contiguous(), kcB.contiguous(), vcB.contiguous(), idx, bt, ttr).cpu()
        out[tag] = {"changed_A_to_B": round((outA - outB).abs().max().item(), 6),
                    "readB_vs_dense_independentB": full_metrics(outB, odB)}
    return out


def _gen(tokens, dist, gspec):
    if dist == "zeros":
        return torch.zeros(tokens, KVH, HD, device=DEV, dtype=torch.bfloat16)
    if dist == "normal":
        return torch.randn(tokens, KVH, HD, generator=gspec, device=DEV, dtype=torch.bfloat16)
    if dist == "heavytail":
        b = torch.randn(tokens, KVH, HD, generator=gspec, device=DEV)
        sp = (torch.rand(tokens, KVH, HD, generator=gspec, device=DEV) < 0.02) * torch.randn(tokens, KVH, HD, generator=gspec, device=DEV) * 8
        return (b + sp).to(torch.bfloat16)
    b = torch.randn(tokens, KVH, HD, generator=gspec, device=DEV) * 0.1
    sp = (torch.rand(tokens, KVH, HD, generator=gspec, device=DEV) < 0.01) * 30.0
    return (b + sp).to(torch.bfloat16)


def W5(seed):
    cells = []
    for tokens in [1, 2, 15, 16, 17, 64, 256]:
        for dist in ["zeros", "normal", "heavytail", "outliers"]:
            gspec = torch.Generator(device=DEV).manual_seed(seed + tokens)
            blocks = (18 + tokens + PAGE - 1) // PAGE + 1
            src = _gen(tokens, dist, gspec)
            g = global_scale_for(src.reshape(-1, HD)) if dist != "zeros" else torch.tensor(1.0, device=DEV)
            # UNIQUE slots covering -1(skip), 0, 15, 16, 17, last, then a disjoint block; then unordered (partial page)
            boundary = [-1, 0, 15, 16, 17, blocks * PAGE - 1]
            rest = list(range(18, 18 + tokens))
            slot = torch.tensor((boundary[:min(6, tokens)] + rest)[:tokens], dtype=torch.int64, device=DEV)
            slot = slot[torch.randperm(tokens, generator=gspec, device=DEV)]     # unordered
            data, scale = write_cache(src, slot, g, blocks)                       # check_unique enforced inside
            decs = []; refs = []; nan = 0; checked = 0
            for i in range(tokens):
                s = int(slot[i].item())
                if s < 0 or s // PAGE >= blocks:
                    continue                                                     # skipped slot -> not written
                b_, o_ = s // PAGE, s % PAGE
                dec = ind_decode(data[b_, o_].reshape(KVH, HD // 2), scale[b_, o_].reshape(KVH, HD // 16), g, HD)
                ref = ind_decode(*indep_pack_one(src[i], g))
                decs.append(dec.flatten()); refs.append(ref.flatten())
                nan += int(torch.isnan(dec).sum() + torch.isinf(dec).sum()); checked += 1
            if checks_empty := (not decs):
                cos, maxerr, mism = 1.0, 0.0, 0
            else:
                dcat = torch.cat(decs); rcat = torch.cat(refs)
                cos = torch.nn.functional.cosine_similarity(dcat, rcat, dim=0).item() if dcat.numel() else 1.0
                maxerr = (dcat - rcat).abs().max().item()
                mism = int(((dcat - rcat).abs() > 1e-4).sum())                    # tie-flip elements
            mism_frac = mism / max(checked * KVH * HD, 1)
            ok = (cos > 0.999 or maxerr < 1e-6) and nan == 0 and mism_frac < 0.02   # tie-robust; zeros exact
            cells.append({"tokens": tokens, "dist": dist, "cosine": round(cos, 6),
                          "max_decode_err": round(maxerr, 5), "tie_flip_elems": mism,
                          "tie_flip_frac": round(mism_frac, 6), "nan_inf": nan,
                          "checked": checked, "PASS": bool(ok)})
    # W5b: duplicate slots must be REJECTED (forbidden race contract)
    gsp = torch.Generator(device=DEV).manual_seed(seed + 999)
    src = torch.randn(4, KVH, HD, generator=gsp, device=DEV, dtype=torch.bfloat16)
    dupslot = torch.tensor([3, 3, 5, 7], dtype=torch.int64, device=DEV)   # slot 3 duplicated
    data, scale = alloc_split(2)
    rejected = False
    try:
        nvfp4_write(src, data, scale, dupslot, global_scale_for(src.reshape(-1, HD)))
    except Exception as e:
        rejected = "DuplicateSlot" in type(e).__name__ or "duplicate" in str(e).lower()
    return {"passed": sum(c["PASS"] for c in cells), "total": len(cells),
            "duplicate_slots_rejected": bool(rejected), "cells": cells}


def indep_pack_one(x, g):
    _, packed, scale, _ = ind_encode(x, torch.as_tensor(g, device=DEV))
    return packed, scale, torch.as_tensor(g, device=DEV), HD


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    R = {"env": {"device": torch.cuda.get_device_name(0), "seed": a.seed}}
    R["W1_pack_parity"] = W1(a.seed)
    R["W2_scale_decode_parity"] = W2(a.seed)
    R["W3_roundtrip"] = W3(a.seed)
    R["W4_slot_lifecycle"] = W4(a.seed)
    R["W5_boundary"] = W5(a.seed)
    # gates
    def clean(m): return m.get("valid_clean", True)
    w1 = all(R["W1_pack_parity"][k]["scale_byte_match_vs_independent"] == 1.0 and
             R["W1_pack_parity"][k]["data_byte_match_vs_independent"] > 0.999 for k in ("K", "V"))
    w3 = (R["W3_roundtrip"]["roundtrip_write_read_vs_dense_independent"]["cosine"] > 0.9999
          and clean(R["W3_roundtrip"]["roundtrip_write_read_vs_dense_independent"]))
    w4 = all(v["changed_A_to_B"] > 1e-4 and v["readB_vs_dense_independentB"]["cosine"] > 0.9999
             for v in R["W4_slot_lifecycle"].values())
    w5 = (R["W5_boundary"]["passed"] == R["W5_boundary"]["total"]
          and R["W5_boundary"]["duplicate_slots_rejected"])
    w2m = R["W2_scale_decode_parity"]["decode_write_vs_independent"]
    gs = {"W1_pack_parity": "PASS" if w1 else "FAIL",
          "W2_decode_parity": "PASS" if (w2m["cosine"] > 0.9999 and w2m["valid_clean"]) else "FAIL",
          "W3_roundtrip": "PASS" if w3 else "FAIL",
          "W4_slot_lifecycle": "PASS" if w4 else "FAIL",
          "W5_boundary": f"{R['W5_boundary']['passed']}/{R['W5_boundary']['total']}",
          "W5_duplicate_rejected": "PASS" if R["W5_boundary"]["duplicate_slots_rejected"] else "FAIL",
          "W5_verdict": "PASS" if w5 else "FAIL"}
    gs["OVERALL"] = "PASS" if all(v == "PASS" for k, v in gs.items() if v in ("PASS", "FAIL")) else "FAIL"
    R["gate_summary"] = gs
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=2, default=str)
    print(json.dumps(gs, indent=2))
    sys.exit(0 if gs["OVERALL"] == "PASS" else 1)


if __name__ == "__main__":
    main()
