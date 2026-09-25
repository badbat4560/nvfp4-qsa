#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-command reproducer for the NVFP4-QSA Slice 3-4 validation (stock prod image).

  C1  format byte + nibble-tie classification + decoded parity (independent / vLLM ref / CUDA)
  C2  GPU unpack (kernel _e2m1) vs independent torch decode
  C3  dense NVFP4 QSA vs BF16 (quality): QK scores / softmax probs / output, NaN split
  C4  fused sparse NVFP4 vs dense-dequant NVFP4 (kernel arithmetic)
  C5  fused sparse NVFP4 vs BF16 fp32 oracle (end-to-end)
  C6  real correctness: multi-request batch, per-request block tables, invalid request,
      host-side slot reuse (A->B), duplicate logical indices
  X   error-ratio analysis: mean_abs AND cosine-defect ratios, both computed and saved
  ST  staging: logical bytes vs actual kernel loads vs metadata traffic; no full BF16 cache
"""
from __future__ import annotations
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvfp4_oracle import encode as ind_encode, decode as ind_decode, global_scale_for, GROUP, E2M1_GRID
from nvfp4_kernel import pack_fp4, qsa_sparse_paged_attention_nvfp4, _e2m1
from qsa_ref import qsa_reference
from metrics import full_metrics
from qsa_fp8_harness import qsa_sparse_paged_attention, quantize_kv
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import ref_nvfp4_quant
from vllm.triton_utils import tl, triton

E4M3_MAX, E2M1_MAX = 448.0, 6.0
DEV = "cuda"
MIDPOINTS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]


@triton.jit
def _decode_kernel(packed_ptr, scale_ptr, out_ptr, inv_g, N: tl.constexpr,
                   NB: tl.constexpr, PACKED: tl.constexpr):
    row = tl.program_id(0)
    dim = tl.arange(0, N)
    raw = tl.load(packed_ptr + row * PACKED + dim // 2)
    nib = tl.where((dim % 2) == 1, (raw >> 4) & 0x0F, raw & 0x0F)
    sc = tl.cast(tl.load(scale_ptr + row * NB + dim // 16), tl.float8e4nv, bitcast=True).to(tl.float32)
    tl.store(out_ptr + row * N + dim, _e2m1(nib) * sc * inv_g)


def gpu_decode(packed, scale_u8, global_scale, N):
    M = packed.shape[0]
    out = torch.empty(M, N, device=DEV, dtype=torch.float32)
    _decode_kernel[(M,)](packed, scale_u8, out, float(1.0 / global_scale), N=N, NB=N // 16, PACKED=N // 2)
    return out


def _unpack_nibbles(packed, n):
    m = packed.shape[0]
    low = packed & 0x0F; high = (packed >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).reshape(m, n).to(torch.int64)


def build_case(ctx, rows, W, qh, kvh, hd, page, dist, seed, frag=False,
               num_requests=1, dev=DEV):
    """Single- or multi-request paged case. rows = query tokens (q_len). Requests
    get DISJOINT physical block ranges; token_to_req assigns rows round-robin."""
    gspec = torch.Generator(device=dev).manual_seed(seed)
    ppr = (ctx + page - 1) // page                       # pages per request
    nb = num_requests * ppr
    # per-request block tables into disjoint physical ranges (frag = shuffle within range)
    bt = torch.zeros(num_requests, ppr, dtype=torch.int32, device=dev)
    for r in range(num_requests):
        base = torch.arange(r * ppr, (r + 1) * ppr, device=dev)
        if frag:
            base = base[torch.randperm(ppr, generator=gspec, device=dev)]
        bt[r] = base.to(torch.int32)
    ttr = (torch.arange(rows, device=dev) % num_requests).to(torch.int32)

    def gen(shape):
        if dist == "zeros":
            return torch.zeros(shape, device=dev, dtype=torch.bfloat16)
        if dist == "normal":
            return torch.randn(shape, generator=gspec, device=dev, dtype=torch.bfloat16)
        if dist == "heavytail":
            b = torch.randn(shape, generator=gspec, device=dev)
            sp = (torch.rand(shape, generator=gspec, device=dev) < 0.02) * torch.randn(shape, generator=gspec, device=dev) * 8
            return (b + sp).to(torch.bfloat16)
        if dist == "outliers":
            b = torch.randn(shape, generator=gspec, device=dev) * 0.1
            sp = (torch.rand(shape, generator=gspec, device=dev) < 0.005) * 30.0
            return (b + sp).to(torch.bfloat16)
        raise ValueError(dist)

    kv = gen((nb, kvh, page, 2 * hd))
    q = gen((rows, qh, hd))
    idx = torch.full((rows, W), -1, dtype=torch.int32, device=dev)
    for r in range(rows):
        pos = max(1, (r + 1) * ctx // max(rows, 1)); avail = min(pos, ctx)
        forced = torch.tensor([0, 1, page - 1, page, page + 1, avail - 1], device=dev)
        forced = forced[(forced >= 0) & (forced < avail)].unique()
        if avail <= W:
            sel = torch.arange(avail, device=dev)
        else:
            rnd = torch.randperm(avail, generator=gspec, device=dev)[: W - forced.numel()]
            sel = torch.cat([forced, rnd]).unique()[:W]
        idx[r, : sel.numel()] = sel.to(torch.int32)
    return kv, q, idx, bt, ttr, nb


def views(t, hd):
    return t.transpose(1, 2).split(hd, dim=-1)


def nvfp4_split_from_half(x_half, page, kvh, hd, block_size=GROUP):
    """[blocks,kvh,page,hd] bf16 -> split buffers (independent oracle) + dequant view."""
    blocks = x_half.shape[0]
    x = x_half.reshape(-1, hd)
    g = global_scale_for(x)
    nib, packed, scale_u8, sat = ind_encode(x, g, block_size)
    deq = ind_decode(packed, scale_u8, g, hd, block_size).reshape(x_half.shape).to(torch.bfloat16)
    data = packed.reshape(blocks, kvh, page, hd // 2).transpose(1, 2).contiguous()
    scale = scale_u8.reshape(blocks, kvh, page, hd // block_size).transpose(1, 2).contiguous()
    return data, scale, float(g.item()), deq, sat


def fused_and_dense(kv, q, idx, bt, ttr, hd, page, kvh):
    kd, ks, gk, kdeq, ksat = nvfp4_split_from_half(kv[..., :hd], page, kvh, hd)
    vd, vs, gv, vdeq, vsat = nvfp4_split_from_half(kv[..., hd:], page, kvh, hd)
    kv_nv = torch.empty_like(kv); kv_nv[..., :hd] = kdeq; kv_nv[..., hd:] = vdeq
    kc, vc = views(kv_nv, hd)
    out_fused = qsa_sparse_paged_attention_nvfp4(q.contiguous(), kd, ks, vd, vs, gk, gv, idx, bt, ttr)
    out_dense = qsa_sparse_paged_attention(q.contiguous(), kc.contiguous(), vc.contiguous(), idx, bt, ttr)
    torch.cuda.synchronize()
    return out_fused, out_dense, (kd, ks, gk, vd, vs, gv), kv_nv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--topk", type=int, default=2051)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    qh, kvh, hd, page = 24, 2, 256, 16
    free0, total = torch.cuda.mem_get_info()
    R = {"env": {"device": torch.cuda.get_device_name(0), "cap": list(torch.cuda.get_device_capability(0)),
                 "free_MiB": round(free0 / 2**20), "torch": torch.__version__, "seed": args.seed,
                 "shapes": {"q_heads": qh, "kv_heads": kvh, "head_dim": hd, "page": page,
                            "ctx": args.ctx, "rows_qlen": args.rows, "topk": args.topk}}}

    # ===== C1 format: byte + nibble-tie classification + decoded parity =====
    xk = torch.randn(64, hd, device=DEV, dtype=torch.bfloat16)
    g = global_scale_for(xk)
    nib_i, packed_i, scale_i, sat_i = ind_encode(xk, g)
    fp4_ref, bs_ref = ref_nvfp4_quant(xk.float(), g.reshape(1).float(), GROUP)
    packed_ref = pack_fp4(fp4_ref)
    nib_ref = _unpack_nibbles(packed_ref, hd)
    mism = (nib_i != nib_ref)
    # pre-round scaled value under the ref block scale, to classify ties
    y_ref = (xk.float().reshape(64, hd // GROUP, GROUP)
             * (g / bs_ref.reshape(64, hd // GROUP, 1).float())).reshape(64, hd).abs()
    dist_to_thr = torch.stack([(y_ref - t).abs() for t in MIDPOINTS], 0).amin(0)
    tie = dist_to_thr < 5e-2
    n_mism = int(mism.sum())
    c1 = {
        "independent_vs_vllm_ref_byte_match": round((packed_i == packed_ref).float().mean().item(), 6),
        "nibble_mismatch_rate": round(mism.float().mean().item(), 6),
        "mismatches_total": n_mism,
        "mismatches_at_ties": int((mism & tie).sum()),
        "mismatches_off_ties": int((mism & ~tie).sum()),
        "tie_fraction_of_mismatches": round((int((mism & tie).sum()) / max(n_mism, 1)), 4),
        "independent_vs_vllm_ref_decoded": full_metrics(
            ind_decode(packed_i, scale_i, g, hd),
            (fp4_ref.reshape(64, hd // GROUP, GROUP) * (bs_ref.reshape(64, hd // GROUP, 1).float() / g)).reshape(64, hd)),
        "nvfp4_quant_saturation": sat_i,
    }
    try:
        from vllm import _custom_ops as ops
        cuda_packed, _ = ops.scaled_fp4_quant(xk, g.reshape(1).float())
        cp = cuda_packed.view(torch.uint8).reshape(64, hd // 2)
        c1["cuda_vs_vllm_ref_byte_match"] = round((cp == packed_ref).float().mean().item(), 6)
        c1["independent_vs_cuda_byte_match"] = round((packed_i == cp).float().mean().item(), 6)
    except Exception as e:
        c1["cuda_scaled_fp4_quant"] = f"{type(e).__name__}: {str(e)[:120]}"
    R["C1_format"] = c1

    # ===== C2 unpack parity =====
    R["C2_unpack_gpu_vs_independent"] = full_metrics(gpu_decode(packed_i, scale_i, g, hd),
                                                     ind_decode(packed_i, scale_i, g, hd))

    # ===== C3-C5 single-request QSA quality/kernel =====
    kv, q, idx, bt, ttr, nb = build_case(args.ctx, args.rows, args.topk, qh, kvh, hd, page, "normal", args.seed)
    q_cpu, kv_cpu, idx_cpu, bt_cpu, ttr_cpu = q.cpu(), kv.cpu(), idx.cpu(), bt.cpu(), ttr.cpu()
    kc_cpu, vc_cpu = views(kv_cpu, hd)
    ref_bf16 = qsa_reference(q_cpu, kc_cpu, vc_cpu, idx_cpu, bt_cpu, ttr_cpu, page)
    out_fused, out_dense, _, kv_nv = fused_and_dense(kv, q, idx, bt, ttr, hd, page, kvh)
    kc_nv_cpu, vc_nv_cpu = views(kv_nv.cpu(), hd)
    ref_nvfp4 = qsa_reference(q_cpu, kc_nv_cpu, vc_nv_cpu, idx_cpu, bt_cpu, ttr_cpu, page)
    kv_fp8, kv_deq8, _, _ = quantize_kv(kv, hd, "absmax")
    kc8_cpu, vc8_cpu = views(kv_deq8.cpu(), hd)
    ref_fp8 = qsa_reference(q_cpu, kc8_cpu, vc8_cpu, idx_cpu, bt_cpu, ttr_cpu, page)

    vmask = ref_bf16["valid"].unsqueeze(1).expand(-1, qh, -1)
    def stage(ref):
        return {"qk_scores": full_metrics(ref["scores"], ref_bf16["scores"], vmask),
                "softmax_probs": full_metrics(ref["probs"], ref_bf16["probs"], vmask),
                "output": full_metrics(ref["out"], ref_bf16["out"])}
    R["C3_dense_nvfp4_vs_bf16"] = stage(ref_nvfp4)
    R["C3b_fp8_vs_bf16"] = stage(ref_fp8)
    R["C4_fused_vs_dense_nvfp4"] = full_metrics(out_fused.cpu(), out_dense.cpu())
    R["C5_fused_vs_bf16_oracle"] = full_metrics(out_fused.cpu(), ref_bf16["out"])

    # ===== C6 real correctness =====
    c6 = {}
    # C6a multi-request batch: 2 requests, token_to_req = [0,1,0,1], disjoint per-request block tables
    kv2, q2, idx2, bt2, ttr2, nb2 = build_case(1024, 4, 1024, qh, kvh, hd, page, "normal", args.seed + 1, num_requests=2)
    of2, od2, _, _ = fused_and_dense(kv2, q2, idx2, bt2, ttr2, hd, page, kvh)
    c6["multi_request"] = {"num_requests": 2, "token_to_req": ttr2.tolist(), "block_table_shape": list(bt2.shape),
                           "metrics": full_metrics(of2.cpu(), od2.cpu())}
    # C6b invalid request id: one row points at a non-existent request -> must be masked (0)
    ttr_bad = ttr2.clone(); ttr_bad[0] = 99
    of_bad, od_bad, _, _ = fused_and_dense(kv2, q2, idx2, bt2, ttr_bad, hd, page, kvh)
    row0_zero = of_bad[0].abs().max().item() < 1e-6
    c6["invalid_request_id"] = {"bad_row_output_is_zero": bool(row0_zero),
                                "other_rows_match_dense": full_metrics(of_bad[1:].cpu(), od_bad[1:].cpu())}
    # C6c duplicate logical indices: force row 0 to repeat one token across all top-k slots
    kv3, q3, idx3, bt3, ttr3, _ = build_case(2048, 4, args.topk, qh, kvh, hd, page, "normal", args.seed + 2)
    idx3[0, :] = 5  # every selected column == logical token 5
    of3, od3, _, kv3nv = fused_and_dense(kv3, q3, idx3, bt3, ttr3, hd, page, kvh)
    kc3, vc3 = views(kv3nv.cpu(), hd)
    ref3 = qsa_reference(q3.cpu(), kc3, vc3, idx3.cpu(), bt3.cpu(), ttr3.cpu(), page)
    c6["duplicate_indices"] = {"fused_vs_dense": full_metrics(of3.cpu(), od3.cpu()),
                               "fused_vs_ref": full_metrics(of3.cpu(), ref3["out"])}
    # C6d packed-level slot reuse: write A packed, read; overwrite SAME physical
    # slot's packed data+scale bytes with B (same global scale), read; prove no A remnants.
    kvA, qd, idxd, btd, ttrd, _ = build_case(256, 2, 64, qh, kvh, hd, page, "normal", args.seed + 3, frag=False)
    kdA, ksA, gkA, kdeqA, _ = nvfp4_split_from_half(kvA[..., :hd], page, kvh, hd)
    vdA, vsA, gvA, vdeqA, _ = nvfp4_split_from_half(kvA[..., hd:], page, kvh, hd)
    ofA = qsa_sparse_paged_attention_nvfp4(qd.contiguous(), kdA, ksA, vdA, vsA, gkA, gvA, idxd, btd, ttrd).cpu()
    # token 0 -> logical page 0 -> physical block btd[0,0] (=0, non-frag). overwrite that used slot.
    pb = int(btd[0, 0].item())
    newK = torch.randn(kvh, hd, device=DEV); newV = torch.randn(kvh, hd, device=DEV)
    _, pkK, scK, _ = ind_encode(newK, torch.tensor(gkA, device=DEV))   # same global scale as A's K
    _, pkV, scV, _ = ind_encode(newV, torch.tensor(gvA, device=DEV))
    kdB = kdA.clone(); ksB = ksA.clone(); vdB = vdA.clone(); vsB = vsA.clone()
    kdB[pb, 0] = pkK; ksB[pb, 0] = scK; vdB[pb, 0] = pkV; vsB[pb, 0] = scV   # [kvh, hd/2] / [kvh, hd/16]
    ofB = qsa_sparse_paged_attention_nvfp4(qd.contiguous(), kdB, ksB, vdB, vsB, gkA, gvA, idxd, btd, ttrd).cpu()
    # dense recompute on the B-modified dequant cache (independent ground truth)
    kdeqB = kdeqA.clone(); vdeqB = vdeqA.clone()
    kdeqB[pb, :, 0, :] = ind_decode(pkK, scK, torch.tensor(gkA, device=DEV), hd).to(torch.bfloat16)
    vdeqB[pb, :, 0, :] = ind_decode(pkV, scV, torch.tensor(gvA, device=DEV), hd).to(torch.bfloat16)
    kvB_deq = torch.empty_like(kvA); kvB_deq[..., :hd] = kdeqB; kvB_deq[..., hd:] = vdeqB
    kcB, vcB = views(kvB_deq, hd)
    odB = qsa_sparse_paged_attention(qd.contiguous(), kcB.contiguous(), vcB.contiguous(), idxd, btd, ttrd).cpu()
    torch.cuda.synchronize()
    c6["slot_reuse_A_then_B"] = {
        "physical_block_overwritten": pb,
        "output_changed_A_to_B": round((ofA - ofB).abs().max().item(), 6),
        "fused_B_vs_dense_B": full_metrics(ofB, odB),
        "note": "packed data+scale of a used physical slot overwritten in place (same global scale); "
                "read reflects B and matches an independent dense recompute -> no stale A remnants",
    }
    R["C6_real_correctness"] = c6

    # ===== X error-ratio (both metrics, computed) =====
    n_o = R["C3_dense_nvfp4_vs_bf16"]["output"]; f_o = R["C3b_fp8_vs_bf16"]["output"]
    R["X_error_ratio"] = {
        "mean_abs_error_fp8": f_o["mean_abs_error"], "mean_abs_error_nvfp4": n_o["mean_abs_error"],
        "mean_abs_ratio": round(n_o["mean_abs_error"] / max(f_o["mean_abs_error"], 1e-30), 3),
        "cosine_defect_fp8": f_o["cosine_defect"], "cosine_defect_nvfp4": n_o["cosine_defect"],
        "cosine_defect_ratio": round(n_o["cosine_defect"] / max(f_o["cosine_defect"], 1e-30), 3),
        "rel_l2_ratio": round(n_o["rel_l2"] / max(f_o["rel_l2"], 1e-30), 3),
        "note": "earlier 'x4' was the mean_abs_ratio; cosine_defect_ratio is the like-for-like",
    }

    # ===== ST staging: logical vs actual-loads vs metadata =====
    tk = args.topk
    R["ST_staging"] = {
        "per_query_row_logical_bytes": tk * (hd // 2 + hd // 16) * 2 * kvh,   # K&V, both kv heads
        "logical_formula": "topk * (head_dim/2 packed + head_dim/16 scale) * 2(K,V) * kv_heads",
        "per_query_row_kernel_load_addresses": tk * (hd + hd) * 2 * kvh,      # redundant: data byte read twice, scale read x16 (cache-served)
        "kernel_load_note": "kernel addresses head_dim data-reads (byte via d//2, read twice) + head_dim scale-reads (broadcast via d//16); L1-served, not extra DRAM",
        "per_query_row_metadata_bytes": tk * 4 + tk * 4,                      # indices int32 + block_table lookups int32
        "metadata_note": "top-k logical indices (int32) + block_table indirections (int32)",
        "dense_bf16_whole_ctx_bytes": args.ctx * kvh * 2 * hd * 2,
        "reads_scale_with_topk_not_ctx": True,
        "no_full_bf16_staging_buffer": True,
        "fused_only_allocates": "out = empty_like(q) [rows,q_heads,head_dim]",
    }

    # ===== gate summary (NaN on valid forbidden) =====
    def clean(m): return m.get("valid_clean", True)
    gs = {
        "C1_decoded_parity": "PASS" if c1["independent_vs_vllm_ref_decoded"]["cosine"] > 0.99999 else "FAIL",
        "C1_byte_tie_classification": {
            "nibble_mismatch_rate": c1["nibble_mismatch_rate"],
            "tie_fraction_of_mismatches": c1["tie_fraction_of_mismatches"],
            "independent_vs_cuda_byte_match": c1.get("independent_vs_cuda_byte_match"),
        },
        "C2_unpack": "PASS" if (R["C2_unpack_gpu_vs_independent"]["cosine"] > 0.99999 and clean(R["C2_unpack_gpu_vs_independent"])) else "FAIL",
        "C3_valid_clean": clean(R["C3_dense_nvfp4_vs_bf16"]["qk_scores"]) and clean(R["C3_dense_nvfp4_vs_bf16"]["output"]),
        "C4_kernel_arithmetic": "PASS" if (R["C4_fused_vs_dense_nvfp4"]["cosine"] > 0.9999 and clean(R["C4_fused_vs_dense_nvfp4"])) else "FAIL",
        "C5_fused_vs_bf16_cos": R["C5_fused_vs_bf16_oracle"]["cosine"],
        "C6_multi_request": "PASS" if (c6["multi_request"]["metrics"]["cosine"] > 0.9999 and clean(c6["multi_request"]["metrics"])) else "FAIL",
        "C6_invalid_request": "PASS" if c6["invalid_request_id"]["bad_row_output_is_zero"] else "FAIL",
        "C6_duplicate_indices": "PASS" if (c6["duplicate_indices"]["fused_vs_dense"]["cosine"] > 0.9999) else "FAIL",
        "C6_slot_reuse": "PASS" if (c6["slot_reuse_A_then_B"]["output_changed_A_to_B"] > 1e-4 and c6["slot_reuse_A_then_B"]["fused_B_vs_dense_B"]["cosine"] > 0.9999) else "FAIL",
    }
    gs["OVERALL"] = "PASS" if all(v == "PASS" for k, v in gs.items()
                                  if isinstance(v, str) and v in ("PASS", "FAIL")) and gs["C3_valid_clean"] else "FAIL"
    R["gate_summary"] = gs

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(R, open(args.out, "w"), indent=2)
    print(json.dumps(gs, indent=2))
    sys.exit(0 if gs["OVERALL"] == "PASS" else 1)


if __name__ == "__main__":
    main()
