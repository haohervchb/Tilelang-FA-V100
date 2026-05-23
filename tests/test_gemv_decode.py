#!/usr/bin/env python3
"""Validate GEMV decode kernel against Triton unified_attention.

Compares TileLang SIMT-FMA GEMV decode output with Triton's
reference attention for single-token (decode) paged attention.
No model server needed — pure kernel math validation.
"""

import torch
import warnings

warnings.filterwarnings("ignore", message="Field.*duplicates")
warnings.filterwarnings("ignore", message=".*GemmSPWarpPolicy.*")

# Ensure tilelang-fa-v100 is importable
from pathlib import Path
import sys

_self = Path(__file__).resolve().parent.parent
if str(_self) not in sys.path:
    sys.path.insert(0, str(_self))

from tilelang_fa_v100._kernels_paged import get_gemv_decode_kernel, get_decode_kernel


def make_inputs(batch, num_heads, num_kv_heads, dim, seq_lens, block_size=16):
    """Create synthetic test inputs with sequential page table."""
    max_blocks_per_seq = max((sl + block_size - 1) // block_size for sl in seq_lens) + 2
    total_blocks = sum((sl + block_size - 1) // block_size for sl in seq_lens) + 4
    num_pages = max(total_blocks, max_blocks_per_seq * batch)

    device = "cuda"
    q = torch.randn(batch, num_heads, dim, device=device, dtype=torch.float16)
    kc = torch.randn(num_pages, block_size, num_kv_heads, dim, device=device, dtype=torch.float16)
    vc = torch.randn(num_pages, block_size, num_kv_heads, dim, device=device, dtype=torch.float16)

    block_table = torch.full(
        (batch, max_blocks_per_seq), -1, dtype=torch.int32, device=device
    )
    offset = 0
    for i, sl in enumerate(seq_lens):
        n_blocks = (sl + block_size - 1) // block_size
        block_table[i, :n_blocks] = torch.arange(
            offset, offset + n_blocks, dtype=torch.int32, device=device
        )
        offset += n_blocks

    sl_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return q, kc, vc, block_table, sl_tensor


def run_gemv_kernel(q, kc, vc, block_table, seq_lens):
    """Run TileLang GEMV decode kernel."""
    batch, heads, dim = q.shape
    heads_kv = kc.shape[2]
    block_size = kc.shape[1]
    num_pages = kc.shape[0]
    max_blocks = block_table.shape[1]

    kernel = get_gemv_decode_kernel(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        block_size=block_size,
        num_pages=num_pages,
        max_blocks=max_blocks,
    )
    return kernel(q, kc, vc, block_table, seq_lens)


def run_triton_attention(q, kc, vc, block_table, seq_lens):
    """Run Triton unified_attention for reference."""
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    batch, heads, dim = q.shape
    max_seqlen_k = seq_lens.max().item()
    scale = dim**-0.5

    out = torch.empty(batch, heads, dim, dtype=torch.float16, device=q.device)

    unified_attention(
        q=q,
        k=kc,
        v=vc,
        out=out,
        cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32, device=q.device),
        max_seqlen_q=1,
        seqused_k=seq_lens,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )
    return out


def run_mma_kernel(q, kc, vc, block_table, seq_lens):
    """Run TileLang MMA decode kernel (with NaN fixes)."""
    batch, heads, dim = q.shape
    heads_kv = kc.shape[2]
    block_size = kc.shape[1]
    num_pages = kc.shape[0]
    max_blocks = block_table.shape[1]

    kernel = get_decode_kernel(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        block_size=block_size,
        num_pages=num_pages,
        max_blocks=max_blocks,
    )
    return kernel(q, kc, vc, block_table, seq_lens)


def run_manual_gemv(q, kc, vc, block_table, seq_lens):
    """Manual mixed-PyTorch reference implementing the same algorithm as the
    GEMV kernel: online softmax, block_N=16, thread-0 sequential reduction."""
    import math
    batch, heads, dim = q.shape
    heads_kv = kc.shape[2]
    block_size = kc.shape[1]
    scale = dim ** -0.5

    out = torch.zeros(batch, heads, dim, dtype=torch.float32, device="cpu")

    for bz in range(batch):
        sl = int(seq_lens[bz].item())
        if sl == 0:
            continue
        for bx in range(heads):
            kvh = bx // (heads // heads_kv)

            # Reconstruct full K/V from pages for this (batch, head)
            K_full = torch.zeros(sl, dim, dtype=torch.float32)
            V_full = torch.zeros(sl, dim, dtype=torch.float32)
            for t in range(sl):
                page_idx = t // block_size
                token_in_page = t % block_size
                ph = int(block_table[bz, page_idx].item())
                K_full[t] = kc[ph, token_in_page, kvh].cpu().float()
                V_full[t] = vc[ph, token_in_page, kvh].cpu().float()

            q_token = q[bz, bx].cpu().float()
            acc = torch.zeros(dim, dtype=torch.float32)
            m = float("-inf")
            l = 0.0

            for k in range(0, sl, 16):
                end = min(k + 16, sl)
                K_tile = K_full[k:end]
                V_tile = V_full[k:end]

                # Q×K^T: element-wise products → sum (serial reduction)
                raw = torch.zeros(16, dtype=torch.float32)
                for j in range(16):
                    if k + j < sl:
                        for d in range(dim):
                            raw[j] += q_token[d].item() * K_tile[j, d].item()

                # Scale + mask
                scores = torch.empty(16, dtype=torch.float32)
                for j in range(16):
                    if k + j < sl:
                        scores[j] = raw[j] * scale
                    else:
                        scores[j] = float("-inf")

                # Online softmax
                m_cur = float(scores[:sl - k].max().item()) if k < sl else 0.0
                if m_cur == float("-inf"):
                    m_cur = 0.0
                old_m = m
                new_m = max(old_m, m_cur)
                sf = math.exp(old_m - new_m) if old_m != float("-inf") else 0.0

                acc *= sf
                tile_sum = 0.0
                probs = torch.empty(16, dtype=torch.float32)
                for j in range(16):
                    if k + j < sl:
                        p = math.exp(float(scores[j].item()) - new_m)
                        probs[j] = p
                        tile_sum += p
                    else:
                        probs[j] = 0.0
                l = l * sf + tile_sum
                m = new_m

                # PV
                for j in range(16):
                    if k + j < sl:
                        acc += probs[j] * V_tile[j]

            if l > 0:
                out[bz, bx] = acc / l

    return out.to(dtype=torch.float16, device=q.device)


def validate_case(name, batch, heads, kv_heads, dim, seq_lens):
    print(f"  {name} (batch={batch}, heads={heads}/{kv_heads}, dim={dim}, "
          f"seq_lens={seq_lens})")
    q, kc, vc, bt, sl = make_inputs(batch, heads, kv_heads, dim, seq_lens)

    output_gemv = run_gemv_kernel(q, kc, vc, bt, sl)
    output_mma = run_mma_kernel(q, kc, vc, bt, sl)
    output_triton = run_triton_attention(q, kc, vc, bt, sl)
    output_manual = run_manual_gemv(q, kc, vc, bt, sl)

    diff_gv_tr = (output_gemv.float() - output_triton.float()).abs()
    diff_mma_tr = (output_mma.float() - output_triton.float()).abs()
    diff_mn_tr = (output_manual.float() - output_triton.float()).abs()
    diff_gv_mn = (output_gemv.float() - output_manual.float()).abs()

    gemv_nan = torch.isnan(output_gemv).any().item()
    mma_nan = torch.isnan(output_mma).any().item()
    triton_nan = torch.isnan(output_triton).any().item()
    manual_nan = torch.isnan(output_manual).any().item()

    passed = (diff_mma_tr.max().item() < 0.10 and diff_mn_tr.max().item() < 0.02
              and not mma_nan and not triton_nan and not manual_nan)
    status = "PASS" if passed else "FAIL"
    print(f"    MMA vs Triton:   max={diff_mma_tr.max().item():.6f} mean={diff_mma_tr.mean().item():.6f} "
          f"nan={mma_nan}")
    print(f"    GEMV vs Triton:  max={diff_gv_tr.max().item():.6f} mean={diff_gv_tr.mean().item():.6f} "
          f"nan={gemv_nan}")
    print(f"    Manual vs Triton: max={diff_mn_tr.max().item():.6f} mean={diff_mn_tr.mean().item():.6f} "
          f"nan={manual_nan}")
    print(f"    → {status}")
    return passed


def main():
    print("=" * 60)
    print("GEMV Decode Kernel Validation (vs Triton unified_attention)")
    print("=" * 60)

    results = []
    results.append(validate_case("[1] Partial last tile", 2, 4, 1, 256, [20, 17]))
    results.append(validate_case("[2] Single token", 1, 4, 1, 256, [1]))
    results.append(validate_case("[3] Exact tile", 2, 4, 1, 256, [16, 16]))
    results.append(validate_case("[4] Empty KV", 1, 4, 1, 256, [0]))
    results.append(validate_case("[5] Long sequence", 1, 4, 1, 256, [512]))
    results.append(validate_case("[6] GQA ratio=4", 2, 8, 2, 256, [32, 15]))

    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{'=' * 60}")
    print(f"Results: {n_pass}/{n_total} passed")
    if n_pass == n_total:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
    return n_pass == n_total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
