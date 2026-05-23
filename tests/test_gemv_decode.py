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

from tilelang_fa_v100._kernels_paged import get_gemv_decode_kernel


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


def validate_case(name, batch, heads, kv_heads, dim, seq_lens):
    print(f"  {name} (batch={batch}, heads={heads}/{kv_heads}, dim={dim}, "
          f"seq_lens={seq_lens})")
    q, kc, vc, bt, sl = make_inputs(batch, heads, kv_heads, dim, seq_lens)

    output_gemv = run_gemv_kernel(q, kc, vc, bt, sl)
    output_triton = run_triton_attention(q, kc, vc, bt, sl)

    diff = (output_gemv.float() - output_triton.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    gemv_nan = torch.isnan(output_gemv).any().item()
    triton_nan = torch.isnan(output_triton).any().item()

    passed = max_diff < 0.10 and mean_diff < 0.01 and not gemv_nan and not triton_nan
    status = "PASS" if passed else "FAIL"
    print(f"    max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
          f"gemv_nan={gemv_nan}, triton_nan={triton_nan} → {status}")
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
