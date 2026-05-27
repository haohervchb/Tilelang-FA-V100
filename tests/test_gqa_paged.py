#!/usr/bin/env python3
"""GQA validation test for TileLang paged FlashAttention kernel.

Tests the paged attention kernel with Grouped Query Attention (GQA)
configurations matching real models, especially Qwen3.5-122B-A10B
which uses a 16:1 GQA ratio (32 Q heads, 2 KV heads, dim 256).

Usage:
    python tests/test_gqa_paged.py
"""
import sys, os, math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tilelang_fa_v100._kernels_paged import get_paged_kernel, _BEST_CONFIGS


def pytorch_gqa_ref(q, k, v, causal):
    """PyTorch reference attention with GQA support.

    Args:
        q: [B, Hq, M, D]  — query heads
        k: [B, Hkv, N, D] — key heads (fewer than query heads)
        v: [B, Hkv, N, D] — value heads
        causal: bool
    Returns:
        [B, Hq, M, D] attention output
    """
    B, Hq, M, D = q.shape
    Hkv = k.shape[1]
    N = k.shape[2]
    ratio = Hq // Hkv
    scale = 1.0 / math.sqrt(D)

    # Expand K/V heads to match Q heads: [B, Hkv, 1, N, D] -> [B, Hkv, ratio, N, D] -> [B, Hq, N, D]
    k_exp = k.unsqueeze(2).expand(B, Hkv, ratio, N, D).reshape(B, Hq, N, D)
    v_exp = v.unsqueeze(2).expand(B, Hkv, ratio, N, D).reshape(B, Hq, N, D)

    scores = torch.einsum('bhqd,bhkd->bhqk', q.float(), k_exp.float()) * scale
    if causal:
        mask = torch.tril(torch.ones(M, N, device=q.device), diagonal=N - M)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1, dtype=torch.float32)
    return torch.einsum('bhqk,bhkd->bhqd', attn, v_exp.float()).half()


def make_paged_gqa(q_4d, k_4d, v_4d, page_block_size=16):
    """Convert dense 4D Q/K/V to paged layout for the TileLang kernel.

    Args:
        q_4d: [B, Hq, M, D]
        k_4d: [B, Hkv, N, D]
        v_4d: [B, Hkv, N, D]
        page_block_size: page size
    Returns:
        dict with Q_flat, K_cache, V_cache, block_table, seq_lens,
        query_start_loc, prefix_kv_lens
    """
    B, Hq, M, D = q_4d.shape
    Hkv = k_4d.shape[1]
    N = k_4d.shape[2]
    device = q_4d.device

    # Flatten Q: [B, Hq, M, D] -> [B*M, Hq, D]
    Q_flat = q_4d.permute(0, 2, 1, 3).reshape(B * M, Hq, D).contiguous()

    # Create paged K/V cache: [num_pages, page_block_size, Hkv, D]
    num_pages_per_seq = math.ceil(N / page_block_size)
    total_pages = B * num_pages_per_seq
    K_cache = torch.zeros(total_pages, page_block_size, Hkv, D,
                          dtype=torch.float16, device=device)
    V_cache = torch.zeros(total_pages, page_block_size, Hkv, D,
                          dtype=torch.float16, device=device)

    # Fill pages
    for b in range(B):
        for n in range(N):
            page_idx = b * num_pages_per_seq + n // page_block_size
            slot_in_page = n % page_block_size
            K_cache[page_idx, slot_in_page, :, :] = k_4d[b, :, n, :]
            V_cache[page_idx, slot_in_page, :, :] = v_4d[b, :, n, :]

    # Block table: [B, num_pages_per_seq]
    block_table = torch.zeros(B, num_pages_per_seq, dtype=torch.int32, device=device)
    for b in range(B):
        for p in range(num_pages_per_seq):
            block_table[b, p] = b * num_pages_per_seq + p

    # Seq lens
    seq_lens = torch.full((B,), N, dtype=torch.int32, device=device)

    # Query start locations (pure prefill: prefix_kv_lens = 0)
    query_start_loc = torch.zeros(B + 1, dtype=torch.int32, device=device)
    for b in range(B):
        query_start_loc[b + 1] = query_start_loc[b] + M

    prefix_kv_lens = torch.zeros(B, dtype=torch.int32, device=device)

    return {
        'Q_flat': Q_flat,
        'K_cache': K_cache,
        'V_cache': V_cache,
        'block_table': block_table,
        'seq_lens': seq_lens,
        'query_start_loc': query_start_loc,
        'prefix_kv_lens': prefix_kv_lens,
        'num_pages': total_pages,
        'num_tokens': B * M,
    }


def test_gqa_config(B, Hq, Hkv, M, N, D, causal, page_block_size=16,
                    atol=0.01, seed=42):
    """Test a single GQA configuration."""
    ratio = Hq // Hkv
    torch.manual_seed(seed)

    q = torch.randn(B, Hq, M, D, dtype=torch.float16, device='cuda')
    k = torch.randn(B, Hkv, N, D, dtype=torch.float16, device='cuda')
    v = torch.randn(B, Hkv, N, D, dtype=torch.float16, device='cuda')

    # PyTorch reference
    ref_out = pytorch_gqa_ref(q, k, v, causal)

    # TileLang paged kernel
    pd = make_paged_gqa(q, k, v, page_block_size=page_block_size)
    kernel = get_paged_kernel(
        batch=B, heads=Hq, heads_kv=Hkv, dim=D,
        block_size=page_block_size,
        num_pages=pd['num_pages'],
        max_blocks=pd['block_table'].shape[1],
        causal=causal,
    )
    tl_out = kernel(
        pd['Q_flat'], pd['K_cache'], pd['V_cache'],
        pd['block_table'], pd['seq_lens'],
        pd['query_start_loc'], pd['prefix_kv_lens'],
        pd['num_tokens'],
        1.0 / math.sqrt(D),
    )
    # Reshape: [B*M, Hq, D] -> [B, M, Hq, D] -> [B, Hq, M, D]
    tl_out_4d = tl_out.reshape(B, M, Hq, D).permute(0, 2, 1, 3)

    err = (tl_out_4d - ref_out).abs().max().item()
    ok = err < atol
    return ok, err


def test_gqa_chunked_prefill(B, Hq, Hkv, query_len, total_seq_len, D,
                              page_block_size=16, atol=0.01, seed=42):
    """Test GQA with chunked prefill (prefix_kv_lens > 0).

    Simulates a scenario where part of the KV cache is already populated
    and we're processing a new chunk of query tokens.
    """
    ratio = Hq // Hkv
    torch.manual_seed(seed)

    # Full K/V for the entire sequence
    k_full = torch.randn(B, Hkv, total_seq_len, D, dtype=torch.float16, device='cuda')
    v_full = torch.randn(B, Hkv, total_seq_len, D, dtype=torch.float16, device='cuda')

    # Query is only for the last `query_len` tokens
    q = torch.randn(B, Hq, query_len, D, dtype=torch.float16, device='cuda')

    # PyTorch reference: attend to all KV tokens
    ref_out = pytorch_gqa_ref(q, k_full, v_full, causal=True)

    # For TileLang: KV cache has total_seq_len tokens, query has query_len tokens
    # prefix_kv_lens = total_seq_len - query_len (tokens already cached)
    prefix_len = total_seq_len - query_len

    Q_flat = q.permute(0, 2, 1, 3).reshape(B * query_len, Hq, D).contiguous()

    num_pages_per_seq = math.ceil(total_seq_len / page_block_size)
    total_pages = B * num_pages_per_seq
    K_cache = torch.zeros(total_pages, page_block_size, Hkv, D,
                          dtype=torch.float16, device='cuda')
    V_cache = torch.zeros(total_pages, page_block_size, Hkv, D,
                          dtype=torch.float16, device='cuda')

    for b in range(B):
        for n in range(total_seq_len):
            page_idx = b * num_pages_per_seq + n // page_block_size
            slot = n % page_block_size
            K_cache[page_idx, slot, :, :] = k_full[b, :, n, :]
            V_cache[page_idx, slot, :, :] = v_full[b, :, n, :]

    block_table = torch.zeros(B, num_pages_per_seq, dtype=torch.int32, device='cuda')
    for b in range(B):
        for p in range(num_pages_per_seq):
            block_table[b, p] = b * num_pages_per_seq + p

    seq_lens = torch.full((B,), total_seq_len, dtype=torch.int32, device='cuda')
    query_start_loc = torch.zeros(B + 1, dtype=torch.int32, device='cuda')
    for b in range(B):
        query_start_loc[b + 1] = query_start_loc[b] + query_len
    prefix_kv_lens = torch.full((B,), prefix_len, dtype=torch.int32, device='cuda')

    kernel = get_paged_kernel(
        batch=B, heads=Hq, heads_kv=Hkv, dim=D,
        block_size=page_block_size,
        num_pages=total_pages,
        max_blocks=block_table.shape[1],
        causal=True,
    )
    tl_out = kernel(
        Q_flat, K_cache, V_cache,
        block_table, seq_lens,
        query_start_loc, prefix_kv_lens,
        B * query_len,
        1.0 / math.sqrt(D),
    )
    tl_out_4d = tl_out.reshape(B, query_len, Hq, D).permute(0, 2, 1, 3)

    err = (tl_out_4d - ref_out).abs().max().item()
    ok = err < atol
    return ok, err


def main():
    print("=" * 80)
    print("  TILELANG-FA-V100 GQA PAGED ATTENTION VALIDATION")
    print("=" * 80)

    # ─── Part 1: Pure Prefill GQA ─────────────────────────────────────────
    print("\n--- Part 1: Pure Prefill GQA Correctness ---")
    gqa_configs = [
        # (B, Hq, Hkv, M,   N,   D,   causal, pbs,  description)
        (1, 32, 2,  64,  64,  256, True,  16, "Qwen3.5-122B (32/2, D256, causal)"),
        (1, 32, 2,  128, 128, 256, True,  16, "Qwen3.5-122B (32/2, D256, longer)"),
        (1, 32, 2,  64,  64,  256, False, 16, "Qwen3.5-122B (32/2, D256, non-causal)"),
        (2, 32, 2,  64,  64,  256, True,  16, "Qwen3.5-122B B=2 (32/2, D256)"),
        (1, 16, 2,  64,  64,  128, True,  16, "Qwen3.6-35B  (16/2, D128, causal)"),
        (1, 16, 2,  256, 256, 128, True,  16, "Qwen3.6-35B  (16/2, D128, longer)"),
        (1, 24, 4,  64,  64,  128, True,  16, "Qwen3.6-27B  (24/4, D128, causal)"),
        (1, 24, 4,  256, 256, 128, True,  16, "Qwen3.6-27B  (24/4, D128, longer)"),
        (1, 8,  1,  64,  64,  256, True,  16, "Synthetic (8/1, D256, causal)"),
        (1, 8,  1,  128, 128, 256, True,  16, "Synthetic (8/1, D256, longer)"),
        (1, 16, 1,  64,  64,  256, True,  16, "Synthetic TP=2 122B (16/1, D256)"),
        (1, 4,  1,  64,  64,  128, True,  16, "Edge case (4/1, D128)"),
        # Multi-batch
        (4, 32, 2,  32,  32,  256, True,  16, "Qwen3.5-122B B=4 (32/2, D256)"),
        (2, 16, 2,  128, 128, 128, True,  16, "Qwen3.6-35B B=2 (16/2, D128)"),
    ]

    passed = 0
    total = len(gqa_configs)
    for B, Hq, Hkv, M, N, D, causal, pbs, desc in gqa_configs:
        ok, err = test_gqa_config(B, Hq, Hkv, M, N, D, causal, page_block_size=pbs)
        status = "PASS" if ok else "FAIL"
        print(f"  {desc:<45s}  err={err:.6f}  {status}")
        passed += ok

    print(f"\n  Pure Prefill GQA: {passed}/{total} passed")

    # ─── Part 2: Chunked Prefill GQA ──────────────────────────────────────
    print("\n--- Part 2: Chunked Prefill GQA Correctness ---")
    chunked_configs = [
        # (B, Hq, Hkv, query_len, total_seq_len, D, pbs, description)
        (1, 32, 2,  32,  128, 256, 16, "Qwen3.5-122B chunked (prefix=96)"),
        (1, 32, 2,  64,  256, 256, 16, "Qwen3.5-122B chunked (prefix=192)"),
        (1, 16, 2,  32,  128, 128, 16, "Qwen3.6-35B chunked (prefix=96)"),
        (1, 24, 4,  32,  128, 128, 16, "Qwen3.6-27B chunked (prefix=96)"),
        (2, 32, 2,  32,  128, 256, 16, "Qwen3.5-122B B=2 chunked"),
        (1, 8,  1,  16,  64,  256, 16, "Synthetic (8/1) chunked"),
    ]

    chunked_passed = 0
    chunked_total = len(chunked_configs)
    for B, Hq, Hkv, ql, tsl, D, pbs, desc in chunked_configs:
        ok, err = test_gqa_chunked_prefill(B, Hq, Hkv, ql, tsl, D,
                                            page_block_size=pbs)
        status = "PASS" if ok else "FAIL"
        print(f"  {desc:<45s}  err={err:.6f}  {status}")
        chunked_passed += ok

    print(f"\n  Chunked Prefill GQA: {chunked_passed}/{chunked_total} passed")

    # ─── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    all_ok = passed == total and chunked_passed == chunked_total
    print(f"  TOTAL: {passed + chunked_passed}/{total + chunked_total} passed")
    print(f"  VERDICT: {'ALL PASS ✓' if all_ok else 'ISSUES FOUND ✗'}")
    print("=" * 80)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
