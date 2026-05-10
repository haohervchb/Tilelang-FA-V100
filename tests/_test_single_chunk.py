#!/usr/bin/env python3
"""Focused correctness test for TileLang paged kernel.
Single-chunk prefill with various token counts including problematic sizes."""
import warnings, torch, math, torch.nn.functional as F
warnings.filterwarnings('ignore')
import sys, os
sys.path.insert(0, '/home/rah/tilelang-fa-v100')
from tilelang_fa_v100 import tilelang_paged_forward
from tilelang_fa_v100._kernels_paged import _KERNEL_CACHE

def test_single_chunk(B, H, KVH, D, block_size, N, causal, label):
    """Single-chunk prefill test: N total tokens, all processed at once.
    KV cache has N tokens (pre-populated with scattered pages)."""
    torch.manual_seed(42)
    
    # Dense reference K/V
    k_dense = torch.randn(B, KVH, N, D, dtype=torch.float16, device='cuda')
    v_dense = torch.randn(B, KVH, N, D, dtype=torch.float16, device='cuda')
    
    # Q (random)
    q = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
    
    # PyTorch reference
    r = H // KVH
    scale = 1.0 / math.sqrt(D)
    scores = torch.einsum('bhqd,bhkd->bhqk', q.float(), k_dense.repeat_interleave(r, 1).float()) * scale
    if causal:
        mask = torch.tril(torch.ones(N, N, device='cuda'), diagonal=0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1, dtype=torch.float32)
    ref = torch.einsum('bhqk,bhkd->bhqd', attn, v_dense.repeat_interleave(r, 1).float()).half()
    
    # Build vLLM paged cache with scattered pages
    num_pages = int(math.ceil(N / block_size))
    k_cache = torch.zeros(num_pages, block_size, KVH, D, dtype=torch.float16, device='cuda')
    v_cache = torch.zeros(num_pages, block_size, KVH, D, dtype=torch.float16, device='cuda')
    for i in range(num_pages):
        start, end = i * block_size, min((i + 1) * block_size, N)
        for t in range(start, end):
            k_cache[i, t - start] = k_dense[0, :, t]
            v_cache[i, t - start] = v_dense[0, :, t]
    # Scatter pages (vLLM random block_table)
    perm = torch.randperm(num_pages, device='cuda')
    k_cache = k_cache[perm].contiguous()
    v_cache = v_cache[perm].contiguous()
    block_table = perm.view(B, -1).to(torch.int32)
    
    # vLLM metadata
    seq_lens = torch.full((B,), N, dtype=torch.int32, device='cuda')
    query_start_loc = torch.tensor([0, N], dtype=torch.int32, device='cuda')
    prefix_kv_lens = torch.full((B,), 0, dtype=torch.int32, device='cuda')
    q_flat = q.permute(0, 2, 1, 3).reshape(B * N, H, D).contiguous()
    
    out_tl, lse_tl = tilelang_paged_forward(
        q_flat, k_cache, v_cache, block_table, seq_lens,
        query_start_loc, prefix_kv_lens,
        out=None, block_size=block_size,
        causal=causal, num_kv_heads=KVH,
    )
    out_tl_4d = out_tl.reshape(B, N, H, D).permute(0, 2, 1, 3)
    
    err = (out_tl_4d - ref).abs().max().item()
    ok = err < 0.01
    return ok, err, label

# ====== Tests ======
print("=" * 80)
print("SINGLE-CHUNK PREFILL TESTS (scattered pages)")
print("=" * 80)

configs = [
    # Small sanity checks
    (1, 8, 2, 128, 16, 22,  False, "non-causal 22"),
    (1, 8, 2, 128, 16, 256, False, "non-causal 256"),
    (1, 8, 2, 128, 16, 22,  True,  "causal 22"),
    (1, 8, 2, 128, 16, 256, True,  "causal 256"),
    (1, 8, 2, 128, 16, 1024, True, "causal 1024"),
    # Medium
    (1, 8, 2, 128, 16, 4096, True, "causal 4096"),
    (1, 8, 2, 128, 16, 8192, True, "causal 8192"),
    # Close to max-batched-tokens
    (1, 8, 2, 128, 16, 16384, True, "causal 16384"),
    # Problematic reported sizes
    (1, 8, 2, 128, 16, 17000, True, "causal 17000"),
    (1, 8, 2, 128, 16, 21000, True, "causal 21000"),
    (1, 8, 2, 128, 16, 23760, True, "causal 23760"),
    # Non-causal large
    (1, 8, 2, 128, 16, 16384, False, "non-causal 16384"),
    (1, 8, 2, 128, 16, 23760, False, "non-causal 23760"),
    # GQA variant
    (1, 16, 4, 64, 16, 4096, True, "GQA 16:4 D=64 causal 4096"),
]

passed = 0
failed = 0
for B, H, KVH, D, bs, N, causal, label in configs:
    ok, err, lab = test_single_chunk(B, H, KVH, D, bs, N, causal, label)
    if ok:
        passed += 1
        print(f"  ✓ {lab:<40s} err={err:.6f}")
    else:
        failed += 1
        print(f"  ✗ {lab:<40s} err={err:.6f}  FAIL")

print(f"\n{passed}/{passed+failed} passed, cache={len(_KERNEL_CACHE)}")
if failed > 0:
    print("ISSUES FOUND!")
    sys.exit(1)
else:
    print("ALL OK")
