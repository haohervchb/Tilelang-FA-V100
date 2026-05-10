#!/usr/bin/env python3
"""Integration test: simulates vLLM chunked prefill + scattered pages."""
import warnings, torch, math, torch.nn.functional as F
warnings.filterwarnings('ignore')
import sys, os
sys.path.insert(0, '/home/rah/tilelang-fa-v100')
from tilelang_fa_v100._kernels_paged import get_paged_kernel, _KERNEL_CACHE, _BEST_CONFIGS
from tilelang_fa_v100 import tilelang_paged_forward

def pytorch_ref(q, k, v, causal=False):
    """PyTorch reference attention. q: [B,H,M,D], k: [B,KVH,N,D], v: same"""
    B, H, M, D = q.shape
    KVH, N = k.shape[1], k.shape[2]
    r = H // KVH
    scale = 1.0 / math.sqrt(D)
    k_exp = k.repeat_interleave(r, 1)
    v_exp = v.repeat_interleave(r, 1)
    scores = torch.einsum('bhqd,bhkd->bhqk', q.float(), k_exp.float()) * scale
    if causal:
        mask = torch.tril(torch.ones(M, N, device='cuda'), diagonal=N - M)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1, dtype=torch.float32)
    out = torch.einsum('bhqk,bhkd->bhqd', attn, v_exp.float()).half()
    # LSE = log(sum(exp(S - max)))
    lse = scores.logsumexp(dim=-1)
    return out, lse

def make_vllm_cache(B, KVH, N, D, block_size=16, seed=42):
    """Create a vLLM-format paged KV cache with scattered pages."""
    torch.manual_seed(seed)
    # Original dense K/V
    kv_dense = torch.randn(B, KVH, N, D, dtype=torch.float16, device='cuda')
    # Number of pages needed
    num_pages = int(math.ceil(N / block_size))
    # Allocate physical cache
    k_cache = torch.zeros(num_pages, block_size, KVH, D, dtype=torch.float16, device='cuda')
    v_cache = torch.zeros(num_pages, block_size, KVH, D, dtype=torch.float16, device='cuda')
    # Fill page i with token data [i*block_size : (i+1)*block_size)
    for i in range(num_pages):
        start, end = i * block_size, min((i + 1) * block_size, N)
        for t in range(start, end):
            k_cache[i, t - start] = kv_dense[0, :, t]
            v_cache[i, t - start] = kv_dense[0, :, t]
    # Randomly permute pages (vLLM scatters physical pages)
    perm = torch.randperm(num_pages, device='cuda')
    k_scattered = k_cache[perm].contiguous()
    v_scattered = v_cache[perm].contiguous()
    # Build inverse mapping: block_table[0, logical_page] = perm[logical_page]
    # logical_page p should read from physical_page perm[p]
    block_table = perm.view(B, -1).to(torch.int32)
    return kv_dense, k_scattered, v_scattered, block_table

def run_chunked_prefill(B, H, KVH, D, block_size, prompt_len, chunk_size, causal, max_tokens_per_call=None):
    """Simulate vLLM chunked prefill with TileLang paged kernel.
    
    Accumulates KV cache across chunks and compares to PyTorch ref.
    """
    M_total = prompt_len  # total query tokens to process
    N = M_total           # total KV tokens
    kv_dense, k_cache, v_cache, block_table = make_vllm_cache(B, KVH, N, D, block_size)
    
    # For non-causal: prefix = 0 always
    # For causal: chunks overlap, prefix grows as we accumulate cache
    kv_length = 0  # how many KV tokens are cached
    
    for chunk_start in range(0, M_total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, M_total)
        M_chunk = chunk_end - chunk_start
        num_tokens = M_chunk
        
        if causal:
            # Causal: prefix = current kv_length (already computed)
            # query_len = chunk_size
            prefix_len = kv_length
        else:
            prefix_len = 0
        
        # Build q for this chunk
        q = torch.randn(B, H, M_chunk, D, dtype=torch.float16, device='cuda')
        
        # TileLang call
        # Build vLLM metadata
        seq_len = num_tokens + prefix_len  # total KV that should be visible
        seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device='cuda')
        query_start_loc = torch.full((B + 1,), 0, dtype=torch.int32, device='cuda')
        for i in range(B):
            query_start_loc[i + 1] = (i + 1) * M_chunk
        prefix_kv_lens = torch.full((B,), prefix_len, dtype=torch.int32, device='cuda')
        
        q_flat = q.permute(0, 2, 1, 3).reshape(B * M_chunk, H, D).contiguous()
        
        out_tl, lse_tl = tilelang_paged_forward(
            q_flat, k_cache, v_cache, block_table, seq_lens,
            query_start_loc, prefix_kv_lens,
            out=None, block_size=block_size,
            causal=causal, num_kv_heads=KVH,
        )
        out_tl_4d = out_tl.reshape(B, M_chunk, H, D).permute(0, 2, 1, 3)
        
        # PyTorch reference for this chunk
        # Build K/V visible to this chunk: up to seq_len tokens (prefix + chunk)
        kv_visible = kv_dense[:, :, :seq_len]
        ref_out, ref_lse = pytorch_ref(q, kv_visible, kv_visible, causal)
        
        # Compare
        err = (out_tl_4d - ref_out).abs().max().item()
        lse_err = (lse_tl - ref_lse).abs().max().item()
        
        # Update KV length for next chunk (simulate cache growth)
        kv_length += M_chunk
        
        yield {
            'chunk': chunk_start // chunk_size,
            'start': chunk_start, 'end': chunk_end,
            'M_chunk': M_chunk, 'prefix_len': prefix_len,
            'seq_len': seq_len, 'err': err, 'lse_err': lse_err,
        }

# ====== Test Suite ======
print("=" * 80)
print("CHUNKED PREFILL INTEGRATION TEST")
print("=" * 80)

configs = [
    # (B, H, KVH, D, block_size, prompt_len, chunk_size, causal, label)
    (1, 8, 2, 128, 16, 256,  64,  False, "non-causal small"),
    (1, 8, 2, 128, 16, 512,  128, False, "non-causal medium"),
    (1, 8, 2, 128, 16, 256,  64,  True,  "causal small"),
    (1, 8, 2, 128, 16, 1024, 256, True,  "causal medium"),
    (1, 8, 2, 128, 16, 4096, 1024, True, "causal large"),
    (1, 8, 2, 128, 16, 8192, 2048, True, "causal 8K"),
    # Edge cases
    (1, 8, 2, 128, 16, 300,  100, True, "causal non-power-of-two"),
    # The problematic lengths
    (1, 8, 2, 128, 16, 16384, 4096, True, "causal 16K"),
    (1, 8, 2, 128, 16, 21000, 4096, True, "causal 21000"),
    (1, 8, 2, 128, 16, 23760, 4096, True, "causal 23760"),
    # Non-causal large
    (1, 8, 2, 128, 16, 16384, 4096, False, "non-causal 16K"),
    (1, 8, 2, 128, 16, 23760, 4096, False, "non-causal 23760"),
]

all_ok = True
for B, H, KVH, D, bs, prompt_len, chunk_sz, causal, label in configs:
    n_chunks = int(math.ceil(prompt_len / chunk_sz))
    print(f"\n--- {label}: {prompt_len} tok, {chunk_sz}/chunk = {n_chunks} chunks ---")
    
    max_err = 0
    max_lse_err = 0
    chunk_fails = 0
    chunk_count = 0
    
    for result in run_chunked_prefill(B, H, KVH, D, bs, prompt_len, chunk_sz, causal):
        chunk_count += 1
        e = result['err']
        le = result['lse_err']
        max_err = max(max_err, e)
        max_lse_err = max(max_lse_err, le)
        ok = e < 0.01
        if not ok:
            chunk_fails += 1
            print(f"  Chunk {result['chunk']:3d} [{result['start']:5d}-{result['end']:5d}) "
                  f"M={result['M_chunk']:5d} prefix={result['prefix_len']:5d} "
                  f"err={e:.6f} lse_err={le:.6f} {'FAIL' if not ok else ''}")
    
    if chunk_fails == 0:
        print(f"  ALL {chunk_count} chunks PASS  max_err={max_err:.6f}  max_lse_err={max_lse_err:.6f}")
    else:
        print(f"  {chunk_fails}/{chunk_count} chunks FAIL  max_err={max_err:.6f}  max_lse_err={max_lse_err:.6f}")
        all_ok = False

print(f"\n{'='*80}")
_K = len(_KERNEL_CACHE)
print(f"Kernel cache: {_K} entries")
print(f"OVERALL: {'ALL PASS' if all_ok else 'SOME FAILED'}")
