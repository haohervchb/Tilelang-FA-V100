"""Adapter: calls TileLang paged kernel on vLLM's 4D paged K/V cache.
Page-by-page loading handles scattered physical blocks correctly.
Kernel uses T.Parallel for per-page element-wise load (correct for non-consecutive pages).
"""
import math
import warnings
import torch

from ._kernels_paged import get_paged_kernel

warnings.filterwarnings("ignore", message="Field.*duplicates an ancestor field")
warnings.filterwarnings("ignore", message=".*GemmSPWarpPolicy.*")


def paged_forward(q, k_cache, v_cache, block_table, seq_lens,
                  query_start_loc, prefix_kv_lens, out=None,
                  block_size=16, num_kv_heads=None, softmax_scale=None,
                  causal=True):
    num_tokens, num_heads, D = q.shape
    heads_kv = num_kv_heads or k_cache.shape[2]
    B = block_table.shape[0]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)
    if out is None:
        out = torch.empty_like(q)

    num_blocks = k_cache.shape[0]
    max_blocks = block_table.shape[1]

    kernel_batch1 = get_paged_kernel(
        batch=1, heads=num_heads, heads_kv=heads_kv, dim=D,
        block_size=block_size, num_pages=num_blocks,
        max_blocks=max_blocks, causal=causal,
    )

    if B > 1:
        q_lens = (query_start_loc[1:] - query_start_loc[:-1])
        for b in range(B):
            b_start = query_start_loc[b].item()
            b_len = q_lens[b].item()
            if b_len <= 0:
                continue
            b_end = b_start + b_len
            result_b = kernel_batch1(
                q[b_start:b_end], k_cache, v_cache,
                block_table[b:b+1], seq_lens[b:b+1],
                b_len,
            )
            out[b_start:b_end] = result_b
    else:
        result = kernel_batch1(q, k_cache, v_cache, block_table, seq_lens, num_tokens)
        out.copy_(result)

    softmax_lse = torch.empty(num_heads, num_tokens, dtype=torch.float32, device=q.device)
    return out, softmax_lse
