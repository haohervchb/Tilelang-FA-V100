"""Adapter: calls TileLang paged kernel directly on vLLM's 4D paged K/V cache.
No staging buffer copy. Kernel reads page-by-page from [num_blocks, block_size, heads_kv, D].
"""
import math
import warnings
import torch

import tilelang

from ._kernels_paged import _paged_kernel_func_4d

warnings.filterwarnings("ignore", message="Field.*duplicates an ancestor field")
warnings.filterwarnings("ignore", message=".*GemmSPWarpPolicy.*")

_BEST_CONFIGS = {
    64: dict(block_M=32, block_N=128, threads=256, num_stages=0),
    128: dict(block_M=32, block_N=128, threads=256, num_stages=0),
    256: dict(block_M=16, block_N=64, threads=128, num_stages=0),
}

_KERNEL_CACHE = {}


def _get_or_compile(batch, heads, heads_kv, dim, block_size, max_blocks,
                    num_pages, num_tokens, tokens_per_seq, causal):
    """Compile a 4D paged kernel once per unique shape. Cached at module level."""
    key = (heads, heads_kv, dim, block_size, max_blocks, num_pages,
           num_tokens, tokens_per_seq, causal)
    if key not in _KERNEL_CACHE:
        cfg = _BEST_CONFIGS.get(dim, dict(block_M=32, block_N=128, threads=256, num_stages=0))
        kt = tilelang.jit(out_idx=[5])(_paged_kernel_func_4d).compile(
            batch=batch, heads=heads, heads_kv=heads_kv, dim=dim,
            page_block_size=block_size,
            max_blocks_per_seq=max_blocks,
            num_pages=num_pages,
            num_tokens=num_tokens,
            tokens_per_seq=tokens_per_seq,
            is_causal=causal,
            **cfg,
        )
        _KERNEL_CACHE[key] = kt
    return _KERNEL_CACHE[key]


def paged_forward(q, k_cache, v_cache, block_table, seq_lens,
                  query_start_loc, prefix_kv_lens, out=None,
                  block_size=16, num_kv_heads=None, softmax_scale=None,
                  causal=True):
    """Paged forward — reads directly from vLLM's 4D paged K/V cache.

    No staging buffer. Each KV tile reads page-by-page from
    [num_blocks, block_size, heads_kv, D] using the block_table.
    """
    num_tokens, num_heads, D = q.shape
    heads_kv = num_kv_heads or k_cache.shape[2]
    B = block_table.shape[0]
    tokens_per_seq = (query_start_loc[1] - query_start_loc[0]).item()

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)
    if out is None:
        out = torch.empty_like(q)

    num_blocks = k_cache.shape[0]
    max_blocks_per_seq = block_table.shape[1]
    num_pages = num_blocks

    kernel_compiled = _get_or_compile(
        batch=B, heads=num_heads, heads_kv=heads_kv, dim=D,
        block_size=block_size, max_blocks=max_blocks_per_seq,
        num_pages=num_pages, num_tokens=num_tokens,
        tokens_per_seq=tokens_per_seq, causal=causal,
    )

    result = kernel_compiled(q, k_cache, v_cache, block_table, seq_lens)

    softmax_lse = torch.empty(num_heads, num_tokens, dtype=torch.float32, device=q.device)
    return result, softmax_lse
