"""Adapter: converts vLLM 4D paged K/V [blocks, block_size, heads_kv, D] to
TileLang 2D linear [total_tokens, heads_kv, D] with staging buffer.

Kernel is compiled ONCE per unique shape and cached at module level.
Same-num_tokens layers reuse the cached kernel — no per-layer recompilation.
"""
import math
import warnings
import torch

import tilelang

from ._kernels_paged import _paged_kernel_func

warnings.filterwarnings("ignore", message="Field.*duplicates an ancestor field")
warnings.filterwarnings("ignore", message=".*GemmSPWarpPolicy.*")

_BEST_CONFIGS = {
    64: dict(block_M=32, block_N=128, threads=256, num_stages=0),
    128: dict(block_M=32, block_N=128, threads=256, num_stages=0),
    256: dict(block_M=16, block_N=64, threads=128, num_stages=0),
}

_staging_k = None
_staging_v = None
_staging_shape = None

# Compiled kernel cache: keyed by unique shape parameters
# Different num_tokens values get different cache entries,
# but same-num_tokens layers reuse the same compiled kernel.
_KERNEL_CACHE = {}


def _block_size_for_dim(D):
    if D <= 64:
        return 128
    elif D <= 128:
        return 128
    else:
        return 64


def _get_or_compile(batch, heads, heads_kv, dim, page_bs, max_blocks, num_pages,
                    num_tokens, tokens_per_seq, causal):
    """Return a compiled kernel, compiling once per unique shape."""
    key = (heads, heads_kv, dim, page_bs, max_blocks, num_pages, num_tokens, tokens_per_seq, causal)
    if key not in _KERNEL_CACHE:
        cfg = _BEST_CONFIGS.get(dim, dict(block_M=32, block_N=128, threads=256, num_stages=0))
        kt = tilelang.jit(out_idx=[5])(_paged_kernel_func).compile(
            batch=batch, heads=heads, heads_kv=heads_kv, dim=dim,
            page_block_size=page_bs,
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
    global _staging_k, _staging_v, _staging_shape

    num_tokens, num_heads, D = q.shape
    heads_kv = num_kv_heads or k_cache.shape[2]
    B = block_table.shape[0]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)
    if out is None:
        out = torch.empty_like(q)

    page_bs = max(_block_size_for_dim(D), block_size)

    if block_size >= page_bs:
        k_linear = k_cache.reshape(-1, heads_kv, D)
        v_linear = v_cache.reshape(-1, heads_kv, D)
        bt_2d = block_table * page_bs
        num_staged_pages = block_table.shape[1]
    else:
        num_blocks, bs, hkv, d = k_cache.shape
        total_tokens_staged = num_blocks * block_size
        super_pages_needed = (total_tokens_staged + page_bs - 1) // page_bs
        total_staged = super_pages_needed * page_bs
        new_shape = (total_staged, hkv, d)

        if _staging_shape != new_shape:
            _staging_k = torch.empty(total_staged, hkv, d, dtype=k_cache.dtype, device=k_cache.device)
            _staging_v = torch.empty(total_staged, hkv, d, dtype=v_cache.dtype, device=v_cache.device)
            _staging_shape = new_shape

        for i in range(num_blocks):
            dst = i * block_size
            _staging_k[dst:dst + block_size] = k_cache[i]
            _staging_v[dst:dst + block_size] = v_cache[i]

        k_linear = _staging_k
        v_linear = _staging_v
        super_pages_per_seq = super_pages_needed // B
        bt_new = torch.arange(super_pages_needed, dtype=torch.int32, device=k_cache.device).view(B, super_pages_per_seq) * page_bs
        bt_2d = bt_new
        num_staged_pages = super_pages_per_seq

    num_staging_pages_total = k_linear.shape[0] // page_bs if page_bs > 0 else 1
    tokens_per_seq = (query_start_loc[1] - query_start_loc[0]).item()

    kernel_compiled = _get_or_compile(
        batch=B, heads=num_heads, heads_kv=heads_kv, dim=D,
        page_bs=page_bs, max_blocks=num_staged_pages,
        num_pages=num_staging_pages_total,
        num_tokens=num_tokens,
        tokens_per_seq=tokens_per_seq,
        causal=causal,
    )

    result = kernel_compiled(q, k_linear, v_linear, bt_2d, seq_lens)

    softmax_lse = torch.empty(num_heads, num_tokens, dtype=torch.float32, device=q.device)
    return result, softmax_lse
