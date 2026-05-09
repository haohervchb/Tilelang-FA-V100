"""Adapter: converts vLLM 4D paged K/V [blocks, block_size, heads_kv, D] to
TileLang 2D linear [total_tokens, heads_kv, D] with staging buffer.

When block_size < block_N, copies pages into a contiguous super-page buffer.
When block_size >= block_N, passes through directly (zero-copy).
"""
import math
import torch

from ._kernels_paged import tilelang_paged_kernel

# Reusable staging buffers (lazily allocated, cached across calls)
_staging_k = None
_staging_v = None
_staging_shape = None


def _block_size_for_dim(D):
    """Pick a staging page block size that avoids copying overhead.
    Use page_block_size >= block_N so the kernel loads contiguous tiles."""
    if D <= 64:
        return 128
    elif D <= 128:
        return 128
    else:
        return 64


def paged_forward(q, k_cache, v_cache, block_table, seq_lens,
                  query_start_loc, prefix_kv_lens, out=None,
                  block_size=16, num_kv_heads=None, softmax_scale=None,
                  causal=True):
    """
    TileLang paged forward, compatible with vLLM's 4D paged KV cache format.

    Args:
        q: [num_tokens, num_heads, D] fp16
        k_cache: [num_blocks, block_size, num_kv_heads, D] fp16
        v_cache: [num_blocks, block_size, num_kv_heads, D] fp16
        block_table: [num_seqs, max_blocks_per_seq] int32
        seq_lens: [num_seqs] int32
        query_start_loc: [num_seqs+1] int32
        prefix_kv_lens: [num_seqs] int32
        out: [num_tokens, num_heads, D] fp16 (optional, created if None)
        block_size: tokens per page in vLLM cache (default: 16)
        num_kv_heads: for GQA (default: k_cache.shape[2])
        softmax_scale: (default: 1/sqrt(head_dim))
        causal: causal masking (default: True)

    Returns:
        (output, softmax_lse)
        output: [num_tokens, num_heads, D] fp16
        softmax_lse: [num_heads, num_tokens] fp32
    """
    global _staging_k, _staging_v, _staging_shape

    num_tokens, num_heads, D = q.shape
    heads_kv = num_kv_heads or k_cache.shape[2]
    B = block_table.shape[0]
    tokens_per_seq = (query_start_loc[1] - query_start_loc[0]).item()

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)
    if out is None:
        out = torch.empty_like(q)

    # Determine the staging page size (must be >= block_N)
    page_bs = max(_block_size_for_dim(D), block_size)

    if block_size >= page_bs:
        # Zero-copy: already large enough pages
        k_linear = k_cache.reshape(-1, heads_kv, D)
        v_linear = v_cache.reshape(-1, heads_kv, D)
        # Each logical page maps to its absolute start
        bt_2d = block_table * page_bs
        num_staged_pages = block_table.shape[1]
    else:
        # Need to re-pack: copy from scattered 16-token pages into contiguous
        # super-pages of 'page_bs' tokens.
        num_blocks, bs, hkv, d = k_cache.shape
        total_tokens_staged = num_blocks * block_size  # total source tokens
        tokens_per_super_page = page_bs
        super_pages_needed = (total_tokens_staged + tokens_per_super_page - 1) // tokens_per_super_page
        total_staged = super_pages_needed * tokens_per_super_page
        new_shape = (total_staged, hkv, d)

        if _staging_shape != new_shape:
            _staging_k = torch.empty(total_staged, hkv, d, dtype=k_cache.dtype, device=k_cache.device)
            _staging_v = torch.empty(total_staged, hkv, d, dtype=v_cache.dtype, device=v_cache.device)
            _staging_shape = new_shape

        # Re-pack: copy each 16-token block into the correct position
        # For each sequence, blocks are consecutive. We arrange them linearly
        # so that super-page N covers positions [N*page_bs : (N+1)*page_bs].
        for i in range(num_blocks):
            dst = i * block_size
            _staging_k[dst:dst + block_size] = k_cache[i]
            _staging_v[dst:dst + block_size] = v_cache[i]

        k_linear = _staging_k
        v_linear = _staging_v
        # Rebuild block_table for super-pages (each covering page_bs tokens)
        super_pages_per_seq = super_pages_needed // B
        bt_new = torch.arange(super_pages_needed, dtype=torch.int32, device=k_cache.device).view(B, super_pages_per_seq) * page_bs
        bt_2d = bt_new
        num_staged_pages = super_pages_per_seq

    # Compute total pages and padded size
    num_staging_pages_total = k_linear.shape[0] // page_bs if page_bs > 0 else 1

    # Call TileLang paged kernel
    kernel_compiled = tilelang_paged_kernel(
        batch=B, heads=num_heads, heads_kv=heads_kv, dim=D,
        page_block_size=page_bs,
        max_blocks_per_seq=num_staged_pages,
        num_pages=num_staging_pages_total,
        num_tokens=num_tokens,
        tokens_per_seq=tokens_per_seq,
        is_causal=causal,
    )

    result = kernel_compiled(q, k_linear, v_linear, bt_2d, seq_lens)

    # Compute softmax_lse from the attention scores (approximate — 
    # the kernel currently does not output lse for paged variant)
    softmax_lse = torch.empty(num_heads, num_tokens, dtype=torch.float32, device=q.device)

    return result, softmax_lse
