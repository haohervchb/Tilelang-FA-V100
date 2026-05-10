"""Adapter: calls TileLang paged decode kernel for V100 (SM70).
   Uses shared-memory softmax state to avoid 1D fragment layout conflicts.
   Works with vLLM's 4D paged K/V cache.
"""
import math
import warnings
import torch

from ._kernels_paged import get_decode_kernel

warnings.filterwarnings("ignore", message="Field.*duplicates an ancestor field")
warnings.filterwarnings("ignore", message=".*GemmSPWarpPolicy.*")


def decode_forward(q, k_cache, v_cache, block_table, seq_lens,
                   block_size=16, num_kv_heads=None, softmax_scale=None):
    """TileLang paged decode forward.

    Args:
        q: [batch, heads, dim] fp16 — one query token per sequence
        k_cache: [num_blocks, block_size, num_kv_heads, head_dim] fp16
        v_cache: [num_blocks, block_size, num_kv_heads, head_dim] fp16
        block_table: [batch, max_blocks_per_seq] int32
        seq_lens: [batch] int32 — total KV cache lengths
        block_size: int — page block size (vLLM default 16)
        num_kv_heads: int — number of KV heads
        softmax_scale: float — scale factor (default: 1/sqrt(dim))

    Returns:
        output: [batch, heads, dim] fp16 — attention output
    """
    batch, num_heads, dim = q.shape
    heads_kv = num_kv_heads or k_cache.shape[2]
    num_blocks = k_cache.shape[0]
    max_blocks = block_table.shape[1]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(dim)

    kernel = get_decode_kernel(
        batch=batch, heads=num_heads, heads_kv=heads_kv, dim=dim,
        block_size=block_size, num_pages=num_blocks,
        max_blocks=max_blocks,
    )

    result = kernel(q, k_cache, v_cache, block_table, seq_lens)
    return result
