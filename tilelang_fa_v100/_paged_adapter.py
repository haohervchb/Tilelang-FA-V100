"""Adapter: calls TileLang paged kernel via 2D linear K/V view.
Zero-copy reshape: k_cache.reshape(-1, heads_kv, dim) → no staging buffer.
Block table: page indices × block_size = absolute positions.
Kernel uses T.copy (vectorized) for each KV tile, not element-by-element.
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

    # Zero-copy: reshape 4D [num_blocks, block_size, heads_kv, D] to 2D
    # [num_padded_tokens, heads_kv, D]. The view is contiguous row-major:
    # block 0's tokens occupy positions [0, block_size), block 1 at [block_size, 2*block_size), etc.
    k_linear = k_cache.reshape(-1, heads_kv, D)
    v_linear = v_cache.reshape(-1, heads_kv, D)

    # block_table: page indices → absolute start positions in the 2D buffer
    # Each page has block_size tokens. Logical page p starts at position p * block_size.
    bt_abs = block_table * block_size  # cheap GPU multiply

    num_pages = k_cache.shape[0]
    max_blocks = block_table.shape[1]

    kernel_compiled = get_paged_kernel(
        batch=B, heads=num_heads, heads_kv=heads_kv, dim=D,
        block_size=block_size, num_pages=num_pages,
        max_blocks=max_blocks, causal=causal,
    )

    result = kernel_compiled(q, k_linear, v_linear, bt_abs, seq_lens,
                             num_tokens)  # T.int32 runtime param — no recompile

    softmax_lse = torch.empty(num_heads, num_tokens, dtype=torch.float32, device=q.device)
    return result, softmax_lse
