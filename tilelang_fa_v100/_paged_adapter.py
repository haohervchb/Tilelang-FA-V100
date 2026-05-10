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

# Fixed max pages for a stable kernel cache key.
# Must >= model's max_num_blocks. A100/V100: 262144/16 ≈ 16384 pages.
_MAX_PAGES = 32768  # generous upper bound for --max-model-len up to 524288


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

    # Pad cache to _MAX_PAGES if needed (ensures stable kernel compilation)
    if num_blocks < _MAX_PAGES:
        pad = _MAX_PAGES - num_blocks
        k_cached = torch.cat([k_cache, torch.zeros(pad, block_size, heads_kv, D,
                                                    dtype=k_cache.dtype, device=k_cache.device)])
        v_cached = torch.cat([v_cache, torch.zeros(pad, block_size, heads_kv, D,
                                                    dtype=v_cache.dtype, device=v_cache.device)])
    else:
        k_cached = k_cache
        v_cached = v_cache

    # Pad block_table too
    if max_blocks < _MAX_PAGES:
        pad_bt = _MAX_PAGES - max_blocks
        bt_padded = torch.cat([block_table, torch.zeros(B, pad_bt, dtype=torch.int32, device=block_table.device)], dim=1)
    else:
        bt_padded = block_table

    kernel_compiled = get_paged_kernel(
        batch=B, heads=num_heads, heads_kv=heads_kv, dim=D,
        block_size=block_size, num_pages=_MAX_PAGES,
        max_blocks=_MAX_PAGES, causal=causal,
    )

    result = kernel_compiled(q, k_cached, v_cached, bt_padded, seq_lens,
                             num_tokens)

    softmax_lse = torch.empty(num_heads, num_tokens, dtype=torch.float32, device=q.device)
    return result, softmax_lse
