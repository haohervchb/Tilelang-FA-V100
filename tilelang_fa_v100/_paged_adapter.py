"""Adapter: calls TileLang paged kernel directly on vLLM's 4D paged K/V cache.
Kernel uses dynamic tensor shapes — compiles ONCE per (heads, dim, block_size, causal).
No recompilation across prompts with different num_tokens.
"""
import math
import warnings
import torch

from ._kernels_paged import get_paged_kernel

warnings.filterwarnings("ignore", message="Field.*duplicates an ancestor field")
warnings.filterwarnings("ignore", message=".*GemmSPWarpPolicy.*")

# Fixed grid size for the max-tokens runtime parameter.
# Must be >= any batch's actual num_tokens. The kernel grid = ceil(GRID / block_M).
# Blocks beyond actual nt are no-ops.
GRID_MAX_TOKENS = 1 << 16  # 65536, covers --max-num-batched-tokens up to 65536


def paged_forward(q, k_cache, v_cache, block_table, seq_lens,
                  query_start_loc, prefix_kv_lens, out=None,
                  block_size=16, num_kv_heads=None, softmax_scale=None,
                  causal=True):
    """Paged forward — kernel compiled with dynamic tensor shapes.
    
    Compiled ONCE per (heads, heads_kv, dim, block_size, causal) combo.
    No recompilation per prompt. Kernel uses runtime max_tokens for grid.
    """
    num_tokens, num_heads, D = q.shape
    heads_kv = num_kv_heads or k_cache.shape[2]
    B = block_table.shape[0]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)
    if out is None:
        out = torch.empty_like(q)

    num_blocks = k_cache.shape[0]
    max_blocks = block_table.shape[1]

    kernel_compiled = get_paged_kernel(
        batch=B, heads=num_heads, heads_kv=heads_kv, dim=D,
        block_size=block_size, num_pages=num_blocks,
        max_blocks=max_blocks, causal=causal,
    )

    result = kernel_compiled(q, k_cache, v_cache, block_table, seq_lens,
                             GRID_MAX_TOKENS)  # Python int for T.int32

    softmax_lse = torch.empty(num_heads, num_tokens, dtype=torch.float32, device=q.device)
    return result, softmax_lse
