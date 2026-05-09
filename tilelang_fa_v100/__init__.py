"""TileLang-optimized FlashAttention for Tesla V100 (SM70)."""

from .interface import (
    tilelang_flash_attn_func,
    tilelang_flash_attn_gpu,
    tilelang_paged_forward,
)

__version__ = "1.0.0"
__all__ = [
    "tilelang_flash_attn_func",
    "tilelang_flash_attn_gpu",
    "tilelang_paged_forward",
]
