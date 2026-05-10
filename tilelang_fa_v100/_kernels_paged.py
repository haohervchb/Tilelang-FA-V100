"""Paged FlashAttention forward kernel for V100 (SM70).
4D page-by-page loading handles scattered vLLM page blocks correctly.
Dynamic tensor shapes via T.dynamic. Compiles ONCE per (heads, dim, causal).
"""
import math
import torch
import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy


def _paged_kernel_func(batch, heads, heads_kv, dim, page_block_size,
                       max_blocks_per_seq, num_pages, is_causal,
                       block_M=32, block_N=128, num_stages=0, threads=256):
    scale = (1.0 / dim) ** 0.5
    pages_per_tile = block_N // page_block_size
    nt = T.dynamic("nt")

    @T.prim_func
    def main(
        Q: T.Tensor([nt, heads, dim], T.float16),
        K_cache: T.Tensor([num_pages, page_block_size, heads_kv, dim], T.float16),
        V_cache: T.Tensor([num_pages, page_block_size, heads_kv, dim], T.float16),
        block_table: T.Tensor([batch, max_blocks_per_seq], T.int32),
        cache_seqlens: T.Tensor([batch], T.int32),
        max_tokens: T.int32,
        Output: T.Tensor([nt, heads, dim], T.float16),
    ):
        with T.Kernel(T.ceildiv(max_tokens, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], T.float16)
            K_shared = T.alloc_shared([block_N, dim], T.float16)
            V_shared = T.alloc_shared([block_N, dim], T.float16)
            P_shared = T.alloc_shared([block_M, block_N], T.float16)

            acc_s = T.alloc_fragment([block_M, block_N], T.float32)
            acc_s_cast = T.alloc_fragment([block_M, block_N], T.float16)
            acc_o = T.alloc_fragment([block_M, dim], T.float32)
            m_i = T.alloc_fragment([block_M], T.float32)
            m_prev = T.alloc_fragment([block_M], T.float32)
            l_i = T.alloc_fragment([block_M], T.float32)
            sf = T.alloc_fragment([block_M], T.float32)
            row_sum = T.alloc_fragment([block_M], T.float32)

            kv_head = by // (heads // heads_kv)
            start_q = bz * max_tokens + bx * block_M
            kv_offset = cache_seqlens[bz] - nt

            T.copy(Q[start_q: start_q + block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(m_i, -T.infinity(T.float32))
            T.fill(l_i, 0)

            loop_end = (
                T.min(T.ceildiv(cache_seqlens[bz], block_N),
                      T.ceildiv(bx * block_M + block_M + kv_offset, block_N))
                if is_causal
                else T.ceildiv(cache_seqlens[bz], block_N)
            )

            for k in T.Pipelined(loop_end, num_stages=num_stages):
                # Load K page-by-page from 4D paged cache
                # Each page is [page_block_size, heads_kv, dim]; a tile spans pages_per_tile pages
                # Zero-init K_shared so OOB pages (beyond block_table) contribute zeros, not NaN
                T.clear(K_shared)
                for p in T.serial(pages_per_tile):
                    logical_page = T.floordiv(k * block_N, page_block_size) + p
                    if logical_page < max_blocks_per_seq:
                        phys = block_table[bz, logical_page]
                        po = p * page_block_size
                        for i, j in T.Parallel(page_block_size, dim):
                            K_shared[po + i, j] = K_cache[phys, i, kv_head, j]

                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        q_pos = bx * block_M + i
                        kv_pos = k * block_N + j
                        causal_ok = kv_pos <= q_pos + kv_offset
                        seq_ok = kv_pos < cache_seqlens[bz]
                        acc_s[i, j] = T.if_then_else(
                            causal_ok & seq_ok, 0, -T.infinity(acc_s.dtype)
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            k * block_N + j < cache_seqlens[bz],
                            0, -T.infinity(acc_s.dtype)
                        )

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=GemmWarpPolicy.FullRow)
                T.copy(m_i, m_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                # Check_inf: if all scores masked (-inf), clamp to 0 to avoid NaN in sf
                for i in T.Parallel(block_M):
                    m_i[i] = T.if_then_else(
                        m_i[i] == -T.infinity(T.float32),
                        T.cast(0, T.float32),
                        m_i[i]
                    )
                for i in T.Parallel(block_M):
                    m_i[i] = T.max(m_i[i], m_prev[i])
                for i in T.Parallel(block_M):
                    sf[i] = T.exp(m_prev[i] * scale - m_i[i] * scale)
                    l_i[i] *= sf[i]
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= sf[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp(acc_s[i, j] * scale - m_i[i] * scale)
                T.reduce_sum(acc_s, row_sum, dim=1)
                for i in T.Parallel(block_M):
                    l_i[i] += row_sum[i]

                # Load V page-by-page
                # Zero-init V_shared so OOB pages contribute zeros, not NaN
                T.clear(V_shared)
                for p in T.serial(pages_per_tile):
                    logical_page2 = T.floordiv(k * block_N, page_block_size) + p
                    if logical_page2 < max_blocks_per_seq:
                        phys2 = block_table[bz, logical_page2]
                        po2 = p * page_block_size
                        for i, j in T.Parallel(page_block_size, dim):
                            V_shared[po2 + i, j] = V_cache[phys2, i, kv_head, j]

                for i, j in T.Parallel(block_M, block_N):
                    P_shared[i, j] = T.cast(acc_s[i, j], T.float16)
                T.copy(P_shared, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=GemmWarpPolicy.Square)

            for i, j in T.Parallel(block_M, dim):
                if start_q + i < nt:
                    Output[start_q + i, by, j] = T.cast(acc_o[i, j] / l_i[i], T.float16)

    return main


_paged_kernel = tilelang.jit(out_idx=[6])(_paged_kernel_func)


_KERNEL_CACHE = {}

_BEST_CONFIGS = {
    64: dict(block_M=32, block_N=128, threads=256, num_stages=0),
    128: dict(block_M=32, block_N=128, threads=256, num_stages=0),
    256: dict(block_M=32, block_N=64, threads=256, num_stages=0),
}


def get_paged_kernel(batch, heads, heads_kv, dim, block_size, num_pages,
                     max_blocks, causal):
    """Return compiled kernel. Compiles ONCE per (heads, dim, block_size, causal)."""
    key = (heads, heads_kv, dim, block_size, causal)
    if key not in _KERNEL_CACHE:
        cfg = _BEST_CONFIGS.get(dim, dict(block_M=32, block_N=128, threads=256, num_stages=0))
        kt = tilelang.jit(out_idx=[6])(_paged_kernel_func).compile(
            batch=batch, heads=heads, heads_kv=heads_kv, dim=dim,
            page_block_size=block_size,
            max_blocks_per_seq=max_blocks,
            num_pages=num_pages,
            is_causal=causal,
            **cfg,
        )
        _KERNEL_CACHE[key] = kt
    return _KERNEL_CACHE[key]
