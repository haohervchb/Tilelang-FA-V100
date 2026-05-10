"""Paged FlashAttention forward kernel for V100 (SM70).
   4D page-by-page loading handles scattered vLLM page blocks correctly.
   Dynamic tensor shapes via T.dynamic. Compiles ONCE per (heads, dim, causal).
   Supports split-KV for long-sequence parallelism (num_splits > 1).
"""
import math
import torch
import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy


def _paged_kernel_func(batch, heads, heads_kv, dim, page_block_size,
                       max_blocks_per_seq, num_pages, is_causal,
                       block_M=32, block_N=128, num_stages=0, threads=256,
                       num_splits=1):
    scale = (1.0 / dim) ** 0.5
    pages_per_tile = block_N // page_block_size
    nt = T.dynamic("nt")

    if num_splits > 1:
        # ── Split-KV variant ────────────────────────────────────────────────
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
            num_tiles = T.ceildiv(max_tokens, block_M)
            glse = T.alloc_global([nt, num_splits], T.float32)
            op = T.alloc_global([nt, num_splits, dim], T.float16)

            # ── Split phase: each block handles ONE Q-tile, iterates over all splits ─
            with T.Kernel(num_tiles, heads, batch,
                          threads=threads) as (bx, by, bz):
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
                total_kv = cache_seqlens[bz]
                split_len = T.ceildiv(total_kv, num_splits)

                T.copy(Q[start_q: start_q + block_M, by, :], Q_shared)

                # Running state across splits (online softmax)
                max_state = T.alloc_fragment([block_M], T.float32)
                exp_sum = T.alloc_fragment([block_M], T.float32)
                out_sum = T.alloc_fragment([block_M, dim], T.float32)
                old_max = T.alloc_fragment([block_M], T.float32)

                T.fill(max_state, -T.infinity(T.float32))
                T.fill(exp_sum, 0)
                T.fill(out_sum, 0)

                for sid in T.serial(num_splits):
                    split_start = sid * split_len
                    split_end = T.min(split_start + split_len, total_kv)
                    kv_blocks = T.ceildiv(split_end - split_start, block_N)

                    T.fill(acc_o, 0)
                    T.fill(m_i, -T.infinity(T.float32))
                    T.fill(l_i, 0)

                    for k in T.Pipelined(kv_blocks, num_stages=num_stages):
                        T.clear(K_shared)
                        for p in T.serial(pages_per_tile):
                            logical_page = T.floordiv(split_start + k * block_N, page_block_size) + p
                            if logical_page < max_blocks_per_seq:
                                phys = block_table[bz, logical_page]
                                po = p * page_block_size
                                for i, j in T.Parallel(page_block_size, dim):
                                    K_shared[po + i, j] = K_cache[phys, i, kv_head, j]

                        if is_causal:
                            for i, j in T.Parallel(block_M, block_N):
                                q_pos = bx * block_M + i
                                kv_pos = split_start + k * block_N + j
                                acc_s[i, j] = T.if_then_else(
                                    kv_pos <= q_pos + (cache_seqlens[bz] - nt),
                                    T.if_then_else(kv_pos < cache_seqlens[bz], T.cast(0, T.float32), -T.infinity(T.float32)),
                                    -T.infinity(T.float32)
                                )
                        else:
                            for i, j in T.Parallel(block_M, block_N):
                                kv_pos = split_start + k * block_N + j
                                acc_s[i, j] = T.if_then_else(
                                    kv_pos < cache_seqlens[bz], 0, -T.infinity(acc_s.dtype)
                                )

                        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=GemmWarpPolicy.FullRow)
                        T.copy(m_i, m_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for i in T.Parallel(block_M):
                            m_i[i] = T.if_then_else(m_i[i] == -T.infinity(T.float32), T.cast(0, T.float32), m_i[i])
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

                        T.clear(V_shared)
                        for p in T.serial(pages_per_tile):
                            logical_page2 = T.floordiv(split_start + k * block_N, page_block_size) + p
                            if logical_page2 < max_blocks_per_seq:
                                phys2 = block_table[bz, logical_page2]
                                po2 = p * page_block_size
                                for i, j in T.Parallel(page_block_size, dim):
                                    V_shared[po2 + i, j] = V_cache[phys2, i, kv_head, j]

                        for i, j in T.Parallel(block_M, block_N):
                            P_shared[i, j] = T.cast(acc_s[i, j], T.float16)
                        T.copy(P_shared, acc_s_cast)
                        T.gemm(acc_s_cast, V_shared, acc_o, policy=GemmWarpPolicy.Square)

                    # Merge this split into running state (online softmax across splits)
                    for i in T.Parallel(block_M):
                        old_max[i] = max_state[i]
                        new_max = T.max(max_state[i], m_i[i])
                        rescale_old = T.exp((old_max[i] - new_max) * scale)
                        rescale_new = T.exp((m_i[i] - new_max) * scale)
                        exp_sum[i] = exp_sum[i] * rescale_old + l_i[i] * rescale_new
                        max_state[i] = new_max
                    for i, j in T.Parallel(block_M, dim):
                        new_max = T.max(old_max[i], m_i[i])
                        rescale_old = T.exp((old_max[i] - new_max) * scale)
                        rescale_new = T.exp((m_i[i] - new_max) * scale)
                        out_sum[i, j] = out_sum[i, j] * rescale_old + acc_o[i, j] * rescale_new

                for i, j in T.Parallel(block_M, dim):
                    if start_q + i < nt:
                        Output[start_q + i, by, j] = T.cast(out_sum[i, j] / exp_sum[i], T.float16)

    else:
        # ── No-split variant (original kernel) ──────────────────────────────
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
                                k * block_N + j < cache_seqlens[bz], 0, -T.infinity(acc_s.dtype)
                            )

                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=GemmWarpPolicy.FullRow)
                    T.copy(m_i, m_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        m_i[i] = T.if_then_else(m_i[i] == -T.infinity(T.float32), T.cast(0, T.float32), m_i[i])
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
    64: dict(block_M=32, block_N=128, threads=256, num_stages=0, num_splits=1),
    128: dict(block_M=32, block_N=128, threads=256, num_stages=0, num_splits=1),
    256: dict(block_M=32, block_N=64, threads=256, num_stages=0, num_splits=1),
}


def get_paged_kernel(batch, heads, heads_kv, dim, block_size, num_pages,
                     max_blocks, causal):
    """Return compiled kernel. Compiles ONCE per (heads, dim, block_size, causal, config)."""
    cfg = _BEST_CONFIGS.get(dim, dict(block_M=32, block_N=128, threads=256, num_stages=0, num_splits=1))
    key = (heads, heads_kv, dim, block_size, causal,
           cfg["block_M"], cfg["block_N"], cfg["threads"], cfg["num_stages"], cfg["num_splits"])
    if key not in _KERNEL_CACHE:
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
