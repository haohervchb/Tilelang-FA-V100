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
            query_start_loc: T.Tensor([batch + 1], T.int32),
            prefix_kv_lens: T.Tensor([batch], T.int32),
            max_tokens: T.int32,
            sm_scale: T.float32,
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
                q_len = query_start_loc[bz + 1] - query_start_loc[bz]
                start_q = query_start_loc[bz] + bx * block_M
                total_kv = cache_seqlens[bz]
                split_len = T.ceildiv(total_kv, num_splits)

                # Running state across splits (online softmax) — allocated unconditionally
                max_state = T.alloc_fragment([block_M], T.float32)
                exp_sum = T.alloc_fragment([block_M], T.float32)
                out_sum = T.alloc_fragment([block_M, dim], T.float32)
                old_max = T.alloc_fragment([block_M], T.float32)

                with T.If(bx * block_M < q_len), T.Then():
                    T.copy(Q[start_q: start_q + block_M, by, :], Q_shared)

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
                            for i, j in T.Parallel(block_N, dim):
                                gkv = split_start + k * block_N + i
                                lp = T.floordiv(gkv, page_block_size)
                                off = gkv - lp * page_block_size
                                if lp < max_blocks_per_seq:
                                    phys = block_table[bz, lp]
                                    K_shared[i, j] = K_cache[phys, off, kv_head, j]

                            if is_causal:
                                for i, j in T.Parallel(block_M, block_N):
                                    q_pos = bx * block_M + i
                                    kv_pos = split_start + k * block_N + j
                                    acc_s[i, j] = T.if_then_else(
                                        kv_pos <= prefix_kv_lens[bz] + q_pos,
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
                                sf[i] = T.exp(m_prev[i] * sm_scale - m_i[i] * sm_scale)
                                l_i[i] *= sf[i]
                            for i, j in T.Parallel(block_M, dim):
                                acc_o[i, j] *= sf[i]
                            for i, j in T.Parallel(block_M, block_N):
                                acc_s[i, j] = T.exp(acc_s[i, j] * sm_scale - m_i[i] * sm_scale)
                            T.reduce_sum(acc_s, row_sum, dim=1)
                            for i in T.Parallel(block_M):
                                l_i[i] += row_sum[i]

                            T.clear(V_shared)
                            for i, j in T.Parallel(block_N, dim):
                                gkv2 = split_start + k * block_N + i
                                lp2 = T.floordiv(gkv2, page_block_size)
                                off2 = gkv2 - lp2 * page_block_size
                                if lp2 < max_blocks_per_seq:
                                    phys2 = block_table[bz, lp2]
                                    V_shared[i, j] = V_cache[phys2, off2, kv_head, j]

                            for i, j in T.Parallel(block_M, block_N):
                                P_shared[i, j] = T.cast(acc_s[i, j], T.float16)
                            T.copy(P_shared, acc_s_cast)
                            T.gemm(acc_s_cast, V_shared, acc_o, policy=GemmWarpPolicy.Square)

                        # Merge this split into running state (online softmax across splits)
                        for i in T.Parallel(block_M):
                            old_max[i] = max_state[i]
                            new_max = T.max(max_state[i], m_i[i])
                            rescale_old = T.exp((old_max[i] - new_max) * sm_scale)
                            rescale_new = T.exp((m_i[i] - new_max) * sm_scale)
                            exp_sum[i] = exp_sum[i] * rescale_old + l_i[i] * rescale_new
                            max_state[i] = new_max
                        for i, j in T.Parallel(block_M, dim):
                            new_max = T.max(old_max[i], m_i[i])
                            rescale_old = T.exp((old_max[i] - new_max) * sm_scale)
                            rescale_new = T.exp((m_i[i] - new_max) * sm_scale)
                            out_sum[i, j] = out_sum[i, j] * rescale_old + acc_o[i, j] * rescale_new

                    for i, j in T.Parallel(block_M, dim):
                        if start_q + i < query_start_loc[bz + 1]:
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
            query_start_loc: T.Tensor([batch + 1], T.int32),
            prefix_kv_lens: T.Tensor([batch], T.int32),
            max_tokens: T.int32,
            sm_scale: T.float32,
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
                    q_len = query_start_loc[bz + 1] - query_start_loc[bz]
                    start_q = query_start_loc[bz] + bx * block_M

                    with T.If(bx * block_M < q_len), T.Then():
                        T.copy(Q[start_q: start_q + block_M, by, :], Q_shared)
                        T.fill(acc_o, 0)
                        T.fill(m_i, -T.infinity(T.float32))
                        T.fill(l_i, 0)

                        loop_end = (
                            T.min(T.ceildiv(cache_seqlens[bz], block_N),
                                  T.ceildiv(prefix_kv_lens[bz] + bx * block_M + block_M, block_N))
                            if is_causal
                            else T.ceildiv(cache_seqlens[bz], block_N)
                        )

                        for k in T.Pipelined(loop_end, num_stages=num_stages):
                            T.clear(K_shared)
                            for i, j in T.Parallel(block_N, dim):
                                gkv = k * block_N + i
                                lp = T.floordiv(gkv, page_block_size)
                                off = gkv - lp * page_block_size
                                if lp < max_blocks_per_seq:
                                    phys = block_table[bz, lp]
                                    K_shared[i, j] = K_cache[phys, off, kv_head, j]

                            if is_causal:
                                for i, j in T.Parallel(block_M, block_N):
                                    q_pos = bx * block_M + i
                                    kv_pos = k * block_N + j
                                    causal_ok = kv_pos <= prefix_kv_lens[bz] + q_pos
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
                                sf[i] = T.exp(m_prev[i] * sm_scale - m_i[i] * sm_scale)
                                l_i[i] *= sf[i]
                            for i, j in T.Parallel(block_M, dim):
                                acc_o[i, j] *= sf[i]
                            for i, j in T.Parallel(block_M, block_N):
                                acc_s[i, j] = T.exp(acc_s[i, j] * sm_scale - m_i[i] * sm_scale)
                            T.reduce_sum(acc_s, row_sum, dim=1)
                            for i in T.Parallel(block_M):
                                l_i[i] += row_sum[i]

                            T.clear(V_shared)
                            for i, j in T.Parallel(block_N, dim):
                                gkv2 = k * block_N + i
                                lp2 = T.floordiv(gkv2, page_block_size)
                                off2 = gkv2 - lp2 * page_block_size
                                if lp2 < max_blocks_per_seq:
                                    phys2 = block_table[bz, lp2]
                                    V_shared[i, j] = V_cache[phys2, off2, kv_head, j]

                            for i, j in T.Parallel(block_M, block_N):
                                P_shared[i, j] = T.cast(acc_s[i, j], T.float16)
                            T.copy(P_shared, acc_s_cast)
                            T.gemm(acc_s_cast, V_shared, acc_o, policy=GemmWarpPolicy.Square)

                        for i, j in T.Parallel(block_M, dim):
                            if start_q + i < query_start_loc[bz + 1]:
                                Output[start_q + i, by, j] = T.cast(acc_o[i, j] / l_i[i], T.float16)

    return main


_paged_kernel = tilelang.jit(out_idx=[9])(_paged_kernel_func)

_KERNEL_CACHE = {}

_BEST_CONFIGS = {
    64: dict(block_M=32, block_N=128, threads=256, num_stages=0, num_splits=1),
    128: dict(block_M=32, block_N=128, threads=256, num_stages=0, num_splits=1),
    256: dict(block_M=64, block_N=32, threads=256, num_stages=0, num_splits=1),
}


def get_paged_kernel(batch, heads, heads_kv, dim, block_size, num_pages,
                     max_blocks, causal):
    """Return compiled kernel. Compiles ONCE per (heads, dim, block_size, causal, config)."""
    cfg = _BEST_CONFIGS.get(dim, dict(block_M=32, block_N=128, threads=256, num_stages=0, num_splits=1))
    key = (heads, heads_kv, dim, block_size, causal,
           cfg["block_M"], cfg["block_N"], cfg["threads"], cfg["num_stages"], cfg["num_splits"],
           batch, max_blocks, num_pages)
    if key not in _KERNEL_CACHE:
        kt = tilelang.jit(out_idx=[9])(_paged_kernel_func).compile(
            batch=batch, heads=heads, heads_kv=heads_kv, dim=dim,
            page_block_size=block_size,
            max_blocks_per_seq=max_blocks,
            num_pages=num_pages,
            is_causal=causal,
            **cfg,
        )
        _KERNEL_CACHE[key] = kt
    return _KERNEL_CACHE[key]


# ═══════════════════════════════════════════════════════════════════════════════
# Decode kernel (shared-memory softmax to avoid 1D fragment layout conflicts)
# ═══════════════════════════════════════════════════════════════════════════════

def _decode_kernel_func(batch, heads, heads_kv, dim, page_block_size,
                        max_blocks_per_seq, num_pages,
                        block_N=128, num_stages=0, threads=128):
    scale = (1.0 / dim) ** 0.5
    pts = block_N // page_block_size
    block_M = 16  # SM70 MMA minimum; row 0 = real Q, rows 1-15 = zero padding

    @T.prim_func
    def kernel(
        Q: T.Tensor([batch, heads, dim], T.float16),
        Kc: T.Tensor([num_pages, page_block_size, heads_kv, dim], T.float16),
        Vc: T.Tensor([num_pages, page_block_size, heads_kv, dim], T.float16),
        bt: T.Tensor([batch, max_blocks_per_seq], T.int32),
        sl: T.Tensor([batch], T.int32),
        Out: T.Tensor([batch, heads, dim], T.float16),
    ):
        with T.Kernel(heads, batch, threads=threads) as (bx, bz):
            Qs = T.alloc_shared([block_M, dim], T.float16)
            Ks = T.alloc_shared([block_N, dim], T.float16)
            Vs = T.alloc_shared([block_N, dim], T.float16)
            Ps = T.alloc_shared([block_M, block_N], T.float16)

            As = T.alloc_fragment([block_M, block_N], T.float32)
            Ac = T.alloc_fragment([block_M, block_N], T.float16)
            Ao = T.alloc_fragment([block_M, dim], T.float32)

            mi = T.alloc_fragment([block_M], T.float32)
            mp = T.alloc_fragment([block_M], T.float32)
            sf = T.alloc_fragment([block_M], T.float32)
            rs = T.alloc_fragment([block_M], T.float32)

            # Shared memory for l_i — bypasses 1D fragment layout conflict with mi
            li = T.alloc_shared([block_M], T.float32)

            kvh = bx // (heads // heads_kv)

            T.clear(Qs)
            T.copy(Q[bz, bx, :], Qs[0, :])
            T.fill(Ao, 0)
            T.fill(mi, -T.infinity(T.float32))
            T.fill(li, 0)

            for k in T.Pipelined(T.ceildiv(sl[bz], block_N), num_stages=num_stages):
                T.clear(Ks)
                for p in T.serial(pts):
                    lp = T.floordiv(k * block_N, page_block_size) + p
                    if lp < max_blocks_per_seq:
                        ph = bt[bz, lp]
                        po = p * page_block_size
                        for i, j in T.Parallel(page_block_size, dim):
                            Ks[po + i, j] = Kc[ph, i, kvh, j]

                for i, j in T.Parallel(block_M, block_N):
                    As[i, j] = T.if_then_else(k * block_N + j < sl[bz], T.cast(0, T.float32), -T.infinity(T.float32))
                T.gemm(Qs, Ks, As, transpose_B=True, policy=GemmWarpPolicy.FullRow)

                T.copy(mi, mp)
                T.reduce_max(As, mi, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    mi[i] = T.if_then_else(mi[i] == -T.infinity(T.float32), T.cast(0, T.float32), mi[i])
                for i in T.Parallel(block_M):
                    mi[i] = T.max(mi[i], mp[i])
                for i in T.Parallel(block_M):
                    sf[i] = T.exp(mp[i] * scale - mi[i] * scale)
                    li[i] *= sf[i]
                for i, j in T.Parallel(block_M, dim):
                    Ao[i, j] *= sf[i]
                for i, j in T.Parallel(block_M, block_N):
                    As[i, j] = T.exp(As[i, j] * scale - mi[i] * scale)
                T.reduce_sum(As, rs, dim=1)
                for i in T.Parallel(block_M):
                    li[i] += rs[i]

                T.clear(Vs)
                for p in T.serial(pts):
                    lp2 = T.floordiv(k * block_N, page_block_size) + p
                    if lp2 < max_blocks_per_seq:
                        ph2 = bt[bz, lp2]
                        po2 = p * page_block_size
                        for i, j in T.Parallel(page_block_size, dim):
                            Vs[po2 + i, j] = Vc[ph2, i, kvh, j]

                for i, j in T.Parallel(block_M, block_N):
                    Ps[i, j] = T.cast(As[i, j], T.float16)
                T.copy(Ps, Ac)
                T.gemm(Ac, Vs, Ao, policy=GemmWarpPolicy.Square)

            for i, j in T.Parallel(block_M, dim):
                if i == 0:
                    Out[bz, bx, j] = T.if_then_else(
                        li[i] > T.cast(0, T.float32),
                        T.cast(Ao[i, j] / li[i], T.float16),
                        T.cast(0, T.float16))

    return kernel


_DECODE_JIT = tilelang.jit(out_idx=[5])(_decode_kernel_func)
_DECODE_CACHE = {}

_DECODE_BEST_CONFIGS = {
    64:  dict(block_N=128, threads=128, num_stages=0),
    128: dict(block_N=128, threads=128, num_stages=0),
    256: dict(block_N=16,  threads=32,  num_stages=0),
}


def get_decode_kernel(batch, heads, heads_kv, dim, block_size, num_pages,
                      max_blocks):
    cfg = _DECODE_BEST_CONFIGS.get(dim, dict(block_N=128, threads=128, num_stages=0))
    key = (heads, heads_kv, dim, block_size, cfg["block_N"], cfg["threads"], cfg["num_stages"],
           batch, max_blocks, num_pages)
    if key not in _DECODE_CACHE:
        kt = tilelang.jit(out_idx=[5])(_decode_kernel_func).compile(
            batch=batch, heads=heads, heads_kv=heads_kv, dim=dim,
            page_block_size=block_size,
            max_blocks_per_seq=max_blocks,
            num_pages=num_pages,
            **cfg,
        )
        _DECODE_CACHE[key] = kt
    return _DECODE_CACHE[key]


# ── GEMV decode kernel (SIMT FMA, no MMA) ─────────────────────────────────

def _decode_gemv_kernel_func(batch, heads, heads_kv, dim, page_block_size,
                              max_blocks_per_seq, num_pages,
                              block_N=16, num_stages=0, threads=256):
    """SIMT FMA GEMV decode kernel — avoids MMA for single-token decode.

    All computation uses shared memory + T.Parallel + T.serial.
    Zero fragment usage, zero tensor core usage.
    Block_M = 1 (no row padding).
    Reductions done by thread 0 via T.serial loops.
    """
    scale = (1.0 / dim) ** 0.5

    @T.prim_func
    def kernel(
        Q: T.Tensor([batch, heads, dim], T.float16),
        Kc: T.Tensor([num_pages, page_block_size, heads_kv, dim], T.float16),
        Vc: T.Tensor([num_pages, page_block_size, heads_kv, dim], T.float16),
        bt: T.Tensor([batch, max_blocks_per_seq], T.int32),
        sl: T.Tensor([batch], T.int32),
        Out: T.Tensor([batch, heads, dim], T.float16),
    ):
        with T.Kernel(heads, batch, threads=threads) as (bx, bz):
            Qs = T.alloc_shared([dim], T.float16)
            Ks = T.alloc_shared([block_N, dim], T.float16)
            Vs = T.alloc_shared([block_N, dim], T.float16)

            m_shared = T.alloc_shared([1], T.float32)
            l_shared = T.alloc_shared([1], T.float32)
            sf_shared = T.alloc_shared([1], T.float32)
            tsum_shared = T.alloc_shared([1], T.float32)
            m_cur_shared = T.alloc_shared([1], T.float32)

            temp_smem = T.alloc_shared([block_N, dim], T.float32)
            scores_smem = T.alloc_shared([block_N], T.float32)
            acc_v_smem = T.alloc_shared([dim], T.float32)
            acc_partial_smem = T.alloc_shared([dim], T.float32)

            kvh = bx // (heads // heads_kv)

            T.copy(Q[bz, bx, :], Qs)
            T.clear(acc_v_smem)
            T.fill(m_shared, -T.infinity(T.float32))
            T.fill(l_shared, 0)

            for k in T.Pipelined(T.ceildiv(sl[bz], block_N), num_stages=num_stages):
                # ── Load K tile (1 page per iteration) ──
                T.clear(Ks)
                lp = k
                if lp < max_blocks_per_seq:
                    ph = bt[bz, lp]
                    for i, j in T.Parallel(page_block_size, dim):
                        Ks[i, j] = Kc[ph, i, kvh, j]

                # ── Q×K^T GEMV: element-wise products into shared memory ──
                for j, d in T.Parallel(block_N, dim):
                    temp_smem[j, d] = T.cast(Qs[d], T.float32) * T.cast(Ks[j, d], T.float32)

                # ── Reduce across dim (d) for each token j: thread 0 only ──
                for tx in T.Parallel(threads):
                    if tx == 0:
                        for j in T.serial(block_N):
                            for d in T.serial(dim):
                                val = temp_smem[j, d]
                                scores_smem[j] = T.if_then_else(
                                    d == 0, val, scores_smem[j] + val)

                # ── Scale + mask ──
                for j in T.Parallel(block_N):
                    token_idx = k * block_N + j
                    scores_smem[j] = T.if_then_else(
                        token_idx < sl[bz],
                        scores_smem[j] * scale,
                        -T.infinity(T.float32))

                # ── Online softmax: all threads redundantly compute max/sf ──
                # First, thread 0 reduces to shared for score reduction and tsum
                for tx in T.Parallel(threads):
                    if tx == 0:
                        for j in T.serial(block_N):
                            m_cur_shared[0] = T.if_then_else(
                                j == 0,
                                scores_smem[j],
                                T.max(m_cur_shared[0], scores_smem[j]))
                        m_cur_shared[0] = T.if_then_else(
                            m_cur_shared[0] == -T.infinity(T.float32),
                            T.cast(0, T.float32),
                            m_cur_shared[0])
                        old_m = m_shared[0]
                        new_m = T.max(old_m, m_cur_shared[0])
                        m_shared[0] = new_m
                        sf_shared[0] = T.exp(old_m - new_m)

                        for j in T.serial(block_N):
                            tsum_shared[0] = T.if_then_else(
                                j == 0,
                                T.exp(scores_smem[j] - new_m),
                                tsum_shared[0] + T.exp(scores_smem[j] - new_m))

                        l_shared[0] = l_shared[0] * sf_shared[0] + tsum_shared[0]

                # ── Convert scores to softmax probs (needed for PV) ──
                for j in T.Parallel(block_N):
                    scores_smem[j] = T.exp(scores_smem[j] - m_shared[0])

                # ── Rescale accumulator (all threads, use sf from shared) ──
                for d in T.Parallel(dim):
                    acc_v_smem[d] *= sf_shared[0]

                # ── Load V tile ──
                T.clear(Vs)
                if lp < max_blocks_per_seq:
                    ph2 = bt[bz, lp]
                    for i, j in T.Parallel(page_block_size, dim):
                        Vs[i, j] = Vc[ph2, i, kvh, j]

                # ── P×V GEMV: scores[j] * V[j,d] into shared memory ──
                for j, d in T.Parallel(block_N, dim):
                    temp_smem[j, d] = scores_smem[j] * T.cast(Vs[j, d], T.float32)

                # ── Reduce across token dim (j) for each output dim d: thread 0 ──
                for tx in T.Parallel(threads):
                    if tx == 0:
                        for d in T.serial(dim):
                            for j in T.serial(block_N):
                                val = temp_smem[j, d]
                                acc_partial_smem[d] = T.if_then_else(
                                    j == 0, val, acc_partial_smem[d] + val)

                # ── Accumulate into running output (all threads) ──
                for d in T.Parallel(dim):
                    acc_v_smem[d] += acc_partial_smem[d]

            # ── Output (guard against l==0 for empty sequences) ──
            for d in T.Parallel(dim):
                Out[bz, bx, d] = T.if_then_else(
                    l_shared[0] > T.cast(0, T.float32),
                    T.cast(acc_v_smem[d] / l_shared[0], T.float16),
                    T.cast(0, T.float16))

    return kernel


_GEMV_DECODE_JIT = tilelang.jit(out_idx=[5])(_decode_gemv_kernel_func)
_GEMV_DECODE_CACHE = {}

_GEMV_DECODE_BEST_CONFIGS = {
    256: dict(block_N=16, threads=256, num_stages=0),
}


def get_gemv_decode_kernel(batch, heads, heads_kv, dim, block_size, num_pages,
                            max_blocks):
    cfg = _GEMV_DECODE_BEST_CONFIGS.get(dim, dict(block_N=16, threads=256, num_stages=0))
    key = (heads, heads_kv, dim, block_size, cfg["block_N"], cfg["threads"], cfg["num_stages"],
           batch, max_blocks, num_pages)
    if key not in _GEMV_DECODE_CACHE:
        kt = tilelang.jit(out_idx=[5])(_decode_gemv_kernel_func).compile(
            batch=batch, heads=heads, heads_kv=heads_kv, dim=dim,
            page_block_size=block_size,
            max_blocks_per_seq=max_blocks,
            num_pages=num_pages,
            **cfg,
        )
        _GEMV_DECODE_CACHE[key] = kt
    return _GEMV_DECODE_CACHE[key]

