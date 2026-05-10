"""Paged FlashAttention forward kernel for V100 (SM70). Autotuned."""
import math
import torch
import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy
from ._configs import get_paged_configs


def _paged_kernel_func_4d(batch, heads, heads_kv, dim, page_block_size,
                          max_blocks_per_seq, num_pages, num_tokens,
                          tokens_per_seq, is_causal,
                          block_M=32, block_N=128, num_stages=0, threads=256):
    """Paged FA forward reading directly from vLLM's 4D KV cache.
    
    K/V layout: [num_pages, page_block_size, heads_kv, dim]
    block_table: [batch, max_blocks_per_seq] — page indices
    For each KV tile (block_N tokens), reads from ceil(block_N/page_block_size) pages.
    """
    scale = (1.0 / dim) ** 0.5
    pages_per_tile = block_N // page_block_size

    @T.prim_func
    def main(
        Q: T.Tensor([num_tokens, heads, dim], T.float16),
        K_cache: T.Tensor([num_pages, page_block_size, heads_kv, dim], T.float16),
        V_cache: T.Tensor([num_pages, page_block_size, heads_kv, dim], T.float16),
        block_table: T.Tensor([batch, max_blocks_per_seq], T.int32),
        cache_seqlens: T.Tensor([batch], T.int32),
        Output: T.Tensor([num_tokens, heads, dim], T.float16),
    ):
        with T.Kernel(T.ceildiv(tokens_per_seq, block_M), heads, batch, threads=threads) as (bx, by, bz):
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
            start_q = bz * tokens_per_seq + bx * block_M
            kv_offset = cache_seqlens[bz] - tokens_per_seq

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
                # Load K tile from 4D paged cache, page by page
                for p in T.serial(pages_per_tile):
                    logical_page = T.floordiv(k * block_N, page_block_size) + p
                    phys = block_table[bz, logical_page]
                    po = p * page_block_size
                    for i, j in T.Parallel(page_block_size, dim):
                        K_shared[po + i, j] = K_cache[phys, i, kv_head, j]

                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        q_pos = bx * block_M + i
                        kv_pos = k * block_N + j
                        acc_s[i, j] = T.if_then_else(
                            kv_pos <= q_pos + kv_offset,
                            0, -T.infinity(acc_s.dtype)
                        )
                else:
                    T.clear(acc_s)

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=GemmWarpPolicy.FullRow)
                T.copy(m_i, m_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
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

                # Load V tile from 4D paged cache
                for p in T.serial(pages_per_tile):
                    logical_page2 = T.floordiv(k * block_N, page_block_size) + p
                    phys2 = block_table[bz, logical_page2]
                    po2 = p * page_block_size
                    for i, j in T.Parallel(page_block_size, dim):
                        V_shared[po2 + i, j] = V_cache[phys2, i, kv_head, j]

                for i, j in T.Parallel(block_M, block_N):
                    P_shared[i, j] = T.cast(acc_s[i, j], T.float16)
                T.copy(P_shared, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=GemmWarpPolicy.Square)

            for i, j in T.Parallel(block_M, dim):
                if start_q + i < (bz + 1) * tokens_per_seq:
                    Output[start_q + i, by, j] = T.cast(acc_o[i, j] / l_i[i], T.float16)

    return main


_paged_kernel_4d = tilelang.jit(out_idx=[5])(_paged_kernel_func_4d)


def make_paged_from_dense(Q, K, V, page_block_size=16):
    """Convert dense 4D Q/K/V to 4D paged format for testing (vLLM-compatible)."""
    B, H, M, D = Q.shape
    N = K.shape[2]
    q_flat = Q.permute(0, 2, 1, 3).reshape(B * M, H, D).contiguous()
    pages_per_seq = int(math.ceil(N / page_block_size))
    num_pages = B * pages_per_seq
    k_cache = torch.zeros(num_pages, page_block_size, H, D, dtype=torch.float16, device='cuda')
    v_cache = torch.zeros(num_pages, page_block_size, H, D, dtype=torch.float16, device='cuda')
    for i in range(B):
        for j in range(N):
            page = j // page_block_size
            offset = j % page_block_size
            k_cache[i * pages_per_seq + page, offset, :, :] = K[i, :, j, :]
            v_cache[i * pages_per_seq + page, offset, :, :] = V[i, :, j, :]
    block_table = torch.arange(num_pages, dtype=torch.int32, device='cuda').view(B, pages_per_seq)
    seq_lens = torch.full((B,), N, dtype=torch.int32, device='cuda')
    return dict(
        Q_flat=q_flat, K_cache=k_cache, V_cache=v_cache,
        block_table=block_table, seq_lens=seq_lens,
        num_tokens=B * M, num_pages=num_pages, tokens_per_seq=M,
    )
