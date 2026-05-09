"""Dense FlashAttention forward kernels (SM70 V100). Autotuned via @tilelang.autotune."""
import math
import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy
from ._configs import get_forward_configs


def _forward_kernel_func(batch, heads, seq_q, seq_kv, dim, is_causal,
                         block_M=32, block_N=128, num_stages=0, threads=256):
    scale = (1.0 / dim) ** 0.5

    @T.prim_func
    def main(
        Q: T.Tensor([batch, heads, seq_q, dim], T.float16),
        K: T.Tensor([batch, heads, seq_kv, dim], T.float16),
        V: T.Tensor([batch, heads, seq_kv, dim], T.float16),
        Output: T.Tensor([batch, heads, seq_q, dim], T.float16),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
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

            T.copy(Q[bz, by, bx * block_M: (bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(m_i, -T.infinity(T.float32))
            T.fill(l_i, 0)

            loop_end = (
                T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M, block_N))
                if is_causal
                else T.ceildiv(seq_kv, block_N)
            )

            for k in T.Pipelined(loop_end, num_stages=num_stages):
                T.copy(K[bz, by, k * block_N: (k + 1) * block_N, :], K_shared)

                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i
                        kv_idx = k * block_N + j
                        acc_s[i, j] = T.if_then_else(
                            kv_idx <= q_idx + (seq_kv - seq_q),
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

                T.copy(V[bz, by, k * block_N: (k + 1) * block_N, :], V_shared)

                for i, j in T.Parallel(block_M, block_N):
                    P_shared[i, j] = T.cast(acc_s[i, j], T.float16)
                T.copy(P_shared, acc_s_cast)

                T.gemm(acc_s_cast, V_shared, acc_o, policy=GemmWarpPolicy.Square)

            for i, j in T.Parallel(block_M, dim):
                Output[bz, by, bx * block_M + i, j] = acc_o[i, j] / l_i[i]

    return main


def _forward_lse_kernel_func(batch, heads, seq_q, seq_kv, dim, is_causal,
                              block_M=32, block_N=128, num_stages=0, threads=256):
    scale = (1.0 / dim) ** 0.5

    @T.prim_func
    def main(
        Q: T.Tensor([batch, heads, seq_q, dim], T.float16),
        K: T.Tensor([batch, heads, seq_kv, dim], T.float16),
        V: T.Tensor([batch, heads, seq_kv, dim], T.float16),
        Output: T.Tensor([batch, heads, seq_q, dim], T.float16),
        softmax_lse: T.Tensor([batch, heads, seq_q], T.float32),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
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

            T.copy(Q[bz, by, bx * block_M: (bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(m_i, -T.infinity(T.float32))
            T.fill(l_i, 0)

            loop_end = (
                T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M, block_N))
                if is_causal
                else T.ceildiv(seq_kv, block_N)
            )

            for k in T.Pipelined(loop_end, num_stages=num_stages):
                T.copy(K[bz, by, k * block_N: (k + 1) * block_N, :], K_shared)

                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i
                        kv_idx = k * block_N + j
                        acc_s[i, j] = T.if_then_else(
                            kv_idx <= q_idx + (seq_kv - seq_q),
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
                T.copy(V[bz, by, k * block_N: (k + 1) * block_N, :], V_shared)
                for i, j in T.Parallel(block_M, block_N):
                    P_shared[i, j] = T.cast(acc_s[i, j], T.float16)
                T.copy(P_shared, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=GemmWarpPolicy.Square)

            for i, j in T.Parallel(block_M, dim):
                Output[bz, by, bx * block_M + i, j] = acc_o[i, j] / l_i[i]
            for i in T.Parallel(block_M):
                softmax_lse[bz, by, bx * block_M + i] = m_i[i] * scale + T.log(l_i[i])

    return main


# Direct JIT (no autotune, for testing)
kernel_forward = tilelang.jit(out_idx=[3])(_forward_kernel_func)
kernel_forward_lse = tilelang.jit(out_idx=[3, 4])(_forward_lse_kernel_func)


@tilelang.autotune(configs=get_forward_configs, warmup=10, rep=10)
@tilelang.jit(out_idx=[3])
def tilelang_forward(batch, heads, seq_q, seq_kv, dim, is_causal,
                     block_M=32, block_N=128, num_stages=0, threads=256):
    return _forward_kernel_func(batch, heads, seq_q, seq_kv, dim, is_causal,
                                block_M, block_N, num_stages, threads)


@tilelang.autotune(configs=get_forward_configs, warmup=10, rep=10)
@tilelang.jit(out_idx=[3, 4])
def tilelang_forward_lse(batch, heads, seq_q, seq_kv, dim, is_causal,
                          block_M=32, block_N=128, num_stages=0, threads=256):
    return _forward_lse_kernel_func(batch, heads, seq_q, seq_kv, dim, is_causal,
                                    block_M, block_N, num_stages, threads)
