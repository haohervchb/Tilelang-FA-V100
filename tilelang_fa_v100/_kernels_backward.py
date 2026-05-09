"""Dense FlashAttention backward kernels (SM70 V100). Autotuned."""
import math
import torch
import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy
from ._configs import get_backward_dq_configs, get_backward_dkv_configs


def _backward_dq_kernel_func(batch, heads, seq_q, seq_kv, dim, is_causal,
                              block_M=32, block_N=128, num_stages=0, threads=256):
    scale = (1.0 / dim) ** 0.5

    @T.prim_func
    def main(
        Q: T.Tensor([batch, heads, seq_q, dim], T.float16),
        K: T.Tensor([batch, heads, seq_kv, dim], T.float16),
        V: T.Tensor([batch, heads, seq_kv, dim], T.float16),
        dO: T.Tensor([batch, heads, seq_q, dim], T.float16),
        softmax_lse: T.Tensor([batch, heads, seq_q], T.float32),
        row_dot: T.Tensor([batch, heads, seq_q], T.float32),
        dQ: T.Tensor([batch, heads, seq_q, dim], T.float16),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], T.float16)
            dO_shared = T.alloc_shared([block_M, dim], T.float16)
            K_shared = T.alloc_shared([block_N, dim], T.float16)
            V_shared = T.alloc_shared([block_N, dim], T.float16)
            P_shared = T.alloc_shared([block_M, block_N], T.float16)

            acc_s = T.alloc_fragment([block_M, block_N], T.float32)
            acc_s_cast = T.alloc_fragment([block_M, block_N], T.float16)
            acc_dOV = T.alloc_fragment([block_M, block_N], T.float32)
            acc_dQ = T.alloc_fragment([block_M, dim], T.float32)

            lse = T.alloc_fragment([block_M], T.float32)
            rd = T.alloc_fragment([block_M], T.float32)

            T.copy(Q[bz, by, bx * block_M: (bx + 1) * block_M, :], Q_shared)
            T.copy(dO[bz, by, bx * block_M: (bx + 1) * block_M, :], dO_shared)
            T.copy(softmax_lse[bz, by, bx * block_M: (bx + 1) * block_M], lse)
            T.copy(row_dot[bz, by, bx * block_M: (bx + 1) * block_M], rd)

            T.fill(acc_dQ, 0)

            loop_end = (
                T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M, block_N))
                if is_causal
                else T.ceildiv(seq_kv, block_N)
            )

            for k in T.Pipelined(loop_end, num_stages=num_stages):
                T.copy(V[bz, by, k * block_N: (k + 1) * block_N, :], V_shared)
                T.clear(acc_dOV)
                T.gemm(dO_shared, V_shared, acc_dOV, transpose_B=True, policy=GemmWarpPolicy.FullRow)

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

                for i, j in T.Parallel(block_M, block_N):
                    p_val = T.exp(acc_s[i, j] * scale - lse[i])
                    acc_s[i, j] = p_val * (acc_dOV[i, j] - rd[i]) * scale

                for i, j in T.Parallel(block_M, block_N):
                    P_shared[i, j] = T.cast(acc_s[i, j], T.float16)
                T.copy(P_shared, acc_s_cast)

                T.gemm(acc_s_cast, K_shared, acc_dQ, policy=GemmWarpPolicy.Square)

            for i, j in T.Parallel(block_M, dim):
                dQ[bz, by, bx * block_M + i, j] = T.cast(acc_dQ[i, j], T.float16)

    return main


def _backward_dkv_kernel_func(batch, heads, seq_q, seq_kv, dim, is_causal,
                               block_M=32, block_N=128, num_stages=0, threads=256):
    scale = (1.0 / dim) ** 0.5

    @T.prim_func
    def main(
        Q: T.Tensor([batch, heads, seq_q, dim], T.float16),
        K: T.Tensor([batch, heads, seq_kv, dim], T.float16),
        V: T.Tensor([batch, heads, seq_kv, dim], T.float16),
        dO: T.Tensor([batch, heads, seq_q, dim], T.float16),
        softmax_lse: T.Tensor([batch, heads, seq_q], T.float32),
        row_dot: T.Tensor([batch, heads, seq_q], T.float32),
        dK: T.Tensor([batch, heads, seq_kv, dim], T.float16),
        dV: T.Tensor([batch, heads, seq_kv, dim], T.float16),
    ):
        with T.Kernel(T.ceildiv(seq_kv, block_M), heads, batch, threads=threads) as (bx, by, bz):
            K_shared = T.alloc_shared([block_M, dim], T.float16)
            V_shared = T.alloc_shared([block_M, dim], T.float16)
            Q_shared = T.alloc_shared([block_N, dim], T.float16)
            dO_shared = T.alloc_shared([block_N, dim], T.float16)
            P_shared_T = T.alloc_shared([block_M, block_N], T.float16)

            acc_s = T.alloc_fragment([block_N, block_M], T.float32)
            acc_s_cast = T.alloc_fragment([block_M, block_N], T.float16)
            acc_dOV = T.alloc_fragment([block_N, block_M], T.float32)
            acc_dK = T.alloc_fragment([block_M, dim], T.float32)
            acc_dV = T.alloc_fragment([block_M, dim], T.float32)

            lse = T.alloc_fragment([block_N], T.float32)
            rd = T.alloc_fragment([block_N], T.float32)

            start_kv = bx * block_M
            T.copy(K[bz, by, start_kv: start_kv + block_M, :], K_shared)
            T.copy(V[bz, by, start_kv: start_kv + block_M, :], V_shared)

            T.fill(acc_dK, 0)
            T.fill(acc_dV, 0)

            if is_causal:
                first_q_tile = T.max(0, T.ceildiv(start_kv - block_N + 1, block_N))
                num_q_tiles = T.ceildiv(seq_q, block_N) - first_q_tile
            else:
                first_q_tile = 0
                num_q_tiles = T.ceildiv(seq_q, block_N)

            for k in T.Pipelined(num_q_tiles, num_stages=num_stages):
                q_tile_idx = k + first_q_tile
                start_q = q_tile_idx * block_N

                T.copy(Q[bz, by, start_q: start_q + block_N, :], Q_shared)

                if is_causal:
                    for i, j in T.Parallel(block_N, block_M):
                        q_idx = start_q + i
                        kv_idx = start_kv + j
                        acc_s[i, j] = T.if_then_else(
                            kv_idx <= q_idx + (seq_kv - seq_q),
                            0, -T.infinity(acc_s.dtype)
                        )
                else:
                    T.clear(acc_s)

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=GemmWarpPolicy.FullRow)

                T.copy(dO[bz, by, start_q: start_q + block_N, :], dO_shared)
                T.copy(softmax_lse[bz, by, start_q: start_q + block_N], lse)
                T.copy(row_dot[bz, by, start_q: start_q + block_N], rd)

                T.clear(acc_dOV)
                T.gemm(dO_shared, V_shared, acc_dOV, transpose_B=True, policy=GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_N, block_M):
                    p_val = T.exp(acc_s[i, j] * scale - lse[i])
                    dS_val = p_val * (acc_dOV[i, j] - rd[i]) * scale
                    P_shared_T[j, i] = T.cast(p_val, T.float16)
                    acc_s[i, j] = dS_val

                T.copy(P_shared_T, acc_s_cast)
                T.gemm(acc_s_cast, dO_shared, acc_dV, policy=GemmWarpPolicy.Square)
                for i, j in T.Parallel(block_N, block_M):
                    P_shared_T[j, i] = T.cast(acc_s[i, j], T.float16)
                T.copy(P_shared_T, acc_s_cast)
                T.gemm(acc_s_cast, Q_shared, acc_dK, policy=GemmWarpPolicy.Square)

            for i, j in T.Parallel(block_M, dim):
                dK[bz, by, start_kv + i, j] = T.cast(acc_dK[i, j], T.float16)
                dV[bz, by, start_kv + i, j] = T.cast(acc_dV[i, j], T.float16)

    return main


# Direct JIT
_kernel_dq = tilelang.jit(out_idx=[6])(_backward_dq_kernel_func)
_kernel_dkv = tilelang.jit(out_idx=[6, 7])(_backward_dkv_kernel_func)


@tilelang.autotune(configs=get_backward_dq_configs, warmup=10, rep=10, skip_check=True)
@tilelang.jit(out_idx=[6])
def _backward_dq_autotuned(batch, heads, seq_q, seq_kv, dim, is_causal,
                           block_M=32, block_N=128, num_stages=0, threads=256):
    return _backward_dq_kernel_func(batch, heads, seq_q, seq_kv, dim, is_causal,
                                    block_M, block_N, num_stages, threads)


@tilelang.autotune(configs=get_backward_dkv_configs, warmup=10, rep=10, skip_check=True)
@tilelang.jit(out_idx=[6, 7])
def _backward_dkv_autotuned(batch, heads, seq_q, seq_kv, dim, is_causal,
                            block_M=32, block_N=128, num_stages=0, threads=256):
    return _backward_dkv_kernel_func(batch, heads, seq_q, seq_kv, dim, is_causal,
                                     block_M, block_N, num_stages, threads)


def tilelang_backward(Q, K, V, O, dO, softmax_lse, is_causal=False, **kwargs):
    """Compute dQ, dK, dV for FlashAttention."""
    batch, heads, seq_q, dim = Q.shape
    row_dot = (O.float() * dO.float()).sum(dim=-1)
    dq_kernel = _backward_dq_autotuned(batch, heads, seq_q, K.shape[2], dim, is_causal, **kwargs)
    dkv_kernel = _backward_dkv_autotuned(batch, heads, seq_q, K.shape[2], dim, is_causal, **kwargs)
    dQ = dq_kernel(Q, K, V, dO, softmax_lse, row_dot)
    dK, dV = dkv_kernel(Q, K, V, dO, softmax_lse, row_dot)
    return dQ, dK, dV
