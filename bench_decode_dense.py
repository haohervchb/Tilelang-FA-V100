#!/usr/bin/env python3
"""Dense MHA decode benchmark using TileLang autotune vs SDPA vs FlashInfer.
   Uses the proven paged-prefill kernel pattern adapted for dense decode."""

import sys, os, math, time, argparse
import warnings
warnings.filterwarnings("ignore", message="Field.*duplicates")
warnings.filterwarnings("ignore", message=".*GemmSPWarpPolicy.*")

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_TL_HOME = os.path.join(os.path.expanduser("~"), "tilelang")
for p in [_TL_HOME, _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)

import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy

DEVICE = "cuda"
torch.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# TileLang MHA Decode Kernel (one-head-per-tile, dense K/V)
# ═══════════════════════════════════════════════════════════════════════════════

def _kernel_func(batch, heads, seqlen_kv, dim,
                 block_M=32, block_N=128, num_stages=0, threads=256):
    scale = (1.0 / dim) ** 0.5  # standard scale for T.exp

    @T.prim_func
    def main(
        Q: T.Tensor([batch, heads, dim], T.float16),
        K: T.Tensor([batch, seqlen_kv, heads, dim], T.float16),
        V: T.Tensor([batch, seqlen_kv, heads, dim], T.float16),
        Output: T.Tensor([batch, heads, dim], T.float16),
    ):
        # 3D grid matching the proven paged kernel pattern:
        # bx = head index, by = dummy seq dim (1), bz = batch
        with T.Kernel(heads, 1, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], T.float16)
            K_shared = T.alloc_shared([block_N, dim], T.float16)
            V_shared = T.alloc_shared([block_N, dim], T.float16)
            P_shared = T.alloc_shared([block_M, block_N], T.float16)

            acc_s = T.alloc_fragment([block_M, block_N], T.float32)
            acc_s_cast = T.alloc_fragment([block_M, block_N], T.float16)
            acc_o = T.alloc_fragment([block_M, dim], T.float32)
            m_i = T.alloc_fragment([block_M], T.float32)
            m_prev = T.alloc_fragment([block_M], T.float32)
            sf = T.alloc_fragment([block_M], T.float32)
            row_sum = T.alloc_fragment([block_M], T.float32)
            l_i = T.alloc_fragment([block_M], T.float32)

            T.copy(Q[bz, bx, :], Q_shared[0, :])
            T.fill(acc_o, 0)
            T.fill(m_i, -T.infinity(T.float32))
            T.fill(l_i, 0)

            for k in T.serial(T.ceildiv(seqlen_kv, block_N)):
                T.copy(K[bz, k * block_N: (k + 1) * block_N, bx, :], K_shared)
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

                T.copy(V[bz, k * block_N: (k + 1) * block_N, bx, :], V_shared)
                for i, j in T.Parallel(block_M, block_N):
                    P_shared[i, j] = T.cast(acc_s[i, j], T.float16)
                T.copy(P_shared, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=GemmWarpPolicy.Square)

            # Output only the first row (the actual Q token)
            for j in T.Parallel(dim):
                Output[bz, bx, j] = T.cast(acc_o[0, j] / l_i[0], T.float16)

    return main


_kernel_jit = tilelang.jit(out_idx=[3])(_kernel_func)
_KERNEL_CACHE = {}


def get_tl_kernel(batch, heads, seqlen_kv, dim, **cfg):
    key = (heads, dim, cfg["block_M"], cfg["block_N"], cfg["num_stages"], cfg["threads"])
    if key not in _KERNEL_CACHE:
        kt = _kernel_jit.compile(batch=batch, heads=heads, seqlen_kv=seqlen_kv, dim=dim, **cfg)
        _KERNEL_CACHE[key] = kt
    return _KERNEL_CACHE[key]


# ═══════════════════════════════════════════════════════════════════════════════
# Config generator (Volta SM70 MMA compatible only)
# ═══════════════════════════════════════════════════════════════════════════════
def _smem(block_M, block_N, dim):
    return (block_M * dim + 2 * block_N * dim + block_M * block_N) * 2


MAX_SMEM = 86000


def _valid_sm70(block_M, block_N, num_warps):
    M, N = block_M, block_N
    m_warp = num_warps
    if M % (m_warp * 16) != 0:
        max_m = M // 16
        if max_m == 0 or num_warps % max_m != 0:
            return False
        m_warp = max_m
        n_warp = num_warps // m_warp
    else:
        n_warp = 1
    warp_row_tiles = M // m_warp
    warp_col_tiles = N // n_warp
    return warp_row_tiles >= 16 and warp_col_tiles >= 16 and M % 16 == 0


def get_configs(dim):
    configs = []
    for block_M in [16, 32, 64]:
        for block_N in [64, 128, 256]:
            if _smem(block_M, block_N, dim) > MAX_SMEM:
                continue
            for threads in [128, 256]:
                num_warps = threads // 32
                if not _valid_sm70(block_M, block_N, num_warps):
                    continue
                configs.append(dict(
                    block_M=block_M, block_N=block_N,
                    num_stages=0, threads=threads,
                ))
    return configs


# ═══════════════════════════════════════════════════════════════════════════════
# Baselines
# ═══════════════════════════════════════════════════════════════════════════════

def ref_sdpa(q, k, v):
    B, H, D = q.shape
    q4 = q.view(B, H, 1, D)
    k4 = k.permute(0, 2, 1, 3).contiguous()
    v4 = v.permute(0, 2, 1, 3).contiguous()
    scale = 1.0 / math.sqrt(D)
    return F.scaled_dot_product_attention(q4, k4, v4, is_causal=False, scale=scale).squeeze(2).contiguous()


def ref_flashinfer(q, k, v):
    import flashinfer
    B = q.shape[0]
    D = q.shape[-1]
    sm = 1.0 / math.sqrt(D)
    k_nhd = k.permute(0, 2, 1, 3).contiguous()
    v_nhd = v.permute(0, 2, 1, 3).contiguous()
    outs = []
    for b in range(B):
        o, _ = flashinfer.single_decode_with_kv_cache(
            q[b:b+1], k_nhd[b], v_nhd[b], kv_layout="NHD", sm_scale=sm, use_tensor_cores=False)
        outs.append(o)
    return torch.cat(outs, dim=0)


def bench(fn, warmup=10, rep=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep * 1000.0


def assert_close(a, b, name="", rtol=0.02, atol=0.02):
    diff = (a.float() - b.float()).abs().max().item()
    max_b = b.float().abs().max().item() + 1e-8
    ok = diff < atol or diff / max_b < rtol
    status = "PASS" if ok else f"FAIL (max_diff={diff:.6f})"
    if name:
        print(f"  {name}: {status}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--kv-lens", type=str, default="1024,2048,4096,8192,16384")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--skip-autotune", action="store_true")
    parser.add_argument("--skip-flashinfer", action="store_true")
    parser.add_argument("--only-correctness", action="store_true")
    args = parser.parse_args()

    B, H, D = args.batch, args.heads, args.dim
    kv_lens = [int(x.strip()) for x in args.kv_lens.split(",")]

    print("=" * 80)
    print(f"  DENSE MHA DECODE BENCHMARK — TileLang vs SDPA vs FlashInfer")
    print(f"  batch={B}, heads={H}, dim={D}")
    print("=" * 80)

    # ── Correctness ──────────────────────────────────────────────────────────
    print("\n--- Correctness Check (kv_len=256, block_M=16, block_N=128, threads=128) ---")

    q = torch.randn(B, H, D, device=DEVICE, dtype=torch.float16)
    k256 = torch.randn(B, 256, H, D, device=DEVICE, dtype=torch.float16)
    v256 = torch.randn(B, 256, H, D, device=DEVICE, dtype=torch.float16)

    ref = ref_sdpa(q, k256, v256)

    cfg_test = dict(block_M=16, block_N=128, num_stages=0, threads=128)
    tl_kt = get_tl_kernel(B, H, 256, D, **cfg_test)
    tl_out = tl_kt(q, k256, v256)
    assert_close(tl_out, ref, "TileLang")

    if not args.skip_flashinfer:
        try:
            fi_out = ref_flashinfer(q, k256, v256)
            assert_close(fi_out, ref, "FlashInfer")
        except Exception as e:
            print(f"  FlashInfer: ERROR ({e})")

    if args.only_correctness:
        return

    # ── Autotune + Benchmark ─────────────────────────────────────────────────
    configs = get_configs(D)
    print(f"\n--- Autotune: {len(configs)} configs per kv_len ---")

    print(f"{'kv_len':>8s}  {'TL_best(ms)':>12s}  {'TL config':>28s}  {'SDPA(ms)':>10s}", end="")
    if not args.skip_flashinfer:
        print(f"  {'FlashInfer(ms)':>15s}", end="")
    print()

    for kv_len in kv_lens:
        q_kv = torch.randn(B, H, D, device=DEVICE, dtype=torch.float16)
        k_kv = torch.randn(B, kv_len, H, D, device=DEVICE, dtype=torch.float16)
        v_kv = torch.randn(B, kv_len, H, D, device=DEVICE, dtype=torch.float16)

        if not args.skip_autotune:
            best_lat, best_cfg = float("inf"), None
            for cfg in configs:
                try:
                    kt = get_tl_kernel(B, H, kv_len, D, **cfg)
                except Exception:
                    continue
                try:
                    for _ in range(args.warmup):
                        kt(q_kv, k_kv, v_kv)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(args.rep):
                        kt(q_kv, k_kv, v_kv)
                    torch.cuda.synchronize()
                    lat = (time.perf_counter() - t0) / args.rep * 1000.0
                    if lat < best_lat:
                        best_lat, best_cfg = lat, cfg
                except Exception:
                    continue
            tl_lat, tl_cfg = best_lat, best_cfg
        else:
            tl_cfg = dict(block_M=16, block_N=128, num_stages=0, threads=128)
            kt = get_tl_kernel(B, H, kv_len, D, **tl_cfg)
            tl_lat = bench(lambda: kt(q_kv, k_kv, v_kv), args.warmup, args.rep)

        sdpa_lat = bench(lambda: ref_sdpa(q_kv, k_kv, v_kv), args.warmup, args.rep)
        cfg_s = f"bM={tl_cfg.get('block_M')} bN={tl_cfg.get('block_N')} t={tl_cfg.get('threads')}"

        print(f"{kv_len:8d}  {tl_lat:12.4f}  {cfg_s:28s}  {sdpa_lat:10.4f}", end="")

        if not args.skip_flashinfer:
            try:
                fi_lat = bench(lambda: ref_flashinfer(q_kv, k_kv, v_kv), args.warmup, args.rep)
                print(f"  {fi_lat:15.4f}", end="")
            except Exception:
                print(f"  {'N/A':>15s}", end="")
        print()
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
