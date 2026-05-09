#!/usr/bin/env python3
"""Head-to-head: TileLang FA vs reference FA-V100 for correctness and speed."""
import sys, os, torch, math, torch.nn.functional as F, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tilelang_fa_v100
from tilelang_fa_v100._kernels_forward import kernel_forward as tl_fwd_kt
from tilelang_fa_v100._kernels_paged import _paged_kernel as tl_paged_kt, make_paged_from_dense
try:
    import flash_attn_v100_cuda as ref_cuda
    HAS_REF = True
except ImportError:
    HAS_REF = False


def pytorch_ref(q, k, v, causal):
    """PyTorch reference attention."""
    B, H, M, D = q.shape
    N = k.shape[2]
    scale = 1.0 / math.sqrt(D)
    scores = torch.einsum('bhqd,bhkd->bhqk', q.float(), k.float()) * scale
    if causal:
        mask = torch.tril(torch.ones(M, N, device='cuda'), diagonal=N - M)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1, dtype=torch.float32)
    return torch.einsum('bhqk,bhkd->bhqd', attn, v.float()).half()

def bench(fn, warmup=10, rep=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep * 1000

print("=" * 80)
print("  TILELANG-FA-V100 vs REFERENCE FA-V100 VALIDATION")
print("=" * 80)

# =====================================================
# PART 1: Dense forward correctness
# =====================================================
print("\n--- PART 1: Dense Forward Correctness ---")
fwd_configs = [
    (1, 16, 256, 256, 64, False),
    (1, 16, 512, 512, 64, False),
    (1, 16, 1024, 1024, 64, False),
    (1, 16, 512, 512, 128, False),
    (1, 16, 1024, 1024, 128, False),
    (1, 16, 256, 256, 256, False),
    (1, 16, 256, 256, 64, True),
    (1, 16, 512, 512, 128, True),
]
fwd_pass = 0
for B, H, M, N, D, causal in fwd_configs:
    torch.manual_seed(42)
    q = torch.randn(B, H, M, D, dtype=torch.float16, device='cuda')
    k = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
    v = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
    scale = 1.0 / math.sqrt(D)
    bM, bN, thr = (32, 128, 256) if D <= 64 else (32, 64, 256) if D <= 128 else (16, 64, 128)
    kt = tl_fwd_kt.compile(batch=B, heads=H, seq_q=M, seq_kv=N, dim=D, is_causal=causal,
                           block_M=bM, block_N=bN, threads=thr)
    tl_out = kt(q, k, v)
    if HAS_REF:
        ref_out = ref_cuda.fwd(q, k, v, None, None, 0.0, scale, causal, -1, -1, 0.0, False, None)[0]
        err = (tl_out - ref_out).abs().max().item()
        ok = err < 0.01
        print(f"  B={B} H={H} {M}x{N} D={D} causal={causal}  err={err:.6f}  {'PASS' if ok else 'FAIL'}")
        fwd_pass += ok
    else:
        print(f"  B={B} H={H} {M}x{N} D={D} causal={causal}  (no ref)")

print(f"  Forward: {fwd_pass}/{len(fwd_configs) if HAS_REF else 'skip'} passed")

# =====================================================
# PART 2: Paged forward correctness
# =====================================================
print("\n--- PART 2: Paged Forward Correctness (TileLang vs Ref paged_fwd) ---")
paged_configs = [
    (1, 16, 128, 256, 64, False, 32, 128, 128, 128),
    (1, 16, 256, 256, 64, False, 32, 128, 128, 128),
    (1, 16, 512, 512, 64, False, 32, 128, 128, 128),
    (1, 16, 256, 256, 128, False, 32, 64, 128, 128),
    (1, 16, 128, 256, 64, True, 32, 128, 128, 128),
    (1, 16, 128, 512, 256, False, 16, 64, 64, 64),
    (2, 16, 64, 256, 64, False, 32, 128, 128, 128),
    (1, 16, 256, 512, 64, True, 32, 128, 128, 128),
]
paged_pass = 0
paged_total = 0
for B, H, M, N, D, causal, bM, bN, thr, pbs in paged_configs:
    torch.manual_seed(42)
    q = torch.randn(B, H, M, D, dtype=torch.float16, device='cuda')
    k = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
    v = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
    scale = 1.0 / math.sqrt(D)
    paged_total += 1

    # PyTorch reference
    ref_O = pytorch_ref(q, k, v, causal)

    # TileLang paged
    pd = make_paged_from_dense(q, k, v, page_block_size=pbs)
    pkt = tl_paged_kt.compile(
        batch=B, heads=H, heads_kv=H, dim=D,
        page_block_size=pbs,
        max_blocks_per_seq=pd['block_table'].size(1),
        num_pages=pd['num_pages'], num_tokens=pd['num_tokens'],
        tokens_per_seq=pd['tokens_per_seq'], is_causal=causal,
        block_M=bM, block_N=bN, threads=thr or 256,
    )
    tl_O = pkt(pd['Q_flat'], pd['K_cache'], pd['V_cache'],
               pd['block_table'], pd['seq_lens'])
    tl_O_4d = tl_O.reshape(B, M, H, D).permute(0, 2, 1, 3)

    err = (tl_O_4d - ref_O).abs().max().item()
    ok = err < 0.01
    print(f"  B={B} H={H} {M}x{N} D={D} causal={causal} pb={pbs}  err={err:.6f}  {'PASS' if ok else 'FAIL'}")
    paged_pass += ok

print(f"  Paged: {paged_pass}/{len(paged_configs)} passed")

# =====================================================
# PART 3: Performance
# =====================================================
print("\n--- PART 3: Performance Benchmarks ---")

print("\n-- 3a. Dense Forward (TileLang vs Ref, ms) --")
perf_configs = [
    (1, 16, 512, 512, 64, False, 32, 128, 256),
    (1, 16, 1024, 1024, 64, False, 32, 128, 256),
    (1, 16, 2048, 2048, 64, False, 32, 128, 256),
    (1, 16, 1024, 1024, 128, False, 32, 64, 256),
    (1, 16, 2048, 2048, 128, False, 32, 64, 256),
    (1, 16, 512, 512, 256, False, 16, 64, 128),
]
print(f"  {'Problem':<38s} {'TL(ms)':<10s} {'Ref(ms)':<10s} {'Speedup':<8s}")
for B, H, M, N, D, causal, bM, bN, thr in perf_configs:
    torch.manual_seed(42)
    q = torch.randn(B, H, M, D, dtype=torch.float16, device='cuda')
    k = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
    v = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
    scale = 1.0 / math.sqrt(D)
    kt = tl_fwd_kt.compile(batch=B, heads=H, seq_q=M, seq_kv=N, dim=D, is_causal=causal,
                           block_M=bM, block_N=bN, threads=thr)
    tl_ms = bench(lambda: kt(q, k, v))
    name = f"B={B} H={H} {M}x{N} D={D}"
    if HAS_REF:
        ref_ms = bench(lambda: ref_cuda.fwd(q, k, v, None, None, 0.0, scale, causal, -1, -1, 0.0, False, None))
        spd = ref_ms / tl_ms
        print(f"  {name:<38s} {tl_ms:<8.3f}ms {ref_ms:<8.3f}ms {spd:<7.2f}x")
    else:
        print(f"  {name:<38s} {tl_ms:<8.3f}ms {'N/A':<10s}")

print("\n-- 3b. Paged Forward (TileLang vs Ref paged_fwd, ms) --")
paged_perf_configs = [
    (1, 16, 512, 512, 64, False, 32, 128, 128),
    (1, 16, 1024, 1024, 64, False, 32, 128, 128),
    (1, 16, 2048, 2048, 64, False, 32, 128, 128),
    (1, 16, 1024, 1024, 128, False, 32, 64, 128),
    (1, 16, 2048, 2048, 128, False, 32, 64, 128),
]
print(f"  {'Problem':<38s} {'TL(ms)':<10s} {'Ref(ms)':<10s} {'Speedup':<8s}")
for B, H, M, N, D, causal, bM, bN, pbs in paged_perf_configs:
    torch.manual_seed(42)
    q = torch.randn(B, H, M, D, dtype=torch.float16, device='cuda')
    k = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
    v = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
    scale = 1.0 / math.sqrt(D)

    # TileLang paged
    pd = make_paged_from_dense(q, k, v, page_block_size=pbs)
    pkt = tl_paged_kt.compile(
        batch=B, heads=H, heads_kv=H, dim=D,
        page_block_size=pbs,
        max_blocks_per_seq=pd['block_table'].size(1),
        num_pages=pd['num_pages'], num_tokens=pd['num_tokens'],
        tokens_per_seq=pd['tokens_per_seq'], is_causal=causal,
        block_M=bM, block_N=bN, threads=256,
    )
    for _ in range(3):
        pkt(pd['Q_flat'], pd['K_cache'], pd['V_cache'], pd['block_table'], pd['seq_lens'])
    torch.cuda.synchronize()
    tl_ms = bench(lambda: pkt(pd['Q_flat'], pd['K_cache'], pd['V_cache'],
                              pd['block_table'], pd['seq_lens']))

    name = f"B={B} H={H} {M}x{N} D={D}"
    if HAS_REF:
        ref_bs = 16
        ref_np = int(math.ceil(N / ref_bs)) * B
        ref_kc = torch.zeros(ref_np, ref_bs, H, D, dtype=torch.float16, device='cuda')
        ref_vc = torch.zeros(ref_np, ref_bs, H, D, dtype=torch.float16, device='cuda')
        for i in range(B):
            for j in range(N):
                p = i * N + j
                ref_kc[p // ref_bs, p % ref_bs, :, :] = k[i, :, j, :]
                ref_vc[p // ref_bs, p % ref_bs, :, :] = v[i, :, j, :]
        ref_bt = torch.arange(ref_np, dtype=torch.int32, device='cuda').view(B, ref_np // B)
        ref_sl = torch.full((B,), N, dtype=torch.int32, device='cuda')
        ref_qsl = torch.arange(0, (B + 1) * M, M, dtype=torch.int32, device='cuda')
        ref_pkl = torch.full((B,), 0, dtype=torch.int32, device='cuda')
        ref_qf = q.permute(0, 2, 1, 3).reshape(B * M, H, D).contiguous()
        ref_out = torch.zeros_like(ref_qf)
        for _ in range(3):
            ref_cuda.paged_fwd(ref_qf, ref_kc, ref_vc, ref_bt, ref_sl,
                               ref_qsl, ref_pkl, ref_out, H, ref_bs, scale, causal)
        torch.cuda.synchronize()
        ref_ms = bench(lambda: ref_cuda.paged_fwd(ref_qf, ref_kc, ref_vc, ref_bt, ref_sl,
                                                  ref_qsl, ref_pkl, ref_out, H, ref_bs, scale, causal))
        spd = ref_ms / tl_ms
        print(f"  {name:<38s} {tl_ms:<8.3f}ms {ref_ms:<8.3f}ms {spd:<7.2f}x")
    else:
        print(f"  {name:<38s} {tl_ms:<8.3f}ms {'N/A':<10s}")

print("\n" + "=" * 80)
if HAS_REF:
    ok = fwd_pass == len(fwd_configs) and paged_pass == len(paged_configs)
    print(f"  CORRECTNESS: Forward {fwd_pass}/{len(fwd_configs)}, Paged {paged_pass}/{len(paged_configs)}")
    print(f"  VERDICT: {'ALL PASS - READY' if ok else 'ISSUES FOUND'}")
else:
    print(f"  Reference FA-V100 not found. Install with: pip install /path/to/flash-attention-v100-ai-bond/")
print("=" * 80)
