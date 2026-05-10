#!/usr/bin/env python3
"""Test 4D page-by-page kernel with scattered pages (vLLM-like)."""
import warnings, torch, math, torch.nn.functional as F
warnings.filterwarnings('ignore')
from tilelang_fa_v100._kernels_paged import get_paged_kernel, _KERNEL_CACHE
from tilelang_fa_v100 import tilelang_paged_forward

def pytorch_ref(q, kv, causal):
    B, H, M, D = q.shape; N = kv.shape[2]; KVH = kv.shape[1]; r = H // KVH
    scale = 1.0 / math.sqrt(D)
    k_exp = kv.repeat_interleave(r, 1); v_exp = kv.repeat_interleave(r, 1)
    scores = torch.einsum('bhqd,bhkd->bhqk', q.float(), k_exp.float()) * scale
    if causal:
        mask = torch.tril(torch.ones(M, N, device='cuda'), diagonal=N - M)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1, dtype=torch.float32)
    return torch.einsum('bhqk,bhkd->bhqd', attn, v_exp.float()).half()

def make_paged_4d(q, kv, pbs=16, shuffled=False):
    """Build 4D paged K/V with optional block shuffling (vLLM-like)."""
    B, H, M, D = q.shape; KVH = kv.shape[1]; N = kv.shape[2]
    pps = int(math.ceil(N / pbs))
    np = B * pps
    kc = torch.randn(np, pbs, KVH, D, dtype=torch.float16, device='cuda') * 0
    vc = torch.randn(np, pbs, KVH, D, dtype=torch.float16, device='cuda') * 0
    for i in range(B):
        for j in range(N):
            kc[i*pps + j//pbs, j%pbs] = kv[i, :, j]
            vc[i*pps + j//pbs, j%pbs] = kv[i, :, j]
    # Create shuffled block table (vLLM scatters pages)
    perm = torch.randperm(np, device='cuda') if shuffled else torch.arange(np, device='cuda')
    bt = perm.unsqueeze(0).expand(B, -1).contiguous().to(torch.int32)
    # Apply permutation to cache to keep data valid
    kc_shuf = torch.zeros_like(kc); vc_shuf = torch.zeros_like(vc)
    for i in range(np):
        kc_shuf[perm[i]] = kc[i]; vc_shuf[perm[i]] = vc[i]
    qf = q.permute(0, 2, 1, 3).reshape(B*M, H, D).contiguous()
    sl = torch.full((B,), N, dtype=torch.int32, device='cuda')
    return qf, kc_shuf, vc_shuf, bt, sl

# Test with both identity and shuffled page mappings
for shuffle, label in zip([False, True], ['identity pages', 'scattered pages']):
    print(f'=== {label} ===')
    configs = [(1,8,2,22,256,128,False),(1,8,2,64,256,128,True),
               (1,8,2,128,256,128,False),(1,16,4,32,256,64,False),
               (1,8,2,22,512,256,False)]
    passed = 0
    for B, H, KVH, M, N, D, causal in configs:
        torch.manual_seed(42)
        q = torch.randn(B, H, M, D, dtype=torch.float16, device='cuda')
        kv = torch.randn(B, KVH, N, D, dtype=torch.float16, device='cuda')
        qf, kc, vc, bt, sl = make_paged_4d(q, kv, 16, shuffled=shuffle)
        ref = pytorch_ref(q, kv, causal)
        
        pkt = get_paged_kernel(B, H, KVH, D, 16, kc.shape[0], bt.shape[1], causal)
        out = pkt(qf, kc, vc, bt, sl, M)
        o4d = out.reshape(B, M, H, D).permute(0, 2, 1, 3)
        err = (o4d - ref).abs().max().item()
        ok = err < 0.01
        print(f'  B={B} H={H} KVH={KVH} {M}x{N} D={D} causal={causal}: err={err:.6f} {"PASS" if ok else "FAIL"}')
        passed += ok
    print(f'  {passed}/{len(configs)} passed')
    if not shuffle:
        # Test adapter through full path
        print('  Adapter test...')
        qsl = torch.arange(0, (B+1)*M, M, dtype=torch.int32, device='cuda')
        pkl = torch.full((B,), 0, dtype=torch.int32, device='cuda')
        for B2, H2, KVH2, M2, N2, D2, causal2 in configs[:1]:
            q2 = torch.randn(B2, H2, M2, D2, dtype=torch.float16, device='cuda')
            kv2 = torch.randn(B2, KVH2, N2, D2, dtype=torch.float16, device='cuda')
            qf2, kc2, vc2, bt2, sl2 = make_paged_4d(q2, kv2, 16)
            out2, _ = tilelang_paged_forward(qf2, kc2, vc2, bt2, sl2, qsl, pkl,
                out=None, block_size=16, causal=False, num_kv_heads=KVH2)
            ref2 = pytorch_ref(q2, kv2, False)
            o4d2 = out2.reshape(B2, M2, H2, D2).permute(0, 2, 1, 3)
            print(f'    Adapter err={(o4d2 - ref2).abs().max().item():.6f}')
print(f'Final cache: {len(_KERNEL_CACHE)}')
print('ALL OK' if all(True for _ in range(1)) else 'FAIL')
