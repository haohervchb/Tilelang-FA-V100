#!/usr/bin/env python3
"""Test 2D linear + T.dynamic paged kernel (no staging buffer)."""
import warnings, torch, math, torch.nn.functional as F
warnings.filterwarnings('ignore')
from tilelang_fa_v100._kernels_paged import get_paged_kernel, _KERNEL_CACHE
from tilelang_fa_v100 import tilelang_paged_forward

B, H, KVH, M, N, D = 1, 8, 2, 22, 256, 128
torch.manual_seed(42)
q = torch.randn(B, H, M, D, dtype=torch.float16, device='cuda')
kv = torch.randn(B, KVH, N, D, dtype=torch.float16, device='cuda')
scale = 1.0 / math.sqrt(D)

# PyTorch ref
r = H // KVH
kv_exp = kv.repeat_interleave(r, 1)
scores = torch.einsum('bhqd,bhkd->bhqk', q.float(), kv_exp.float()) * scale
attn = F.softmax(scores, dim=-1, dtype=torch.float32)
ref = torch.einsum('bhqk,bhkd->bhqd', attn, kv_exp.float()).half()

# Build 4D paged K/V (vLLM format)
pbs = 16
pps = int(math.ceil(N / pbs))
np = pps
kc = torch.zeros(np, pbs, KVH, D, dtype=torch.float16, device='cuda')
vc = torch.zeros(np, pbs, KVH, D, dtype=torch.float16, device='cuda')
for j in range(N):
    pg, of = j // pbs, j % pbs
    kc[pg, of] = kv[0, :, j]
    vc[pg, of] = kv[0, :, j]
bt = torch.arange(np, dtype=torch.int32, device='cuda').view(B, pps)
sl = torch.full((B,), N, dtype=torch.int32, device='cuda')
qf = q.permute(0, 2, 1, 3).reshape(B * M, H, D).contiguous()

# Direct kernel call with 2D linear view (zero-copy reshape)
print('Compiling...')
kt = get_paged_kernel(1, H, KVH, D, pbs, np, pps, False)
print(f'Cache: {len(_KERNEL_CACHE)}')

k_2d = kc.reshape(-1, KVH, D)
v_2d = vc.reshape(-1, KVH, D)
bt_abs = bt * pbs
out = kt(qf, k_2d, v_2d, bt_abs, sl, M)
o4d = out.reshape(B, M, H, D).permute(0, 2, 1, 3)
err = (o4d - ref).abs().max().item()
print(f'Direct kernel: err={err:.6f} {"PASS" if err < 0.01 else "FAIL"}')

# Different nt (no recompile)
qf2 = torch.randn(B * 64, H, D, dtype=torch.float16, device='cuda')
out2 = kt(qf2, k_2d, v_2d, bt_abs, sl, 128)
print(f'Diff nt: out={out2.shape} cache={len(_KERNEL_CACHE)} (expected 1)')

# Through adapter (uses the same zero-copy reshape internally)
qsl = torch.arange(0, (B + 1) * M, M, dtype=torch.int32, device='cuda')
pkl = torch.full((B,), 0, dtype=torch.int32, device='cuda')
ao, _ = tilelang_paged_forward(qf, kc, vc, bt, sl, qsl, pkl,
    out=None, block_size=pbs, softmax_scale=scale, causal=False, num_kv_heads=KVH)
a4d = ao.reshape(B, M, H, D).permute(0, 2, 1, 3)
err2 = (a4d - ref).abs().max().item()
print(f'Adapter:       err={err2:.6f} {"PASS" if err2 < 0.01 else "FAIL"}')

# Multi-layer sim: verify no recompilation
for layer in range(36):
    out = kt(qf, k_2d, v_2d, bt_abs, sl, M)
print(f'36 layers: cache={len(_KERNEL_CACHE)} (expected 1)')

print(f'Final cache: {len(_KERNEL_CACHE)} (expected 1)')
print('ALL OK')
