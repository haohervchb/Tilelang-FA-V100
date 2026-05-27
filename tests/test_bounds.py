import torch
import math
from tilelang_fa_v100._kernels_paged import get_paged_kernel

B, Hq, Hkv, D = 1, 2, 1, 256
M = 19665
N = 19665
pbs = 16

q = torch.randn(B, Hq, M, D, dtype=torch.float16, device='cuda')
k = torch.randn(B, Hkv, N, D, dtype=torch.float16, device='cuda')
v = torch.randn(B, Hkv, N, D, dtype=torch.float16, device='cuda')

Q_flat = q.permute(0, 2, 1, 3).reshape(B * M, Hq, D).contiguous()

num_pages = math.ceil(N / pbs)
K_cache = torch.zeros(num_pages, pbs, Hkv, D, dtype=torch.float16, device='cuda')
V_cache = torch.zeros(num_pages, pbs, Hkv, D, dtype=torch.float16, device='cuda')
block_table = torch.arange(num_pages, dtype=torch.int32, device='cuda').reshape(1, -1)
seq_lens = torch.tensor([N], dtype=torch.int32, device='cuda')
query_start_loc = torch.tensor([0, M], dtype=torch.int32, device='cuda')
prefix_kv_lens = torch.tensor([0], dtype=torch.int32, device='cuda')

kernel = get_paged_kernel(
    batch=B, heads=Hq, heads_kv=Hkv, dim=D,
    block_size=pbs, num_pages=num_pages, max_blocks=num_pages, causal=True,
)

print("Running kernel...")
out = kernel(Q_flat, K_cache, V_cache, block_table, seq_lens, query_start_loc, prefix_kv_lens, M, 1.0 / math.sqrt(D))
torch.cuda.synchronize()
print("Done!")
