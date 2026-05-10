import warnings, torch; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '/home/rah/tilelang-fa-v100')
import tilelang, tilelang.language as T

def mk():
    nt = T.dynamic('nt')
    @T.prim_func
    def k(Q: T.Tensor([nt, 8, 128], T.float16), mt: T.int32,
          O: T.Tensor([nt, 8, 128], T.float16)):
        with T.Kernel(T.ceildiv(mt, 32), 8, 1, threads=256) as (bx, by, bz):
            Qs = T.alloc_shared([32, 128], T.float16)
            sq = bz * mt + bx * 32
            T.copy(Q[sq: sq + 32, by, :], Qs)
            for i, j in T.Parallel(32, 128):
                if sq + i < nt:
                    O[sq + i, by, j] = Qs[i, j]
    return k

kt = tilelang.jit(out_idx=[2])(mk).compile()
Q = torch.randn(256, 8, 128, dtype=torch.float16, device='cuda')
O = kt(Q, 256)
e0 = (Q[0] - O[0]).abs().max().item()
e255 = (Q[255] - O[255]).abs().max().item()
d = (O[0] - O[128]).abs().max().item()
print(f'Q[0]==O[0]: {e0:.6f}')
print(f'Q[255]==O[255]: {e255:.6f}')
print(f'O[0]!=O[128]: {d:.6f}')
print('COPY OK' if d > 0.01 else 'COPY FAIL')
