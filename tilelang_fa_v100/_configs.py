"""Autotune config generators for all V100 TileLang kernels."""
from ._utils import _is_valid_gemm2, _smem_forward, _smem_backward, MAX_SMEM


def get_forward_configs(*args, **kwargs):
    """Generate config space for FA forward autotuning (dense)."""
    dim = args[4] if len(args) > 4 else kwargs.get('dim', 64)
    if not isinstance(dim, int):
        dim = 64
    configs = []
    block_M_vals = [16, 32, 64]
    block_N_vals = [64, 128, 256]
    if dim <= 32:
        block_N_vals.append(512)
    if dim >= 128:
        block_N_vals.extend([32, 48])
    for block_M in block_M_vals:
        for block_N in block_N_vals:
            if _smem_forward(block_M, block_N, dim) > MAX_SMEM:
                continue
            for threads in [64, 128, 256]:
                if threads < block_M:
                    continue
                if not _is_valid_gemm2(block_M, dim, threads // 32):
                    continue
                configs.append(dict(block_M=block_M, block_N=block_N, num_stages=0, threads=threads))
    return configs


def get_backward_dq_configs(*args, **kwargs):
    """Generate config space for backward dQ kernel autotuning."""
    dim = args[4] if len(args) > 4 else kwargs.get('dim', 64)
    if not isinstance(dim, int):
        dim = 64
    configs = []
    block_M_vals = [16, 32, 64]
    block_N_vals = [32, 64, 128, 256]
    if dim <= 32:
        block_N_vals.append(512)
    for block_M in block_M_vals:
        for block_N in block_N_vals:
            if _smem_backward(block_M, block_N, dim) > MAX_SMEM:
                continue
            for threads in [64, 128, 256]:
                if threads < block_M:
                    continue
                if not _is_valid_gemm2(block_M, dim, threads // 32):
                    continue
                configs.append(dict(block_M=block_M, block_N=block_N, num_stages=0, threads=threads))
    return configs


def get_backward_dkv_configs(*args, **kwargs):
    """Generate config space for backward dKV kernel autotuning."""
    return get_backward_dq_configs(*args, **kwargs)


def get_paged_configs(*args, **kwargs):
    """Generate config space for paged forward autotuning."""
    dim = args[3] if len(args) > 3 else kwargs.get('dim', 64)
    if not isinstance(dim, int):
        dim = 64
    configs = []
    block_M_vals = [16, 32, 64]
    block_N_vals = [32, 64, 128, 256]
    if dim <= 32:
        block_N_vals.extend([512])
    for block_M in block_M_vals:
        for block_N in block_N_vals:
            if _smem_forward(block_M, block_N, dim) > MAX_SMEM:
                continue
            for threads in [64, 128, 256]:
                if threads < block_M:
                    continue
                if not _is_valid_gemm2(block_M, dim, threads // 32):
                    continue
                for num_stages in [0, 1, 2]:
                    configs.append(dict(block_M=block_M, block_N=block_N, num_stages=num_stages, threads=threads))
    return configs
