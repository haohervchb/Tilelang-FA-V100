# tilelang-fa-v100

**FlashAttention for NVIDIA V100 (SM70), written entirely in [TileLang](https://github.com/tile-ai/tilelang).**

[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-%3E%3D2.5-orange)](https://pytorch.org)
[![TileLang](https://img.shields.io/badge/tilelang-%3E%3D0.1.9-green)](https://github.com/tile-ai/tilelang)

This package provides **dense forward/backward** (for training) and **paged forward** (for vLLM-style inference) FlashAttention kernels targeting the V100's **SM70** architecture. Every kernel is expressed in TileLang's tensor-IR DSL and compiled to CUDA at runtime — no hand-written CUDA C++ required.

The API is a **drop-in replacement** for `flash_attn_v100` (`flash-attention-v100-ai-bond`), making it easy to swap into existing training and inference pipelines.

---

## Features

- **Dense FlashAttention forward + backward** — full training loop with `torch.autograd.Function`
- **Paged FlashAttention forward** — 4D page-by-page KV cache loading for vLLM integration
- **Causal masking** — diagonal-length-aware for non-square Q/KV
- **GQA / MQA support** — via the paged kernel (`num_kv_heads < num_heads`)
- **Autotuned tile configs** — `@tilelang.autotune` sweeps `block_M`, `block_N`, `threads` within V100's 96 KB shared memory budget
- **Dynamic tensor shapes** — the paged kernel compiles once per `(heads, dim, block_size, causal)` and handles variable sequence lengths via `T.dynamic`
- **API-compatible** — same function signatures as `tilelang_flash_attn_func`, `tilelang_flash_attn_gpu`, `tilelang_paged_forward`

---

## Installation

```bash
pip install tilelang-fa-v100
```

Requires `tilelang>=0.1.9` and `torch>=2.5.0`, both installed automatically.

---

## Usage

### Dense forward (inference)

```python
import torch
from tilelang_fa_v100 import tilelang_flash_attn_func

q = torch.randn(1, 512, 16, 64, dtype=torch.float16, device='cuda')
k = torch.randn(1, 512, 16, 64, dtype=torch.float16, device='cuda')
v = torch.randn(1, 512, 16, 64, dtype=torch.float16, device='cuda')

out = tilelang_flash_attn_func(q, k, v, causal=False)
```

### Dense forward/backward (training)

```python
from tilelang_fa_v100 import tilelang_flash_attn_func

q = torch.randn(1, 512, 16, 64, dtype=torch.float16, device='cuda', requires_grad=True)
k = torch.randn(1, 512, 16, 64, dtype=torch.float16, device='cuda', requires_grad=True)
v = torch.randn(1, 512, 16, 64, dtype=torch.float16, device='cuda', requires_grad=True)

out = tilelang_flash_attn_func(q, k, v, causal=True)
loss = out.sum()
loss.backward()
```

### Paged forward (vLLM inference)

```python
from tilelang_fa_v100 import tilelang_paged_forward

qf = torch.randn(total_tokens, num_heads, dim, dtype=torch.float16, device='cuda')
kc = torch.randn(num_pages, page_size, num_kv_heads, dim, dtype=torch.float16, device='cuda')
vc = torch.randn_like(kc)
bt = torch.arange(num_pages, dtype=torch.int32, device='cuda').view(batch, -1)
sl = torch.full((batch,), seq_len, dtype=torch.int32, device='cuda')
qsl = torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device='cuda')
pkl = torch.full((batch,), 0, dtype=torch.int32, device='cuda')

out, softmax_lse = tilelang_paged_forward(qf, kc, vc, bt, sl, qsl, pkl)
```

---

## API

```python
tilelang_flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None,
                         causal=False, window_size=(-1, -1), softcap=0.0,
                         alibi_slopes=None, deterministic=False,
                         return_attn_probs=False) -> torch.Tensor | tuple
```

FA2-compatible. Input shapes `[B, M, H, D]` (batch, seq, heads, dim). Returns `[B, M, H, D]` or `(out, lse, None)`.

```python
tilelang_paged_forward(q, k_cache, v_cache, block_table, seq_lens,
                       query_start_loc, prefix_kv_lens, out=None,
                       block_size=16, num_kv_heads=None,
                       softmax_scale=None, causal=True) -> (torch.Tensor, torch.Tensor)
```

vLLM-compatible paged attention. Returns `(output, softmax_lse)`.

`tilelang_flash_attn_gpu` is an alias for `tilelang_flash_attn_func`.

---

## Architecture

All kernels follow the standard **FlashAttention online-softmax tiling** pattern:

1. Load a tile of Q into shared memory.
2. Iterate over KV tiles; for each tile:
   - Load K from global → shared, compute `S = Q @ K^T` with warp-level GEMM.
   - Online safe softmax: track running `m_i` (row max) and `l_i` (row sum), rescale `O` and `l_i`.
   - Load V from global → shared, compute `O += softmax(S) @ V`.
3. Write `O = O / l_i` back to global memory.

The backward pass uses the saved softmax LSE and recomputes attention probabilities on the fly to avoid storing the full `M × N` matrix.

### V100-specific tuning

- **Shared memory bound**: V100 (SM70) has **96 KB** shared memory per block. The autotuner (`_configs.py`) prunes configurations that exceed `MAX_SMEM = 86000` bytes.
- **Warp layout**: `GemmWarpPolicy.FullRow` / `Square` schedules are validated with `_is_valid_gemm2` to ensure the warp arrangement fits the GEMM shape on SM70.
- **No async copy**: SM70 lacks `cp.async`; all copies use synchronous `T.copy` with `num_stages=0`.

### Paged kernel (vLLM)

The paged kernel reads KV caches stored as a 4D tensor `[num_pages, page_block_size, heads_kv, dim]` — the same layout vLLM uses. Physical page indices are resolved through a `block_table`, and each tile loads its KV data **page-by-page** inside the kernel:

```python
for p in T.serial(pages_per_tile):
    logical_page = k * block_N // page_block_size + p
    phys = block_table[bz, logical_page]
    K_shared[po + i, j] = K_cache[phys, i, kv_head, j]
```

This correctly handles **scattered (non-consecutive) physical pages** in vLLM's allocator. The kernel uses `T.dynamic` for the total number of query tokens, compiling once and dispatching for any sequence length.

---

## Performance

Numbers below are from a V100-SXM2 (vs `flash_attn_v100`, Tesla T4 kernel built for SM70). Measured with `torch.float16`, FP32 accumulation.

### Dense forward (ms, lower is better)

| Problem | TileLang | Reference | Speedup |
|---|---|---|---|
| B=1 H=16 512×512 D=64 | — | — | — |
| B=1 H=16 1024×1024 D=64 | — | — | — |
| B=1 H=16 2048×2048 D=64 | — | — | — |
| B=1 H=16 1024×1024 D=128 | — | — | — |

Run `python tests/test_compare_ref.py` on your hardware for live results.

---

## Why TileLang?

TileLang is a Python DSL that compiles tensor programs to CUDA kernels via TVM's TIR infrastructure. Writing FlashAttention in TileLang means:

- **No handwritten CUDA** — kernel logic stays in Python, easy to read and modify.
- **Autotuning** — `@tilelang.autotune` sweeps tile sizes at import time to find the fastest configuration for the specific GPU.
- **Rapid iteration** — changes to tiling strategies, masking, or memory layout take seconds to test.

---

## Related Projects

- **[flash-attention](https://github.com/Dao-AILab/flash-attention)** — the original FlashAttention (Hopper/Ampere).
- **[flash-attention-v100-ai-bond](https://github.com/AI-Bond/flash-attention-v100-ai-bond)** — reference CUDA implementation for V100; this project's API is compatible with it.
- **[TileLang](https://github.com/tile-ai/tilelang)** — the DSL that makes this package possible.

---

## Acknowledgments

Special thanks to the **TileLang team and community** for building an incredible DSL that makes GPU kernel development accessible and productive. This project would not exist without their work on the TileLang compiler, autotuning infrastructure, and SM70 support.

Thanks also to the **FlashAttention authors** (Tri Dao et al.) for the algorithmic breakthroughs, and to the **AI-Bond / flash-attention-v100-ai-bond** project for providing a well-tested V100 reference implementation that guided our development.

---

## License

MIT
