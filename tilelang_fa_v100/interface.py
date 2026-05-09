"""High-level FA2-compatible API for TileLang FlashAttention V100."""
import math
import torch
import torch.nn.functional as F

from ._kernels_forward import tilelang_forward, tilelang_forward_lse, kernel_forward_lse
from ._kernels_backward import tilelang_backward
from ._paged_adapter import paged_forward as _paged_forward


class FlashAttnTileLangVFunc(torch.autograd.Function):
    """Autograd function for TileLang FlashAttention (training)."""

    @staticmethod
    def forward(ctx, q, k, v, dropout_p, softmax_scale, causal, window_size,
                softcap, alibi_slopes, deterministic, return_softmax):
        layout = q.shape  # (B, M, H, D)
        q_ = q.permute(0, 2, 1, 3).contiguous()
        k_ = k.permute(0, 2, 1, 3).contiguous()
        v_ = v.permute(0, 2, 1, 3).contiguous()
        B, H, M, D = q_.shape
        N = k_.shape[2]

        if dropout_p != 0.0:
            raise NotImplementedError("dropout not supported")
        if alibi_slopes is not None:
            raise NotImplementedError("alibi not supported")
        if softcap != 0.0:
            raise NotImplementedError("softcap not supported")

        scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)
        window_left, window_right = window_size if window_size is not None else (-1, -1)

        fwd_kernel = tilelang_forward_lse(B, H, M, N, D, causal)
        out_, lse_ = fwd_kernel(q_, k_, v_)

        out = out_.permute(0, 2, 1, 3).contiguous()

        ctx.save_for_backward(q_, k_, v_, out_, lse_)
        ctx.causal = causal
        ctx.softmax_scale = scale

        if return_softmax:
            return out, lse_, None
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        q_, k_, v_, out_, lse_ = ctx.saved_tensors
        dout_ = dout.permute(0, 2, 1, 3).contiguous() if dout.dim() == 4 else dout
        dq_, dk_, dv_ = tilelang_backward(q_, k_, v_, out_, dout_, lse_, is_causal=ctx.causal)
        dq = dq_.permute(0, 2, 1, 3)
        dk = dk_.permute(0, 2, 1, 3)
        dv = dv_.permute(0, 2, 1, 3)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def tilelang_flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None,
                              causal=False, window_size=(-1, -1), softcap=0.0,
                              alibi_slopes=None, deterministic=False,
                              return_attn_probs=False):
    """FA2-compatible flash attention function.

    Args:
        q, k, v: [B, M, H, D] (FA2 layout)
        causal: causal masking
        softmax_scale: (default: 1/sqrt(D))
        return_attn_probs: return (out, lse, None) instead of just out

    Returns:
        out: [B, M, H, D] fp16
        or (out, lse, None) if return_attn_probs
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    return FlashAttnTileLangVFunc.apply(
        q, k, v, dropout_p, softmax_scale, causal,
        window_size, softcap, alibi_slopes, deterministic, return_attn_probs,
    )


def tilelang_paged_forward(q, k_cache, v_cache, block_table, seq_lens,
                            query_start_loc, prefix_kv_lens, out=None,
                            block_size=16, num_kv_heads=None,
                            softmax_scale=None, causal=True):
    """Paged FlashAttention forward for vLLM integration.

    Args match the reference FA-V100 paged_fwd API.
    """
    return _paged_forward(
        q, k_cache, v_cache, block_table, seq_lens,
        query_start_loc, prefix_kv_lens, out=out,
        block_size=block_size, num_kv_heads=num_kv_heads,
        softmax_scale=softmax_scale, causal=causal,
    )


tilelang_flash_attn_gpu = tilelang_flash_attn_func
