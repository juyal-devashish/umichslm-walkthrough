"""Learned per-head logit sinks for FULL-attention layers (GQA and MLA).

A logit sink adds one softmax competitor per head with logit ``s_h`` and an
all-zero value vector (gpt-oss; StreamingLLM 2309.17453; Gu et al. 2410.10781).
It drains attention mass like a sink token but needs no key, no value and no
sequence position, so the layer stays on the FA2 (varlen) path::

    y = softmax([q.K^T * scale, s_h]) @ [V, 0] = sigmoid(lse - s_h) * o

where ``o``/``lse`` are the ordinary attention output and log-sum-exp.

THE GRADIENT. ``flash_attn_func(..., return_attn_probs=True)`` hands back the
LSE, but flash-attn's autograd Functions IGNORE any gradient that reaches it
(``backward(ctx, dout, *args)``; checked against 2.8.3). Computing
``sigmoid(lse - s) * o`` on top of the public API therefore trains ``s_h`` but
drops the ``dlse`` term from dq/dk -- the part that teaches queries how much
mass to give the sink. :class:`_FlashSinkFunc` gets it exactly at no extra
attention cost. With ``sig = sigmoid(lse - s)``, ``dout = sig * dy`` and the
missing term is ``dlse = sig * (1 - sig) * <dy, o>``. FA2's backward uses
``out`` only for ``D = rowsum(dout * out)`` and computes
``dS = P * (dP - D)``; the lse term adds ``dlse * P``, i.e. it replaces ``D``
by ``D - dlse = sig^2 <dy, o> = rowsum(dout * y)``. So calling FA2's own
backward kernel with ``out := y`` (the sink-scaled output) is exact.

GENERAL LSE MERGES. swa_scribe joins out-of-sequence branches (sink bank,
scribes, block-sparse far keys, logit sinks) to the window branch by an exact
log-sum-exp merge, which needs a DIFFERENTIABLE window LSE. :func:`flash_attn_with_lse`
supplies it: same forward as flash_attn, and a backward that folds ``dlse`` into
FA2's kernel by padding the head dim by 8 -- ``out`` gets a column of ones and
``dout`` a column of ``-dlse`` (with zero q/k/v columns, so P and dP are
untouched), which makes the kernel's ``D = rowsum(dout * out)`` exactly
``D - dlse``. Exact for any downstream use of the LSE; costs a +8 head-dim
backward on the layers that need it (the SWA band, which is cheap).

SDPA/eager: :func:`extend_kv_with_sink` appends a zero key, a zero value and a
mask column biased by ``s_h`` -- the native joint softmax, used as the
correctness reference (tests/test_attn_sink.py).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

try:  # flash-attn internals (2.8.x): the same custom ops flash_attn_func wraps.
    from flash_attn.flash_attn_interface import (
        _wrapped_flash_attn_backward,
        _wrapped_flash_attn_forward,
        _wrapped_flash_attn_varlen_backward,
        _wrapped_flash_attn_varlen_forward,
    )

    _HAS_FLASH_INTERNALS = True
except Exception:  # pragma: no cover - CPU-only installs
    _HAS_FLASH_INTERNALS = False


def _row_scale(lse: torch.Tensor, sink: torch.Tensor, varlen: bool) -> torch.Tensor:
    """sigmoid(lse - s_h) laid out to broadcast against the flash output.

    Batched: lse [B,H,S] -> [B,S,H,1]. Varlen: lse [H,T] -> [T,H,1].
    """
    s = sink.to(torch.float32)
    if varlen:
        return torch.sigmoid(lse.float() - s.view(-1, 1)).transpose(0, 1).unsqueeze(-1)
    return torch.sigmoid(lse.float() - s.view(1, -1, 1)).transpose(1, 2).unsqueeze(-1)


class _FlashSinkFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sink, softmax_scale, causal, window, cu_seqlens, max_seqlen):
        varlen = cu_seqlens is not None
        if varlen:
            out, lse, _, rng = _wrapped_flash_attn_varlen_forward(
                q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, 0.0, softmax_scale,
                causal=causal, window_size_left=window[0], window_size_right=window[1],
                softcap=0.0, alibi_slopes=None, return_softmax=False,
            )
            lse = lse[:, : q.shape[0]]
        else:
            out, lse, _, rng = _wrapped_flash_attn_forward(
                q, k, v, 0.0, softmax_scale, causal=causal, window_size_left=window[0],
                window_size_right=window[1], softcap=0.0, alibi_slopes=None,
                return_softmax=False,
            )
        sig = _row_scale(lse, sink, varlen)
        y = (out.float() * sig).to(out.dtype)
        ctx.save_for_backward(q, k, v, out, y, lse, sig, rng, cu_seqlens)
        ctx.softmax_scale, ctx.causal, ctx.window = softmax_scale, causal, window
        ctx.max_seqlen, ctx.varlen = max_seqlen, varlen
        return y

    @staticmethod
    def backward(ctx, dy):
        q, k, v, out, y, lse, sig, rng, cu_seqlens = ctx.saved_tensors
        dout = (dy.float() * sig).to(q.dtype).contiguous()
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        if ctx.varlen:
            _wrapped_flash_attn_varlen_backward(
                dout, q, k, v, y, lse, dq, dk, dv, cu_seqlens, cu_seqlens,
                ctx.max_seqlen, ctx.max_seqlen, 0.0, ctx.softmax_scale, ctx.causal,
                ctx.window[0], ctx.window[1], 0.0, None, False, rng_state=rng,
            )
        else:
            _wrapped_flash_attn_backward(
                dout, q, k, v, y, lse, dq, dk, dv, 0.0, ctx.softmax_scale, ctx.causal,
                ctx.window[0], ctx.window[1], 0.0, None, False, rng_state=rng,
            )
        # d y / d s_h = -sig (1 - sig) o, summed over every row of that head.
        g = (dy.float() * out.float()).sum(-1, keepdim=True) * sig * (1.0 - sig)
        dsink = -g.squeeze(-1).reshape(-1, g.shape[-2]).sum(0)
        return dq, dk, dv, dsink, None, None, None, None, None


def flash_attn_with_sink(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window: Tuple[int, int] = (-1, -1),
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> torch.Tensor:
    """FA2 attention with a per-head logit sink and the exact gradient.

    Layout as flash_attn: q/k/v ``[B,S,H,D]`` (``[T,H,D]`` with ``cu_seqlens``),
    GQA-native K/V allowed; ``sink`` is ``[H]``. No dropout (the sink arms run
    attention_dropout=0, like every sweep arm). Head dims must be multiples of 8
    (the sweep's 64/128/192 are); V may be zero-padded by the caller, as MLA's
    flash path already does.
    """
    if not _HAS_FLASH_INTERNALS:
        raise RuntimeError("flash_attn_with_sink needs flash-attn 2.x internals")
    if q.shape[-1] % 8 != 0 or v.shape[-1] % 8 != 0:
        raise ValueError(f"head dims must be multiples of 8, got {q.shape[-1]}/{v.shape[-1]}")
    if (cu_seqlens is None) != (max_seqlen is None):
        raise ValueError("pass cu_seqlens and max_seqlen together")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    return _FlashSinkFunc.apply(
        q.contiguous(), k.contiguous(), v.contiguous(), sink, float(softmax_scale),
        bool(causal), (int(window[0]), int(window[1])), cu_seqlens,
        None if max_seqlen is None else int(max_seqlen),
    )


class _FlashLSEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, softmax_scale, causal, window, cu_seqlens, max_seqlen):
        varlen = cu_seqlens is not None
        if varlen:
            out, lse, _, rng = _wrapped_flash_attn_varlen_forward(
                q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, 0.0, softmax_scale,
                causal=causal, window_size_left=window[0], window_size_right=window[1],
                softcap=0.0, alibi_slopes=None, return_softmax=False,
            )
            lse = lse[:, : q.shape[0]]
        else:
            out, lse, _, rng = _wrapped_flash_attn_forward(
                q, k, v, 0.0, softmax_scale, causal=causal, window_size_left=window[0],
                window_size_right=window[1], softcap=0.0, alibi_slopes=None,
                return_softmax=False,
            )
        ctx.save_for_backward(q, k, v, out, lse, rng, cu_seqlens)
        ctx.softmax_scale, ctx.causal, ctx.window = softmax_scale, causal, window
        ctx.max_seqlen, ctx.varlen = max_seqlen, varlen
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, out, lse, rng, cu_seqlens = ctx.saved_tensors
        if dlse is None:
            pad = 0
        else:
            pad = 8
            # dlse [B,H,S] (batched) / [H,T] (varlen) -> a column aligned with out.
            col = (dlse.transpose(0, 1) if ctx.varlen else dlse.transpose(1, 2)).unsqueeze(-1)
            z = lambda t: torch.nn.functional.pad(t, (0, pad))  # noqa: E731
            q, k, v = z(q), z(k), z(v)
            ones = torch.ones_like(out[..., :1])
            out = torch.cat([out, ones, out.new_zeros(*out.shape[:-1], pad - 1)], dim=-1)
            dout = torch.cat(
                [dout, (-col).to(dout.dtype), dout.new_zeros(*dout.shape[:-1], pad - 1)], dim=-1
            )
        dout = dout.contiguous()
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        if ctx.varlen:
            _wrapped_flash_attn_varlen_backward(
                dout, q, k, v, out, lse, dq, dk, dv, cu_seqlens, cu_seqlens,
                ctx.max_seqlen, ctx.max_seqlen, 0.0, ctx.softmax_scale, ctx.causal,
                ctx.window[0], ctx.window[1], 0.0, None, False, rng_state=rng,
            )
        else:
            _wrapped_flash_attn_backward(
                dout, q, k, v, out, lse, dq, dk, dv, 0.0, ctx.softmax_scale, ctx.causal,
                ctx.window[0], ctx.window[1], 0.0, None, False, rng_state=rng,
            )
        if pad:
            dq, dk, dv = dq[..., :-pad], dk[..., :-pad], dv[..., :-pad]
        return dq, dk, dv, None, None, None, None, None


def flash_attn_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window: Tuple[int, int] = (-1, -1),
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``flash_attn(_varlen)_func(..., return_attn_probs=True)``'s (out, lse), with an
    LSE that carries its gradient. lse is ``[B,H,S]`` batched, ``[H,T]`` varlen (the
    layouts flash-attn returns). No dropout; head dims multiples of 8."""
    if not _HAS_FLASH_INTERNALS:
        raise RuntimeError("flash_attn_with_lse needs flash-attn 2.x internals")
    if q.shape[-1] % 8 != 0 or v.shape[-1] != q.shape[-1]:
        raise ValueError(f"need equal head dims, multiple of 8; got {q.shape[-1]}/{v.shape[-1]}")
    if (cu_seqlens is None) != (max_seqlen is None):
        raise ValueError("pass cu_seqlens and max_seqlen together")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5  # BEFORE any backward padding
    return _FlashLSEFunc.apply(
        q.contiguous(), k.contiguous(), v.contiguous(), float(softmax_scale), bool(causal),
        (int(window[0]), int(window[1])), cu_seqlens,
        None if max_seqlen is None else int(max_seqlen),
    )


def extend_kv_with_sink(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    sink: torch.Tensor,
    q_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SDPA/eager sink: one zero key/value column biased by ``s_h``.

    key/value ``[B,H,K,D]`` (already GQA-expanded); ``attention_mask`` additive
    ``[B or 1, 1 or H, q_len, K]`` or None (= plain causal, materialized here).
    Returns the extended key, value and a float mask ``[B,H,q_len,K+1]`` in the
    key dtype. Pass ``is_causal=False`` afterwards: causality lives in the mask.
    """
    bsz, nh, kv_len, _ = key_states.shape
    dev = key_states.device
    if attention_mask is None:
        allowed = torch.ones(q_len, kv_len, dtype=torch.bool, device=dev).tril(kv_len - q_len)
        attention_mask = torch.zeros(1, 1, q_len, kv_len, device=dev)
        attention_mask = attention_mask.masked_fill(~allowed, torch.finfo(torch.float32).min)
    mask = attention_mask.to(torch.float32).expand(bsz, nh, q_len, kv_len)
    col = sink.to(torch.float32).view(1, nh, 1, 1).expand(bsz, nh, q_len, 1)
    mask = torch.cat([mask, col], dim=-1).to(key_states.dtype)
    zk = key_states.new_zeros(bsz, nh, 1, key_states.shape[-1])
    zv = value_states.new_zeros(bsz, nh, 1, value_states.shape[-1])
    return torch.cat([key_states, zk], dim=2), torch.cat([value_states, zv], dim=2), mask
