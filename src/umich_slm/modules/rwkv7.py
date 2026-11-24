"""RWKV-7 "Goose" time-mix as a drop-in for the DeltaNet slots of a hybrid.

RWKV-7 (arXiv 2503.14456) is a generalized delta rule with VECTOR decay, a vector
in-context learning rate, and decoupled removal/replacement keys. Per head, with a
key-by-value state S:

    S_t = diag(exp(w_t)) S_{t-1} + b_t (a_t^T S_{t-1}) + k_t v_t^T
    o_t = S_t^T r_t

where w_t < 0 is the per-key-channel log decay, a_t = -kk_t (the normalized removal
key) and b_t = kk_t * alpha_t (the in-context learning rate). That is FLA's DPLR form
(``fla.ops.generalized_delta_rule`` / ``fla.ops.rwkv7.chunk_rwkv7``), and
``_reference_wkv`` below is the exact per-step recurrence it computes.

WHAT IS SWAPPED, AND WHAT IS NOT. Only the time-mix (the token mixer) replaces
GatedDeltaNet. RWKV-7's channel-mix (squared-ReLU FFN with its own token shift) is NOT
used: the block keeps our SwiGLU FFN, so a ``_rwkv7`` twin differs from its parent in
the recurrent mixer only. Two RWKV-7 pieces live inside the time-mix and are kept:

- token shift (a learned per-channel lerp toward the previous token) replaces our
  short depthwise conv; it resets at document starts like the conv does;
- the value residual ("v_first"): every RWKV layer after the first lerps its values
  toward the FIRST RWKV layer's values, with a learned data-dependent gate. That needs
  state shared across layers within one forward. It rides a plain dict
  (``_v_first_store``) bound by :func:`bind_value_residual`: the first RWKV layer
  overwrites ``store["v"]`` on every forward, later layers read it. Layers run in
  order and backward recomputation runs in reverse, so under non-reentrant gradient
  checkpointing every reader sees the value written in the same forward. With no
  store bound, or ``rwkv7_value_residual=False``, the residual is off and the
  ``v_lora`` gate is not created (so DDP sees no unused parameters).

INITIALIZATION is part of the architecture here (decay spectrum, zero o_proj, LoRA
biases); the model's generic ``_init_weights`` would overwrite it with normal(0.02) and
zero biases. :meth:`RWKV7Mixer.reset_buffers` re-applies it: ``_init_weights`` calls
``reset_buffers`` on every module after visiting its children, so the RWKV init lands
last. It is skipped for parameters already loaded from a checkpoint
(``_is_hf_initialized``).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deltanet import DeltaNetCache

RWKV7_FLA_AVAILABLE = False
try:
    from fla.ops.rwkv7 import chunk_rwkv7

    RWKV7_FLA_AVAILABLE = True
except ImportError:  # CPU / no FLA: the exact reference recurrence runs instead
    chunk_rwkv7 = None

# exp(-0.5): the decay ceiling of RWKV-7, w = -exp(-0.5) * sigmoid(.) in log space,
# so per-step decay lies in (0.545, 1). Also keeps FLA's safe_gate range [-5, 0).
_W_SCALE = math.exp(-0.5)


class _LoRA(nn.Module):
    """``up(act(down(x)))``: RWKV-7's low-rank data-dependent projections."""

    def __init__(self, dim_in, dim_out, rank, activation=None, bias=True):
        super().__init__()
        self.down = nn.Linear(dim_in, rank, bias=False)
        self.up = nn.Linear(rank, dim_out, bias=bias)
        self.act = {None: nn.Identity(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}[activation]

    def forward(self, x):
        return self.up(self.act(self.down(x)))

    @torch.no_grad()
    def init_rwkv(self, bias=None):
        nn.init.zeros_(self.down.weight)
        w = self.up.weight
        gain = math.sqrt(w.shape[1] / w.shape[0]) if w.shape[1] > w.shape[0] else 1.0
        w32 = torch.empty_like(w, dtype=torch.float32)
        nn.init.orthogonal_(w32, gain=0.1 * gain)
        w.copy_(w32.to(w.dtype))
        if self.up.bias is not None:
            if bias is None:
                self.up.bias.zero_()
            else:
                self.up.bias.copy_(bias.to(self.up.bias.dtype))


def _lora_rank(scale: float, hidden: int, factor: float = 1.0) -> int:
    return max(32, int(round(scale * hidden**0.5 * factor / 32) * 32))


def first_rwkv_layer(config) -> int:
    """Index of the first layer that resolves to an RWKV-7 mixer (owns v_first)."""
    from .transformer_block import _should_use_deltanet

    for i in range(int(config.num_hidden_layers)):
        if _should_use_deltanet(config, i):
            return i
    return 0


def bind_value_residual(model: nn.Module) -> int:
    """Share one v_first store across every RWKV7Mixer in ``model``.

    Call once after construction (and again after any module surgery). Returns the
    number of mixers bound. Idempotent.
    """
    store: dict = {}
    n = 0
    for m in model.modules():
        if isinstance(m, RWKV7Mixer):
            m._v_first_store = store
            n += 1
    return n


def _reference_wkv(r, w, k, v, a, b, state=None, doc_start=None):
    """Exact RWKV-7 / DPLR recurrence, fp32, per step.

    r, w, k, a, b: [B, T, H, K]; v: [B, T, H, V]; state: [B, H, K, V] or None;
    doc_start: [B, T] bool, True where a new document starts (state reset to 0).
    Returns (o [B, T, H, V] in v's dtype, final state [B, H, K, V] fp32).
    """
    B, T, H, K = r.shape
    V = v.shape[-1]
    r, w, k, v, a, b = (x.float() for x in (r, w, k, v, a, b))
    S = r.new_zeros(B, H, K, V) if state is None else state.float().clone()
    out = []
    for t in range(T):
        if doc_start is not None:
            keep = (~doc_start[:, t]).float().view(B, 1, 1, 1)
            S = S * keep
        sa = torch.einsum("bhk,bhkv->bhv", a[:, t], S)
        S = (
            S * w[:, t].exp().unsqueeze(-1)
            + b[:, t].unsqueeze(-1) * sa.unsqueeze(-2)
            + k[:, t].unsqueeze(-1) * v[:, t].unsqueeze(-2)
        )
        out.append(torch.einsum("bhk,bhkv->bhv", r[:, t], S))
    return torch.stack(out, dim=1).to(v.dtype), S


class RWKV7Mixer(nn.Module):
    """RWKV-7 time-mix with the GatedDeltaNet interface (``deltanet_variant="rwkv7"``).

    Reads the model-stashed ``_intra_doc_start`` ([B, L] bool) and the shared
    ``_delta_cu_seqlens`` exactly like GatedDeltaNet: the recurrent state and the
    token shift both reset at document starts. Head layout follows GatedDeltaNet
    (``deltanet_num_heads`` or ``num_attention_heads``) so a twin matches its parent's
    heads; RWKV-7's own default is head_dim 64.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = 0 if layer_idx is None else int(layer_idx)
        self.hidden_size = h = int(config.hidden_size)
        self.num_heads = int(
            getattr(config, "deltanet_num_heads", None) or config.num_attention_heads
        )
        if h % self.num_heads:
            raise ValueError(f"hidden_size {h} not divisible by num_heads {self.num_heads}")
        self.head_dim = h // self.num_heads
        self.num_layers = int(config.num_hidden_layers)
        self._intra_doc_start = None
        self._delta_cu_seqlens = None
        self._v_first_store: Optional[dict] = None

        self.is_first = self.layer_idx == first_rwkv_layer(config)
        self.value_residual = bool(getattr(config, "rwkv7_value_residual", True))

        factor = self.head_dim / 64
        self.x_r = nn.Parameter(torch.zeros(1, 1, h))
        self.x_w = nn.Parameter(torch.zeros(1, 1, h))
        self.x_k = nn.Parameter(torch.zeros(1, 1, h))
        self.x_v = nn.Parameter(torch.zeros(1, 1, h))
        self.x_a = nn.Parameter(torch.zeros(1, 1, h))
        self.x_g = nn.Parameter(torch.zeros(1, 1, h))
        self.k_k = nn.Parameter(torch.zeros(h))
        self.k_a = nn.Parameter(torch.zeros(h))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        self.r_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)
        self.w_lora = _LoRA(h, h, _lora_rank(2.5, h, factor), activation="tanh")
        self.a_lora = _LoRA(h, h, _lora_rank(2.5, h, factor), activation=None)
        self.g_lora = _LoRA(h, h, _lora_rank(5.0, h), activation="sigmoid", bias=False)
        self.v_lora = (
            _LoRA(h, h, _lora_rank(1.7, h, factor), activation=None)
            if (self.value_residual and not self.is_first)
            else None
        )
        self.g_norm = nn.GroupNorm(self.num_heads, h, eps=self.head_dim * 1e-5, affine=True)
        self.init_rwkv_weights()

    # ------------------------------------------------------------------ init
    @torch.no_grad()
    def init_rwkv_weights(self):
        """RWKV-7's reference init (BlinkDL RWKV-LM / FLA RWKV7Attention)."""
        h, L, li = self.hidden_size, self.num_layers, self.layer_idx
        dev = self.x_r.device
        r01 = li / max(L - 1, 1)
        r10 = 1.0 - li / L
        n = torch.arange(h, device=dev, dtype=torch.float32)
        ddd = (n / h).view(1, 1, h)
        linear = n / max(h - 1, 1) - 0.5
        zig = ((n % self.head_dim) - (self.head_dim - 1) / 2) / max((self.head_dim - 1) / 2, 1)
        zig = zig * zig.abs()
        www = -6 + 6 * (n / max(h - 1, 1)) ** (1 + r01**0.3)
        for p, e in (
            (self.x_r, 0.2), (self.x_w, 0.9), (self.x_k, 0.7),
            (self.x_v, 0.7), (self.x_a, 0.9), (self.x_g, 0.2),
        ):
            p.copy_((1.0 - ddd.pow(e * r10)).to(p.dtype))
        self.k_a.fill_(1.02)
        self.r_k.fill_(-0.04)
        self.k_k.copy_((0.71 - linear * 0.1).to(self.k_k.dtype))
        self.w_lora.init_rwkv(www + 0.5 + zig * 2.5)
        self.a_lora.init_rwkv(-0.19 + zig * 0.3 + linear * 0.4)
        self.g_lora.init_rwkv()
        if self.v_lora is not None:
            self.v_lora.init_rwkv(0.73 - linear * 0.4)
        self.g_norm.weight.fill_(((li + 1) / L) ** 0.7)
        self.g_norm.bias.zero_()
        for lin, gain in ((self.r_proj, 1.0), (self.k_proj, 0.1), (self.v_proj, 1.0)):
            w32 = torch.empty_like(lin.weight, dtype=torch.float32)
            nn.init.orthogonal_(w32, gain=gain)
            lin.weight.copy_(w32.to(lin.weight.dtype))
        self.o_proj.weight.zero_()

    def reset_buffers(self):
        """Hook called by UMichSLMPreTrainedModel._init_weights after this module's
        children were re-initialised: put RWKV-7's init back, unless loaded."""
        if any(getattr(p, "_is_hf_initialized", False) for p in self.parameters()):
            return
        self.init_rwkv_weights()

    # --------------------------------------------------------------- helpers
    def _token_shift(self, x: torch.Tensor, prev: Optional[torch.Tensor]) -> torch.Tensor:
        """``x_{t-1} - x_t`` with a zero predecessor at row and document starts."""
        shifted = F.pad(x[:, :-1], (0, 0, 1, 0))
        if prev is not None:
            shifted = torch.cat([prev.view(x.shape[0], 1, -1).to(x.dtype), shifted[:, 1:]], 1)
        doc_start = self._intra_doc_start
        if doc_start is not None and doc_start.shape[-1] == x.shape[1] and prev is None:
            shifted = shifted * (~doc_start).unsqueeze(-1).to(x.dtype)
        return shifted - x

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)

    def _cu_seqlens(self, B: int, L: int) -> Optional[torch.Tensor]:
        doc_start = self._intra_doc_start
        if doc_start is None or doc_start.shape[-1] != L:
            return None
        cu = self._delta_cu_seqlens
        if cu is None:
            ds = doc_start.clone()
            ds[:, 0] = True
            starts = torch.nonzero(ds.reshape(-1), as_tuple=False).flatten()
            cu = torch.cat([starts, starts.new_tensor([B * L])])
        return cu.to(torch.int32) if cu.dtype != torch.int32 else cu

    # --------------------------------------------------------------- forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[DeltaNetCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        depth_kv_cache=None,
        pause_query_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None, Optional[DeltaNetCache]]:
        """``pause_query_states`` (modules/pause_stream.py): the receptance r comes from
        these rows (through the same token shift) while every write-side quantity
        (w, k, v, a, g) comes from ``hidden_states``, which the block fills with the
        STATE stream for both halves -- the prediction stream reads S_t and writes
        nothing."""
        B, L, _ = hidden_states.shape
        if pause_query_states is not None and (use_cache or past_key_value is not None):
            raise NotImplementedError("pause_query_states supports full-sequence passes only")
        state = prev = None
        if past_key_value is not None:
            state = past_key_value.recurrent_state
            if past_key_value.conv_state is not None:
                prev = past_key_value.conv_state[..., -1]

        x = hidden_states
        dx = self._token_shift(x, prev)
        xw, xk, xv, xa, xg = (
            torch.addcmul(x, dx, p) for p in (self.x_w, self.x_k, self.x_v, self.x_a, self.x_g)
        )
        if pause_query_states is not None:
            p_shift = self._token_shift(pause_query_states, None)
            xr = torch.addcmul(pause_query_states, p_shift, self.x_r)
        else:
            xr = torch.addcmul(x, dx, self.x_r)

        r = self.r_proj(xr)
        w = -_W_SCALE * torch.sigmoid(self.w_lora(xw).float())
        k = self.k_proj(xk)
        v = self.v_proj(xv)
        store = self._v_first_store if self.value_residual else None
        if store is not None:
            if self.is_first:
                store["v"] = v
            elif self.v_lora is not None and store.get("v") is not None:
                v = torch.lerp(v, store["v"].to(v.dtype), torch.sigmoid(self.v_lora(xv)))
        alpha = torch.sigmoid(self.a_lora(xa))
        g = self.g_lora(xg)

        kk = F.normalize(self._heads(k * self.k_k), dim=-1, p=2.0)
        k = k * (1 + (alpha - 1) * self.k_a)
        r_h, k_h, v_h, a_h = self._heads(r), self._heads(k), self._heads(v), self._heads(alpha)
        w_h = w.view(B, L, self.num_heads, self.head_dim)
        a_in, b_in = -kk, kk * a_h

        cu = self._cu_seqlens(B, L) if state is None and not use_cache else None
        if RWKV7_FLA_AVAILABLE and x.is_cuda and state is None and not use_cache:
            dt = r_h.dtype
            if cu is not None:
                flat = lambda t: t.reshape(1, B * L, *t.shape[2:])  # noqa: E731
                o, _ = chunk_rwkv7(
                    r=flat(r_h), w=flat(w_h).to(dt), k=flat(k_h), v=flat(v_h),
                    a=flat(a_in).to(dt), b=flat(b_in).to(dt), scale=1.0,
                    cu_seqlens=cu, safe_gate=True, chunk_size=64,
                )
                o = o.reshape(B, L, self.num_heads, self.head_dim)
            else:
                o, _ = chunk_rwkv7(
                    r=r_h, w=w_h.to(dt), k=k_h, v=v_h, a=a_in.to(dt), b=b_in.to(dt),
                    scale=1.0, safe_gate=True, chunk_size=64,
                )
            new_state = None
        else:
            doc_start = self._intra_doc_start
            if doc_start is not None and (doc_start.shape[-1] != L or state is not None):
                doc_start = None
            o, new_state = _reference_wkv(r_h, w_h, k_h, v_h, a_in, b_in, state, doc_start)

        o = self.g_norm(o.reshape(B * L, -1).to(v.dtype)).view(B, L, -1)
        corr = (r_h * k_h * self.r_k).sum(-1, keepdim=True) * v_h
        o = (o + corr.reshape(B, L, -1)) * g
        o = self.o_proj(o)

        present = None
        if use_cache:
            present = DeltaNetCache(
                recurrent_state=new_state,
                conv_state=hidden_states[:, -1:].transpose(1, 2).contiguous(),
            )
        return o, None, present
