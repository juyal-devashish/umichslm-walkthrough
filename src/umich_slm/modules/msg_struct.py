"""Message-structured MLA (P1) and per-message digest slots (P2).

FABLE review 2026-11-04, proposals P1 and P2
(``/shared/6/projects/umich-slm/tmp/handoff/FABLE_REVIEW_2026-11-04.md``). Both act on
the MLA layers of a hybrid (``deltanet_mla``); DeltaNet layers are untouched. Arms:
``deltanet_mla_turnpos`` (P1), ``deltanet_mla_digest`` (P2) and their composition
``deltanet_mla_turnpos_digest``.

MESSAGE STRUCTURE (shared, :func:`compute_message_structure`). Once per forward, from
``input_ids`` alone (no host syncs; cumsum / cummax / gather only, static shapes):

* ``m``: message index, +1 at every ``<|start|>``, reset at document boundaries (the
  intra-doc ``document_ids``). Tokens before a document's first ``<|start|>``, and every
  token of a document with no Harmony markers, have ``m = 0``.
* ``o``: offset within the message (``pos - pos[latest <|start|>]``); ``o = pos`` where
  ``m = 0``. ``pos`` is the model's position_ids AFTER the intra-doc reset and any
  ``mla_position_jitter``, so both carry over.
* ``pinned``: tokens of the first system message's tool block (from the first
  ``namespace`` token through the message terminator; the template renders the block
  last in that message) and of the first user message. Strictly causal: a message's
  role is the token after ``<|start|>``, so the ``<|start|>`` token itself is never
  pinned (its role is not known yet at that position).
* digest slots: one per COMPLETE message (``<|start|>`` ... first of ``<|end|>``,
  ``<|call|>``, ``<|return|>``), at most ``digest_max_messages`` per row; later ones
  are dropped (no digest; their raw tokens are still in the window).

P1 ROPE (:class:`MessageRotaryEmbedding`). The MLA rope slice (64 dims = 32 pairs at
every screening tier) is rotated by ``o`` on all pairs with the stock frequencies
(YaRN included), and the lowest-frequency ``msg_rope_frac`` of the pairs (default half)
is rotated ADDITIONALLY by ``m`` at base ``msg_rope_theta`` (default 1000). Because 2-D
rotations compose additively, each m-pair carries angle ``o*f_i + m*g_i``: still a
relative encoding, in the 2-D coordinate (o, m). Departure from the review's literal
"half by o, half by m": with ``m = 0`` the literal split leaves half the rope dims
position-free, so it does NOT reduce to the parent on plain text; the additive form
does, exactly (``m = 0`` -> cos 1, sin 0), and the m-pairs are the ones whose o-angle is
smallest, so m dominates them. ``msg_index_jitter = J > 1`` (training only) rebuilds
``m`` on a random half of the rows as a running sum of per-message gaps drawn from
[1, J], the ``mla_position_jitter`` idea applied to the message axis (extrapolation to
300+ messages).

P1 PINNED KEYS. A learned per-head scalar ``b_h`` (zero init) is added to the MLA
logit of every pinned key. FA2 has no per-key bias, so the bias is folded into the
dot product: q gets one extra dim of 1 and k gets ``b_h * pinned_j / scale`` (padded to
8 extra dims for FA2), with the softmax scale passed explicitly from the ORIGINAL head
dim. ``scale * (q.k + b_h pinned_j / scale) = scale q.k + b_h pinned_j``: exact, one
kernel call, gradient to ``b_h`` through dk. Cost: the MLA q/k head dim goes 192 -> 200,
which FA2 runs on its 224 tile (+17% on the MLA attention kernel only; the projections
and the 3/4 DeltaNet layers are unchanged). bf16 rounds the bias column like any other
key element (relative 2^-9).

P2 DIGESTS (:class:`DigestWriter`). After layer ``digest_source_layer`` (default: the
first MLA layer), ``k`` learned queries per message cross-attend over that message's
hidden states (a segment softmax by scatter_reduce/scatter_add, O(S*k*d), no per-message
loop) and write ``k`` digest vectors (k = ``digest_slots_tool`` for tool-role messages,
``digest_slots_other`` otherwise; both query sets have the larger k, extra slots are
masked off). Every MLA layer then attends over the last ``digest_window`` raw tokens of
its own document; MLA layers ABOVE the source layer also attend over the digests of all
earlier complete messages in the same document, visible strictly after the message's
terminator (``t > t_end``). A digest is a virtual token: each reading layer projects it
through its own ``kv_down_proj``/``k_up``/``v_up``/``k_rope_proj``, so it costs exactly
one cached MLA token per slot at decode, and it is rotated at its terminator's position
(``o``/``m`` coordinates under P1, plain position otherwise). MLA layers at or below the
source layer cannot read digests in a parallel training pass (the digests do not exist
yet in that forward), so they are window-only. The "keep the task" auxiliary loss of the
review is DEFERRED (not implemented).

ATTENTION PATHS (:func:`msg_attention_forward`, entered from a thin hook in mla.py):

* FA2 (CUDA, flash class): raw keys via ``flash_attn_with_lse`` (varlen with the model's
  intra-doc ``cu_seqlens``, sliding window ``(W-1, 0)``), digests by an eager [S, M*k]
  attention, both joined by an exact log-sum-exp merge (the LSE carries its gradient,
  attn_sink.py). The logit sink, if on, is a third merge term.
* Dense (CPU, sdpa/eager classes, or anything the FA2 path cannot express): one SDPA call
  over [raw keys ; digest keys] with a dense mask. This is the reference.

NOT SUPPORTED (validated in the config): cached decode (the msg arms raise on
``past_length > 0``; prefill is fine), mla_diff_attn, RePo, MoD, pause/feedback streams,
and prefix tokens (swa_globals). The msg path ignores a 2-D padding mask; right padding
is harmless under causal attention, left padding is not.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_sink import _HAS_FLASH_INTERNALS, flash_attn_with_lse
from .rotary import RotaryEmbedding, _maybe_extend_cache, rotate_half

# Production SuperBPE (vocab 56000) ids, verified against load_default_tokenizer() by
# tests/test_msg_struct.py. Role headers tokenize alone because the tokenizer splits on
# <|...|> tags; "tool to=assistant" starts with "tool " (2586), "tool call_id=" with
# "tool call" (7929). tool_block: every vocab entry containing "namespace".
DEFAULT_MSG_TOKEN_IDS: Dict[str, List[int]] = {
    "start": [4],
    "end": [5, 8, 9],  # <|end|> <|call|> <|return|>
    "system": [1073, 3053],
    "user": [1367, 1212],
    "tool": [1058, 2586, 7929],
    "tool_block": [11229, 13511, 17987],
    # P12 decision tokens (pause_decision_only)
    "assistant": [16622],
    "channel": [7],
    "message": [6],
    "commentary": [5064],
    "final": [22696],
}


def msg_token_ids(config) -> Dict[str, List[int]]:
    ids = dict(DEFAULT_MSG_TOKEN_IDS)
    ids.update(getattr(config, "msg_token_ids", None) or {})
    return ids


# ----------------------------------------------------------------------------- structure


def _last_index(flag: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Index of the latest True at or before each position (-1 if none)."""
    return torch.cummax(torch.where(flag, idx, torch.full_like(idx, -1)), dim=1).values


def _doc_count(x: torch.Tensor, doc_first_idx: torch.Tensor) -> torch.Tensor:
    """Inclusive running sum of ``x`` that restarts at every document start."""
    x = x.long()
    c = x.cumsum(1)
    return c - (c - x).gather(1, doc_first_idx)


def compute_message_structure(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    document_ids: Optional[torch.Tensor],
    ids: Dict[str, List[int]],
    max_messages: int = 0,
) -> Dict[str, torch.Tensor]:
    """Per-token message coordinates, pin mask and digest-slot bookkeeping.

    Returns a dict of ``[B, S]`` tensors (``doc``, ``m``, ``o``, ``in_msg``, ``closing``,
    ``pinned``, ``tool``, ``slot``) plus, when ``max_messages > 0``, ``[B, M]`` slot
    tensors ``end_pos`` (-1 = no complete message), ``slot_tool``.
    """
    B, S = input_ids.shape
    dev = input_ids.device
    idx = torch.arange(S, device=dev).unsqueeze(0).expand(B, S)
    doc = (
        document_ids.long()
        if document_ids is not None
        else torch.zeros(B, S, dtype=torch.long, device=dev)
    )
    first = torch.ones(B, S, dtype=torch.bool, device=dev)
    first[:, 1:] = doc[:, 1:] != doc[:, :-1]
    dfi = _last_index(first, idx)  # >= 0: column 0 is always a document start

    def isin(x, key):
        return torch.isin(x, torch.tensor(ids[key], dtype=x.dtype, device=dev))

    is_start = isin(input_ids, "start")
    is_end = isin(input_ids, "end")
    m = _doc_count(is_start, dfi)
    ls = _last_index(is_start, idx).clamp(min=0)  # meaningful only where m > 0
    pos = position_ids.long()
    o = torch.where(m > 0, pos - pos.gather(1, ls), pos)

    ends = is_end.long().cumsum(1)
    ends_excl = ends - is_end.long()
    in_msg = (m > 0) & ((ends_excl - ends_excl.gather(1, ls)) == 0)
    closing = in_msg & is_end

    # Role = the token after <|start|>. Known at t only once t > ls (causal); the digest
    # writer may look ahead inside its message (its digest is read only after the end).
    role_tok = input_ids.gather(1, (ls + 1).clamp(max=S - 1))
    role_known = in_msg & (idx > ls)
    is_role_pos = torch.zeros_like(is_start)
    is_role_pos[:, 1:] = is_start[:, :-1] & ~first[:, 1:]
    sys_n = _doc_count(is_role_pos & isin(input_ids, "system"), dfi)
    user_n = _doc_count(is_role_pos & isin(input_ids, "user"), dfi)
    mk = isin(input_ids, "tool_block").long()
    mk_c = mk.cumsum(1)
    in_block = (mk_c - (mk_c - mk).gather(1, ls)) >= 1
    pinned = (role_known & isin(role_tok, "system") & (sys_n == 1) & in_block) | (
        role_known & isin(role_tok, "user") & (user_n == 1)
    )
    tool = in_msg & isin(role_tok, "tool")

    out = dict(
        doc=doc, m=m, o=o, in_msg=in_msg, closing=closing, pinned=pinned, tool=tool,
        is_start=is_start, doc_first_idx=dfi, last_start=ls, role_tok=role_tok,
        role_known=role_known, idx=idx,
    )
    if max_messages > 0:
        M = int(max_messages)
        r = is_start.long().cumsum(1) - 1  # row-level message ordinal
        keep = in_msg & (r < M)
        out["slot"] = torch.where(keep, r, torch.full_like(r, M))
        dst = torch.where(closing & (r < M), r, torch.full_like(r, M))
        end_pos = torch.full((B, M + 1), -1, dtype=torch.long, device=dev)
        # Real slots get exactly one writer (one closing end per message); non-closing
        # tokens all land in the dump column M, which is sliced off.
        end_pos = end_pos.scatter(1, dst, torch.where(dst < M, idx, torch.full_like(idx, -1)))
        out["end_pos"] = end_pos[:, :M]
        out["slot_tool"] = tool.gather(1, out["end_pos"].clamp(min=0))
    return out


def compute_decision_mask(
    input_ids: torch.Tensor,
    document_ids: Optional[torch.Tensor],
    ids: Dict[str, List[int]],
    final_tokens: int = 16,
) -> torch.Tensor:
    """P12 decision tokens, ``[B, S]`` bool (``pause_decision_only``).

    In an assistant message: its (first) ``<|channel|>`` token, where the model chooses
    analysis / commentary / final; and after it, every token of a ``commentary`` message
    through its terminator (the tool call), and the header plus the first
    ``final_tokens`` content tokens of a ``final`` message (counted from its first
    ``<|message|>``). Strictly causal: the channel NAME is the token after
    ``<|channel|>``, so the ``<|channel|>`` token is a decision token whatever follows.
    """
    B, S = input_ids.shape
    pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
    st = compute_message_structure(input_ids, pos, document_ids, ids)
    ls, idx, in_msg = st["last_start"], st["idx"], st["in_msg"]
    dev = input_ids.device

    def isin(x, key):
        return torch.isin(x, torch.tensor(ids[key], dtype=x.dtype, device=dev))

    def in_msg_count(x):  # inclusive count of x since this message's <|start|>
        c = x.long().cumsum(1)
        return c - c.gather(1, ls)

    ch = isin(input_ids, "channel")
    first_ch = ch & (in_msg_count(ch) == 1) & in_msg
    lc = _last_index(first_ch, idx)
    has_ch = in_msg & (lc > ls)
    ch_name = input_ids.gather(1, (lc + 1).clamp(min=0, max=S - 1))
    assistant = st["role_known"] & isin(st["role_tok"], "assistant")
    at_channel = assistant & first_ch
    after = assistant & has_ch & (idx > lc)
    commentary = after & isin(ch_name, "commentary")
    seen_msg = in_msg_count(isin(input_ids, "message")) >= 1
    n_content = in_msg_count(seen_msg)  # 1 at the first <|message|>
    final = after & isin(ch_name, "final") & (n_content <= int(final_tokens) + 1)
    return at_channel | commentary | final


def jitter_message_index(
    m: torch.Tensor, is_start: torch.Tensor, doc_first_idx: torch.Tensor, max_gap: int,
    frac: float = 0.5,
) -> torch.Tensor:
    """``msg_index_jitter``: on a random ``frac`` of rows, m = running sum of per-message
    gaps drawn from [1, max_gap], restarting per document. Order is preserved, m = 0
    stays 0, and the other rows keep the true index (the eval-time geometry)."""
    B, S = m.shape
    gaps = torch.randint(1, max_gap + 1, (B, S), device=m.device) * is_start.long()
    run = _doc_count(gaps, doc_first_idx)
    rows = torch.rand(B, 1, device=m.device) < frac
    return torch.where(rows, run, m)


def message_rope_freqs(rope_dim: int, theta: float, frac: float, device) -> torch.Tensor:
    """Per-pair m-frequencies, ``[rope_dim // 2]``: zero on the high-o-frequency pairs, a
    standard RoPE ladder at base ``theta`` on the lowest-frequency ``frac`` of them."""
    half = rope_dim // 2
    n = max(1, min(half, int(round(frac * half))))
    g = torch.zeros(half, dtype=torch.float32, device=device)
    g[half - n :] = theta ** (-torch.arange(n, dtype=torch.float32, device=device) / n)
    return g


def _cos_sin(m: torch.Tensor, g: torch.Tensor):
    ang = m.float().unsqueeze(-1) * g  # [B, T, half]
    emb = torch.cat((ang, ang), dim=-1).unsqueeze(1)  # [B, 1, T, dim]
    return emb.cos(), emb.sin()


class MsgContext:
    """Per-forward message state shared by every MLA layer of one model.

    ONE persistent object per model whose fields are reassigned each forward, so a
    compiled block sees a stable object (guards on tensor metadata, not identity).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.doc = None  # [B,S] document ids (zeros without intra-doc masking)
        self.has_doc = False
        self.pos = None  # [B,S] model position ids
        self.o = None  # [B,S] rope o-coordinate (offset in message; = pos where m = 0)
        self.m_cos = self.m_sin = None  # [B,1,S,dr] m-rotation (None: P1 rope off)
        self.pinned = None  # [B,S] float (None: pin bias off)
        self.window = None  # raw-key window W (None: unbounded)
        self.slot = self.tool = None  # [B,S] writer inputs
        self.dig_valid = None  # [B,M,K] bool
        self.dig_pos = None  # [B,M*K] rope o-coordinate of each digest slot
        self.dig_m_cos = self.dig_m_sin = None  # [B,1,M*K,dr]
        self.dig_vis = None  # [B,S,M*K] bool: query t may read slot
        self.digest_h = None  # [B,M,K,d], filled after the source layer


class MessageRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding whose MLA rotation reads the (o, m) coordinates of ``_msg_ctx``.

    Installed by swapping ``__class__`` on an MLA layer's existing rotary (no new state),
    so YaRN rescopes and ``isinstance(..., RotaryEmbedding)`` checks are unchanged. With
    no context bound it is the parent class bit for bit.
    """

    _msg_ctx: Optional[MsgContext] = None

    def forward(self, q, k, position_ids):
        ctx = self._msg_ctx
        if ctx is None:
            return super().forward(q, k, position_ids)
        q, k = super().forward(q, k, ctx.o)
        if ctx.m_cos is None:
            return q, k
        cos, sin = ctx.m_cos.to(q.dtype), ctx.m_sin.to(q.dtype)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def rotate_keys(rotary: RotaryEmbedding, x: torch.Tensor, pos: torch.Tensor, m_cos=None, m_sin=None):
    """Rotate keys ``x [B, H, T, dim]`` at arbitrary positions ``pos [B, T]`` (digest slots)."""
    _maybe_extend_cache(rotary, pos)
    cos = rotary.cos_cached[pos].unsqueeze(1)
    sin = rotary.sin_cached[pos].unsqueeze(1)
    scale = 1.0
    if rotary.rope_type == "yarn" and rotary.apply_mscale:
        cos, sin, scale = cos * rotary.mscale, sin * rotary.mscale, rotary.mscale
    x = rotary._rotate(x, cos, sin, scale)
    if m_cos is not None:
        c, s = m_cos.to(x.dtype), m_sin.to(x.dtype)
        x = (x * c) + (rotate_half(x) * s)
    return x


# ------------------------------------------------------------------------------ digests


class DigestWriter(nn.Module):
    """k learned queries per message attend over that message's hidden states.

    Multi-head (the model's attention heads), one query set for tool-role messages and
    one for the rest. Segment softmax over each message's tokens by scatter, so the cost
    is O(S * k * d) with a fixed slot count and no data-dependent shapes.
    """

    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        self.num_heads = config.num_attention_heads
        if d % self.num_heads:
            raise ValueError(f"hidden_size {d} not divisible by heads {self.num_heads}")
        self.k_tool = int(getattr(config, "digest_slots_tool", 8))
        self.k_other = int(getattr(config, "digest_slots_other", 4))
        self.k = max(self.k_tool, self.k_other)
        self.max_messages = int(getattr(config, "digest_max_messages", 128))
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.in_norm = nn.RMSNorm(d, eps=eps)
        self.kv_proj = nn.Linear(d, 2 * d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.out_norm = nn.RMSNorm(d, eps=eps)
        # [role (0 = other, 1 = tool), slot, d]
        self.queries = nn.Parameter(torch.randn(2, self.k, d) * config.initializer_range)

    def forward(self, h: torch.Tensor, slot: torch.Tensor, tool: torch.Tensor) -> torch.Tensor:
        B, S, d = h.shape
        H, K, M = self.num_heads, self.k, self.max_messages
        dh = d // H
        key, val = self.kv_proj(self.in_norm(h)).chunk(2, dim=-1)
        key = key.view(B, S, H, dh).float()
        val = val.view(B, S, H, dh).float()
        q = self.queries.view(2, K, H, dh).float()
        s_all = torch.einsum("bshd,rkhd->bshrk", key, q) / math.sqrt(dh)
        s = torch.where(tool.view(B, S, 1, 1), s_all[..., 1, :], s_all[..., 0, :])  # [B,S,H,K]
        ix = slot.view(B, S, 1, 1).expand(B, S, H, K)
        mx = torch.full((B, M + 1, H, K), float("-inf"), device=h.device)
        mx = mx.scatter_reduce(1, ix, s.detach(), reduce="amax", include_self=True)
        e = torch.exp(s - mx.gather(1, ix))
        den = torch.zeros(B, M + 1, H, K, device=h.device).scatter_add(1, ix, e)
        num = torch.zeros(B, M + 1, H, K, dh, device=h.device).scatter_add(
            1, ix.unsqueeze(-1).expand(B, S, H, K, dh), e.unsqueeze(-1) * val.unsqueeze(3)
        )
        dig = (num / den.clamp_min(1e-30).unsqueeze(-1))[:, :M]  # [B,M,H,K,dh]
        dig = dig.permute(0, 1, 3, 2, 4).reshape(B, M, K, d).to(h.dtype)
        return self.out_norm(self.o_proj(dig))


class MessageStructure(nn.Module):
    """Model-level owner of the P1/P2 machinery: installs the per-layer pieces, builds the
    per-forward context, and runs the digest writer after the source layer."""

    def __init__(self, config, model: nn.Module):
        super().__init__()
        from .mla import UMichSLMMLAAttention

        self.ids = msg_token_ids(config)
        self.rope = bool(getattr(config, "msg_rope", False))
        self.rope_theta = float(getattr(config, "msg_rope_theta", 1000.0))
        self.rope_frac = float(getattr(config, "msg_rope_frac", 0.5))
        self.jitter = int(getattr(config, "msg_index_jitter", 0) or 0)
        self.pin = bool(getattr(config, "msg_pin_bias", False))
        self.digest = bool(getattr(config, "msg_digest", False))
        self.window = int(getattr(config, "digest_window", 2048)) if self.digest else None

        mla_idx = [
            i for i, layer in enumerate(model.layers)
            if isinstance(getattr(layer, "self_attn", None), UMichSLMMLAAttention)
        ]
        if not mla_idx:
            raise ValueError("msg_rope / msg_pin_bias / msg_digest need MLA layers")
        src = int(getattr(config, "digest_source_layer", -1))
        self.source_layer = mla_idx[0] if src < 0 else src
        if self.digest and not any(i > self.source_layer for i in mla_idx):
            raise ValueError(f"no MLA layer above digest_source_layer={self.source_layer}")
        self.writer = DigestWriter(config) if self.digest else None
        self.ctx = MsgContext()
        self.rope_dim = model.layers[mla_idx[0]].self_attn.qk_rope_head_dim

        # Per-layer install: plain attributes / parameters on the existing MLA modules.
        self._mla = []
        for i in mla_idx:
            attn = model.layers[i].self_attn
            attn._msg_ctx = None
            attn._msg_digest_read = self.digest and i > self.source_layer
            if self.rope and not attn.mla_nope:
                attn.rotary_emb.__class__ = MessageRotaryEmbedding
            if self.pin:
                attn.msg_pin_bias = nn.Parameter(torch.zeros(attn.num_heads))
            self._mla.append(attn)

    def _bind(self, ctx: Optional[MsgContext]):
        for attn in self._mla:
            attn._msg_ctx = ctx
            if isinstance(attn.rotary_emb, MessageRotaryEmbedding):
                attn.rotary_emb._msg_ctx = ctx

    def begin(self, input_ids, position_ids, document_ids, past_length: int) -> MsgContext:
        """Build this forward's context and bind it to every MLA layer."""
        if past_length > 0:
            raise NotImplementedError(
                "msg_rope / msg_pin_bias / msg_digest have no incremental-decode path "
                "(message state is not carried in the cache); run full forwards"
            )
        if input_ids is None:
            raise ValueError("the message-structure arms need input_ids (not inputs_embeds)")
        B, S = input_ids.shape
        M = self.writer.max_messages if self.writer is not None else 0
        st = compute_message_structure(input_ids, position_ids, document_ids, self.ids, M)
        ctx = self.ctx
        ctx.reset()
        ctx.doc, ctx.has_doc = st["doc"], document_ids is not None
        ctx.pos = position_ids
        ctx.window = self.window
        m = st["m"]
        if self.rope:
            if self.jitter > 1 and self.training:
                m = jitter_message_index(m, st["is_start"], st["doc_first_idx"], self.jitter)
            g = message_rope_freqs(self.rope_dim, self.rope_theta, self.rope_frac, m.device)
            ctx.m_cos, ctx.m_sin = _cos_sin(m, g)
            ctx.o = st["o"]
        else:
            ctx.o = position_ids
        if self.pin:
            ctx.pinned = st["pinned"].float()
        if self.writer is not None:
            K = self.writer.k
            ctx.slot, ctx.tool = st["slot"], st["tool"]
            end_pos = st["end_pos"]  # [B,M]
            valid = end_pos >= 0
            n_k = torch.where(st["slot_tool"], self.writer.k_tool, self.writer.k_other)
            kvalid = valid.unsqueeze(-1) & (
                torch.arange(K, device=m.device).view(1, 1, K) < n_k.unsqueeze(-1)
            )
            ctx.dig_valid = kvalid
            ep = end_pos.clamp(min=0)
            dig_doc = st["doc"].gather(1, ep)
            t = torch.arange(S, device=m.device).view(1, S, 1, 1)
            vis = (
                kvalid.unsqueeze(1)
                & (dig_doc.view(B, 1, M, 1) == st["doc"].view(B, S, 1, 1))
                & (end_pos.view(B, 1, M, 1) < t)
            )
            ctx.dig_vis = vis.reshape(B, S, M * K)
            rep = lambda x: x.unsqueeze(-1).expand(B, M, K).reshape(B, M * K)  # noqa: E731
            ctx.dig_pos = rep(ctx.o.long().gather(1, ep))
            if ctx.m_cos is not None:
                ctx.dig_m_cos, ctx.dig_m_sin = _cos_sin(rep(m.gather(1, ep)), g)
        self._bind(ctx)
        return ctx

    def write(self, hidden_states: torch.Tensor, checkpoint: bool = False) -> None:
        """Run the digest writer on the source layer's output; fills ``ctx.digest_h``."""
        ctx = self.ctx
        if checkpoint:
            ctx.digest_h = torch.utils.checkpoint.checkpoint(
                self.writer, hidden_states, ctx.slot, ctx.tool, use_reentrant=False
            )
        else:
            ctx.digest_h = self.writer(hidden_states, ctx.slot, ctx.tool)


def build_message_structure(config, model) -> Optional[MessageStructure]:
    if not (
        getattr(config, "msg_rope", False)
        or getattr(config, "msg_pin_bias", False)
        or getattr(config, "msg_digest", False)
    ):
        return None
    return MessageStructure(config, model)


# ---------------------------------------------------------------------------- attention

_PIN_PAD = 8  # FA2 head dims must be multiples of 8


def _augment_pinned(q, k, bias, pinned, scale):
    """Fold a per-head bias on pinned keys into the dot product (exact; see module doc).
    q/k ``[B, H, S, D]`` -> ``[B, H, S, D + 8]``."""
    B, H, S, _ = q.shape
    qx = q.new_zeros(B, H, S, _PIN_PAD)
    qx[..., 0] = 1.0
    kcol = (bias.float().view(1, H, 1) * pinned.float().unsqueeze(1) / scale).to(k.dtype)
    kx = torch.cat([kcol.unsqueeze(-1), k.new_zeros(B, H, S, _PIN_PAD - 1)], dim=-1)
    return torch.cat([q, qx], dim=-1), torch.cat([k, kx], dim=-1)


def _digest_kv(attn, ctx: MsgContext, d_qk: int, dtype):
    """Each reading layer projects the digests through its OWN MLA K/V maps.
    Returns kd ``[B, H, M*K, d_qk]`` (zero-padded to the augmented width) and vd."""
    h = ctx.digest_h
    B, M, K, d = h.shape
    T = M * K
    hf = h.reshape(B, T, d)
    H = attn.num_heads
    c = attn.kv_layernorm(attn.kv_down_proj(hf))
    k_nope = attn.k_up_proj(c).view(B, T, H, attn.qk_nope_head_dim)
    vd = attn.v_up_proj(c).view(B, T, H, attn.v_head_dim)
    kr = attn.k_rope_proj(hf)
    if hasattr(attn, "_k_rope_heads"):
        kr = attn._k_rope_heads(kr, B, T)
    else:
        kr = kr.view(B, T, H, attn.qk_rope_head_dim)
    k_nope, kr = attn._norm_k(k_nope, kr)
    kr = kr.transpose(1, 2)
    if not attn.mla_nope:
        kr = rotate_keys(attn.rotary_emb, kr, ctx.dig_pos, ctx.dig_m_cos, ctx.dig_m_sin)
    kd = torch.cat([k_nope.transpose(1, 2).to(kr.dtype), kr], dim=-1).to(dtype)
    if d_qk > kd.shape[-1]:
        kd = F.pad(kd, (0, d_qk - kd.shape[-1]))
    return kd, vd.transpose(1, 2).to(dtype)


def _raw_allowed(ctx: MsgContext, S: int, device) -> torch.Tensor:
    """[B, 1, S, S] bool: same document, causal, inside the window."""
    i = torch.arange(S, device=device)
    rel = i.view(S, 1) - i.view(1, S)
    ok = rel >= 0
    if ctx.window is not None:
        ok = ok & (rel < ctx.window)
    same = ctx.doc.unsqueeze(2) == ctx.doc.unsqueeze(1)
    return (ok.unsqueeze(0) & same).unsqueeze(1)


def _dense_attention(q, k, v, kd, vd, vis, sink, ctx, scale, attention_mask):
    """Reference path: one SDPA over [raw ; digest] keys with a dense mask."""
    B, H, S, _ = q.shape
    allowed = _raw_allowed(ctx, S, q.device)
    if attention_mask is not None:
        allowed = allowed & (attention_mask > torch.finfo(attention_mask.dtype).min / 2)
    keys, vals = k, v
    if kd is not None:
        keys, vals = torch.cat([k, kd], dim=2), torch.cat([v, vd], dim=2)
        allowed = torch.cat([allowed.expand(B, 1, S, S), vis.unsqueeze(1)], dim=-1)
    neg = torch.finfo(q.dtype).min
    mask = torch.zeros(allowed.shape, dtype=q.dtype, device=q.device).masked_fill(~allowed, neg)
    if sink is not None:
        mask = mask.expand(B, H, S, mask.shape[-1])
        mask = torch.cat([mask, sink.to(q.dtype).view(1, H, 1, 1).expand(B, H, S, 1)], dim=-1)
        keys = torch.cat([keys, keys.new_zeros(B, H, 1, keys.shape[-1])], dim=2)
        vals = torch.cat([vals, vals.new_zeros(B, H, 1, vals.shape[-1])], dim=2)
    return F.scaled_dot_product_attention(q, keys, vals, attn_mask=mask, scale=scale)


def _flash_attention(attn, q, k, v, kd, vd, vis, sink, ctx, scale):
    """FA2 raw (windowed, varlen) + eager digest part, exact LSE merge. q/k/v [B,H,S,D]."""
    B, H, S, D = q.shape
    dv = v.shape[-1]
    qf, kf = q.transpose(1, 2), k.transpose(1, 2)
    vf = F.pad(v, (0, D - dv)).transpose(1, 2)
    orig = q.dtype
    if orig not in (torch.float16, torch.bfloat16):
        qf, kf, vf = qf.bfloat16(), kf.bfloat16(), vf.bfloat16()
    win = (ctx.window - 1, 0) if ctx.window is not None else (-1, -1)
    cu = getattr(attn, "_intra_doc_cu_seqlens", None)
    if cu is not None:
        o, lse = flash_attn_with_lse(
            qf.reshape(B * S, H, D), kf.reshape(B * S, H, D), vf.reshape(B * S, H, D),
            softmax_scale=scale, causal=True, window=win,
            cu_seqlens=cu.to(q.device), max_seqlen=int(attn._intra_doc_max_seqlen),
        )
        o = o.view(B, S, H, D)
        lse = lse.view(H, B, S).transpose(0, 1)
    else:
        o, lse = flash_attn_with_lse(qf, kf, vf, softmax_scale=scale, causal=True, window=win)
    o = o.transpose(1, 2)[..., :dv]  # [B,H,S,dv]
    if kd is None and sink is None:
        return o.to(orig)
    terms = [lse.float()]
    ld = None
    if kd is not None:
        ld = torch.einsum("bhsd,bhtd->bhst", q.float(), kd.float()) * scale
        ld = ld.masked_fill(~vis.unsqueeze(1), float("-inf"))
        terms.append(torch.logsumexp(ld, dim=-1))
    if sink is not None:
        terms.append(sink.float().view(1, H, 1).expand(B, H, S))
    tot = torch.logsumexp(torch.stack(terms, 0), dim=0)  # finite: the diagonal raw key
    out = torch.exp(terms[0] - tot).unsqueeze(-1) * o.float()
    if ld is not None:
        out = out + torch.exp(ld - tot.unsqueeze(-1)) @ vd.float()
    return out.to(orig)


def msg_attention_forward(attn, q, k, v, q_src, attention_mask=None):
    """The MLA attention of the msg arms, from mla.py's hook. q/k/v ``[B, H, S, D]`` (RoPE
    already applied through the message rotary). Returns the projected layer output."""
    ctx = attn._msg_ctx
    B, H, S, _ = q.shape
    scale = attn._yarn_logit_scale() / math.sqrt(attn.qk_head_dim)
    bias = getattr(attn, "msg_pin_bias", None)
    if bias is not None and ctx.pinned is not None:
        q, k = _augment_pinned(q, k, bias, ctx.pinned, scale)
    kd = vd = vis = None
    if getattr(attn, "_msg_digest_read", False):
        if ctx.digest_h is None:
            raise RuntimeError("digest-reading MLA layer ran before the digest writer")
        kd, vd = _digest_kv(attn, ctx, q.shape[-1], q.dtype)
        vis = ctx.dig_vis
    sink = attn.attn_logit_sink
    use_flash = (
        _HAS_FLASH_INTERNALS
        and q.is_cuda
        and getattr(attn, "_supports_varlen_intra_doc", False)
        and attention_mask is None
        and (not ctx.has_doc or getattr(attn, "_intra_doc_cu_seqlens", None) is not None)
    )
    if use_flash:
        out = _flash_attention(attn, q, k, v, kd, vd, vis, sink, ctx, scale)
    else:
        out = _dense_attention(q, k, v, kd, vd, vis, sink, ctx, scale, attention_mask)
    out = out.transpose(1, 2).reshape(B, S, H * attn.v_head_dim)
    return attn.o_proj(attn._gate_out(out, q_src))
