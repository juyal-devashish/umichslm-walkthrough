"""P1 ``dual_stream``: a narrow fast stream plus a wide, sparse slow stream.

The idea (docs/dyncompute_redesign.md, sketch P1). A standard transformer layer
fuses two different scaling axes -- compute per token and parameter capacity --
so buying more capacity always costs proportionally more compute. This arm
separates them. Every token goes through the narrow FAST stream (the ordinary
backbone, dense kernels, no routing). Every ``k`` layers, a small fraction of
tokens additionally passes through a WIDE slow stream. Parameters grow a lot;
compute grows a little.

What makes this version ours rather than a re-implementation of the dual-path
line: **the slow stream's only long-range channel is the register bank**. It
does not cross-attend to the full fast-stream sequence. Instead it
  * self-attends among the selected ("hard") tokens, causally by their ORIGINAL
    positions, and
  * cross-attends into the register bank,
while the fast stream reads the same registers on every SWA layer. Registers
therefore become the bus between the two streams: the fast stream's only access
to deep computation is through what the slow stream leaves in memory. That also
keeps the cost at O(k^2 + k*R) instead of the O(k*S) a full cross-attention
would cost -- at S=4096 the latter would materialise ~53M mask entries per layer
and dominate everything the arm was supposed to save.

COMPUTE IS FIXED, ALLOCATION IS LEARNED. The gather width is static
(``frac * S`` tokens every time), so this arm's ms/step is deterministic and
does not drift with the router -- the dynamism is in WHICH tokens get the wide
block, not how many. That is the same bargain MoE makes, and it is deliberate:
a variable width would recompile every step and make wall-clock unpredictable.

CAUSALITY. Selection uses a learned THRESHOLD (a per-token rule, so a token's
own selection never depends on other tokens) capped at a static width with
causal clipping: when a row selects more than the width, the EARLIEST
selections win. Inclusion at position p therefore depends only on decisions at
positions <= p. This is the ``kmod`` pattern already used in this codebase, and
it is chosen over top-k precisely because a top-k over the sequence would make a
token's routing depend on its future. Attention among selected tokens is masked
by original position, so a selected token never sees a later one.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rmsnorm import get_rmsnorm_class
from .swiglu import SwiGLUMLP


class SlowStreamBlock(nn.Module):
    """One wide block applied to a causally-selected subset of tokens."""

    def __init__(self, config, width_mult: float = 2.0, frac: float = 0.2):
        super().__init__()
        d = config.hidden_size
        self.d = d
        self.dw = int(round(d * float(width_mult)))
        self.frac = float(frac)
        self.n_heads = max(1, self.dw // max(1, d // max(1, config.num_attention_heads)))
        self.head_dim = self.dw // self.n_heads
        norm_cls = get_rmsnorm_class(
            zero_centered=getattr(config, "use_zero_centered_rmsnorm", False)
        )
        self.in_norm = norm_cls(d, eps=config.rms_norm_eps)
        self.router = nn.Linear(d, 1, bias=False)
        # Learned selection threshold: a per-token rule, so selection is causal.
        self.register_buffer("route_bias", torch.zeros(1), persistent=True)
        # Loss-free occupancy control, ported from MixtureOfDepths (2026-08-25).
        #
        # WHY THIS EXISTS. Without it `route_bias` is a zero buffer that NOTHING
        # updates, so the router has to learn to push scores above a hard 0 on its
        # own -- and it never did. Measured on the finished ar10m cell
        # (checkpoint-31069), the max score over 16,384 real tokens is NEGATIVE at
        # every checkpoint from step 2,485 on, i.e. the slow stream selected zero
        # tokens for essentially the whole run. With `valid` all-False the
        # write-back is exactly zero, so no gradient reached the slow stream and
        # weight decay annihilated it: q/k/v/o/ffn all ended at ~1e-4 norm against
        # the fast stream's ~3e+1. The arm paid its full +22% wall-clock (the
        # gather width is static) to compute nothing, and its apparent 0.038-nat
        # win over swa_sinkdiet was really the +15% token budget its inflated N
        # bought under D = TPR * N.
        #
        # This is the same failure as the mat_ffn collapse and has the same cause:
        # the router only gets gradient through tokens that ARE selected, so a
        # router that stops selecting can never learn to start again.
        #
        # The fix is MoD's, not an aux loss: nudge a NON-learned bias from realized
        # statistics. An aux penalty would trade off against the LM objective and
        # needs a weight to tune; this converges on the rate itself and has only a
        # step size. Sign differs from MoD because the comparison differs -- MoD
        # tests `score + bias > 0` (bias DOWN to route less), this tests
        # `score > bias` (bias UP to route less).
        # Design follows mat_ffn's `route_mode="threshold"` (swiglu.py:323), not
        # MoD's, in two respects that turn out to be the difference between a
        # controller that works and one that only works eventually:
        #
        #  1. THE SCORE IS BOUNDED. mat_ffn compares a normalized tail against a
        #     theta clamped to [0,1], so lr=0.01 can traverse the entire decision
        #     range in ~100 steps. Comparing a RAW linear output instead leaves
        #     theta an unbounded range to chase: from a fully dead router the
        #     recovery rate is only lr*frac per step, so the observed drift to a
        #     -0.85 median score would take ~4,000 steps to undo at MoD's 1e-3.
        #     Squashing through sigmoid makes the range finite and revival fast,
        #     and makes death structurally impossible: if the router drives every
        #     score to -inf, sigmoid -> 0, theta clamps at 0, and `> 0` fires for
        #     everything, which the controller then pulls back.
        #  2. lr defaults to mat_ffn's 1e-2, not MoD's 1e-3.
        #
        # It differs from mat_ffn on ONE point, deliberately. mat_ffn updates from
        # POST-clip occupancy because its per-sequence integer clip has a ceiling
        # (~735 of a 768 budget) that pre-clip targeting could never reach, which
        # stalled it at 664-675. dual_stream's clip has no such gap -- width is
        # exactly round(frac*S) -- so targeting the PRE-clip fire rate is both
        # reachable and stronger: it forces the ROUTER to allocate rather than
        # letting the earliest-wins clip do it. If the clip is allocating, the arm
        # is not the mechanism it claims to be.
        self.route_bias_lr = float(getattr(config, "dual_stream_route_bias_lr", 0.0) or 0.0)
        if self.route_bias_lr > 0.0:
            # Bounded decision space: theta lives in [0,1] against sigmoid(score).
            # 0.5 == "score > 0", i.e. the old rule, so step 0 is unchanged.
            with torch.no_grad():
                self.route_bias.fill_(0.5)
        self.up = nn.Linear(d, self.dw, bias=False)
        self.q_proj = nn.Linear(self.dw, self.dw, bias=False)
        self.k_proj = nn.Linear(self.dw, self.dw, bias=False)
        self.v_proj = nn.Linear(self.dw, self.dw, bias=False)
        self.o_proj = nn.Linear(self.dw, self.dw, bias=False)
        self.bank_k = nn.Linear(d, self.dw, bias=False)
        self.bank_v = nn.Linear(d, self.dw, bias=False)
        self.mid_norm = norm_cls(self.dw, eps=config.rms_norm_eps)
        self.ffn = SwiGLUMLP(self.dw, self.dw * 2, getattr(config, "hidden_act", "silu"))
        self.down = nn.Linear(self.dw, d, bias=False)
        # Zero-init write-back: at step 0 the arm is BITWISE the fast stream
        # alone, so any measured difference is the slow stream doing work.
        nn.init.zeros_(self.down.weight)
        # Marker required: HF's `_init_weights` re-inits every nn.Linear after
        # construction, which silently overwrote this zero until 2026-08-21 --
        # so the "at step 0 the arm is BITWISE the fast stream" claim above was
        # false as built, and a random `down` was writing a 2x-wide stream into
        # the residual from step 0.
        self.down._zero_init_weight = True
        # dual_stream_kmax (2026-08-25): band the slow stream's SELF-attention to
        # the previous `kmax` SELECTED tokens. The bank stays fully visible.
        #
        # Why this and not a cap on the selection count: k = frac * S, so the
        # k x k self-attention is frac^2 * S^2 -- still QUADRATIC, just with a
        # 0.04 constant. Modelled at ar10m, the slow stream is ~7% of the SWA
        # attention work at 4K but ~47% at 32K (it grows 56x while the windowed
        # fast stream grows 9x), because nothing caps k the way the 1024 band
        # caps the fast stream. Capping the COUNT instead would be truncation,
        # not allocation: with earliest-wins clipping the first `kmax` candidates
        # would consume the whole budget and the tail of a long document would
        # get no slow-stream compute at all. A band keeps every token eligible
        # and makes the cost k*kmax -- linear in S.
        #
        # Inert whenever k <= kmax (no band is built), so at any S where
        # frac*S <= kmax this arm is BITWISE `dual_stream`.
        _km = getattr(config, "dual_stream_kmax", None)
        self.kmax = int(_km) if _km else None
        self._last_frac = None  # realized (== frac; static width)
        self._last_threshold_frac = None  # how often the router fired
        self.collect_stats = False  # counters cost a host sync; opt in
        self._forced_frac = None

    # ---------------------------------------------------------------- selection
    def _select(self, x):
        """Causal threshold, static width, earliest-wins clipping.

        Returns [B, width] sorted original positions. The priority trick gives
        all three properties in one topk: selected tokens outrank unselected
        ones, and WITHIN each group earlier positions outrank later ones, so a
        row that over-selects keeps its earliest choices and a row that
        under-selects is padded with its earliest unselected positions. The
        gather width is a compile-time constant for the whole run.
        """
        b, s, _ = x.shape
        width = max(1, int(round((self._forced_frac or self.frac) * s)))
        score = self.router(self.in_norm(x)).squeeze(-1).float()  # [B,S]
        if self._forced_frac is not None:
            keep = torch.zeros_like(score, dtype=torch.bool)
            keep[:, :: max(1, s // width)] = True  # pinned occupancy
        else:
            if self.route_bias_lr > 0.0:
                keep = torch.sigmoid(score) > self.route_bias
                if self.training:
                    self._update_route_bias(keep)
            else:
                keep = score > self.route_bias
        # Earliest-wins clip: a candidate at r survives iff at most `width`
        # candidates occur at positions <= r, which depends only on tokens <= r.
        rank = keep.long().cumsum(dim=1)
        valid_full = keep & (rank <= width)
        order = torch.arange(s, device=x.device, dtype=torch.float32)
        priority = valid_full.float() * float(s + 1) + (s - order).unsqueeze(0)
        idx = priority.topk(width, dim=1).indices.sort(dim=1).values
        # `valid` marks which of the `width` gathered rows are REAL candidates
        # rather than padding. This is load-bearing for causality, not cosmetic:
        # the number of padding slots depends on the TOTAL candidate count over
        # the whole sequence -- a global quantity -- so a new candidate at p can
        # displace a padding row at q < p. Measured on the first version of this
        # module, that leaked at 4 of 6 probe positions (up to 2.8e-1). Padding
        # rows are therefore zeroed out of BOTH the write-back and the attention
        # keys below, which makes the output independent of the padding choice.
        valid = valid_full.gather(1, idx)
        # float(tensor) is a device->host SYNC. These are diagnostics, so they
        # are opt-in: the bench and eval turn them on, training never pays.
        if self.collect_stats:
            self._last_threshold_frac = float(keep.float().mean().detach())
            self._last_frac = float(valid_full.float().mean().detach())
        return idx, score, valid

    @torch.no_grad()
    def _update_route_bias(self, keep: torch.Tensor) -> None:
        """Nudge the threshold so the realized fire rate tracks ``frac``.

        Tracks ``keep`` (the raw threshold rate) rather than the post-clip count:
        the goal is for the router to fire on about ``frac`` of tokens so the
        earliest-wins clip rarely binds. If the clip is doing the allocating, the
        router is not the mechanism.

        Stays a tensor op -- no ``.item()`` -- so it neither syncs nor breaks the
        compiled graph.
        """
        realized = keep.float().mean()
        self.route_bias += self.route_bias_lr * (realized - self.frac)
        self.route_bias.clamp_(0.0, 1.0)

    # ------------------------------------------------------------------ forward
    def forward(self, x, bank: Optional[torch.Tensor] = None, doc_ids=None):
        b, s, d = x.shape
        idx, score, valid = self._select(x)
        k = idx.shape[1]
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, d)
        xs = x.gather(1, gather_idx)  # [B,k,d]

        h = self.up(self.in_norm(xs))  # [B,k,dw]
        q = self.q_proj(h).view(b, k, self.n_heads, self.head_dim).transpose(1, 2)
        kk = self.k_proj(h).view(b, k, self.n_heads, self.head_dim).transpose(1, 2)
        vv = self.v_proj(h).view(b, k, self.n_heads, self.head_dim).transpose(1, 2)

        # Causal by ORIGINAL position; same-document when doc ids are supplied.
        pos = idx.unsqueeze(-1)  # [B,k,1]
        ok = pos.transpose(1, 2) <= pos  # [B,k,k]
        # Padding rows must not be visible as KEYS either -- a real token at q
        # attending to a padded token at r < q would reintroduce the same
        # global-count dependence the write-back gate removes.
        ok = ok & valid.unsqueeze(1)
        if doc_ids is not None:
            dsel = doc_ids.gather(1, idx).unsqueeze(-1)
            ok = ok & (dsel == dsel.transpose(1, 2))
        if self.kmax is not None and k > self.kmax:
            # Band in SELECTED-SLOT space. `idx` is sorted ascending, so slot
            # order is position order and a slot band is "the previous kmax
            # selected tokens". Causal already holds via `ok`; this only narrows.
            slot = torch.arange(k, device=x.device)
            ok = ok & ((slot.view(1, k, 1) - slot.view(1, 1, k)) < self.kmax)
        if bank is not None and bank.numel():
            r = bank.shape[0] if bank.dim() == 2 else bank.shape[1]
            bt = bank if bank.dim() == 3 else bank.unsqueeze(0).expand(b, -1, -1)
            bk = self.bank_k(bt).view(b, r, self.n_heads, self.head_dim).transpose(1, 2)
            bv = self.bank_v(bt).view(b, r, self.n_heads, self.head_dim).transpose(1, 2)
            kk = torch.cat([bk, kk], dim=2)
            vv = torch.cat([bv, vv], dim=2)
            # The bank is out-of-sequence: every selected token may read it.
            ok = torch.cat([ok.new_ones(b, k, r), ok], dim=-1)
        # Pass the BOOL mask straight through. Materialising it as a float
        # additive mask allocated a [B, k, k+r] fp32 tensor and ran a
        # masked_fill over it once per slow block per step; SDPA accepts a bool
        # mask directly (True = keep), which is 4x less memory traffic and drops
        # the fill entirely. At S=4096 with frac=0.2, k is ~819, so this is a
        # real allocation, not a rounding error.
        a = F.scaled_dot_product_attention(q, kk, vv, attn_mask=ok.unsqueeze(1))
        h = h + self.o_proj(a.transpose(1, 2).reshape(b, k, self.dw))
        h = h + self.ffn(self.mid_norm(h))

        # Straight-through gate so the router receives gradient.
        g = torch.sigmoid(score.gather(1, idx)).unsqueeze(-1).to(h.dtype)
        upd = self.down(h) * (g + (1.0 - g.detach())) * valid.unsqueeze(-1).to(h.dtype)
        return x.scatter_add(1, gather_idx, upd.to(x.dtype))
