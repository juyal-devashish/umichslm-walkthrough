"""S1 ``hourglass``: learned dynamic pooling as the compute allocator.

The idea (docs/dyncompute_redesign.md, sketch S1). Rounds 1-7 measured three
things about dynamic compute in this codebase:

  1. every routing point costs ~30 ms of fixed machinery, so 11 cheap decisions
     cost far more than one expensive one;
  2. moving routed tokens is bandwidth-bound and lives in the backward
     (~250 ms/step of ``[B,S,h]`` scatter_add at 11 stages);
  3. ADDING effective layers costs ~12.5 ms each, but REMOVING sequence
     positions from expensive layers is nearly free -- with a sliding window,
     attention is O(S*W) and the FFN is O(S), so both fall linearly.

This arm is what (3) looks like taken seriously. Thin layers run at full length
at the bottom and top; in between, the sequence is pooled to a FIXED number of
segments and the expensive middle stack -- where the banks live -- runs on the
short sequence. There is no token gather and no scatter: pooling is one
segment-mean and unpooling is one gather, both dense ops.

Where the DYNAMISM comes from. The number of segments is fixed (so shapes are
static and compile-friendly -- variable segment counts would recompile every
step), but WHICH positions start a segment is content-dependent: a learned
boundary head scores every token and the top-K become segment starts. Dense,
hard regions therefore get more boundaries and are represented at finer
granularity -- more middle-stack compute per token -- while boilerplate is
compressed harder. Total compute is fixed and its ALLOCATION is learned, which
is the same bargain MoE makes with parameters.

CAUSALITY -- the part that is easy to get wrong. A segment mean pools tokens
that include the future of earlier tokens in the same segment, so handing that
pooled vector back to those tokens would leak the future and quietly inflate
every loss number. This module therefore applies the Hourglass shift: a token
in segment j receives the middle stack's output for segment ``j-1``, never j.
Tokens in segment 0 receive zero. Each token still sees its own local context
through the full-resolution outer layers; the middle stack supplies deep,
compressed context strictly from the past.

Documents. Boundaries are FORCED at every document start, so no segment ever
straddles two documents, and the segment-level attention mask is causal AND
same-document. Without that, pooling would mix documents and defeat the
intra-doc masking the rest of the stack is careful about.
"""

from typing import Optional

import torch
import torch.nn as nn

from .rmsnorm import get_rmsnorm_class


class HourglassMiddle(nn.Module):
    """Pool -> deep stack -> (shifted) unpool, inserted at one layer boundary.

    Holds its own middle decoder layers; the parent model's ``layers`` list
    carries only the full-resolution outer layers.
    """

    def __init__(self, config, layer_cls, num_layers: int, pool_ratio: int, base_idx: int = 0):
        super().__init__()
        self.pool_ratio = max(2, int(pool_ratio))
        self.hidden_size = config.hidden_size
        self.layers = nn.ModuleList(layer_cls(config, base_idx + i) for i in range(int(num_layers)))
        norm_cls = get_rmsnorm_class(
            zero_centered=getattr(config, "use_zero_centered_rmsnorm", False)
        )
        self.pool_norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.boundary = nn.Linear(config.hidden_size, 1, bias=False)
        # Straight-through gate so the boundary head gets gradient: the pooled
        # vector is scaled by sigmoid(score) with value 1.0 in the forward.
        self.out_scale = nn.Parameter(torch.ones(1))
        # Learned boundary threshold. A per-token test keeps selection causal;
        # see _segment_ids for why a global top-K does not.
        self.boundary_bias = nn.Parameter(torch.zeros(1))
        # Attention path for the middle stack. The segment-level constraint is
        # causal AND same-document, and the FA2 non-varlen path silently DROPS
        # additive masks (flash_attn_func gets causal=True and nothing else), so
        # a naive FA2 layer would leak across documents -- the same failure the
        # flex path had once. Two correct options:
        #   attn_mode="sdpa"    materialise a [B,1,K,K] additive mask (safe,
        #                       O(K^2) memory);
        #   attn_mode="varlen"  segments are DOC-PURE by construction (boundaries
        #                       are forced at every document start), so the same
        #                       constraint is exactly FA2 varlen over
        #                       segment-level cu_seqlens -- O(K) and no mask
        #                       tensor at all.
        # varlen is the fast path and the default; sdpa is kept for the A/B and
        # as the fallback when doc ids are unavailable.
        self.attn_mode = str(getattr(config, "hourglass_attn_mode", "varlen"))
        self._set_sdpa(self.attn_mode != "varlen")

        self._last_segments = None
        self._last_boundary_frac = None
        # Counters that need a device->host copy are opt-in; the bench and the
        # eval probes turn them on, training never pays for them.
        self.collect_stats = False

    def _set_sdpa(self, force: bool):
        for layer in self.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "_intra_doc_force_sdpa"):
                attn._intra_doc_force_sdpa = bool(force)

    def _set_varlen(self, seg_doc, n_seg):
        """Publish segment-level cu_seqlens so the middle stack runs FA2 varlen.

        Returns True when the varlen path is armed. Segments are doc-pure, so a
        document's segments are a contiguous run and the block-diagonal causal
        structure varlen implements is exactly the mask we would otherwise
        materialise.
        """
        if self.attn_mode != "varlen" or seg_doc is None:
            return False
        from .doc_masking import build_batched_cu_seqlens

        cu, mx = build_batched_cu_seqlens(seg_doc)
        for layer in self.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None or not hasattr(attn, "_intra_doc_cu_seqlens"):
                return False
            attn._intra_doc_cu_seqlens = cu
            attn._intra_doc_max_seqlen = mx
            attn._intra_doc_force_sdpa = False
        return True

    def _clear_varlen(self):
        for layer in self.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "_intra_doc_cu_seqlens"):
                attn._intra_doc_cu_seqlens = None
                attn._intra_doc_max_seqlen = None

    # ------------------------------------------------------------------ pooling
    def _segment_ids(self, x, doc_ids):
        """[B,S] segment index in [0, K), K = S // pool_ratio.

        CAUSAL boundary selection. The obvious implementation -- score every
        token and take a global top-K -- is NOT causal: top-K is a whole-
        sequence order statistic, so perturbing a token at position p can evict
        a boundary at q < p, shifting the segment ids of earlier tokens and
        letting information flow backwards. Measured on the first version of
        this module: perturbing token 400 changed the logits at positions < 400
        by up to 7.7e-2. The first causality test passed only because that
        particular perturbation happened not to move the top-K.

        The causal rule instead is threshold + static width + earliest-wins,
        the same pattern the ``kmod`` arm and the dual-stream router use:
          * a position is a candidate boundary iff its score exceeds a learned
            threshold (a per-token test -- no dependence on other tokens), or it
            starts a document (forced, so a segment never spans two documents);
          * candidates are taken in POSITION order until the budget K is spent,
            so whether q is a boundary depends only on scores at positions <= q;
          * if a row underspends, the earliest non-candidates pad it out, which
            can only add boundaries LATER than the last real one.
        Segment ids at positions < p are therefore invariant to any change at p,
        and the segment count stays exactly K so shapes are static.
        """
        b, s, _ = x.shape
        k = max(1, s // self.pool_ratio)
        score = self.boundary(self.pool_norm(x)).squeeze(-1).float()  # [B,S]
        forced = torch.zeros_like(score, dtype=torch.bool)
        forced[:, 0] = True
        if doc_ids is not None:
            forced[:, 1:] |= doc_ids[:, 1:] != doc_ids[:, :-1]
        cand = forced | (score > self.boundary_bias)
        # Earliest-wins: keep the first k candidates of each row.
        rank = cand.long().cumsum(dim=1)
        starts = cand & (rank <= k)
        # Pad short rows with their earliest non-candidates. Priority puts real
        # candidates first, then earlier positions -- one topk gives both.
        order = torch.arange(s, device=x.device, dtype=torch.float32)
        priority = starts.float() * float(s + 1) + (s - order).unsqueeze(0)
        idx = priority.topk(k, dim=1).indices
        starts = torch.zeros_like(cand)
        starts.scatter_(1, idx, True)
        starts[:, 0] = True
        starts |= forced & (forced.long().cumsum(dim=1) <= k)
        seg = (starts.long().cumsum(dim=1) - 1).clamp(min=0)  # [B,S]
        # OPTIMIZATION (2026-08-20): the segment count is EXACTLY k by
        # construction -- the selection takes k starts and the cumsum runs
        # 0..k-1 -- so `seg.max().item()` was a per-layer HOST SYNC computing a
        # number already known at trace time. Removing it also lets the pooled
        # shapes stay compile-time constants. The counters are now recorded
        # lazily and only when someone asks (bench/eval), never on the hot path.
        seg = seg.clamp(max=k - 1)
        if self.collect_stats:
            self._last_segments = k
            self._last_boundary_frac = float(starts.float().mean().item())
        else:
            self._last_segments = k
        return seg, starts, k, score

    def _pool(self, x, seg, n_seg):
        """Segment mean -> [B, n_seg, d]."""
        b, s, d = x.shape
        out = x.new_zeros(b, n_seg, d)
        cnt = x.new_zeros(b, n_seg, 1)
        out.scatter_add_(1, seg.unsqueeze(-1).expand(-1, -1, d), x)
        cnt.scatter_add_(1, seg.unsqueeze(-1), torch.ones_like(cnt[:, :1]).expand(-1, s, -1))
        return out / cnt.clamp(min=1.0)

    def _seg_doc(self, seg, doc_ids, n_seg):
        if doc_ids is None:
            return None
        b = seg.shape[0]
        out = seg.new_full((b, n_seg), -1)
        out.scatter_(1, seg, doc_ids)  # segments are doc-pure by construction
        return out

    def _seg_mask(self, seg_doc, n_seg, device, dtype):
        """Causal AND same-document mask over segments, [B or 1, 1, K, K]."""
        causal = torch.ones(n_seg, n_seg, dtype=torch.bool, device=device).tril()
        if seg_doc is None:
            keep = causal.view(1, 1, n_seg, n_seg)
        else:
            same = seg_doc.unsqueeze(-1) == seg_doc.unsqueeze(-2)  # [B,K,K]
            keep = (same & causal).unsqueeze(1)
        mask = torch.zeros(keep.shape, device=device, dtype=torch.float32)
        mask.masked_fill_(~keep, torch.finfo(torch.float32).min)
        return mask.to(dtype)

    # ------------------------------------------------------------------ forward
    def forward(self, hidden_states, position_ids=None, doc_ids=None, **kw):
        b, s, d = hidden_states.shape
        seg, starts, n_seg, score = self._segment_ids(hidden_states, doc_ids)

        pooled = self._pool(hidden_states, seg, n_seg)
        seg_doc = self._seg_doc(seg, doc_ids, n_seg)
        used_varlen = self._set_varlen(seg_doc, n_seg)
        attn = (
            None
            if used_varlen
            else self._seg_mask(seg_doc, n_seg, hidden_states.device, hidden_states.dtype)
        )
        pos = torch.arange(n_seg, device=hidden_states.device).unsqueeze(0).expand(b, -1)

        h = pooled
        for layer in self.layers:
            out = layer(
                hidden_states=h,
                attention_mask=attn,
                position_ids=pos,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            h = out[0] if isinstance(out, tuple) else out

        if used_varlen:
            self._clear_varlen()  # never let a stale map leak into the next call

        # CAUSAL UNPOOL: segment j reads the stack's output for segment j-1.
        # Reading j would hand a token the mean of its own (future-containing)
        # segment. Segment 0 has no predecessor and gets zeros.
        prev = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        up = prev.gather(1, seg.unsqueeze(-1).expand(-1, -1, d))

        # Straight-through boundary gate: value 1.0 forward, gradient to the
        # boundary head so the top-K selection is trainable at all.
        g = torch.sigmoid(score).unsqueeze(-1).to(up.dtype)
        up = up * (g + (1.0 - g.detach()))
        return hidden_states + self.out_scale.to(up.dtype) * up
