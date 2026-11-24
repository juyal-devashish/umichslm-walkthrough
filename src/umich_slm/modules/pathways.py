"""Mixture of Pathways (MoP) — dynamic-compute chunk 2, Pitch 2.

"Depth-specialized experts, one committed choice."

MoD re-decides token fate at every block. MoP makes ONE causal commitment per
token among parallel pathways of different *depth* — experts that differ in
compute, not just parameters:

  - Shared bottom layers (dense) build a representation.
  - A token-choice router commits each token to a lane from
    ``pathway_depths`` (e.g. [0, 2, 4]: express / standard / deep-thought).
    The decision reads only the token's own (causal) hidden state, so there
    is exactly zero train/infer routing mismatch by construction.
  - Lane blocks cross-attend to the shared-stream export (K/V from the last
    shared-bottom layer, under the model's causal/doc mask), never to other
    lane tokens — an express token's context is still fully visible to
    deep-lane tokens, and no token is ever missing from anyone's K/V.
  - Lanes rejoin the residual stream through a per-lane RMSNorm + zero-init
    learned scale (safe insertion: at init the model is bitwise the shared
    stack alone), then shared top layers + LM head finish.

Because lane blocks never mix lane tokens with each other (cross-attention
reads only the routing-independent shared export, and the MLP is
position-wise), computing every lane densely and selecting per token at the
merge is bitwise identical to gathering each lane's tokens — the realized
implementation does the former (simple, correct); gather batching is a kernel
optimization deferred to the efficiency pass. Algorithmic accounting bills
the *expected* lane cost (see ``umich_slm.compute_accounting``).

Cost-aware load balancing (the novel twist): instead of balancing token
counts, the aux loss balances EXPECTED COMPUTE against a budget —
``((sum_i f_i * depth_i - budget) / max_depth)^2`` plus a mild entropy term —
so an easy batch legitimately sends more tokens express and a hard batch
spends more. Emitted through the existing ``mod_aux_loss`` channel (F4).

Deep-lane starvation guard: for the first ``pathway_floor_steps`` training
steps, the deepest lane is guaranteed at least ``pathway_floor_frac`` of
content tokens (the most deep-preferring tokens by router logit are forced),
then released.

Parameter-sharing ladder (each sweep arm changes exactly one thing):

  - ``pathway_share_kv`` — every lane block cross-attends to the SAME frozen
    export, so K/V are projected once by a shared head and reused by every
    block in every lane (common "how to read the shared stream" signal; also
    saves the per-block K/V params and FLOPs). Arm: ``mop_kv``.
  - ``pathway_variant="nested"`` — lanes are PREFIXES of one shared lane
    stack: deep = [B1..B4], standard exits after B2, express before B1. The
    early blocks are the common low/intermediate processing (trained by
    standard + deep tokens together); depth specialization lives in the tail
    blocks. Arm: ``mop_nested``.
  - ``pathway_variant="recursive"`` — one weight-tied block; lane depth is
    the ITERATION count of that block (0/2/4 passes). Maximal signal sharing:
    every routed token trains the same parameters, and depth is pure compute.
    Arm: ``mop_rec`` (the MoR-flavored end of the spectrum).

All variants preserve the core invariants: one causal commitment, no
lane-token mixing (dense compute == gathered, bitwise), budget-balanced aux.

KV-cache decode is deferred like Arm B/GWT (config forces
``use_cache=False``): lane cross-attention over cached shared-stream K/V is
standard cache semantics but needs its own cache plumbing — an inference
integration, not a screening prerequisite. Routing itself is decode-stable
(per-token argmax of causal state; no batch coupling).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rmsnorm import get_rmsnorm_class
from .swiglu import SwiGLUMLP

# Gather widths are rounded up to this multiple so flex_attention (compiled with
# dynamic=False) sees a handful of shapes rather than a new one every step. Also
# a valid FlexAttention Q_LEN granularity (multiple of the 128 block size).
_GATHER_BUCKET = 512


def pathway_budget_blocks(config) -> float:
    """Resolve the compute budget in expected lane-blocks per token."""
    budget = float(getattr(config, "pathway_budget", 0.0) or 0.0)
    if budget > 0:
        return budget
    depths = list(getattr(config, "pathway_depths", None) or [0, 2, 4])
    return sum(depths) / len(depths)


class PathwayRouter(nn.Module):
    """Token-choice lane router: one causal commitment per token."""

    def __init__(self, hidden_size: int, num_lanes: int):
        super().__init__()
        self.router = nn.Linear(hidden_size, num_lanes, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router(x)
        probs = F.softmax(logits.float(), dim=-1).to(x.dtype)
        lane = logits.argmax(dim=-1)
        return lane, probs, logits


def _maybe_chunk_route(mod, lane, logits):
    """D2 `chunk_mop`: one lane commitment per CONTIGUOUS CHUNK of tokens.

    Motivation is the measured cost model: a per-token depth router costs ~30 ms
    of fixed machinery per stage plus a bandwidth-bound scatter in the backward.
    Deciding once per 128-256 tokens amortizes the router ~200x and turns each
    tier's token set into near-contiguous runs, which the permute dispatch turns
    into slices instead of gathers.

    CAUSALITY: the chunk's lane comes from its FIRST token's logits, never from
    a mean over the chunk. Mean-pooling would let tokens later in the chunk
    decide the routing of tokens earlier in it -- the same future-leak the
    top-k MoD router was criticised for (docs, submit_scaling_curve.sh header).
    Deciding from the first token is strictly causal and also cheaper.

    Quality caveat, measured BEFORE this was written: against the trained
    swa_scribe_mopd router only 16% (block 128) / 13% (block 256) of per-token
    depth variance is between-chunk, so a chunk router gives up most of the
    allocation precision. See docs/dyncompute_redesign.md sketch D2.
    """
    c = int(getattr(mod, "route_chunk", 1) or 1)
    if c <= 1 or logits is None or logits.dim() != 3:
        return lane
    b, t, k = logits.shape
    n = (t // c) * c
    if n <= 0:
        return lane
    head = logits[:, :n].view(b, n // c, c, k)[:, :, 0, :]   # FIRST token per chunk
    chunk_lane = head.argmax(dim=-1).repeat_interleave(c, dim=1)  # [B, n]
    lane = lane.clone()
    lane[:, :n] = chunk_lane.to(lane.dtype)
    return lane


def _broadcast_chunk_first(lane, chunk: int, num_prefix: int):
    """Re-impose chunk-uniform lanes using each chunk's FIRST token.

    The deep-lane starvation floor promotes INDIVIDUAL tokens, which silently
    breaks D2's contract -- measured: with the floor armed, every 128-token
    chunk carried 2 distinct lanes, so the arm was not actually chunk-routed
    and would have benched as ordinary mopr with a slightly different router.
    Re-broadcasting after the floor keeps the floor's purpose (a chunk whose
    first token was promoted stays promoted, so the deep lane still gets
    roughly floor_frac of the mass) while restoring chunk uniformity.
    Using the FIRST token, not the max over the chunk, keeps the rule causal.
    """
    b, t = lane.shape
    start = int(num_prefix)
    n = ((t - start) // chunk) * chunk
    if n <= 0:
        return lane
    seg = lane[:, start : start + n].view(b, n // chunk, chunk)
    lane = lane.clone()
    lane[:, start : start + n] = seg[:, :, :1].expand(-1, -1, chunk).reshape(b, n)
    return lane


def _maybe_force_lane(mod, x, lane):
    """Bench hook: replace the router's argmax with the budget-consistent
    allocation (utils/forced_routing.py). Inert unless a bench pinned the
    module, and never imported on a training path -- the import is local so a
    training process never pays for it."""
    if getattr(mod, "_forced_lane_fracs", None):
        from ..utils.forced_routing import apply_forced_lane

        forced = apply_forced_lane(mod, x)
        if forced is not None:
            return forced.to(lane.dtype)
    return lane


class SharedKVProjection(nn.Module):
    """K/V projected ONCE from the (already normed) export and shared by every
    lane block (``pathway_share_kv``). Returns head-shaped K/V at the NATIVE
    kv-head count — the attention calls use ``enable_gqa`` instead of a
    ``repeat_interleave`` materialization (the sweep shapes are MQA/GQA:
    kv=1 vs 7-11 q heads, so expanding K/V cost 7-11x the memory traffic).
    """

    def __init__(self, config):
        super().__init__()
        h = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = h // self.num_heads
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", None) or self.num_heads)
        kv_dim = self.num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(h, kv_dim, bias=False)
        self.v_proj = nn.Linear(h, kv_dim, bias=False)

    def forward(self, export: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = export.shape
        k = self.k_proj(export).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(export).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return k, v


class PathwayLaneBlock(nn.Module):
    """One lane block: cross-attention to the shared export + SwiGLU MLP.

    GQA matching the backbone's KV head count; no position encoding of its own
    (positions are already baked into the shared-stream representations, and
    query/key rows are index-aligned under the model's causal mask).

    With ``shared_kv=True`` the block owns no K/V projections — the caller
    passes precomputed (k, v) from :class:`SharedKVProjection`.
    """

    def __init__(self, config, shared_kv: bool = False):
        super().__init__()
        h = config.hidden_size
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.num_heads = config.num_attention_heads
        self.head_dim = h // self.num_heads
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", None) or self.num_heads)
        if self.head_dim * self.num_heads != h:
            raise ValueError(f"hidden_size {h} not divisible by num_attention_heads")
        self.shared_kv = bool(shared_kv)
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(h, h, bias=False)
        if not self.shared_kv:
            self.k_proj = nn.Linear(h, kv_dim, bias=False)
            self.v_proj = nn.Linear(h, kv_dim, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)

        norm_class = get_rmsnorm_class(
            zero_centered=getattr(config, "use_zero_centered_rmsnorm", False)
        )
        self.input_layernorm = norm_class(h, eps=eps)
        self.post_attention_layernorm = norm_class(h, eps=eps)
        self.mlp = SwiGLUMLP(
            hidden_size=h,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(config, "hidden_act", "silu"),
            bias=getattr(config, "mlp_bias", False),
        )

    def forward(
        self,
        x: torch.Tensor,
        export: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        block_mask: Optional[object] = None,
    ) -> torch.Tensor:
        """``x`` may be a GATHERED subset of the sequence (``[B, k, H]``) while
        ``export``/``kv`` always span the full sequence — lane blocks read only
        the shared stream, so a query subset changes nothing about the keys.
        ``block_mask`` (FlexAttention) already encodes causality in ORIGINAL
        position space; ``attention_mask`` is the SDPA fallback and must already
        be gathered to ``[B, 1, k, T]`` by the caller.
        """
        b, t, h = x.shape
        q = (
            self.q_proj(self.input_layernorm(x))
            .view(b, t, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        if kv is not None:
            k, v = kv
        else:
            if self.shared_kv:
                raise ValueError("shared_kv block requires precomputed kv")
            s = export.shape[1]
            k = self.k_proj(export).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(export).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # K/V stay at the native kv-head count; the kernels group q heads onto
        # kv heads in-kernel (enable_gqa) instead of reading a repeat_interleave
        # materialization -- 7-11x less K/V memory traffic on the MQA shapes.
        gqa = k.shape[1] != self.num_heads
        if gqa and block_mask is not None and q.shape[2] < 128:
            # Short-q flex dispatches to the flex-DECODING kernel, which has no
            # valid GQA autotune configs on some backends (NoValidChoicesError,
            # seen on sm86 / torch 2.11). Expand instead — sub-128-row calls
            # are negligible, and production lane widths are bucketed >= 512.
            rep = self.num_heads // k.shape[1]
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
            gqa = False
        if block_mask is not None:
            # Block-sparse fused kernel: skips fully-masked score blocks (causal
            # + document structure), so this never materialises the [B,1,S,S]
            # additive mask the SDPA branch below reads once per lane block.
            from .flex_swa import flex_sdpa

            attn = flex_sdpa(q, k, v, block_mask, enable_gqa=gqa)
        elif attention_mask is not None:
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, enable_gqa=gqa
            )
        else:
            # No 4D mask supplied (e.g. FA2 config): queries and keys are
            # index-aligned over the same sequence, so plain causal is exact.
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
        x = x + self.o_proj(attn.transpose(1, 2).reshape(b, t, h))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Pathways(nn.Module):
    """The full MoP mechanism: router + parallel depth lanes + merge + aux.

    Diagnostics stashed per forward (consumed by dyncompute_diagnostics):
      - ``_last_lane_assignment``: [B, T] committed lane per token.
      - ``_last_lane_fractions``: realized hard fraction per lane.
      - ``_last_expected_cost``: differentiable expected blocks/token (float).
    """

    def __init__(self, config):
        super().__init__()
        h = config.hidden_size
        eps = getattr(config, "rms_norm_eps", 1e-6)
        depths = list(getattr(config, "pathway_depths", None) or [0, 2, 4])
        if not depths or any(d < 0 for d in depths) or max(depths) == 0:
            raise ValueError(f"pathway_depths must be non-negative with max > 0, got {depths}")
        # Memory lane (pathway_memory_lane, requires pathway_bank_export): one
        # extra routed lane whose block reads ONLY the bank rows appended to
        # the stage export — "consult memory" as a per-token choice. It is NOT
        # part of the nested compute stack: _tier_plan and the tier loops
        # exclude its index; its depth counts in the router budget.
        self.memory_lane_index: Optional[int] = None
        if bool(getattr(config, "pathway_memory_lane", False)):
            md = int(getattr(config, "pathway_memory_lane_depth", 1))
            if md in depths:
                raise ValueError(
                    f"pathway_memory_lane_depth {md} collides with a compute depth {depths}"
                )
            self.memory_lane_index = len(depths)
            depths = depths + [md]
        self.depths = depths
        self.num_lanes = len(depths)
        self.deep_lane = max(
            (i for i in range(self.num_lanes) if i != self.memory_lane_index),
            key=lambda i: depths[i],
        )
        # Permute-not-gather dispatch (math-identical; see forward): needs the
        # sorted-by-lane order to make depth tiers contiguous suffixes.
        self.permute_dispatch = bool(getattr(config, "pathway_permute_dispatch", False))
        if self.permute_dispatch and depths != sorted(depths):
            raise ValueError(
                f"pathway_permute_dispatch requires pathway_depths monotone in lane "
                f"index, got {depths}"
            )
        self.budget = pathway_budget_blocks(config)
        self.aux_weight = float(getattr(config, "pathway_aux_loss_weight", 0.01))
        self.entropy_weight = float(getattr(config, "pathway_entropy_weight", 0.001))
        self.route_chunk = int(getattr(config, "pathway_route_chunk", 1) or 1)
        self.floor_frac = float(getattr(config, "pathway_floor_frac", 0.0))
        self.floor_steps = int(getattr(config, "pathway_floor_steps", 0))

        self.variant = str(getattr(config, "pathway_variant", "disjoint"))
        self.share_kv = bool(getattr(config, "pathway_share_kv", False))
        # Pure-performance switches: both paths are mathematically identical to
        # the dense/SDPA original (lane blocks read only the shared export, so a
        # query subset is exact, and the flex BlockMask encodes the same
        # causal ∧ intra-doc predicate the additive mask did). Off => the
        # original code path, kept for A/B timing and for CPU fallback.
        self.gather_lanes = bool(getattr(config, "pathway_gather_lanes", True))
        self.use_flex_attention = bool(getattr(config, "pathway_flex_attention", True))
        # Interval-form lane BlockMask (math-identical; cheaper mask_mod in the
        # fwd/bwd flex kernels + one shared per-step lo table). Off => the
        # general causal∧doc∧window predicate, kept for A/B timing.
        self.interval_mask = bool(getattr(config, "pathway_interval_mask", True))
        self.export_interval = int(getattr(config, "pathway_export_interval", 1))
        # Position of this stage in the stacked ModuleList; set by the model
        # when stages are built (0 for a single-stage MoP).
        self._stage_index = 0
        # Gather-width bucket: rows pad to the per-tier row max rounded up to
        # this. Smaller = less padding compute, more distinct compile/flex
        # shapes; 512 was the original module constant. Tunable per arm.
        self.gather_bucket = int(getattr(config, "pathway_gather_bucket", _GATHER_BUCKET))
        # Static per-tier gather widths (mop-stack-k). Aligned with
        # _tier_plan() order; validated once the tier plan is known.
        fracs = getattr(config, "pathway_fixed_tier_fracs", None)
        self.fixed_tier_fracs = None if fracs is None else [float(f) for f in fracs]
        if self.fixed_tier_fracs is not None:
            n_tiers = len(sorted({d for d in depths if d > 0}))
            if len(self.fixed_tier_fracs) != n_tiers:
                raise ValueError(
                    f"pathway_fixed_tier_fracs needs one entry per positive-depth tier "
                    f"({n_tiers} for depths {depths}), got {self.fixed_tier_fracs}."
                )
            if any(not (0.0 < f <= 1.0) for f in self.fixed_tier_fracs):
                raise ValueError(
                    f"pathway_fixed_tier_fracs must be in (0, 1], got {self.fixed_tier_fracs}."
                )
        # One async DtoH of all tier widths per forward (see
        # _prefetch_tier_counts) instead of a blocking sync per tier.
        self._tier_count_pinned: Optional[torch.Tensor] = None
        self._tier_count_event = None
        self._tier_counts_pending = False
        # Lane cross-attention band. The lane blocks were the ONLY full-context
        # attention in these architectures -- every backbone layer is windowed at
        # swa_window -- which both cost ~2.3x the attention work of a backbone
        # layer and confounded the sweep: `mopd` was the sole arm able to attend
        # past the band, so a win over `swa_scribe` could not be attributed to
        # MoP. None inherits swa_window; 0 restores the old full-context lanes.
        lw = getattr(config, "pathway_lane_window", None)
        if lw is None:
            lw = getattr(config, "swa_window", None) or getattr(config, "sliding_window", None) or 0
        self.lane_window = int(lw)
        self._lane_device = None
        self._lane_batch = None
        self._lane_dtype = torch.float32
        self._use_flex = False

        self.router = PathwayRouter(h, self.num_lanes) if self.num_lanes > 1 else None
        norm_class = get_rmsnorm_class(
            zero_centered=getattr(config, "use_zero_centered_rmsnorm", False)
        )
        self.export_norm = norm_class(h, eps=eps)
        self.shared_kv_proj = SharedKVProjection(config) if self.share_kv else None

        # Lane parameterization by variant:
        #   disjoint  — each lane owns its full block stack (max specialization)
        #   nested    — ONE stack of max(depths) blocks; lanes exit at their
        #               depth (early blocks shared by standard+deep tokens)
        #   recursive — one weight-tied block iterated depth times (max sharing)
        self.lanes = None
        self.lane_stack = None
        if self.variant == "disjoint":
            self.lanes = nn.ModuleList(
                nn.ModuleList(PathwayLaneBlock(config, shared_kv=self.share_kv) for _ in range(d))
                for d in depths
            )
        elif self.variant == "nested":
            self.lane_stack = nn.ModuleList(
                PathwayLaneBlock(config, shared_kv=self.share_kv) for _ in range(max(depths))
            )
        elif self.variant == "recursive":
            tied = PathwayLaneBlock(config, shared_kv=self.share_kv)
            self.lane_stack = nn.ModuleList([tied] * max(depths))
        else:
            raise ValueError(
                f"pathway_variant must be disjoint | nested | recursive, got {self.variant!r}"
            )
        # Memory lane block: single PathwayLaneBlock applied
        # pathway_memory_lane_depth times, attending only to bank columns.
        self.memory_block = (
            PathwayLaneBlock(config, shared_kv=self.share_kv)
            if self.memory_lane_index is not None
            else None
        )
        self.merge_norms = nn.ModuleList(
            norm_class(h, eps=eps) if d > 0 else nn.Identity() for d in depths
        )
        # Zero-init merge scales: at init every lane contributes nothing, so
        # the model equals its shared stack bitwise (safe insertion). Plain
        # parameters are untouched by _init_weights, so zeros survive.
        self.merge_scales = nn.ParameterDict(
            {str(i): nn.Parameter(torch.zeros(h)) for i, d in enumerate(depths) if d > 0}
        )
        # Incremented by the MODEL forward (outside gradient checkpointing,
        # which would double-count on recompute). Persistent: floor release
        # survives checkpoint resume.
        self.register_buffer("_train_steps", torch.zeros((), dtype=torch.long), persistent=True)

        self._last_lane_assignment: Optional[torch.Tensor] = None
        # Kept as DEVICE tensors and converted only when read (see the properties
        # below). `.tolist()` / `float()` are host syncs, and these are pure
        # diagnostics -- they were costing 3 syncs per stage per forward (9 on the
        # 3-stage mopd arms) to populate numbers only scripts/dyncompute_diagnostics
        # ever reads.
        self._last_lane_counts: Optional[torch.Tensor] = None
        self._last_expected_cost_t: Optional[torch.Tensor] = None

    def iter_lane_blocks(self):
        """Yield one entry per executable block SLOT (objects repeat for the
        weight-tied recursive variant). Compute accounting bills slots at the
        expected lane cost and dedups objects for unique-N."""
        if self.lanes is not None:
            for lane in self.lanes:
                yield from lane
        else:
            yield from self.lane_stack

    @property
    def _last_lane_fractions(self):
        """Realized hard fraction per lane, materialized on demand.

        A property rather than a stored list so the host sync happens only when a
        caller actually reads it (diagnostics / MFU billing), not on every forward.
        """
        counts = self._last_lane_counts
        if counts is None:
            return None
        total = max(int(counts.sum()), 1)
        return [float(c) / total for c in counts.tolist()]

    @property
    def _last_expected_cost(self):
        """Differentiable expected blocks/token from the last forward, on demand."""
        cost = self._last_expected_cost_t
        return None if cost is None else float(cost)

    def _tier_plan(self) -> List[Tuple[int, int]]:
        """Group the nested lane stack into runs of blocks sharing one token set.

        Nested/recursive lanes are PREFIXES of one stack, so the set of tokens
        still executing only shrinks at a depth boundary: with depths [0, 2, 4],
        blocks 1-2 are executed by every token routed to depth >= 2 and blocks
        3-4 only by the depth-4 tokens. Returns ``[(depth_threshold, n_blocks)]``
        in execution order.
        """
        cuts = sorted(
            {d for i, d in enumerate(self.depths) if d > 0 and i != self.memory_lane_index}
        )
        plan, prev = [], 0
        for d in cuts:
            plan.append((d, d - prev))
            prev = d
        return plan

    def _lane_masks(
        self, idx, k, seq_len, attention_mask, doc_ids, flex_cache, cache_key, bank_cols=0
    ):
        """(block_mask, sdpa_mask) for a lane block whose queries are ``idx``.

        ``idx=None`` means the queries are the full sequence. Prefers the flex
        block-sparse path; falls back to gathering rows out of the dense additive
        mask when FlexAttention is unavailable (CPU / older torch).

        ``bank_cols`` (pathway_bank_export): the export carries that many bank
        rows appended AFTER the sequence — position-free, visible to every
        query (kv columns >= seq_len are unmasked).
        """
        window = self.lane_window
        if self._use_flex and bank_cols > 0 and not (
            self.interval_mask and (doc_ids is not None or idx is not None)
        ):
            # Only the interval mask knows how to admit bank columns; other
            # flex predicates fall back to the additive path below.
            pass
        elif self._use_flex:
            if self.interval_mask and (doc_ids is not None or idx is not None):
                # Interval form: under packed docs the lane predicate is a
                # contiguous kv range per query row (see make_interval_mask_mod)
                # — cheaper mask_mod gathers inside the fwd/bwd flex kernels.
                # The (doc_ids None, idx None) case stays on the process-wide
                # shape-cached causal mask below (built once, ever).
                from .flex_swa import get_interval_block_mask

                lo_pos = self._lane_lo_positions(doc_ids, seq_len, flex_cache)  # [B, S]
                if idx is None:
                    lo = lo_pos
                    hi = (
                        torch.arange(seq_len, device=self._lane_device)
                        .expand(self._lane_batch, seq_len)
                    )
                else:
                    lo = torch.gather(lo_pos, 1, idx)
                    hi = idx
                return (
                    get_interval_block_mask(
                        lo,
                        hi,
                        kv_len=seq_len + bank_cols,
                        step_cache=flex_cache,
                        cache_key=cache_key + (window, "interval", bank_cols),
                        bank_start=seq_len if bank_cols > 0 else None,
                    ),
                    None,
                )
            from .flex_swa import get_causal_doc_block_mask

            bm = get_causal_doc_block_mask(
                doc_ids=doc_ids,
                q_len=k,
                kv_len=seq_len,
                device=self._lane_device,
                q_pos=idx,
                batch=self._lane_batch,
                step_cache=flex_cache,
                cache_key=cache_key + (window,),
                window=window,
            )
            return bm, None

        base = attention_mask
        if base is not None and idx is not None:
            b = base.shape[0]
            m = base.expand(b, base.shape[1], seq_len, seq_len)
            rows = idx[:, None, :, None].expand(b, m.shape[1], k, seq_len)
            base = torch.gather(m, 2, rows)
        if window > 0:
            band = self._band_mask(idx, k, seq_len, window)
            base = band if base is None else base + band
        if bank_cols > 0 and base is not None:
            # Bank columns are appended after the sequence and always visible.
            pad_shape = base.shape[:-1] + (bank_cols,)
            base = torch.cat([base, base.new_zeros(pad_shape)], dim=-1)
        return None, base

    def _bank_only_masks(self, idx, k, seq_len, bank_cols, flex_cache, cache_key):
        """Masks for the memory lane: ONLY the bank columns are visible.

        Reuses the interval machinery with an empty sequence interval
        (lo == seq_len > hi) plus the always-visible bank tail."""
        if self._use_flex:
            from .flex_swa import get_interval_block_mask

            dev = self._lane_device
            lo = torch.full((self._lane_batch, k), seq_len, dtype=torch.long, device=dev)
            hi = idx if idx is not None else (
                torch.arange(k, device=dev).expand(self._lane_batch, k)
            )
            return (
                get_interval_block_mask(
                    lo,
                    hi,
                    kv_len=seq_len + bank_cols,
                    step_cache=flex_cache,
                    cache_key=cache_key + ("bank_only", bank_cols),
                    bank_start=seq_len,
                ),
                None,
            )
        neg = torch.finfo(torch.float32).min
        sm = torch.zeros(1, 1, 1, seq_len + bank_cols, device=self._lane_device)
        sm[..., :seq_len] = neg
        return None, sm

    def _lane_lo_positions(self, doc_ids, seq_len, flex_cache):
        """``[B, S]`` lowest kv position each ORIGINAL position may attend to:
        ``max(doc_segment_start, pos - window + 1)``. Built once per forward
        and shared across every stage and tier through the step ``flex_cache``
        (all lane masks derive their rows from this one table)."""
        key = ("lane_lo_pos", self.lane_window, seq_len)
        if flex_cache is not None:
            cached = flex_cache.get(key)
            if cached is not None:
                return cached
        dev = self._lane_device
        pos = torch.arange(seq_len, device=dev)
        if self.lane_window > 0:
            lo = (pos - (self.lane_window - 1)).clamp_min(0)
        else:
            lo = torch.zeros_like(pos)
        lo = lo.expand(self._lane_batch, seq_len)
        if doc_ids is not None:
            from .flex_swa import doc_segment_starts

            lo = torch.maximum(lo, doc_segment_starts(doc_ids))
        else:
            lo = lo.contiguous()
        if flex_cache is not None:
            flex_cache[key] = lo
        return lo

    def _band_mask(self, idx, k, seq_len, window):
        """Additive [1 or B, 1, k, S] mask banding queries to ``window`` back."""
        dev = self._lane_device
        kv = torch.arange(seq_len, device=dev)
        q = torch.arange(k, device=dev)[:, None] if idx is None else idx.unsqueeze(-1)
        dist = q - kv  # broadcast to [k, S] or [B, k, S]
        allowed = (dist >= 0) & (dist < window)
        neg = torch.finfo(self._lane_dtype).min
        out = torch.zeros_like(allowed, dtype=self._lane_dtype).masked_fill_(~allowed, neg)
        return out.unsqueeze(0).unsqueeze(0) if idx is None else out.unsqueeze(1)

    def _lane_deltas(
        self,
        x: torch.Tensor,
        export: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
        lane: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.Tensor] = None,
        flex_cache: Optional[dict] = None,
    ) -> dict:
        """Per-lane transformed states {lane_index: h_lane} for lanes with
        depth > 0. Disjoint runs each lane's own stack; nested/recursive run
        the single shared stack once, capturing each exit depth."""
        out: dict = {}
        seq_len = x.shape[1]
        bank_cols = export.shape[1] - seq_len  # pathway_bank_export rows
        if self.lanes is not None:
            # Disjoint lanes: each lane owns a private stack, so there is no
            # shared prefix to tier and the gather would have to be per-lane.
            # Not implemented — only the flex mask applies here. The sweep's MoP
            # arms are all `nested` (swa_scribe_mop*/mopd*), so this path is
            # currently unused; `pathway_gather_lanes` is a no-op for it.
            bm, sm = self._lane_masks(
                None, seq_len, seq_len, attention_mask, doc_ids, flex_cache,
                ("dense", seq_len), bank_cols=bank_cols,
            )
            for i, blocks in enumerate(self.lanes):
                if len(blocks) == 0:
                    continue
                h_lane = x
                for block in blocks:
                    h_lane = block(h_lane, export, sm, kv=kv, block_mask=bm)
                out[i] = (h_lane, None)
            return out

        if lane is not None and self.gather_lanes:
            return self._lane_deltas_gathered(
                x, export, attention_mask, kv, lane, doc_ids, flex_cache
            )

        bm, sm = self._lane_masks(
            None, seq_len, seq_len, attention_mask, doc_ids, flex_cache,
            ("dense", seq_len), bank_cols=bank_cols,
        )
        exit_for_depth: dict = {}
        h_cur = x
        for step, block in enumerate(self.lane_stack, start=1):
            h_cur = block(h_cur, export, sm, kv=kv, block_mask=bm)
            exit_for_depth[step] = h_cur
        for i, d in enumerate(self.depths):
            if d > 0 and i != self.memory_lane_index:
                out[i] = (exit_for_depth[d], None)
        if self.memory_block is not None and bank_cols > 0:
            bm_m, sm_m = self._bank_only_masks(
                None, seq_len, seq_len, bank_cols, flex_cache, ("dense_mem", seq_len)
            )
            h_mem = x
            for _ in range(self.depths[self.memory_lane_index]):
                h_mem = self.memory_block(h_mem, export, sm_m, kv=kv, block_mask=bm_m)
            out[self.memory_lane_index] = (h_mem, None)
        return out

    @staticmethod
    def _compact_index(sel: torch.Tensor, k: int) -> torch.Tensor:
        """``[B, k]`` row indices putting the selected rows first, in order.

        The tail beyond each row's own count is padding that points at real (just
        unselected) rows rather than a dummy slot, so every gathered query row
        still admits at least its own key -- a fully-masked row would make the
        attention kernel emit NaN. Padding is discarded at the merge, where the
        lane selector is already zero for those tokens.
        """
        n = sel.shape[1]
        pos = torch.arange(n, device=sel.device).expand(sel.shape[0], n)
        return torch.argsort(torch.where(sel, pos, pos + n), dim=1)[:, :k]

    def _lane_deltas_gathered(
        self,
        x: torch.Tensor,
        export: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
        lane: torch.Tensor,
        doc_ids: Optional[torch.Tensor],
        flex_cache: Optional[dict],
    ) -> dict:
        """Nested lanes, running each block only over the tokens routed that deep.

        Mathematically identical to the dense path: a lane block cross-attends to
        the shared-stream export and its MLP is position-wise, so lane tokens
        never mix with each other and dropping a query row cannot change any
        other row's output. At ``pathway_budget=2.0`` over depths [0, 2, 4] this
        executes 2.0 block-passes per token instead of 4.0.

        Everything stays COMPACT: tier t+1's token set is a subset of tier t's, so
        it is gathered straight out of tier t's compact rows rather than scattered
        back to [B, T, H] first. Returns ``{lane: (h_compact, orig_positions)}``;
        the caller merges in compact space and scatters once.
        """
        b, _, h = x.shape
        seq_len = x.shape[1]
        bank_cols = export.shape[1] - seq_len  # pathway_bank_export rows
        lane_full = lane  # original [B, S] map (memory-lane gather needs it)
        depth_of_lane = torch.as_tensor(self.depths, device=lane.device)
        d_c = depth_of_lane[lane]  # committed depth, aligned with the current rows

        out: dict = {}
        blocks = list(self.lane_stack)
        cursor = 0
        h_c: Optional[torch.Tensor] = None
        idx: Optional[torch.Tensor] = None  # original positions of the compact rows
        t0 = x.shape[1]
        tier_kmaxes = self._consume_tier_counts()  # one early sync, or None (fallback)
        for tier, (thresh, n_blocks) in enumerate(self._tier_plan()):
            sel = d_c >= thresh
            if self.fixed_tier_fracs is not None:
                # mop-stack-k: static width, causal positional clip. Rows routed
                # deeper than the width simply miss the deep delta (they are
                # absent from idx, so the merge never touches them); _compact_
                # index keeps the EARLIEST selected rows, so inclusion depends
                # only on earlier routing decisions.
                k = min(sel.shape[1], max(1, int(math.ceil(self.fixed_tier_fracs[tier] * t0))))
            elif tier_kmaxes is not None:
                k = self._bucket_width(tier_kmaxes[tier], sel.shape[1], self.gather_bucket)
            else:
                k = self._gather_width(sel, sel.shape[1], self.gather_bucket)
            if k == 0:
                cursor += n_blocks
                continue
            sub = self._compact_index(sel, k)  # indices into the CURRENT rows
            idx = sub if idx is None else torch.gather(idx, 1, sub)
            source = x if h_c is None else h_c
            h_c = torch.gather(source, 1, sub.unsqueeze(-1).expand(b, k, h))
            d_c = torch.gather(d_c, 1, sub)

            bm, sm = self._lane_masks(
                idx, k, seq_len, attention_mask, doc_ids, flex_cache,
                (id(self), tier, k), bank_cols=bank_cols,
            )
            for _ in range(n_blocks):
                h_c = blocks[cursor](h_c, export, sm, kv=kv, block_mask=bm)
                cursor += 1
            for i, d in enumerate(self.depths):
                if d == thresh and i != self.memory_lane_index:
                    out[i] = (h_c, idx)
        if self.memory_block is not None and bank_cols > 0:
            mem = self.memory_lane_index
            sel_m = lane_full == mem
            if tier_kmaxes is not None and len(tier_kmaxes) > len(self._tier_plan()):
                k_m = self._bucket_width(tier_kmaxes[-1], seq_len, self.gather_bucket)
            else:
                k_m = self._gather_width(sel_m, seq_len, self.gather_bucket)
            if k_m > 0:
                sub_m = self._compact_index(sel_m, k_m)
                h_m = torch.gather(x, 1, sub_m.unsqueeze(-1).expand(b, k_m, h))
                bm_m, sm_m = self._bank_only_masks(
                    sub_m, k_m, seq_len, bank_cols, flex_cache, (id(self), "mem", k_m)
                )
                for _ in range(self.depths[mem]):
                    h_m = self.memory_block(h_m, export, sm_m, kv=kv, block_mask=bm_m)
                out[mem] = (h_m, sub_m)
        return out

    @staticmethod
    def _bucket_width(kmax: int, t: int, bucket: int = _GATHER_BUCKET) -> int:
        if kmax <= 0:
            return 0
        return min(t, ((kmax + bucket - 1) // bucket) * bucket)

    @staticmethod
    def _gather_width(sel: torch.Tensor, t: int, bucket: int = _GATHER_BUCKET) -> int:
        """Padded per-row gather width, rounded up to a fixed bucket.

        Token-choice routing is an argmax, not a fixed-capacity top-k, so rows
        disagree on how many tokens they send deep and the gather must pad to the
        row max. Rounding that up to a multiple of the bucket keeps the number
        of distinct shapes small — flex_attention is compiled with
        ``dynamic=False``, so an unbucketed width would trigger a recompile on
        almost every step.

        NOTE: ``int(...)`` here is a blocking device->host sync. The forward
        path avoids it via :meth:`_prefetch_tier_counts` (one async copy per
        forward); this remains as the CPU / fallback path.
        """
        kmax = int(sel.sum(dim=1).max())
        return Pathways._bucket_width(kmax, t, bucket)

    def _prefetch_tier_counts(self, lane: torch.Tensor) -> None:
        """Issue ONE async DtoH copy of every tier's row-max count.

        All tier gather widths derive from the FINAL lane map (tier t+1's token
        set is a subset of tier t's, so per-tier counts are invariant under the
        compaction), so they can all be computed here — before the shared-KV
        projection and the lane gathers are queued — and read once inside
        ``_lane_deltas_gathered``. This replaces one blocking ``.max()`` sync
        per tier (stacked mopd: 6/step) with a single pinned-memory copy whose
        latency hides under the projections' GPU time.
        """
        self._tier_counts_pending = False
        if (
            not lane.is_cuda
            or not self.gather_lanes
            or self.lanes is not None
            or self.fixed_tier_fracs is not None  # static widths: no counts needed at all
            or torch.compiler.is_compiling()
        ):
            return
        plan = self._tier_plan()
        if not plan:
            return
        depth_of_lane = torch.as_tensor(self.depths, device=lane.device)
        d_all = depth_of_lane[lane]  # [B, T]
        thr = torch.as_tensor([p[0] for p in plan], device=lane.device)
        counts = (d_all.unsqueeze(0) >= thr.view(-1, 1, 1)).sum(dim=2).amax(dim=1)
        if self.memory_lane_index is not None:
            mem_count = (lane == self.memory_lane_index).sum(dim=1).amax().reshape(1)
            counts = torch.cat([counts, mem_count])
        n = counts.shape[0]
        if self._tier_count_pinned is None or self._tier_count_pinned.numel() != n:
            self._tier_count_pinned = torch.empty(n, dtype=counts.dtype, pin_memory=True)
            self._tier_count_event = torch.cuda.Event()
        self._tier_count_pinned.copy_(counts, non_blocking=True)
        self._tier_count_event.record()
        self._tier_counts_pending = True

    def _consume_tier_counts(self) -> Optional[List[int]]:
        if not self._tier_counts_pending:
            return None
        self._tier_counts_pending = False
        # Waits only for work up to the recorded event (the router + count
        # kernels + the copy), not for anything queued after the prefetch.
        self._tier_count_event.synchronize()
        return [int(v) for v in self._tier_count_pinned.tolist()]

    def _apply_deep_floor(
        self, lane: torch.Tensor, logits: torch.Tensor, num_prefix: int
    ) -> torch.Tensor:
        """Force the most deep-preferring content tokens into the deep lane
        until it holds at least ``floor_frac`` of content tokens per row."""
        t_content = lane.shape[1] - num_prefix
        need = int(math.ceil(self.floor_frac * t_content))
        if need <= 0:
            return lane
        deep_logits = logits[:, num_prefix:, self.deep_lane]
        top = deep_logits.topk(min(need, t_content), dim=1).indices + num_prefix
        lane = lane.clone()
        lane.scatter_(1, top, self.deep_lane)
        return lane

    def _merge_dense(self, y, x, h_lane, lane, probs, i: int):
        """Dense-path merge of one lane's delta into the stream. Extracted so
        compile_dyncompute_blocks can fuse the elementwise chain — the round-4
        profile put ~250 ms/step of mopr's cost in the BACKWARD of these eager
        merge chains (11 stages x full-seq intermediates)."""
        delta = self.merge_norms[i](h_lane - x) * self.merge_scales[str(i)]
        gate = probs[..., i : i + 1]
        selected = (lane == i).unsqueeze(-1).to(x.dtype)
        return y + selected * gate * delta

    def _merge_gathered(self, y, x, h_lane, idx, lane, probs, i: int):
        """Compact merge: the delta is nonzero only where lane == i, and those
        tokens are exactly the ones this tier gathered, so normalising and
        scaling the FULL sequence here would repeat the discard-after-compute
        the gather just removed. Padding rows carry a different depth, hence
        a different lane, so `selected` already zeroes them -- scatter_add is
        safe because a row's gather indices are distinct by construction.
        Extracted for per-stage compile (bucketed k -> few graph variants)."""
        b = x.shape[0]
        k = idx.shape[1]
        idx_h = idx.unsqueeze(-1).expand(b, k, x.shape[-1])
        delta = self.merge_norms[i](h_lane - torch.gather(x, 1, idx_h))
        delta = delta * self.merge_scales[str(i)]
        gate = torch.gather(probs[..., i], 1, idx).unsqueeze(-1)
        selected = (torch.gather(lane, 1, idx) == i).unsqueeze(-1).to(x.dtype)
        return y.scatter_add(1, idx_h, selected * gate * delta)

    def _forward_permuted(
        self, x, export, attention_mask, kv, lane, probs, doc_ids, flex_cache
    ) -> torch.Tensor:
        """Permute-not-gather dispatch + merge (math-identical to the gathered
        path + scatter merge). ONE stable sort by lane per stage makes every
        depth tier a contiguous SUFFIX of the sorted sequence (depths are
        validated monotone in lane index), so tier token sets are slices —
        views, not gathers — deltas merge with sliced adds in sorted space,
        and ONE inverse permutation restores order. Two full-seq row movements
        replace the per-tier gathers plus per-lane scatter_adds that the
        round-4/6 profiling identified as the bandwidth-bound residual.
        Bucketed slice widths may include a few shallower rows at the slice
        head; their merge contribution is zeroed by ``selected`` exactly as
        bucket padding is on the gathered path."""
        b, t, h = x.shape
        bank_cols = export.shape[1] - t
        order = torch.argsort(lane, dim=1, stable=True)  # ascending lane index
        inv = torch.empty_like(order)
        inv.scatter_(1, order, torch.arange(t, device=order.device).expand(b, t))
        order_h = order.unsqueeze(-1).expand(b, t, h)
        x_s = torch.gather(x, 1, order_h)
        lane_s = torch.gather(lane, 1, order)
        probs_s = torch.gather(probs, 1, order.unsqueeze(-1).expand_as(probs))
        y_s = x_s.clone()

        blocks = list(self.lane_stack)
        cursor = 0
        h_c: Optional[torch.Tensor] = None
        prev_k = t
        tier_kmaxes = self._consume_tier_counts()
        depth_of_lane = torch.as_tensor(self.depths, device=lane.device)
        d_s = depth_of_lane[lane_s]
        for tier, (thresh, n_blocks) in enumerate(self._tier_plan()):
            if tier_kmaxes is not None:
                k = self._bucket_width(tier_kmaxes[tier], t, self.gather_bucket)
            else:
                k = self._gather_width(d_s >= thresh, t, self.gather_bucket)
            k = min(k, prev_k)  # tiers are nested suffixes
            if k == 0:
                cursor += n_blocks
                continue
            s0 = t - k
            idx = order[:, s0:]
            h_c = x_s[:, s0:] if h_c is None else h_c[:, prev_k - k :]
            bm, sm = self._lane_masks(
                idx, k, t, attention_mask, doc_ids, flex_cache,
                (id(self), "perm", tier, k), bank_cols=bank_cols,
            )
            for _ in range(n_blocks):
                h_c = blocks[cursor](h_c, export, sm, kv=kv, block_mask=bm)
                cursor += 1
            prev_k = k
            sl = slice(s0, t)
            for i, d in enumerate(self.depths):
                if d != thresh or i == self.memory_lane_index:
                    continue
                delta = self.merge_norms[i](h_c - x_s[:, sl]) * self.merge_scales[str(i)]
                gate = probs_s[:, sl, i : i + 1]
                selected = (lane_s[:, sl] == i).unsqueeze(-1).to(x.dtype)
                y_s = torch.cat([y_s[:, :s0], y_s[:, sl] + selected * gate * delta], dim=1)
        return torch.gather(y_s, 1, inv.unsqueeze(-1).expand(b, t, h))

    def _stage_prologue(self, x: torch.Tensor, bank_rows: Optional[torch.Tensor] = None):
        """Export norm + shared-KV projection + router in one static-shape
        graph. No host syncs, no data-dependent shapes — safe to
        ``torch.compile`` per stage (``compile_dyncompute_blocks``); the
        prefix-force / deep-floor edits and the tier-count prefetch stay in
        the eager caller (in-place edit + D2H copy).

        ``bank_rows`` (pathway_bank_export): the register bank ``[N, h]``
        joins the export as extra KV rows — position-free, visible to every
        lane query (masks derive the bank width from export vs x length)."""
        export = self.export_norm(x)
        if bank_rows is not None:
            bank = self.export_norm(bank_rows).to(export.dtype)
            export = torch.cat([export, bank.unsqueeze(0).expand(x.shape[0], -1, -1)], dim=1)
        kv = self.shared_kv_proj(export) if self.shared_kv_proj is not None else None
        if self.router is None:
            b, t, _ = x.shape
            lane = x.new_zeros(b, t, dtype=torch.long)
            probs = x.new_ones(b, t, 1)
            logits = None
        else:
            lane, probs, logits = self.router(x)
        lane = _maybe_chunk_route(self, lane, logits)
        lane = _maybe_force_lane(self, x, lane)
        return export, kv, lane, probs, logits

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_prefix: int = 0,
        aux_mode: str = "local",
        doc_ids: Optional[torch.Tensor] = None,
        flex_cache: Optional[dict] = None,
        mask_is_causal_doc: bool = False,
        export_cache: Optional[dict] = None,
        bank_rows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[object]]:
        """``aux_mode="local"`` (default) returns (y, aux) exactly as a
        single-stage MoP. ``aux_mode="parts"`` returns
        (y, (expected_cost, entropy_penalty)) so a stacked model can balance
        one GLOBAL budget across stages (a token may spend its depth in a few
        stages and coast elsewhere) instead of S independent per-stage
        budgets; parts are None outside training / without a router.

        ``export_cache`` (pathway_export_interval > 1): stages whose
        ``_stage_index % interval != 0`` reuse the export/kv computed by the
        last recomputing stage (STALE by the intervening spine layers — a
        screened functional change). The model passes one per-forward dict."""
        b, t, _ = hidden_states.shape
        x = hidden_states
        _reuse_export = (
            export_cache is not None
            and self.export_interval > 1
            and (getattr(self, "_stage_index", 0) % self.export_interval) != 0
            and "export" in export_cache
        )
        # Resolve the attention backend for this forward. The flex path REBUILDS
        # the predicate from (causality, doc_ids) rather than reading the caller's
        # additive mask, so it is only valid when the caller certifies that its
        # mask is exactly that. A padding-bearing mask (ragged batch) is not
        # recoverable from doc_ids, and silently ignoring it would let padded
        # positions into the lane context -- so it stays on SDPA.
        from .flex_swa import flex_available

        self._use_flex = (
            self.use_flex_attention
            and flex_available()
            and hidden_states.is_cuda
            and (attention_mask is None or mask_is_causal_doc)
        )
        self._lane_device = hidden_states.device
        self._lane_batch = b
        self._lane_dtype = attention_mask.dtype if attention_mask is not None else hidden_states.dtype

        if _reuse_export:
            export = export_cache["export"]
            kv = export_cache.get("kv")
            if self.router is None:
                lane = x.new_zeros(b, t, dtype=torch.long)
                probs = x.new_ones(b, t, 1)
                logits = None
            else:
                lane, probs, logits = self.router(x)
                lane = _maybe_chunk_route(self, lane, logits)
                lane = _maybe_force_lane(self, x, lane)
        else:
            # Static-shape eager glue (export norm + shared K/V + router) lives
            # in one method so compile_dyncompute_blocks can fuse it into a
            # single graph per stage — it is otherwise ~30 tiny eager launches
            # repeated per stage, about half the measured per-stage machinery.
            export, kv, lane, probs, logits = self._stage_prologue(x, bank_rows)
            if export_cache is not None and self.export_interval > 1:
                export_cache["export"] = export
                export_cache["kv"] = kv

        if self.router is not None:
            # Persistent globals / meta prefix: force the deep lane (same
            # spirit as MoD's force_route_prefix — the prefix is not a
            # routing choice and must traverse the reasoning path).
            if num_prefix > 0:
                lane[:, :num_prefix] = self.deep_lane
            if (
                self.training
                and self.floor_steps > 0
                and self.floor_frac > 0.0
                and int(self._train_steps) < self.floor_steps
            ):
                lane = self._apply_deep_floor(lane, logits, num_prefix)
            if self.route_chunk > 1:
                # The floor (and the prefix force) work per token; D2 needs the
                # commitment to stay per CHUNK, so re-impose it here. See
                # _broadcast_chunk_first.
                lane = _broadcast_chunk_first(lane, self.route_chunk, num_prefix)

        # Lane map is final past this point: queue the tier-width DtoH copy now
        # so it overlaps the merge/gather work below (see the method doc).
        self._prefetch_tier_counts(lane)

        if (
            self.permute_dispatch
            and lane is not None
            and self.gather_lanes
            and self.lanes is None
            and self.fixed_tier_fracs is None
        ):
            # Math-identical permuted dispatch: sort once, tiers as suffix
            # slices, merge in sorted space, un-permute once (2 full-seq row
            # movements instead of per-tier gathers + per-lane scatter_adds).
            y = self._forward_permuted(
                x, export, attention_mask, kv, lane, probs, doc_ids, flex_cache
            )
        else:
            y = x
            lane_states = self._lane_deltas(
                x, export, attention_mask, kv, lane=lane, doc_ids=doc_ids, flex_cache=flex_cache
            )
            for i, (h_lane, idx) in lane_states.items():
                if idx is None:
                    y = self._merge_dense(y, x, h_lane, lane, probs, i)
                    continue
                y = self._merge_gathered(y, x, h_lane, idx, lane, probs, i)

        self._last_lane_assignment = lane.detach()
        self._last_lane_counts = torch.bincount(
            lane[:, num_prefix:].reshape(-1), minlength=self.num_lanes
        ).detach()

        aux = None
        if self.router is not None and self.training:
            content_probs = probs[:, num_prefix:].float()
            depth_t = torch.tensor(
                self.depths, dtype=content_probs.dtype, device=content_probs.device
            )
            expected_cost = (content_probs.mean(dim=(0, 1)) * depth_t).sum()
            self._last_expected_cost_t = expected_cost.detach()
            ent = -(content_probs * (content_probs + 1e-9).log()).sum(-1).mean()
            entropy_penalty = (math.log(self.num_lanes) - ent) / math.log(self.num_lanes)
            if aux_mode == "parts":
                return y, (expected_cost, entropy_penalty)
            max_depth = float(max(self.depths))
            aux = self.aux_weight * ((expected_cost - self.budget) / max_depth) ** 2
            if self.entropy_weight > 0:
                aux = aux + self.entropy_weight * entropy_penalty
            aux = aux.to(hidden_states.dtype)

        if aux_mode == "parts":
            return y, None
        return y, aux
