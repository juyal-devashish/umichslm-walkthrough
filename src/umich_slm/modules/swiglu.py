"""SwiGLU MLP implementation.

Reference: https://arxiv.org/abs/2002.05202
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    """SwiGLU-based MLP as used in modern transformer architectures (LLaMA, Mistral, etc.).

    SwiGLU(x) = (Swish(xW_gate) * xW_up) W_down

    Where Swish (also known as SiLU) is: swish(x) = x * sigmoid(x)

    This activation function has been shown to improve training dynamics compared
    to standard FFN with ReLU or GELU activations.

    Args:
        hidden_size: Input and output dimension.
        intermediate_size: Hidden dimension of the MLP.
            For parameter parity with standard FFN, use: intermediate_size = int(8/3 * hidden_size)
            rounded to a multiple of 256 for efficiency.
        hidden_act: Activation function (default "silu" for Swish).
        bias: Whether to use bias in linear projections.

    Shape:
        - Input: (*, hidden_size)
        - Output: (*, hidden_size)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Gate projection (for the gating mechanism)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        # Up projection (the value that gets gated)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        # Down projection (back to hidden_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

        # Activation function (SiLU = Swish with beta=1)
        self.act_fn = self._get_activation(hidden_act)

    def _get_activation(self, hidden_act: str) -> nn.Module:
        """Get the activation function module."""
        activations = {
            "silu": nn.SiLU(),
            "swish": nn.SiLU(),  # Swish is the same as SiLU
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
        }
        if hidden_act not in activations:
            raise ValueError(
                f"Unknown activation: {hidden_act}. " f"Supported: {list(activations.keys())}"
            )
        return activations[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SwiGLU MLP.

        Args:
            x: Input tensor of shape (*, hidden_size).

        Returns:
            Output tensor of shape (*, hidden_size).
        """
        # SwiGLU: (activation(gate(x)) * up(x)) @ down
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"bias={self.gate_proj.bias is not None}"
        )


class MatFormerSwiGLUMLP(SwiGLUMLP):
    """SwiGLU MLP with Matryoshka (MatFormer) nested FFNs.

    The weights are allocated at the full ``intermediate_size``, but a forward pass
    may use only the **first** ``active_size`` units of the intermediate dimension —
    the first ``active_size`` columns of ``gate_proj``/``up_proj`` and the first
    ``active_size`` rows of ``down_proj``. Because the slice is always the top-left
    sub-matrix, smaller FFNs are nested inside the larger one (Gemma-3n/4 MatFormer).

    - ``total`` params correspond to the full ``intermediate_size``.
    - ``active`` params/compute correspond to the selected ``active_size``.

    During training, pass ``granularities`` and call :meth:`sample_active` each step
    (or per layer) to randomly pick a nested size so every slice stays competent. For
    eval/inference set a fixed ``active_size``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        bias: bool = False,
        active_size=None,
        granularities=None,
    ):
        super().__init__(hidden_size, intermediate_size, hidden_act, bias)
        self.granularities = sorted(granularities) if granularities else None
        self._active_size = self._clamp(active_size) if active_size else intermediate_size

    def _clamp(self, n: int) -> int:
        return max(1, min(int(n), self.intermediate_size))

    @property
    def active_size(self) -> int:
        return self._active_size

    def set_active_size(self, n) -> None:
        """Fix the active intermediate size (None -> full)."""
        self._active_size = self.intermediate_size if n is None else self._clamp(n)

    def sample_active(self, generator=None) -> int:
        """Randomly pick one of the configured granularities as the active size."""
        if not self.granularities:
            return self._active_size
        idx = int(torch.randint(0, len(self.granularities), (1,), generator=generator).item())
        self._active_size = self._clamp(self.granularities[idx])
        return self._active_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self._active_size
        if a >= self.intermediate_size:
            return super().forward(x)
        # Nested top-left slice: first `a` intermediate units only.
        gate_b = None if self.gate_proj.bias is None else self.gate_proj.bias[:a]
        up_b = None if self.up_proj.bias is None else self.up_proj.bias[:a]
        gate = self.act_fn(F.linear(x, self.gate_proj.weight[:a], gate_b))
        up = F.linear(x, self.up_proj.weight[:a], up_b)
        return F.linear(gate * up, self.down_proj.weight[:, :a], self.down_proj.bias)


def compute_intermediate_size(hidden_size: int, multiple_of: int = 256) -> int:
    """Compute the intermediate size for SwiGLU to match standard FFN parameter count.

    For a standard FFN with 4x expansion: params = 2 * hidden_size * 4 * hidden_size
    For SwiGLU with 3 projections: params = 3 * hidden_size * intermediate_size

    To match: intermediate_size = 8/3 * hidden_size

    Args:
        hidden_size: The hidden dimension of the model.
        multiple_of: Round up to this multiple for efficiency.

    Returns:
        The computed intermediate size.
    """
    intermediate_size = int(8 * hidden_size / 3)
    # Round up to multiple
    intermediate_size = ((intermediate_size + multiple_of - 1) // multiple_of) * multiple_of
    return intermediate_size


ROUTE_CTRLS = ("quantile", "quantile_nested", "wish", "post_clip")


class RoutedMatFormerSwiGLUMLP(MatFormerSwiGLUMLP):
    """W1 ``mat_ffn``: PER-TOKEN nested-width routing (docs/dyncompute_redesign.md).

    :class:`MatFormerSwiGLUMLP` above implements MatFormer as published
    (2310.07707): the nested widths exist, but ONE width is selected per forward
    pass -- per *model*, at deployment. Flextron (2406.10260) routes
    input-adaptively but only as a post-training conversion, and MoNE routes per
    token over nested experts in *vision*. This class fills the open square:
    a router picks the nested width **per token**, trained end to end from step 0
    in a causal decoder LM.

    Why it can be cheap where depth routing is not. The measured cost of our
    stacked depth router is ~30 ms/stage of fixed machinery plus a bandwidth-
    bound ``[B,S,h]`` scatter_add in the backward (docs/arch_naming.md round 4).
    Nested width has neither: attention is untouched, so there is no KV export
    and no merge, and because the widths are NESTED each tier is an *incremental
    column slice* of the same weight tensor. Tier 0 runs dense on every token
    (no indexing at all); only the incremental slices touch a subset of rows.

    Cost, uniform routing over {1/4, 1/2, 1}: 1.0*(1/4) + (2/3)*(1/4) +
    (1/3)*(1/2) = 0.58x the dense FFN, on ~2/3 of block FLOPs.

    Router gradient: tier k's contribution is scaled by a straight-through gate
    equal to P(choice >= k) -- value 1.0 in the forward, so the arithmetic is
    exactly the nested slice, with the router's gradient flowing through the
    gate. A cost-aware aux loss holds the mean routed width at ``budget``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        bias: bool = False,
        granularities=None,
        budget: float = 0.0,
        aux_weight: float = 0.01,
        dispatch: str = "sorted",
        tier_fracs=None,
        aux_mode: str = "mean",
        explore_frac: float = 0.0,
        gate_mode: str = "ste",
        mixture_frac: float = 1.0,
        route_mode: str = "argmax",
        route_thresh_lr: float = 0.01,
        route_ctrl: str = "wish",
        cap_headroom: float = 1.0,
        explore_from: str = "target",
        tier_scale_init=None,
        prior_bias_init: bool = False,
        track_occupancy: bool = True,
        warmup_stages: int = 4,
        warmup_top_weight: float = 1.0,
    ):
        super().__init__(hidden_size, intermediate_size, hidden_act, bias, None, granularities)
        widths = self.granularities or [intermediate_size]
        self.widths = [max(1, min(int(w), intermediate_size)) for w in widths]
        self.router = nn.Linear(hidden_size, len(self.widths), bias=bool(prior_bias_init))
        # SMALL-RANDOM init, NOT zeros. With zero weights and no bias every
        # token's logits are exactly equal and argmax degenerates to tier 0, so
        # (a) every token takes the CHEAPEST width, breaking the menu-symmetry
        # rule -- a random-init bench would report the cheapest cost, not the
        # realistic one -- and (b) tiers >0 stay empty, so the straight-through
        # gate has no members and the router receives no gradient at all except
        # through the aux loss. Verified: zero-init gave tier fracs [1,0,0] and
        # a None router grad. A small random init spreads tokens ~uniformly, so
        # the realized mean width starts at the menu mean == default budget.
        nn.init.normal_(self.router.weight, std=0.02)
        # Optional: start the router AT the target prior instead of at whatever
        # a random projection happens to produce. With `prior_bias_init` the
        # router carries a bias initialised to log(pi), so step 0's marginal IS
        # the prior and any drift away from it is something the model chose
        # rather than something it started with.
        self.budget = float(budget) if budget else float(sum(self.widths)) / len(self.widths)
        self.aux_weight = float(aux_weight)
        self._last_aux = None
        self._last_mean_width = None
        self._last_fracs = None
        # Bench/eval hook: pin every token to one tier index (see the realized-
        # routing bench mode). None = use the router.
        self._forced_tier = None
        # "sorted" = contiguous-slice dispatch (fast). "gather" = the original
        # index_select/index_add_ path, retained for the optimization A/B.
        self.dispatch = dispatch
        self._static_fracs = None
        self.collect_stats = False  # counters cost a host sync; opt in
        # ---- always-on occupancy telemetry (2026-08-23) ----
        # `collect_stats` above is a per-step host sync, so training never sets
        # it -- which is how both ar10m nest arms trained 26k steps with a router
        # collapsed to 98.6% tier 0 and NOTHING in the logs said so. This counter
        # is the cheap always-on replacement: pure on-device index_add_, no
        # .item(), and `track_occupancy` is fixed for the whole run so the branch
        # is a compile-time constant (toggling it per step would recompile).
        # The trainer drains and zeroes `_occ_counts` every N steps -- that is
        # the only host sync, amortized 1/N.
        self.track_occupancy = bool(track_occupancy)
        if self.track_occupancy:
            # int64, NOT float32. `.to(torch.bfloat16)` -- which the trainer
            # calls on the whole model -- converts every FLOATING-POINT buffer,
            # and a bf16 counter has an 8-bit mantissa: past 256 an increment of
            # 1.0 rounds away entirely. Every tier then saturates at the same
            # value and the reported occupancy is exactly uniform, which is both
            # wrong and extremely convincing. Integer buffers are left alone by
            # a dtype cast, and the counts are exact.
            self.register_buffer(
                "_occ_counts", torch.zeros(len(self.widths), dtype=torch.long), persistent=False
            )
            # Same counts split by sequence HALF (row 0 = first half of each
            # sequence). Occupancy alone cannot tell a content router from a
            # position schedule -- the post-clip controller held mean width
            # 767.9/768 while granting 2460 to the first decile and 384 to the
            # last six. A content router's two halves match; that schedule's
            # first-half mean width is ~3x its second-half.
            self.register_buffer(
                "_occ_pos_counts",
                torch.zeros(2, len(self.widths), dtype=torch.long),
                persistent=False,
            )
            # The PRE-clip assignment (the router's wish), same units. Occupancy
            # below target has two separable causes -- the controller setting
            # theta too high (wish < target) and the per-sequence clip demoting
            # overflow (granted < wish) -- and only this pair tells them apart.
            self.register_buffer(
                "_wish_counts", torch.zeros(len(self.widths), dtype=torch.long), persistent=False
            )
        # ---- 2026-08-22 nesting round (docs/mat_ffn_nesting.md) ----
        # TARGET tier occupancy. None keeps the legacy closest-to-uniform vector
        # that `budget_fractions` derives from `budget`; a list here is the
        # SKEWED prior ("small tiers used more often"), smallest tier first.
        self.tier_fracs = [float(x) for x in tier_fracs] if tier_fracs else None
        self.aux_mode = str(aux_mode)
        # Permanent exploration floor -- see `mat_ffn_explore_frac` in
        # configuration_umichslm.py for why zero occupancy is an absorbing state
        # that the LM gradient alone can never leave. 0.0 is bitwise identical
        # to the pre-2026-08-22 behaviour.
        self.explore_frac = float(explore_frac)
        # "ste"     -- straight-through, forward gate == 1 (the shipped path).
        # "mixture" -- exact soft mixture at train time; see _forward_mixture.
        self.gate_mode = str(gate_mode)
        # Anneal from the exact mixture (0.0) to the hard masked staircase (1.0).
        #
        # Mixture fixes the collapse -- toy eval mean width 651 vs the collapsed
        # baseline's 388, tier 3 occupancy 0.0407 vs 0.0002 -- but it trains a
        # DIFFERENT function from the one that is evaluated, in two ways at once:
        # it has no membership mask (that is the whole point: non-members must
        # get gradient) and no `_causal_budget_clip`, so training is not even
        # capacity-constrained. Measured cost: val_lm_loss 5.41 against ~5.15.
        #
        # Sharpening the softmax does NOT close this -- verified, the gap GROWS
        # as tau falls, because softness was never the difference. Interpolating
        # the mask is what closes it: start unmasked so the router gets its
        # counterfactual, finish masked and clipped so the final model is the one
        # that ships.
        self.gate_anneal = 0.0
        # PARTIAL MIXTURE. The mixture gate is the only thing that prevents the
        # collapse, but it costs full width for every token (~4x budget FLOPs).
        # The anneal sweep showed occupancy survives at anneal 0.83 -- the
        # counterfactual attenuated to a SIXTH of full strength -- and only dies
        # at exactly 1.0, where the non-member gate becomes zero. So the
        # mechanism tolerates far less signal than it is currently given.
        #
        # `mixture_frac` gives FULL counterfactual to a random rho-fraction of
        # tokens and the hard gate to the rest, rather than weak counterfactual
        # to all of them. Content-independent per-token draw, so it is causally
        # safe for the same reason the exploration floor is.
        #
        # 1.0 (default) = every token, i.e. the original full mixture. 0.0 =
        # none, i.e. exactly the hard gate. NOT `or`-defaulted anywhere: 0.0 is
        # falsy and would silently become 1.0, inverting the meaning.
        self.mixture_frac = float(mixture_frac)
        # ---- OPTION B: threshold routing (structural occupancy) ----
        # `_causal_budget_clip` enforces an UPPER bound on occupancy and can only
        # DEMOTE. Nothing enforces a lower bound, which is exactly why collapse
        # is stable once entered: the router walks down and nothing walks it back.
        #
        # "threshold" replaces argmax with `tier >= k iff tail_k > theta_k`, where
        # theta is a CONTROLLER-adapted buffer: theta_k += eta * (realized_k -
        # target_k). If the router pushes every score down, theta follows it down
        # and occupancy stays on target. Collapse becomes structurally impossible
        # rather than something the router has to be taught to avoid.
        #
        # This decouples the two jobs the argmax router was doing at once:
        #   HOW MUCH width gets used  -> structural, set by the prior
        #   WHICH tokens get it       -> learned, a ranking
        # Only the first was broken. The mixture gate fixes it by supplying
        # counterfactual gradient at ~1.21x model FLOPs; this fixes it for free,
        # at the cost of the router only ever learning a ranking.
        #
        # Causal: within a step theta is FIXED, so a token's tier depends only on
        # its own score. The controller update uses the step's realized fractions
        # and lands on the NEXT step, so nothing at position p is influenced by
        # position > p. At inference theta is frozen, i.e. a plain fixed threshold.
        self.route_mode = str(route_mode)
        self.route_thresh_lr = float(route_thresh_lr)
        # HOW theta is adapted. Every threshold cell before 2026-09-18 used
        # "post_clip", and in ALL of them theta ratcheted to 0 (all 9 A40-grid
        # checkpoints: every theta < 0.0015, reached within ~8% of training):
        #   * it measures occupancy AFTER `_causal_budget_clip`, and the clip
        #     bounds that at <= the target, so the error is never positive and
        #     theta can only fall -- there is no restoring force from below;
        #   * it also ran during nested-dropout warmup, where tiers are SAMPLED
        #     and the clip trims the sampling overflow, so theta slid toward 0
        #     before the router had routed a single token.
        # With theta ~ 0 every token WISHES for the wide tiers and the clip,
        # earliest-position-wins, hands them out by ARRIVAL ORDER. Measured on
        # ar10m tpr350 at S=4096: granted width by position decile
        # 2460/1382/768/768/384/384/384/384/384/384 in every layer -- a
        # position schedule, not a content router. Mean width was still exactly
        # the budget, which is why "occupancy 767.9/768" certified it.
        #
        # "quantile" is the DeltaNet dyn-rank controller: theta_k is an
        # EMA of the (1 - target_k) empirical quantile of the router's PRE-clip
        # tail score, so the WISH fraction lands on the target from either side
        # regardless of how the score distribution is shaped, and warmup cannot
        # bias it (it targets the FINAL prior, not the staged one). The clip is
        # then a safety cap that only trims per-sequence overflow -- see
        # `cap_headroom`. "post_clip" is kept only to reproduce old cells.
        #
        # "quantile_nested" fixes an UNDER-FILL in "quantile". Routing is nested
        # (`cumprod(above)`: tier >= k needs EVERY threshold <= k passed), but
        # "quantile" sets each theta_k from tail_k's MARGINAL distribution. The
        # tokens clearing theta_k are not all among those clearing theta_{k-1},
        # so the joint tier->=k share lands below target. Toy, 2026-09-18: wish
        # 659 against a 768 budget, e.g. one layer at theta = [0.004, 1e-4,
        # 2e-4], where tier 2's own test is nearly free but tier 1's admits 40%.
        # The nested form sets theta_k among the tokens that PASS theta_1..k-1
        # in this batch, so the joint share is the target by construction.
        #
        # "wish" feeds back on the routed PRE-clip share itself (nested, with
        # the strict `>` routing uses): theta_k *= exp(lr * (wish_k - t_k)/t_k).
        # A quantile of the score is only a proxy for that share -- it is off
        # under ties (bf16 logits quantize the tail) and a linear EMA of
        # per-batch quantiles that swing by factors is biased upward -- while
        # this converges to E[wish_k] = t_k whatever the score looks like.
        # Pre-clip, so unlike "post_clip" it has a restoring force from both
        # sides; multiplicative, so it spans the 1e-5..1 range theta occupies.
        # DEFAULT since 2026-09-18. Toy A/B, training-time mean width against a
        # 768 budget (wish -> granted): quantile 687 -> 643, quantile_nested
        # 723 -> 667, wish 769 -> 688. What is left is the per-sequence clip
        # (~10%); only `cap_headroom` recovers that, at more compute than width.
        self.route_ctrl = str(route_ctrl)
        if self.route_ctrl not in ROUTE_CTRLS:
            raise ValueError(f"route_ctrl must be one of {ROUTE_CTRLS}, got {route_ctrl!r}")
        # Per-sequence cap = min(1, headroom * target_k) of the row. 1.0 is the
        # legacy cap (and keeps every existing checkpoint's eval bit-identical).
        # Under the quantile controller the WISH averages the target, so at 1.0
        # roughly every other sequence overflows and its tail is demoted by
        # position; headroom lets a sequence with more hard tokens than average
        # keep them. It also sizes `_forward_sorted`'s static windows, so it
        # bills up to headroom x the incremental tier FLOPs.
        self.cap_headroom = float(cap_headroom)
        if self.cap_headroom < 1.0:
            raise ValueError(f"cap_headroom must be >= 1.0, got {cap_headroom}")
        # "target" (default, unchanged) or "uniform" -- see the explore branch.
        self.explore_from = str(explore_from)
        # Per-tier output scale (LayerScale/ReZero on each incremental block).
        # `tier_scale_init=0.0` makes ADDING a tier an exact no-op at init, so a
        # wider tier can never hurt before its columns have learned anything --
        # which removes the early "this untrained slice is noise" gradient that
        # is one candidate driver of the collapse. None disables the parameter
        # entirely (bitwise-identical to the pre-2026-08-23 path).
        self.tier_scale_init = tier_scale_init
        if tier_scale_init is None:
            self.tier_scale = None
        else:
            init = torch.full((len(self.widths),), float(tier_scale_init))
            init[0] = 1.0  # tier 0 always contributes; it IS the base model
            self.tier_scale = nn.Parameter(init)
        # Nested-dropout / sandwich warmup. `warmup_stage` is an INTEGER stage
        # index in [0, warmup_stages] set once per step by MatFFNWarmupController;
        # stage s uses prior lerp(uniform, target, s / warmup_stages), so stage 0
        # is uniform (every width trained on 1/k of tokens, the widest included)
        # and stage `warmup_stages` is the target skew. It is an int and not a
        # float because `_tier_cap` sizes a STATIC window in `_forward_sorted`:
        # a continuously varying cap would recompile the graph every step, where
        # a staged one recompiles `warmup_stages + 1` times in total.
        self.warmup_stages = max(1, int(warmup_stages))
        self.warmup_stage = self.warmup_stages  # default: no warmup, target prior
        self.warmup_sampling = False  # controller turns this on during warmup
        # LARGEST-GRANULARITY UPWEIGHT (MatFormer 2310.07707 appendix). Re-shapes
        # the warmup's STARTING prior from uniform to normalize([1,...,1,lambda]);
        # the anneal endpoint stays exactly `tier_fracs`. 1.0 == off, and off is
        # bitwise identical to the pre-2026-08-22 behaviour. See tier_prior().
        self.warmup_top_weight = float(warmup_top_weight)
        # Router bias init at log(pi). Makes step 0's marginal EXACTLY the target
        # prior, so any drift away from it is something the model chose rather
        # than an accident of the random projection it started from. Must run
        # last: tier_prior() reads tier_fracs/warmup_stage, both set above.
        if self.route_mode == "threshold":
            # Must run last: tier_prior() reads tier_fracs and warmup_stage.
            self.register_buffer(
                "route_thresh",
                torch.tensor(
                    [sum(self.tier_prior()[j:]) for j in range(1, len(self.widths))],
                    dtype=torch.float32,
                ),
                persistent=True,
            )
            # Latest (realized - target) occupancy error, written by forward and
            # consumed by `step_route_thresholds` once per optimizer step. NOT
            # persistent: it is a one-step scratch slot, and checkpointing it
            # would make a resume replay an update it had already applied. Zeros
            # so the first apply -- before any forward has run -- is a no-op.
            self.register_buffer(
                "route_thresh_err",
                torch.zeros(len(self.widths) - 1, dtype=torch.float32),
                persistent=False,
            )
            # Latest pre-clip quantile per tier, the "quantile" controller's
            # scratch slot -- same forward-measures / step-applies split and the
            # same reason (grad-ckpt recompute must see an unchanged theta).
            # NaN = "no forward yet": the step leaves theta alone rather than
            # pulling it toward an arbitrary init value on a resume.
            self.register_buffer(
                "route_thresh_q",
                torch.full((len(self.widths) - 1,), float("nan"), dtype=torch.float32),
                persistent=False,
            )
        if prior_bias_init and self.router.bias is not None:
            with torch.no_grad():
                pi = torch.tensor(self.tier_prior(), dtype=torch.float32).clamp_min(1e-9)
                self.router.bias.copy_(pi.log())

    _FP32_BUFFERS = ("route_thresh", "route_thresh_err", "route_thresh_q")

    def _apply(self, fn, recurse=True):
        """Keep the controller state fp32 through `.to(bfloat16)` / `.half()`.

        A whole-model dtype cast converts every floating buffer. bf16 spacing
        near 0.4 is ~0.002, so a 1%-per-step controller move rounds away unless
        the error is large -- updates survive only near 0, where bf16 is fine,
        which by itself biases theta downward. The torchtitan path trains under
        autocast with fp32 params and never hit this; the toy screen and bench,
        which cast the model, did.
        """
        keep = {
            n: b
            for n, b in self._buffers.items()
            if n in self._FP32_BUFFERS and b is not None and b.dtype == torch.float32
        }
        out = super()._apply(fn, recurse)
        for n, old in keep.items():
            new = self._buffers[n]
            if new is not None and new.dtype != torch.float32:
                self._buffers[n] = old.to(device=new.device)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        logits = self.router(flat.to(self.router.weight.dtype))
        probs = torch.softmax(logits.float(), dim=-1)
        forced = None
        if getattr(self, "_forced_tier_fracs", None):
            from ..utils.forced_routing import apply_forced_tier

            forced = apply_forced_tier(self, flat.shape[0], flat.device)
        if forced is not None:
            choice = forced
        elif self._forced_tier is None and self.warmup_sampling and self.training:
            # NESTED-DROPOUT WARMUP (Rippel et al. 2014; the sandwich rule of
            # Universally Slimmable Networks solves the same problem with an
            # extra full-width pass). Draw each token's tier from the staged
            # prior instead of the router's argmax, so every width is trained on
            # its share of tokens from step 0 -- an untrained router cannot
            # starve the widest tier before it has learned anything.
            #
            # The ROUTER STILL LEARNS during this phase: tier k's contribution
            # is scaled by the straight-through gate tail[:, k], so the router
            # receives "would more width have helped this token" gradient no
            # matter who made the assignment. That is why the hand-over at the
            # end of warmup is not a cold start.
            #
            # bucketize on a [k] cumulative prior rather than multinomial: one
            # uniform draw per row, no host sync, and compile-friendly.
            cdf = torch.tensor(self.tier_prior(), device=flat.device, dtype=torch.float32).cumsum(0)
            u = torch.rand(flat.shape[0], device=flat.device, dtype=torch.float32)
            choice = torch.bucketize(u, cdf.clamp(max=1.0), right=True).clamp_(
                max=len(self.widths) - 1
            )
        elif self._forced_tier is None and self.route_mode == "threshold":
            _tail = probs.flip(-1).cumsum(-1).flip(-1)
            above = _tail[:, 1:] > self.route_thresh.view(1, -1).to(_tail.dtype)
            # leading run of satisfied thresholds keeps the assignment NESTED
            choice = torch.cumprod(above.long(), dim=1).sum(dim=1)
        elif self._forced_tier is None:
            choice = logits.argmax(dim=-1)
            if self.explore_frac > 0.0 and self.training:
                # Override an eps-fraction of tokens with a draw from the target
                # prior, so no tier can reach zero occupancy and freeze its own
                # gradient. Same bucketize-on-a-CDF trick as the warmup sampler:
                # one uniform draw per row, no host sync, compile-friendly. The
                # draws are content-independent, so nothing moves between
                # positions; the causal clip below still applies on top.
                # Which distribution the eps-fraction is drawn FROM matters a
                # lot under a skewed prior: drawing from the TARGET gives the
                # widest tier only eps*0.06 = 0.6% of tokens, which is what the
                # ar10m cells got, and 98.4% of that tier's occupancy was this
                # floor alone. Drawing UNIFORM gives it eps/k = 2.5% -- 4x the
                # gradient and 4x the anti-decay pressure on the wide columns --
                # at the cost of a floor whose shape does not match the budget.
                _pri = self.tier_fracs if self.tier_fracs else self.tier_prior()
                if self.explore_from == "uniform":
                    _pri = [1.0 / len(self.widths)] * len(self.widths)
                cdf = torch.tensor(
                    _pri,
                    device=flat.device,
                    dtype=torch.float32,
                ).cumsum(0)
                u = torch.rand(flat.shape[0], device=flat.device, dtype=torch.float32)
                sampled = torch.bucketize(u, cdf.clamp(max=1.0), right=True).clamp_(
                    max=len(self.widths) - 1
                )
                take = (
                    torch.rand(flat.shape[0], device=flat.device, dtype=torch.float32)
                    < self.explore_frac
                )
                choice = torch.where(take, sampled, choice)
        else:
            choice = torch.full(
                (flat.shape[0],), int(self._forced_tier), device=flat.device, dtype=torch.long
            )
        # cost-aware aux.
        #
        # "mean" (legacy) holds the expected routed width at the budget. It
        # constrains ONLY THE MEAN, and infinitely many distributions share a
        # mean -- including the degenerate bimodal one where half the tokens sit
        # at tier 0, half at the widest, and the middle tiers are never used.
        # That failure is invisible to this term.
        #
        # "kl" pins the whole shape: KL(batch-marginal router distribution ||
        # target prior). Same term the standard MoE load-balancing loss uses,
        # generalized from a uniform prior to a skewed one.
        #
        # Either way the aux is a BACKSTOP, not the mechanism: `_causal_budget_clip`
        # already enforces occupancy as a hard constraint (this is MoNE's
        # argument for capacity over an auxiliary loss, 2407.19985). "none" says
        # so explicitly.
        wvec = torch.tensor(self.widths, device=flat.device, dtype=torch.float32)
        mean_w = (probs * wvec).sum(-1).mean()
        if self.aux_mode == "none":
            self._last_aux = None
        elif self.aux_mode == "kl":
            pi = torch.tensor(self.tier_fracs, device=flat.device, dtype=torch.float32).clamp_min(
                1e-9
            )
            marg = probs.mean(0).clamp_min(1e-9)
            self._last_aux = self.aux_weight * (marg * (marg.log() - pi.log())).sum()
        else:
            self._last_aux = (
                self.aux_weight * ((mean_w - self.budget) / self.intermediate_size) ** 2
            )
        # Both of these are device->host SYNCS, once per layer per step. They
        # are diagnostics only, so they are opt-in (bench/eval set
        # collect_stats); the training path never pays for them.
        if self.collect_stats:
            self._last_mean_width = float(wvec[choice].mean().detach()) if choice.numel() else None
            self._last_fracs = [
                float((choice == k).float().mean().detach()) for k in range(len(self.widths))
            ]

        tail = probs.flip(-1).cumsum(-1).flip(-1)  # tail[:, k] = P(choice >= k)
        # The clip is hoisted here (not inside _forward_sorted) so that every
        # dispatch computes the SAME function -- gate P depends on that. But it
        # must NOT run under _forced_tier: that override exists so the parity
        # tests can pin every token to one tier and reproduce a dense SwiGLU or
        # a MatFormer prefix exactly, and clipping to the budget share silently
        # demotes most of those tokens. Hoisting it unconditionally broke
        # test_mat_ffn_nested_slices_are_exact (max|diff| 0.41 vs a 1e-5 bar).
        # _forced_tier_fracs is deliberately still clipped: that is the bench
        # path, and every gate number to date was measured with the clip on.
        if self._forced_tier is None:
            if self.track_occupancy:
                _w = choice.detach().reshape(-1)
                self._wish_counts.index_add_(
                    0, _w, torch.ones_like(_w, dtype=self._wish_counts.dtype)
                )
            choice = self._causal_budget_clip(choice, shape)
        if (
            self.route_mode == "threshold"
            and self.route_ctrl in ("quantile", "quantile_nested")
            and self.training
            and self._forced_tier is None
        ):
            # Measured on `probs`, i.e. BEFORE the clip and independent of how
            # this step's tiers were assigned (router or warmup sampler), so the
            # same statistic is valid in both phases. `copy_` keeps it idempotent
            # under grad-ckpt recompute; `step_route_thresholds` applies it.
            with torch.no_grad():
                fr = self.tier_fracs if self.tier_fracs else self.tier_prior()
                tl = probs.detach().flip(-1).cumsum(-1).flip(-1)[:, 1:].float()
                if tl.shape[0] > (1 << 24):  # torch.quantile's input limit
                    tl = tl[:: (tl.shape[0] >> 24) + 1]
                qs = torch.tensor(
                    [min(1.0, max(0.0, 1.0 - sum(fr[j:]))) for j in range(1, len(self.widths))],
                    device=tl.device,
                    dtype=torch.float32,
                )
                if self.route_ctrl == "quantile_nested":
                    # Condition on passing the lower tiers without a boolean
                    # gather (static shapes, compile-safe): failed rows are set
                    # to -1, below any tail in [0, 1], and the quantile is still
                    # taken over ALL rows at the unconditional level 1 - f_k.
                    # While the passing set holds >= f_k of rows, exactly f_k
                    # of rows -- all of them passers -- land above it.
                    qv = []
                    alive = torch.ones_like(tl[:, 0], dtype=torch.bool)
                    for j in range(tl.shape[1]):
                        col = torch.where(alive, tl[:, j], torch.full_like(tl[:, j], -1.0))
                        t = torch.quantile(col, qs[j])
                        qv.append(t)
                        alive = alive & (tl[:, j] > t)
                    self.route_thresh_q.copy_(torch.stack(qv))
                else:
                    self.route_thresh_q.copy_(
                        torch.stack([torch.quantile(tl[:, j], qs[j]) for j in range(tl.shape[1])])
                    )
        if (
            self.route_mode == "threshold"
            and self.route_ctrl == "wish"
            and self.training
            and self._forced_tier is None
        ):
            # Recomputed from `probs` rather than read off `choice`: during the
            # warmup `choice` is SAMPLED, and the controller must see what the
            # thresholds would route. Relative error, so every tier moves at the
            # same log-rate whatever its target share.
            with torch.no_grad():
                fr = self.tier_fracs if self.tier_fracs else self.tier_prior()
                tl = probs.detach().flip(-1).cumsum(-1).flip(-1)[:, 1:].float()
                wish = torch.cumprod(
                    (tl > self.route_thresh.view(1, -1)).long(), dim=1
                ).sum(dim=1)
                tgt = torch.tensor(
                    [sum(fr[j:]) for j in range(1, len(self.widths))],
                    device=tl.device,
                    dtype=torch.float32,
                ).clamp_min(1e-6)
                got = torch.stack([(wish >= j).float().mean() for j in range(1, len(self.widths))])
                # Clipped: tier 3's relative error spans [-1, 1/t - 1] ~ [-1, 16],
                # and an unclipped step overshoots by e^(16 lr) and limit-cycles.
                self.route_thresh_err.copy_(((got - tgt) / tgt).clamp(-1.0, 1.0))
        if (
            self.route_mode == "threshold"
            and self.route_ctrl == "post_clip"
            and self.training
            and self._forced_tier is None
        ):
            # LEGACY -- ratchets theta to 0; see `route_ctrl` in __init__.
            # Controller MEASUREMENT, AFTER the clip on purpose. The theta write
            # itself lives in `step_route_thresholds`, called once per optimizer
            # step -- see the gradient-checkpointing note below.
            #
            # Measuring before the clip -- which is what the first version did --
            # drives the router's WISHES to the prior, and the clip then demotes
            # the per-sequence overflow, so REALIZED occupancy lands below target
            # and stays there. Measured: mean width stalled at 664-675 against a
            # 768 budget with no sign of closing.
            #
            # Targeting post-clip occupancy instead makes the loop self-correcting:
            # if realized is under target, theta falls, more tokens want width,
            # and the clip binds -- at which point realized EQUALS the cap. The
            # ceiling is then whatever the clip permits, which `random_prior`
            # (forced exactly at the prior) showed is ~735 rather than 768,
            # because the cap is enforced per sequence with integer rounding.
            #
            # THIS BLOCK MUST NOT MUTATE ANYTHING FORWARD READS. Until 2026-08-29
            # it did `self.route_thresh += lr * (realized - tgt)` right here, and
            # under gradient checkpointing that is a correctness bug, not a
            # nuance: a checkpointed block's forward runs TWICE per step -- once
            # to produce activations, once recomputed during backward -- so the
            # recompute read an already-updated theta, routed at least one token
            # to a different tier, and changed the nested-slice widths. Different
            # slice shapes mean a different compiled specialization and a
            # different saved-tensor list, which surfaces as
            #   CheckpointError: Recomputed values ... have different metadata
            # with EVERY position mismatched (positions 0/1 literally swapped),
            # not one tensor resized. Observed on an A6000 at S=8192 (job 56494)
            # and a Blackwell at S=32768 (job 59094314). Data-dependent, because
            # a small theta move is absorbed whenever `_causal_budget_clip`'s
            # per-sequence integer cap binds -- which is why 4096 never tripped
            # it and one 32768 cell survived while its twin died at step ~15.
            #
            # `copy_` is the load-bearing verb: it is idempotent, so the
            # recompute writing the same slot again changes nothing, and the
            # error it stores feeds no shape and no routing decision.
            with torch.no_grad():
                tgt = torch.tensor(
                    [sum(self.tier_prior()[j:]) for j in range(1, len(self.widths))],
                    device=choice.device,
                    dtype=torch.float32,
                )
                realized = torch.stack(
                    [(choice >= j).float().mean() for j in range(1, len(self.widths))]
                ).to(torch.float32)
                self.route_thresh_err.copy_(realized - tgt)
        # On-device occupancy accumulation, placed AFTER _causal_budget_clip so
        # it records the assignment that is actually DISPATCHED rather than the
        # router's pre-clip wish. No host sync and no branch on a per-step value
        # (see `track_occupancy`), so it is free on the training path. Detached:
        # telemetry, never a gradient path.
        if self.track_occupancy:
            _c = choice.detach().reshape(-1)
            self._occ_counts.index_add_(0, _c, torch.ones_like(_c, dtype=self._occ_counts.dtype))
            if len(shape) == 3 and int(shape[1]) >= 2:
                _s = int(shape[1])
                _half = (torch.arange(_s, device=_c.device) >= _s // 2).long()
                _idx = _half.repeat(int(shape[0])) * len(self.widths) + _c
                self._occ_pos_counts.view(-1).index_add_(
                    0, _idx, torch.ones_like(_idx, dtype=self._occ_pos_counts.dtype)
                )
        if (
            self.gate_mode == "mixture"
            and self.training
            and self._forced_tier is None
            and float(getattr(self, "gate_anneal", 0.0) or 0.0) < 1.0
        ):
            # Train-time exact mixture; eval keeps the hard argmax path so that
            # realized occupancy still measures the LEARNED policy.
            #
            # CUTOVER at anneal >= 1: the annealed mixture is then bit-exact with
            # _forward_sorted (test asserts torch.equal), so continuing to run it
            # would compute the identical function at FULL WIDTH for every token
            # -- about 4x the budget FLOPs -- for nothing. Falling through makes
            # the cost profile "expensive while the router is learning, normal
            # once it has", instead of 4x for the whole run.
            out = self._forward_mixture(flat, tail, choice)
        elif self.dispatch in ("sorted", "grouped") and self._forced_tier is None:
            out = self._forward_sorted(flat, choice, tail, shape)
        else:
            # _forced_tier is an exactness/debug override (used by the parity
            # tests to reproduce a dense SwiGLU or a MatFormer prefix exactly).
            # The sorted path deliberately CLIPS each tier to its static budget
            # share, so it cannot honour "put every token in the widest tier";
            # route those requests through the gather path.
            out = self._forward_gathered(flat, choice, tail)
        if self.down_proj.bias is not None:
            out = out + self.down_proj.bias
        return out.reshape(shape)

    def _causal_budget_clip(self, choice, shape):
        """Hold the realized tier occupancy to the budget, causally, PER SEQUENCE.

        Applied in `forward` so EVERY dispatch sees the same assignment. It used
        to live inside the sorted path only, which meant `gather` and `sorted`
        computed different functions -- so a dispatch A/B was silently comparing
        two models rather than two schedules.

        A token keeps tier k only if at most cap_k tokens at positions <= its own
        WITHIN ITS OWN SEQUENCE also wanted tier >= k (earliest wins). That is a
        prefix property, so a later token can never change an earlier token's
        width.

        THE `PER SEQUENCE` PART IS THE WHOLE POINT (fixed 2026-08-21). This ran
        `cumsum(0)` over the FLATTENED [B*S] batch, with a pooled cap of
        frac*B*S. Two things were wrong with that, and the second is why a
        multi-position causality probe caught it:
          * a token's width depended on the other SEQUENCES in its batch, so the
            model computed a different function of the same sequence depending on
            what it was batched with -- and eval was not reproducible;
          * because the flatten is row-major, row 1's EARLY positions sit after
            row 0's LATE positions in the cumsum, so perturbing a token at
            position p changed logits at positions < p in the other rows. It
            probed as a backward leak (0.079 at position 128, 0.057 at 170, clean
            at 20/40/64/90/150) even though nothing inside a single sequence ever
            saw its own future.
        Occupancy is unchanged: the same fraction, applied per row instead of
        pooled. Making the cap per-row also makes each sequence's tier widths
        independent of batch composition, which is what the sorted dispatch's
        static shapes assume anyway.
        """
        b, s = int(shape[0]), int(shape[1])
        c = choice.view(b, s)
        tier = torch.zeros_like(c)
        alive = torch.ones_like(c, dtype=torch.bool)
        for k in range(1, len(self.widths)):
            # `alive` makes the clip NESTED, and that is load-bearing rather
            # than tidy. Without it (the form shipped until 2026-08-21) each
            # tier was capped independently, so a token could FAIL the k=1 cap
            # and still PASS k=2 -- its `keep` was credited anyway and it came
            # out with tier >= 1. Realized occupancy then exceeded the cap that
            # `_forward_sorted` sizes its static window from: measured at
            # b=3,s=192, tier 1 wanted 443 rows against a 432-row window, so 11
            # tokens that had selected the tier fell outside it and silently
            # lost that contribution. WHICH 11 depends on the stable sort order,
            # i.e. on every other token's choice -- so perturbing one token
            # changed the output of tokens BEFORE it (0.058 at 2 of 8 probe
            # positions, and only under `sorted`; `gather` uses no window and
            # was exactly 0.0). Dropping out at the first failed tier and
            # staying out keeps every per-tier count <= cap, which is exactly
            # the invariant the window assumes.
            want = (c >= k) & alive
            keep = want & (want.long().cumsum(1) <= self._tier_cap(k, s))
            alive = keep
            tier = tier + keep.long()
        return tier.reshape(-1)

    def _tier_cap(self, k: int, s: int) -> int:
        """Per-SEQUENCE row budget for tier k. The one definition of the cap.

        `_causal_budget_clip` enforces it per row and `_forward_sorted` sizes its
        static window as b * this. They MUST agree: the sorted window relies on
        every token that selected tier >= k landing inside the first `cnt` rows
        of the tier-descending sort, and only the cap makes that true. When they
        disagreed (2026-08-21, briefly: cap per row = round(f*s) but window =
        round(f*b*s), which differs whenever the per-row round goes UP), the
        overflow tokens fell outside the window and silently lost their tier-k
        contribution -- and WHICH tokens overflowed depended on the sort order,
        hence on other tokens. It probed as a 0.058 backward leak at 2 positions
        under `sorted` while `gather`, which uses no window, was exactly 0.0.
        """
        f = min(1.0, self._tier_frac(k) * self.cap_headroom)
        return min(s, max(1, int(round(f * s))))

    # ------------------------------------------------------------- dispatchers
    def _tier_slice(self, xs, prev, w):
        """One incremental nested column slice [prev:w] applied to rows ``xs``."""
        gate_b = None if self.gate_proj.bias is None else self.gate_proj.bias[prev:w]
        up_b = None if self.up_proj.bias is None else self.up_proj.bias[prev:w]
        g = self.act_fn(F.linear(xs, self.gate_proj.weight[prev:w], gate_b))
        u = F.linear(xs, self.up_proj.weight[prev:w], up_b)
        return F.linear(g * u, self.down_proj.weight[:, prev:w])

    def _forward_mixture(self, flat, tail, choice):
        """Exact soft mixture over tiers, annealable to the hard staircase.

        With NESTED widths, sum_k p_k * FFN_k == sum_j tail_j * inc_j, so this is
        the same slice loop as the hard path with the real tail_j instead of the
        STE 1.0 and -- at anneal 0 -- no membership mask. That missing mask is
        the entire mechanism: it is what lets a token that did NOT pick tier j
        still produce gradient about tier j, which the shipped STE path cannot.

        `gate_anneal` in [0,1] interpolates to the hard gate, so training can
        start with full counterfactual signal and finish computing exactly what
        eval computes. At anneal 1 the gate is `mask` in the forward with
        gradient still flowing through tail on members -- identical to
        `_forward_sorted`.

        Full-width compute for every token, so this buys router quality, not
        FLOPs.
        """
        a = float(getattr(self, "gate_anneal", 0.0) or 0.0)
        _rho = getattr(self, "mixture_frac", 1.0)
        rho = 1.0 if _rho is None else float(_rho)
        use_mix = None
        if rho < 1.0:
            use_mix = (
                torch.rand(flat.shape[0], device=flat.device, dtype=torch.float32) < rho
            ).unsqueeze(-1)
        y = None
        prev = 0
        for k, w in enumerate(self.widths):
            if w <= prev:
                continue
            inc = self._tier_slice(flat, prev, w)
            if self.tier_scale is not None:
                inc = inc * self.tier_scale[k].to(inc.dtype)
            gt = tail[:, k].unsqueeze(-1).to(inc.dtype)
            mask = (choice >= k).unsqueeze(-1).to(inc.dtype)
            # forward value `mask`, gradient still through tail on members
            ste = (gt + (1.0 - gt.detach())) * mask
            if a <= 0.0:
                g = gt
            else:
                g = (1.0 - a) * gt + a * ste
            if use_mix is not None:
                # full counterfactual for the sampled tokens, hard gate for the
                # rest -- NOT a weakened gate for everyone.
                g = torch.where(use_mix, gt, ste)
            y = inc * g if y is None else y + inc * g
            prev = w
        return y

    def _forward_gathered(self, flat, choice, tail):
        """NAIVE dispatch: index_select / index_add_ per tier.

        Kept for the optimization-pass A/B. Two problems, both measured:
        ``(choice >= k).nonzero()`` is a HOST SYNC per tier per layer, and the
        row counts vary every step so the gather shapes churn. At 100m this arm
        ran 1.94x dense while its own backbone (swa_sinkdiet, no width routing)
        ran 1.15x -- the routing was costing ~1.7x instead of saving ~0.8x.
        """
        out = torch.zeros_like(flat)
        prev = 0
        for k, w in enumerate(self.widths):
            if w <= prev:
                continue
            if k == 0:
                out = out + self._tier_slice(flat, prev, w).to(out.dtype)
            else:
                sel = (choice >= k).nonzero(as_tuple=True)[0]  # <- host sync
                if sel.numel():
                    y = self._tier_slice(flat.index_select(0, sel), prev, w)
                    gt = tail[:, k].index_select(0, sel).unsqueeze(-1).to(y.dtype)
                    y = y * (gt + (1.0 - gt.detach()))
                    out.index_add_(0, sel, y.to(out.dtype))
            prev = w
        return out

    def _forward_sorted(self, flat, choice, tail, shape):
        """OPTIMIZED dispatch: sort by tier once, then contiguous slices.

        The widths are NESTED, so tier k's token set is a subset of tier k-1's.
        Sorting rows by tier DESCENDING therefore makes every tier a contiguous
        PREFIX of the sorted array, and each incremental column slice becomes a
        plain dense GEMM on a view -- no gather, no scatter_add, no nonzero().
        This is the same trick round 7's permute dispatch used on the MoP lanes
        (worth -69 ms/step there).

        Row counts are STATIC (``ceil(frac_k * N)`` from the budget), so the
        shapes are compile-time constants and the layer keeps ONE dynamo graph
        instead of recompiling as occupancy drifts. Rows inside a tier's static
        window that did not actually select it are zeroed by ``valid`` rather
        than skipped, which keeps the arithmetic exact while keeping the shape
        fixed -- the same bargain dual_stream makes.

        Causality is unaffected: ``choice`` is a per-token argmax computed
        before any sorting, and a permutation cannot move information backwards.
        """
        n = flat.shape[0]
        _b, _s = int(shape[0]), int(shape[1])
        # (clip already applied in forward -- see _causal_budget_clip)
        # CAUSAL earliest-wins clip BEFORE sorting. Sorting by tier and taking a
        # prefix is not by itself causal: membership would then depend on how
        # many higher-tier tokens exist ANYWHERE, so a later token could evict an
        # earlier one. Clipping on a prefix count first makes tier membership a
        # function of positions <= p only; the sort afterwards is a permutation
        # of an already-decided assignment and cannot move information backwards.
        order = torch.argsort(choice, descending=True, stable=True)
        inv = torch.empty_like(order)
        inv.scatter_(0, order, torch.arange(n, device=flat.device))
        xs_all = flat.index_select(0, order)
        ch_s = choice.index_select(0, order)
        tail_s = tail.index_select(0, order)

        # Incremental nested slices on contiguous PREFIXES, accumulated with
        # in-place slice adds instead of rebuilding the output tensor.
        #
        # Why not compute each row group once at its own full width: the tier
        # windows are STATIC (fixed fractions of N), so a row's position inside
        # window k depends on how many OTHER rows chose tier >= k -- including
        # rows at later positions. Sizing the compute off that window is
        # therefore non-causal, and the multi-position probe caught it
        # immediately. The `valid` mask below is what keeps it causal: a row
        # inside window k that did not actually select tier k contributes zero,
        # so the result depends only on each row's own (causally clipped)
        # choice.
        #
        # The win here is dropping torch.cat. Rebuilding [N, h] once per tier
        # cost 470 MB/layer at deep80 (N=65536, h=896, bf16) -- ~33 GB/step over
        # 70 routed layers -- on top of the gather and the inverse. In-place
        # adds on the prefix touch only cnt_k rows each.
        if self.dispatch == "grouped":
            got = self._forward_grouped(flat, choice, tail, order, inv, xs_all, ch_s, tail_s, n)
            if got is not None:
                return got
        y = self._tier_slice(xs_all, 0, self.widths[0]).to(xs_all.dtype)
        prev = self.widths[0]
        for k in range(1, len(self.widths)):
            w = self.widths[k]
            if w <= prev:
                continue
            # b * per-sequence cap, NOT round(frac*n): see _tier_cap.
            cnt = min(n, self._tier_cap(k, _s) * _b)
            part = self._tier_slice(xs_all[:cnt], prev, w)
            if self.tier_scale is not None:
                part = part * self.tier_scale[k].to(part.dtype)
            gt = tail_s[:cnt, k].unsqueeze(-1).to(part.dtype)
            part = part * (gt + (1.0 - gt.detach()))
            part = part * (ch_s[:cnt] >= k).unsqueeze(-1).to(part.dtype)
            # In-place prefix add. The previous form was
            #   torch.index_add(y, 0, torch.arange(cnt), part)
            # which allocated an index tensor every tier of every layer and then
            # ran a general SCATTER to do what a contiguous slice add does
            # directly. Safe in autograd: y is a non-leaf (an F.linear output)
            # and linear's backward needs its INPUT and weight, not its output.
            y[:cnt] = y[:cnt] + part.to(y.dtype)
            prev = w
        return y.index_select(0, inv)

    # ------------------------------------------------------ grouped dispatch
    def _uniform_increments(self):
        """Increment widths, or None when the menu is not uniform.

        The grouped path needs every tier's GEMM to share K and N and differ
        only in row count -- that is exactly `torch._grouped_mm`'s contract
        (aten::_grouped_mm(self, mat2, offs, bias, out_dtype)). Uniform
        increments are what make N identical across tiers, so the default menu
        is quarters of intermediate_size rather than {1/4, 1/2, 1}.
        """
        incs, prev = [], 0
        for w in self.widths:
            incs.append(w - prev)
            prev = w
        return incs if len(set(incs)) == 1 else None

    def _forward_grouped(self, flat, choice, tail, order, inv, xs_all, ch_s, tail_s, n):
        """All tiers in THREE kernels instead of 3 per tier.

        The tiers are nested prefixes of the sorted rows, so tier j covers rows
        [0:cnt_j] with cnt_1 >= cnt_2 >= ... Those row sets OVERLAP, which a
        grouped GEMM cannot express directly, so the input prefixes are
        concatenated into one ragged batch and `offs` marks the boundaries.
        That costs sum(cnt_j) rows of input copy -- about 2.5N at four uniform
        tiers -- and buys 3 kernel launches instead of 12, with the weights
        streamed once per tier rather than once per tier per call.

        Whether that trade wins is an empirical question: if the FFN is
        weight-traffic bound rather than launch bound, the extra input copy can
        cost more than the saved launches. Falls back to the loop on any error
        so an unsupported dtype or shape degrades instead of failing a run.
        """
        incs = self._uniform_increments()
        if incs is None:
            return None
        # torch._grouped_mm HAS NO WORKING BACKWARD (measured 2026-08-23 on
        # A6000/sm_86, torch 2.11):
        #     RuntimeError: Invalid strides/sizes, got [0, 0] for strides and
        #     [1024, 464] for sizes
        # This path has never actually run -- it is gated behind
        # dispatch="grouped", which no arm sets, AND `_uniform_increments()`
        # returns None for both production menus (skew4 increments
        # [384,384,768,1536], flat4 [320,320,256,320]). So the crash has been
        # masked twice over. Anyone who set dispatch="grouped" on a uniform
        # menu would get a correct forward and then die in backward hours into
        # a run. Fall back to the loop whenever a gradient will be taken; the
        # inference/bench path is still free to use it.
        if torch.is_grad_enabled() and (
            flat.requires_grad
            or self.gate_proj.weight.requires_grad
            or self.up_proj.weight.requires_grad
            or self.down_proj.weight.requires_grad
        ):
            return None
        inc = incs[0]
        G = len(self.widths)
        cnts = [n] + [self._tier_cap(k, n) for k in range(1, G)]
        try:
            a_cat = torch.cat([xs_all[:c] for c in cnts], dim=0)
            offs = torch.tensor(
                list(torch.tensor(cnts).cumsum(0).tolist()),
                device=flat.device,
                dtype=torch.int32,
            )
            wg = torch.stack([self.gate_proj.weight[j * inc : (j + 1) * inc].t() for j in range(G)])
            wu = torch.stack([self.up_proj.weight[j * inc : (j + 1) * inc].t() for j in range(G)])
            wd = torch.stack(
                [self.down_proj.weight[:, j * inc : (j + 1) * inc].t() for j in range(G)]
            )
            g = torch._grouped_mm(a_cat, wg, offs)
            u = torch._grouped_mm(a_cat, wu, offs)
            act = self.act_fn(g) * u
            parts = torch._grouped_mm(act, wd, offs)
        except Exception:
            return None

        y = None
        start = 0
        for j, c in enumerate(cnts):
            part = parts[start : start + c]
            if j > 0:
                gt = tail_s[:c, j].unsqueeze(-1).to(part.dtype)
                part = part * (gt + (1.0 - gt.detach()))
                part = part * (ch_s[:c] >= j).unsqueeze(-1).to(part.dtype)
                y[:c] = y[:c] + part.to(y.dtype)
            else:
                y = part.to(xs_all.dtype)
            start += c
        return y.index_select(0, inv)

    def tier_row_scale(self, device=None, dtype=None):
        """Per-ROW usage frequency of the FFN weights, shape [intermediate_size].

        Row r of gate_proj/up_proj (and column r of down_proj) belongs to tier k
        where widths[k-1] <= r < widths[k], and that block only receives gradient
        from tokens routed to tier >= k -- a fraction `_tier_frac(k)` of them.
        Under the default skewed prior that is 1.00 / 0.40 / 0.18 / 0.06, so the
        widest block (also the LARGEST, 1536 of 3072 columns) trains on 16.7x
        fewer tokens than the narrowest.

        Decoupled weight decay does not know this: it shrinks every parameter on
        every step regardless of whether the parameter was used. Over an
        ar10m-length run (26,037 steps, lr 1.75e-3, wd 0.1) that is a factor of
        exp(-lr*wd*steps) ~ 0.01, and tier 3 has 16.7x fewer gradient updates to
        push back with. Under the MEASURED collapse it was far worse -- tier 3
        saw 0.02% of tokens, so those columns were decayed toward zero with
        essentially nothing resisting, which plausibly makes the collapse
        self-reinforcing: the wide columns rot, which makes them genuinely
        useless, which confirms the router's decision to avoid them.

        Scaling decay by this vector equalises decay-per-update across tiers.
        """
        scale = torch.ones(self.intermediate_size, dtype=torch.float32)
        prev = 0
        for k, w in enumerate(self.widths):
            if w > prev:
                scale[prev:w] = float(self._tier_frac(k))
            prev = w
        if prev < self.intermediate_size:
            # columns past the widest tier are never routed to at all
            scale[prev:] = float(self._tier_frac(len(self.widths) - 1))
        return scale.to(device=device, dtype=dtype) if device or dtype else scale

    def tier_prior(self):
        """Per-tier target occupancy at the CURRENT warmup stage. Sums to 1.

        With no `tier_fracs` this is the legacy closest-to-uniform vector that
        hits `budget` and the warmup stage is ignored -- the existing `mat_ffn`
        arm is bitwise unchanged.

        With `tier_fracs` set, stage s interpolates base -> target:
        stage 0 is the BASE prior (every tier, the widest included, gets its
        share of the tokens, which is what keeps it from being undertrained)
        and stage `warmup_stages` is the target skew.

        The base is uniform (1/k each) unless `warmup_top_weight` != 1, in which
        case it is normalize([1, ..., 1, lambda]) -- MatFormer's appendix knob
        (2310.07707), which tunes the per-granularity sampling probabilities and
        reports that upweighting the largest granularity helps the big model at
        modest cost to the small submodels. The per-token analogue works because
        the nested prefix is SHARED: tier 0's columns take gradient from every
        token, so raising the widest tier's share changes the mixture on the
        shared columns toward the widest configuration. It acts ONLY during
        warmup -- the endpoint is untouched, so the steady state, E[w] and the
        matched-N solve are unchanged, and the extra warmup FLOPs are already
        priced by the submitter's `warm0` ms/step term.
        """
        if self.tier_fracs is None:
            from ..utils.forced_routing import budget_fractions

            return budget_fractions(self.widths, self.budget)
        k = len(self.widths)
        s = min(max(int(self.warmup_stage), 0), self.warmup_stages)
        t = s / float(self.warmup_stages)
        lam = float(getattr(self, "warmup_top_weight", 1.0) or 1.0)
        if lam == 1.0:
            base = [1.0 / k] * k
        else:
            w = [1.0] * (k - 1) + [lam]
            tot = sum(w)
            base = [x / tot for x in w]
        return [b + t * (p - b) for b, p in zip(base, self.tier_fracs)]

    def _tier_frac(self, k: int) -> float:
        """Static share of rows reaching tier >= k, at the current stage.

        Cached per stage, not globally: the cap sizes `_forward_sorted`'s static
        window, so it must move with the annealed prior or the warmup's uniform
        assignment would be clipped straight back down to the target skew --
        which is exactly the starvation the warmup exists to prevent.
        """
        key = int(self.warmup_stage) if self.tier_fracs is not None else -1
        if self._static_fracs is None or self._static_fracs[0] != key:
            f = self.tier_prior()
            self._static_fracs = (key, [sum(f[i:]) for i in range(len(f))])
        return self._static_fracs[1][k]


def step_route_thresholds(model) -> int:
    """Apply one threshold-controller update per optimizer step; return #modules.

    The other half of the fix described in ``RoutedMatFormerSwiGLUMLP.forward``.
    Forward MEASURES the post-clip occupancy error into ``route_thresh_err`` and
    touches nothing routing depends on; this applies it. Call it once per
    optimizer step, at the TOP of the step, so the update computed from step
    k-1's forward governs step k -- which is the causality the controller was
    always documented to have ("lands on the NEXT step") and, before 2026-08-29,
    did not: under gradient checkpointing the write landed on the current step's
    own recompute and crashed it.

    GAIN CHANGED WITH THE FIX, and every mat_ffn_thresh cell before 2026-08-29
    predates it. theta used to move once per FORWARD, i.e. ``2 * grad_accum``
    times per optimizer step under grad-ckpt (twice per micro-batch, the second
    time off a distribution the first update had already shifted). It now moves
    once. The controller's FIXED POINT is unchanged -- it still settles where
    realized == target -- so the reported occupancy (767.9/768 at ar10m) should
    reproduce; only the approach is ~4x slower in steps, which is nothing over a
    40k-step run but is worth knowing if a short run looks under-occupied.

    Safe and free to call on any model: arms without a threshold-routed nested
    FFN match nothing and return 0.
    """
    n = 0
    for m in model.modules():
        if not isinstance(m, RoutedMatFormerSwiGLUMLP) or m.route_mode != "threshold":
            continue
        with torch.no_grad():
            if getattr(m, "route_ctrl", "post_clip") in ("quantile", "quantile_nested"):
                q = m.route_thresh_q
                ema = (1.0 - m.route_thresh_lr) * m.route_thresh + m.route_thresh_lr * q
                m.route_thresh.copy_(torch.where(torch.isnan(q), m.route_thresh, ema))
            elif getattr(m, "route_ctrl", "post_clip") == "wish":
                # wish above target -> raise theta. The floor keeps a theta that
                # reached 0 able to grow back (0 * exp(.) = 0 forever).
                m.route_thresh.copy_(
                    m.route_thresh.clamp_min(1e-9)
                    * torch.exp(m.route_thresh_lr * m.route_thresh_err)
                )
            else:
                m.route_thresh += m.route_thresh_lr * m.route_thresh_err
            m.route_thresh.clamp_(0.0, 1.0)
        n += 1
    return n


def mat_ffn_occupancy_payload(model) -> dict:
    """Drain the mat_ffn tier counters into log keys; ``{}`` when there are none.

    Follows `gather_width_payload`'s contract: keys are ABSENT for arms without a
    routed nested FFN rather than logged as zeros. Call this on a logging step
    only -- it is the single host sync the counters cost, and it ZEROES them, so
    each report covers exactly the steps since the previous call.

    Exists because the ar10m nest arms trained 26k steps with the router
    collapsed to 98.6% tier 0 and no metric said so; occupancy was only
    recoverable by a separate forward-pass job over the final checkpoint.
    """
    payload = _threshold_payload(model)
    mats = [
        m
        for m in model.modules()
        if isinstance(m, RoutedMatFormerSwiGLUMLP) and getattr(m, "track_occupancy", False)
    ]
    if not mats:
        return payload
    total = None
    pos = None
    wish = None
    for m in mats:
        c = m._occ_counts
        total = c.clone() if total is None else total + c
        c.zero_()
        wc = getattr(m, "_wish_counts", None)
        if wc is not None:
            wish = wc.clone() if wish is None else wish + wc
            wc.zero_()
        pc = getattr(m, "_occ_pos_counts", None)
        if pc is not None:
            pos = pc.clone() if pos is None else pos + pc
            pc.zero_()
    n = float(total.sum().item())
    if n <= 0:
        return payload
    fracs = (total.double() / n).tolist()
    widths = mats[0].widths
    payload.update({f"mat_ffn/tier{k}": float(v) for k, v in enumerate(fracs)})
    payload["mat_ffn/mean_width"] = float(sum(w * f for w, f in zip(widths, fracs)))
    payload["mat_ffn/budget"] = float(mats[0].budget)
    if wish is not None and float(wish.sum().item()) > 0:
        wf = (wish.double() / float(wish.sum().item())).tolist()
        # pre-clip; mean_width - wish_mean_width is what the clip took away
        payload["mat_ffn/wish_mean_width"] = float(sum(w * f for w, f in zip(widths, wf)))
    if pos is not None:
        wv = torch.tensor(widths, dtype=torch.float64)
        halves = pos.double().cpu()
        cnt = halves.sum(1)
        if bool((cnt > 0).all()):
            mw = (halves * wv).sum(1) / cnt
            payload["mat_ffn/mean_width_first_half"] = float(mw[0])
            payload["mat_ffn/mean_width_second_half"] = float(mw[1])
            # ~1 for a content router; the post-clip positional schedule was ~3
            payload["mat_ffn/width_half_ratio"] = float(mw[0] / mw[1])
    return payload


THETA_FLOOR = 1e-3


def _threshold_payload(model) -> dict:
    """Threshold-controller state as log keys; ``{}`` for non-threshold arms.

    theta was never logged, and every A40-grid mat_ffn_thresh cell sat at
    theta < 0.0015 from ~8% of training on -- visible at the first log line had
    anyone recorded it. `mat_ffn/theta_floor_frac` is the fraction of
    (layer, tier) thresholds within THETA_FLOOR of 0; 1.0 means the router no
    longer decides anything and the clip is routing by position.
    """
    th = [
        m.route_thresh.detach().float().cpu()
        for m in model.modules()
        if isinstance(m, RoutedMatFormerSwiGLUMLP) and hasattr(m, "route_thresh")
    ]
    if not th:
        return {}
    t = torch.stack(th)  # [layers, tiers-1]
    out = {}
    for k in range(t.shape[1]):
        out[f"mat_ffn/theta{k + 1}_mean"] = float(t[:, k].mean())
        out[f"mat_ffn/theta{k + 1}_min"] = float(t[:, k].min())
        out[f"mat_ffn/theta{k + 1}_max"] = float(t[:, k].max())
    out["mat_ffn/theta_floor_frac"] = float((t < THETA_FLOOR).float().mean())
    return out
