"""Mixture of Depths (MoD) Module.

This module implements Mixture of Depths as described in:
"Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"
(Raposo et al., 2024) - https://arxiv.org/abs/2404.02258

Key concepts:
- Dynamic compute allocation: tokens can skip layers based on learned routing
- Top-k routing: only k tokens (based on capacity) are processed per layer
- Auxiliary loss: trains router for autoregressive inference
- Significant FLOPs reduction with minimal performance impact

The router assigns a scalar weight to each token. Tokens with top-k weights
are processed through the layer; others pass through via residual connection.
"""

import collections
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput

from .doc_masking import PREFIX_ID


@dataclass
class MoDOutput(ModelOutput):
    """Output type for Mixture of Depths layers.

    Attributes:
        hidden_states: Output hidden states.
        router_weights: Router weights for each token.
        selected_mask: Boolean mask indicating which tokens were selected.
        aux_loss: Auxiliary loss for training router for inference.
    """

    hidden_states: torch.FloatTensor = None
    router_weights: Optional[torch.FloatTensor] = None
    selected_mask: Optional[torch.BoolTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None


class MoDRouter(nn.Module):
    """Router for Mixture of Depths.

    Computes a scalar routing weight for each token via linear projection.
    During training, uses top-k selection; during inference, can use
    learned threshold via auxiliary loss or auxiliary predictor.

    Args:
        hidden_size: Dimension of hidden representations.
        capacity: Fraction of tokens to process (0.0-1.0).
        aux_loss_weight: Weight for auxiliary loss.
        use_aux_predictor: Use auxiliary MLP for inference routing.
        gate_fn: Squashing applied to the raw logit *before* it multiplies the
            block output in ``MoDGate``. "raw" (default) reproduces the original
            behaviour: the unbounded logit is the gain. "sigmoid" bounds it to
            (0, 1); "tanh" gives p-MoD's zero-centered ``tanh_alpha * tanh(r)``.
            Top-k, the aux BCE and the inference threshold always use the raw
            logit — sigmoid/tanh are monotone so the top-k ranking is unchanged.
        tanh_alpha: Scale for gate_fn="tanh" (p-MoD's TanhNorm uses 0.2).
        capacity_warmup_steps: Linearly anneal the routed fraction 1.0 ->
            ``capacity`` over this many training steps (0 = off, the default).
    """

    def __init__(
        self,
        hidden_size: int,
        capacity: float = 0.125,
        aux_loss_weight: float = 0.01,
        use_aux_predictor: bool = False,
        gate_fn: str = "raw",
        tanh_alpha: float = 0.2,
        capacity_warmup_steps: int = 0,
        router_mode: str = "topk",
        router_chunk: int = 128,
        route_bias_lr: float = 1e-3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.capacity = capacity
        self.aux_loss_weight = aux_loss_weight
        self.use_aux_predictor = use_aux_predictor
        if router_mode not in {"topk", "threshold"}:
            raise ValueError(f"router_mode must be topk|threshold, got {router_mode!r}")
        self.router_mode = router_mode
        self.router_chunk = int(router_chunk)
        self.route_bias_lr = float(route_bias_lr)
        if router_mode == "threshold":
            # Loss-free capacity control (DeepSeek-V3 style): a NON-learned bias
            # nudged from realized statistics, so the routed fraction tracks
            # `capacity` without ever perturbing the LM gradient. A squared
            # penalty on the mean would fight the LM loss and is weakest exactly
            # when furthest from target; this hits the rate directly.
            self.register_buffer("route_bias", torch.zeros(()), persistent=True)
        if gate_fn not in {"raw", "sigmoid", "tanh"}:
            raise ValueError(f"gate_fn must be raw|sigmoid|tanh, got {gate_fn!r}")
        self.gate_fn = gate_fn
        self.tanh_alpha = tanh_alpha
        # Per-forward gather hints (see forward); read by _run_gathered_mod_block.
        self._last_uniform_count: Optional[int] = None
        self._last_capacity: float = float(capacity)
        self.capacity_warmup_steps = int(capacity_warmup_steps)

        # Router projection: hidden_size -> 1 scalar weight per token
        self.router = nn.Linear(hidden_size, 1, bias=False)

        # Only materialize the step counter when annealing is on, so the
        # state_dict of every existing config is byte-identical. Persistent so
        # the anneal resumes correctly after a preemption (the job script
        # auto-resumes from the newest checkpoint).
        if self.capacity_warmup_steps > 0:
            self.register_buffer(
                "_train_step", torch.zeros((), dtype=torch.long), persistent=True
            )

        # Optional auxiliary predictor for inference
        if use_aux_predictor:
            self.aux_predictor = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, 1),
            )
        else:
            self.aux_predictor = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        use_cache: bool = False,
        force_route_prefix: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute routing decisions.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            use_cache: If True, use inference mode (no top-k, use threshold).
            force_route_prefix: Number of leading positions that must be
                force-routed (always selected). Used by the SWA + persistent
                globals path so the prepended global tokens always traverse
                every block. The aux loss is computed only over the remaining
                content positions so the router is supervised only on tokens
                where it actually has a meaningful choice.

        Returns:
            Tuple of (router_weights, selected_mask, aux_loss).
            - router_weights: [batch, seq_len] scalar weights
            - selected_mask: [batch, seq_len] bool mask of selected tokens
            - aux_loss: Auxiliary loss for training (None during inference)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Raw logit. This drives top-k, the aux BCE and the inference threshold;
        # only the *gate gain* is squashed (see _gate_gain).
        router_weights = self.router(hidden_states).squeeze(-1)

        prefix = max(0, int(force_route_prefix))
        if prefix > seq_len:
            prefix = seq_len
        content_len = seq_len - prefix

        capacity = self._current_capacity()

        # Calculate k based on capacity, computed over the *content* portion
        # so capacity describes "fraction of content tokens that get processed".
        k = max(1, int(content_len * capacity)) if content_len > 0 else 0

        aux_loss = None

        # Stashed for the gather path: a `topk` selection has EXACTLY prefix+k
        # per row by construction, so mod_gather_index can skip its count sync;
        # a training-mode `threshold` selection is ragged (None). Capacity rides
        # along for the fixed-width (kmod) gather. Both are plain python values
        # derived from the same locals the selection used — no drift possible.
        self._last_uniform_count: Optional[int] = None
        self._last_capacity: float = float(capacity)

        if self.training:
            # Training mode: use top-k selection on the content slice; the
            # forced prefix is set to True unconditionally.
            selected_mask = torch.zeros(
                batch_size, seq_len, dtype=torch.bool, device=hidden_states.device
            )
            if prefix > 0:
                selected_mask[:, :prefix] = True
            if content_len > 0 and k > 0:
                content_weights = router_weights[:, prefix:]
                selected_mask[:, prefix:] = self._select(content_weights, k, capacity)
                if self.router_mode == "topk":
                    self._last_uniform_count = prefix + k

            # Compute auxiliary loss only over the content slice — the prefix
            # has no learnable choice so there is nothing to supervise there.
            if content_len > 0:
                targets = selected_mask[:, prefix:].float()
                aux_loss = (
                    F.binary_cross_entropy_with_logits(
                        router_weights[:, prefix:], targets, reduction="mean"
                    )
                    * self.aux_loss_weight
                )
            else:
                aux_loss = router_weights.new_zeros(())

        else:
            # Inference mode
            if use_cache and seq_len == 1:
                # Autoregressive generation: single token. The single token
                # must not be a "global" (we should never decode at a global
                # position), so prefix logic doesn't apply here.
                if self.aux_predictor is not None:
                    pred_logits = self.aux_predictor(hidden_states).squeeze(-1)
                    selected_mask = pred_logits > 0
                else:
                    selected_mask = torch.sigmoid(router_weights) > 0.5
            else:
                # Batch inference or prefill: use top-k on the content slice.
                selected_mask = torch.zeros(
                    batch_size, seq_len, dtype=torch.bool, device=hidden_states.device
                )
                if prefix > 0:
                    selected_mask[:, :prefix] = True
                if content_len > 0 and k > 0:
                    content_weights = router_weights[:, prefix:]
                    _, top_k_indices = torch.topk(content_weights, k, dim=-1)
                    selected_mask[:, prefix:].scatter_(1, top_k_indices, True)
                    self._last_uniform_count = prefix + k

        return self._gate_gain(router_weights), selected_mask, aux_loss

    def _select(self, weights: torch.Tensor, k: int, capacity: float) -> torch.Tensor:
        """Pick the routed tokens from per-token scores ``weights`` ``[B, L]``.

        The SCORE is always a pure per-token function of a causally built hidden
        state. What differs between modes is the SELECTION RULE, and that is where
        causality is won or lost.

        ``topk`` (default, MoD as published)
            ``argtopk`` over the whole sequence. Token t is selected iff its score
            outranks all but k-1 others -- an order statistic over every position,
            INCLUDING t+1..L-1. So a token's fate depends on its future, and under
            packed documents on its neighbours' documents. Kept as the default
            because it is what the published architecture does and what every
            existing checkpoint was trained with.

        (A chunk-local top-k was tried and REJECTED: ranking inside fixed chunks
        bounds the lookahead but does nothing for packed documents, because any
        chunk straddling a boundary still has the two documents competing. With
        median document ~688 tokens against a 4096 sequence that is almost every
        boundary; measured leak 4.7e-2 vs top-k's 5.0e-2, i.e. no improvement.
        Doc-aligned chunks would fix it but require ragged groups -- exactly what
        the chunking was meant to avoid.)

        ``threshold`` (option A)
            ``score + route_bias > 0``: a pure per-token decision, so strictly
            causal and free of any cross-document channel. This is also the rule
            inference already uses, so training and generation finally agree.
            Capacity is met only in expectation, which makes the selection count
            ragged -- see ``mod_gather_index``.
        """
        mode = self.router_mode
        if mode == "threshold":
            selected = (weights + self.route_bias) > 0
            if self.training:
                self._update_route_bias(selected, capacity)
            return selected

        selected = torch.zeros_like(weights, dtype=torch.bool)
        top_k_indices = torch.topk(weights, k, dim=-1).indices
        return selected.scatter_(1, top_k_indices, True)

    @torch.no_grad()
    def _update_route_bias(self, selected: torch.Tensor, capacity: float) -> None:
        """Nudge the routing bias so the realized fraction tracks ``capacity``.

        Pure statistics, no gradient: over-routing pushes the bias down, under-
        routing pushes it up. Deliberately NOT an aux loss -- a penalty term would
        trade off against the LM objective and needs a weight to tune, whereas this
        converges on the rate itself and has only a step size.
        """
        realized = selected.float().mean()
        self.route_bias -= self.route_bias_lr * (realized - capacity)

    def _current_capacity(self) -> float:
        """Routed fraction for this step, honoring the optional warmup anneal.

        The counter ticks per training *forward pass* (micro-batch), not per
        optimizer step — under gradient accumulation the two differ by
        ``grad_accum``. Size ``capacity_warmup_steps`` accordingly.
        """
        if self.capacity_warmup_steps <= 0 or not self.training:
            return self.capacity
        step = int(self._train_step.item())
        self._train_step += 1
        frac = min(1.0, step / float(self.capacity_warmup_steps))
        # Start dense (route everything) and anneal down to the target capacity.
        return 1.0 + (self.capacity - 1.0) * frac

    def _gate_gain(self, router_weights: torch.Tensor) -> torch.Tensor:
        """Squash the raw logit into the multiplicative gain used by MoDGate.

        "raw" leaves the logit unbounded and sign-flipping, which is the single
        most-reproduced MoD failure in the wild (fp16 inf; NaN through RMSNorm)
        and lets the BCE — which wants a calibrated logit — fight the LM loss,
        which wants a useful gain. Bounding the gain makes the two objectives
        agree: sigmoid(r)->1 is simultaneously a full-strength gain and a large
        positive logit.
        """
        if self.gate_fn == "sigmoid":
            return torch.sigmoid(router_weights)
        if self.gate_fn == "tanh":
            return self.tanh_alpha * torch.tanh(router_weights)
        return router_weights


class MoDGate(nn.Module):
    """Gating mechanism for MoD that applies router weights to outputs.

    The gate multiplies the block output by the router weight, allowing
    gradients to flow through the routing decision.

    Args:
        hidden_size: Dimension of hidden representations.
        symmetric_reweight: p-MoD's STRing. Off by default, matching the
            original paper (§3.4: router weights are included "only ... along
            the computational path for those tokens that do not bypass"), which
            leaves the router with *zero* gradient from the tokens it skips.
            When on, skipped tokens become ``gain * x + x`` so the router is
            supervised on them too. Intended to pair with gate_fn="tanh": the
            gain is then zero-centered, so a skipped token is perturbed by
            ``1 +/- tanh_alpha`` rather than rescaled by an unbounded factor.
    """

    def __init__(self, hidden_size: int, symmetric_reweight: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.symmetric_reweight = symmetric_reweight

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        router_weights: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply gating to combine block output with residual.

        For selected tokens: output = router_weight * block_output + hidden_states
        For skipped tokens: output = hidden_states

        Args:
            hidden_states: Original input [batch, seq_len, hidden_size].
            block_output: Output from attention+MLP [batch, seq_len, hidden_size].
            router_weights: Router weights [batch, seq_len].
            selected_mask: Selection mask [batch, seq_len].

        Returns:
            Gated output [batch, seq_len, hidden_size].
        """
        # Expand router weights for broadcasting
        router_weights = router_weights.unsqueeze(-1)  # [batch, seq_len, 1]
        selected_mask = selected_mask.unsqueeze(-1)  # [batch, seq_len, 1]

        # Apply gating: selected tokens get weighted block output
        # router_weight * block_output + residual for selected
        # just residual for skipped (or gain * residual + residual under STRing)
        skipped = (
            router_weights * hidden_states + hidden_states
            if self.symmetric_reweight
            else hidden_states
        )
        gated_output = torch.where(
            selected_mask,
            router_weights * block_output + hidden_states,
            skipped,
        )

        return gated_output


def is_mod_layer(layer_idx: int, mod_every_n_layers: int) -> bool:
    """Determine if a layer should use MoD routing.

    Args:
        layer_idx: Index of the layer (0-based).
        mod_every_n_layers: Apply MoD every N layers. 0 disables.

    Returns:
        True if this layer should use MoD routing.
    """
    if mod_every_n_layers <= 0:
        return False
    # Apply MoD to layers where (layer_idx % mod_every_n_layers) != 0
    # This means layer 0 is always full, layer 1 is MoD (if every_n=2), etc.
    # Actually per paper: "routing every other block" suggests 1, 3, 5, etc.
    return (layer_idx % mod_every_n_layers) == 1


def compute_mod_block_plan(
    num_layers: int,
    block_size: int,
    io_blocks_dense: bool = True,
) -> Tuple[list, list]:
    """Partition layers into fixed-size blocks and pick which are MoD-routed.

    Block-atomic MoD (the stacking-plan design) treats a contiguous run of
    ``block_size`` layers as one routing unit: a single router decision at the
    block head selects tokens, and only those tokens traverse the *whole* block
    (the rest bypass all ``block_size`` layers via a residual gate at the block
    boundary). This mirrors the "4-layer block is the atomic unit of duplication"
    rule in ``STACKING_IMPLEMENTATION_PLAN.md``.

    Args:
        num_layers: Total decoder layers.
        block_size: Layers per block (>=1). The final block may be partial when
            ``num_layers`` is not a multiple of ``block_size``.
        io_blocks_dense: If True, the first and last block stay dense (no MoD),
            matching the design's "I/O blocks remain dense" rule. With <=2 blocks
            this leaves *no* MoD blocks — the caller should surface that.

    Returns:
        (blocks, mod_blocks) where ``blocks`` is the list of per-block layer-index
        lists (the full partition) and ``mod_blocks`` is the subset that are
        MoD-routed. Each entry of ``mod_blocks`` is a list of layer indices; its
        first element is the block head (owns the router), its last is the tail
        (where the gate is applied).
    """
    bs = max(1, int(block_size))
    blocks: list = []
    start = 0
    while start < num_layers:
        end = min(start + bs, num_layers)
        blocks.append(list(range(start, end)))
        start = end

    mod_blocks: list = []
    last = len(blocks) - 1
    for bi, members in enumerate(blocks):
        if io_blocks_dense and (bi == 0 or bi == last):
            continue
        mod_blocks.append(members)
    return blocks, mod_blocks


# Distinct gathered widths seen this run, and how often each occurred.
#
# WHY THIS IS TRACKED: torch.compile runs with ``dynamic=False``, so every new
# width forces a recompile of the decoder-layer frame. Past
# ``torch._dynamo.config.cache_size_limit`` Dynamo gives up and runs that frame
# EAGER for the rest of the job -- and ``apply_compile`` sets
# ``suppress_errors=True``, so it does so with no exception and no log line. The
# cell just quietly loses the ~1.5x compile speedup. In a sweep that exists to
# compare routing mechanisms, that would read as "the causal-router arm is
# slower" -- an architecture conclusion drawn from a compiler artifact.
#
# `k` is already a host-side int here, so counting costs nothing extra.
_GATHER_WIDTHS: "collections.Counter[int]" = collections.Counter()

# One-time (per process) verification that a router-certified uniform count
# matches the mask before the sync-free gather path is trusted (see
# mod_gather_index). Non-empty once verified.
_EXPECTED_COUNT_VERIFIED: set = set()


def record_gather_width(width: int) -> None:
    _GATHER_WIDTHS[int(width)] += 1


def gather_width_report() -> Dict[str, Any]:
    """Distinct gathered widths seen so far, for the training-loop log.

    ``distinct`` is the number that matters: compare it against
    ``torch._dynamo.config.cache_size_limit``. Fixed-capacity top-k never reaches
    this code at all (its rows are uniform), so an empty report is expected and
    correct for every arm except the threshold-router one.
    """
    if not _GATHER_WIDTHS:
        return {}
    widths = sorted(_GATHER_WIDTHS)
    return {
        "distinct": len(widths),
        "min": widths[0],
        "max": widths[-1],
        "counts": {int(w): int(_GATHER_WIDTHS[w]) for w in widths},
    }


def gather_width_payload() -> Dict[str, Any]:
    """Gathered-MoD width churn as log keys, or ``{}`` when there is none.

    Only a threshold router reaches the ragged path (top-k rows are uniform), so
    the keys are ABSENT for every other arm rather than logged as zeros.
    """
    report = gather_width_report()
    if not report:
        return {}
    return {
        "mod/gather_widths_distinct": report["distinct"],
        "mod/gather_width_min": report["min"],
        "mod/gather_width_max": report["max"],
    }


_CHURN_WARNED = [False]


def gather_width_warning(cache_size_limit: int) -> Optional[str]:
    """Message to print ONCE when distinct widths approach Dynamo's budget.

    Returns ``None`` when there is nothing to say, or after it has already been
    said once. Lives here rather than in the trainer so it is importable without
    the ``torchtitan_train`` package (whose name collides with
    ``scripts/torchtitan_train.py`` depending on sys.path order).
    """
    report = gather_width_report()
    if not report or _CHURN_WARNED[0]:
        return None
    distinct = report["distinct"]
    limit = int(cache_size_limit)
    if distinct < max(1, limit - 4):
        return None
    _CHURN_WARNED[0] = True
    return (
        f"WARNING: {distinct} distinct gathered MoD widths seen, and "
        f"torch._dynamo.config.cache_size_limit={limit}. Each width forces a "
        f"recompile; past the limit Dynamo silently falls back to eager for the "
        f"rest of the run. Raise the limit or widen mod_gather_bucket."
    )


def reset_gather_widths() -> None:
    _GATHER_WIDTHS.clear()
    _CHURN_WARNED[0] = False


def mod_gather_index(
    selected_mask: torch.Tensor,
    bucket: int = 128,
    expected_count: Optional[int] = None,
    fixed_width: Optional[int] = None,
) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Sorted per-row indices of the selected tokens, for gather/scatter MoD.

    Block-atomic MoD with real FLOP savings runs the block over ONLY the selected
    tokens (a compact ``[B, k, H]`` sequence). A fixed-capacity router (``topk`` /
    fixed-capacity ``topk``) plus a fixed forced prefix gives every row the same count
    ``k``, so the gather is a dense ``[B, k]`` index with no padding. A
    ``threshold`` router meets capacity only in expectation, so rows disagree and
    the gather is padded to a bucketed width instead.

    Args:
        selected_mask: Boolean ``[B, T]`` selection mask.
        bucket: Round the padded width up to this multiple, so the compact shape
            takes a handful of values across a run rather than a new one per step.

    Returns:
        ``(idx, valid)`` where ``idx`` is ``[B, k]`` of selected positions sorted
        ascending per row (keeping the compact sequence in original causal order),
        and ``valid`` is ``[B, k]`` bool marking real slots — ``None`` when every
        slot is real, which is the uniform-count fast path. Returns ``None`` if
        nothing is selected (caller falls back to the gate path).

    Sync-free fast paths (either kwarg set skips the count transfer below —
    the ONLY remaining device->host sync on the MoD hot path):
        expected_count: the caller (the router, via ``_last_uniform_count``)
            certifies every row selected exactly this many tokens — true by
            construction for `topk` selection. Verified once per process
            against the mask, then trusted.
        fixed_width: gather at this STATIC width regardless of per-row counts
            (`mod_fixed_gather`, the kmod arm): rows above it are clipped to
            their EARLIEST-selected tokens — a causal truncation, inclusion at
            position p depends only on selections at <= p — and rows below it
            pad with invalid slots exactly like the ragged path. One shape for
            the whole run, no syncs, no recompiles.
    """
    if fixed_width is not None:
        n = int(selected_mask.shape[1])
        k = min(n, max(1, int(fixed_width)))
        pos = torch.arange(n, device=selected_mask.device).expand_as(selected_mask)
        order = torch.argsort(torch.where(selected_mask, pos, pos + n), dim=1)[:, :k]
        valid = torch.gather(selected_mask, 1, order)
        return order, valid

    if expected_count is not None:
        k = int(expected_count)
        if k <= 0:
            return None
        if not torch.compiler.is_compiling() and not _EXPECTED_COUNT_VERIFIED:
            counts = selected_mask.sum(dim=1)
            lo, hi = torch.stack((counts.min(), counts.max())).tolist()
            if not (lo == hi == k):
                raise AssertionError(
                    f"mod_gather_index: router certified a uniform count of {k} but the "
                    f"mask has counts in [{lo}, {hi}] — the sync-free gather would "
                    "silently mis-select; fix the router hint."
                )
            _EXPECTED_COUNT_VERIFIED.add(True)
        _, idx = torch.topk(selected_mask.float(), k, dim=1)
        idx, _ = torch.sort(idx, dim=1)
        return idx, None

    counts = selected_mask.sum(dim=1)
    # ONE sync, not two. Reading `counts[0].item()` and then `(counts == k).all()`
    # stalls the pipeline twice to answer a single question ("are the rows
    # uniform, and how wide?"); min/max answer both from one transfer.
    lo, hi = torch.stack((counts.min(), counts.max())).tolist()
    k = int(hi)
    if k > 0 and lo == hi:
        # Every selected entry is 1.0, every other 0.0, so top-k picks exactly the
        # selected set; sort ascending to restore original-position order.
        _, idx = torch.topk(selected_mask.float(), k, dim=1)
        idx, _ = torch.sort(idx, dim=1)
        return idx, None

    # Ragged rows. A threshold router (router_mode="threshold") meets capacity only
    # in expectation, so rows disagree on k -- and returning None here would drop
    # the caller onto the DENSE gate path, which runs every layer and masks, i.e.
    # throws away the entire point of MoD. Pad to the row max instead, bucketed so
    # the gathered shape churns over a handful of values rather than a new one each
    # step. Padding slots are marked invalid and the caller must both exclude them
    # as attention KEYS and zero their scatter contribution.
    kmax = k  # already read above; re-reading counts.max() would sync a second time
    if kmax <= 0:
        return None
    bucket = max(1, int(bucket))
    k = min(int(selected_mask.shape[1]), ((kmax + bucket - 1) // bucket) * bucket)
    record_gather_width(k)
    # Stable compaction: selected keep their position, unselected are pushed past
    # the end, so an ascending sort puts the selected first in original order.
    n = selected_mask.shape[1]
    pos = torch.arange(n, device=selected_mask.device).expand_as(selected_mask)
    order = torch.argsort(torch.where(selected_mask, pos, pos + n), dim=1)[:, :k]
    valid = torch.gather(selected_mask, 1, order)
    return order, valid


def build_mod_compact_masks(
    positions: torch.Tensor,
    num_globals: int,
    window: int,
    dtype: torch.dtype,
    doc_ids: Optional[torch.Tensor] = None,
    valid: Optional[torch.Tensor] = None,
    want_windowed: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Additive attention masks for a gathered (compact) MoD block sequence.

    Given the *original* positions of the gathered tokens, build the visibility
    masks in original-position space so the sliding window is preserved even
    though the compact sequence is non-contiguous. Two masks are returned because
    a ``swa_globals`` block mixes windowed (SWA) and full-causal (MLA) layers.

    ``doc_ids`` is REQUIRED for correctness whenever documents are packed. Without
    it this mask tests only ``pj <= pi`` on positions that
    ``build_intra_doc_position_ids`` has already RESET per document -- so a token
    at document-2 position 5 sees a document-1 token at position 3 and the leak
    looks perfectly causal to the mask. Measured before this argument existed:
    perturbing document A moved document B's outputs by ~6e-2 in every MoD arm,
    versus exactly 0.0 with MoD off. The dense path never had this bug because it
    consumes the model's document-aware 4-D mask; the compact path overrides that
    mask with this one, so the document rule has to be re-applied here.

    Args:
        positions: ``[B, k]`` original position of each gathered token.
        num_globals: leading positions that are persistent globals (always
            attendable). A token at original position ``< num_globals`` is a
            global.
        window: SWA band width (content token i sees j iff ``i - j < window``).
        dtype: float dtype for the additive mask.
        doc_ids: optional ``[B, k]`` gather of the per-token document ids
            (``PREFIX_ID`` on prefix rows). When given, content may only attend
            within its own document; ``PREFIX_ID`` rows stay visible across all
            documents, matching ``build_doc_mask``.

    Returns:
        ``(causal_mask, windowed_mask)`` each ``[B, 1, k, k]`` additive (0 keep,
        large-negative block). ``causal_mask`` is plain causal (for MLA layers);
        ``windowed_mask`` adds the globals+sliding-window constraint (for SWA).
        ``windowed_mask`` is ``None`` when ``want_windowed=False`` — the caller
        covers the SWA members with a FlexAttention BlockMask, which takes
        unconditional dispatch priority over the additive mask, so materialising
        the second ``[B, 1, k, k]`` tensor would be dead weight.
    """
    pi = positions[:, :, None]  # [B, k, 1] query positions
    pj = positions[:, None, :]  # [B, 1, k] key positions
    causal = pj <= pi

    # Identifying a global by POSITION (``p < num_globals``) is only valid when
    # positions are absolute. Under intra-doc masking they are not:
    # build_intra_doc_position_ids restarts every document at 0, so the first
    # ``num_globals`` tokens of EVERY document look like globals. Use the document
    # id (PREFIX_ID marks the prefix) whenever it is available.
    if doc_ids is not None:
        dq = doc_ids[:, :, None]
        dk = doc_ids[:, None, :]
        is_global_q = dq == PREFIX_ID
        is_global_k = dk == PREFIX_ID
        same_doc = dq == dk
    else:
        is_global_q = pi < num_globals
        is_global_k = pj < num_globals
        same_doc = torch.ones_like(causal)

    # A global QUERY sees only globals -- matching GlobalTokens.build_attention_mask,
    # where `kv_idx <= q_idx < N` confines a global row to other globals in ABSOLUTE
    # index space. The compact path compares per-document positions instead, so
    # without this rule a global at position 5 attends to the first 6 tokens of
    # every packed document, absorbs their content, and hands it to later documents
    # through the (legitimately cross-document) global read. That was the leak: it
    # is not the window or the causal test, it is the prefix acting as a bridge.
    global_rule = is_global_q & is_global_k & causal
    windowed = None
    if want_windowed:
        # Band convention is ``< window``, matching GlobalTokens.build_attention_mask
        # (``content_distance < window``) and flex_swa's mask_mods. This builder used
        # ``<= window`` and so ran one position wider than the dense path for the same
        # layer; the two now agree.
        in_band = causal & same_doc & ((pi - pj) < window)
        content_windowed = (~is_global_q) & (is_global_k | in_band)
        windowed = global_rule | content_windowed
    # MLA member: same structure, no band.
    content_causal = (~is_global_q) & (is_global_k | (causal & same_doc))
    causal = global_rule | content_causal

    if valid is not None:
        # Padding slots of a ragged (threshold-router) gather must be invisible as
        # KEYS or they inject an unselected token's content into every real row.
        # Their own query rows are left attendable so no row is fully masked --
        # softmax over an all-masked row is NaN -- and their output is discarded by
        # the zeroed gate in the caller.
        keep_k = valid[:, None, :]
        self_pair = torch.eye(valid.shape[1], dtype=torch.bool, device=valid.device)
        causal = (causal & keep_k) | self_pair
        if windowed is not None:
            windowed = (windowed & keep_k) | self_pair

    neg = torch.finfo(dtype).min
    zero = torch.zeros((), dtype=dtype, device=positions.device)
    full = torch.full((), neg, dtype=dtype, device=positions.device)
    causal_mask = torch.where(causal, zero, full)
    windowed_mask = torch.where(windowed, zero, full) if windowed is not None else None
    return (
        causal_mask[:, None, :, :],
        windowed_mask[:, None, :, :] if windowed_mask is not None else None,
    )


def compute_mod_aux_loss(
    mod_outputs: Tuple[MoDOutput, ...],
) -> Optional[torch.Tensor]:
    """Compute total MoD auxiliary loss from all layers.

    Args:
        mod_outputs: Tuple of MoDOutput from each MoD layer.

    Returns:
        Total auxiliary loss, or None if no losses.
    """
    total_loss = None
    for output in mod_outputs:
        if output is not None and output.aux_loss is not None:
            if total_loss is None:
                total_loss = output.aux_loss
            else:
                total_loss = total_loss + output.aux_loss
    return total_loss


class MoDProcessor:
    """Utility class for processing tokens through MoD layers efficiently.

    This handles the token selection and recombination logic for MoD,
    allowing efficient processing where only selected tokens go through
    the expensive attention and MLP computations.
    """

    @staticmethod
    def select_tokens(
        hidden_states: torch.Tensor,
        selected_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Select tokens for processing based on mask.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            selected_mask: Boolean mask [batch, seq_len].
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.

        Returns:
            Tuple of (selected_hidden, selected_attn_mask, selected_pos_ids, indices).
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Get indices of selected tokens per batch
        # For simplicity, we'll use a padded approach
        num_selected = selected_mask.sum(dim=1).max().item()

        # Gather selected tokens
        # Create index tensor for gathering
        selected_indices = []
        for b in range(batch_size):
            batch_indices = selected_mask[b].nonzero(as_tuple=True)[0]
            # Pad to max selected
            if len(batch_indices) < num_selected:
                pad_size = num_selected - len(batch_indices)
                batch_indices = F.pad(batch_indices, (0, pad_size), value=0)
            selected_indices.append(batch_indices[:num_selected])

        indices = torch.stack(selected_indices)  # [batch, num_selected]

        # Gather hidden states
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, hidden_size)
        selected_hidden = torch.gather(hidden_states, 1, indices_expanded)

        # Handle attention mask if provided
        selected_attn_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                selected_attn_mask = torch.gather(attention_mask, 1, indices)
            # For 4D attention mask, more complex handling needed

        # Handle position IDs if provided
        selected_pos_ids = None
        if position_ids is not None:
            selected_pos_ids = torch.gather(position_ids, 1, indices)

        return selected_hidden, selected_attn_mask, selected_pos_ids, indices

    @staticmethod
    def scatter_tokens(
        full_hidden: torch.Tensor,
        processed_hidden: torch.Tensor,
        indices: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter processed tokens back to full sequence.

        Args:
            full_hidden: Original hidden states [batch, seq_len, hidden_size].
            processed_hidden: Processed tokens [batch, num_selected, hidden_size].
            indices: Indices of selected tokens [batch, num_selected].
            selected_mask: Selection mask [batch, seq_len].

        Returns:
            Full hidden states with processed tokens scattered back.
        """
        batch_size, seq_len, hidden_size = full_hidden.shape

        # Create output tensor (copy of input for non-selected positions)
        output = full_hidden.clone()

        # Scatter processed tokens back
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, hidden_size)
        output.scatter_(1, indices_expanded, processed_hidden)

        return output
