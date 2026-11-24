"""Per-layer compute accounting for dynamic-compute architectures.

Single source of truth for "how much of each decoder layer does an average
token actually execute". Consumed by:

- ``titan/wrapper.py::num_active_params`` — per-token-active params for MFU,
- ``titan/throughput.py::FlopsCounter`` — the attention-quadratic share,
- ``scripts/scaling_laws.py`` — the IsoFLOP token horizon ``D = C / (6 * N_compute)``.

Before this module, MoD architectures were billed as fully dense (the only
discount was MoE idle experts), so the IsoFLOP sweep over-charged every
dynamic-compute arm and under-fed it tokens. The accounting here is
*algorithmic*: it reflects what the architecture fundamentally computes per
token, not what the current kernel implementation realizes. Wall-clock MFU
(``local-slurm/scaling50m/submit_arch_mfu.sh``) is the ground truth for
realized efficiency; the gap between the two is the implementation tax.

Covered adjustments:

- **MoD routing-fraction discount** — per-layer and block-atomic MoD: routed
  layers only process ``capacity`` of the tokens. With ``mod_mlp_only`` the
  discount applies to the FFN share only (attention stays dense for every
  token).
- **Recurrent-depth iteration multiplier** — ``use_recurrent_depth`` loops the
  core block; core layers are billed ``E[iterations]`` times, and so are the
  loop-boundary mixers (stable injection / loop hyper-connections). (Also the hook
  chunk 2's GWT trunk-iteration accounting reuses.)
- **Workspace trunk (GWT)** — the trunk runs on ``slots`` workspace states per
  chunk of ``trunk_chunk_size`` tokens, iterated ``E[T]`` times: per-token cost
  is ``block_params * trunk_layers * E[T] * slots / chunk_size`` plus the
  per-token workspace read. O(1) in sequence length by construction.
- **Pathways (MoP)** — lane blocks are billed at the *expected* lane cost. The
  cost-aware aux loss drives ``sum_i f_i * depth_i`` to ``pathway_budget``, so
  the budget is the planning expectation; realized fractions (diagnostics) can
  refine it post-hoc via ``FlopsCounter.set_lane_cost_weights``.

Not covered (documented limitation): MoE + MoD on the same layer. The MoE
idle-expert discount is applied separately in ``num_active_params``; configs
combining both on one layer would slightly misattribute the FFN share. No
sweep arm does this and ``_validate_mod_moe_config`` polices the overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .modules.mixture_of_depths import compute_mod_block_plan, is_mod_layer


@dataclass
class LayerComputeWeights:
    """Expected execution fraction of one decoder layer, by component.

    ``attn`` scales the attention *projection* (param-proportional) cost,
    ``attn_sq`` the attention *quadratic* (seq_len^2) cost — for MoD the
    selected-token subset attends only within itself, so the quadratic term
    scales with capacity^2 while projections scale with capacity.
    ``mlp`` scales the FFN cost. Dense layers are (1, 1, 1); a value > 1
    (recurrent depth) means the layer runs more than once per forward.
    """

    attn: float = 1.0
    attn_sq: float = 1.0
    mlp: float = 1.0
    # Everything else in the layer (norms, routers, gates). MoD routers run on
    # every token so routing capacity never discounts this share, but recurrent
    # depth does re-execute it per iteration.
    other: float = 1.0

    @property
    def is_dense(self) -> bool:
        return self.attn == 1.0 and self.attn_sq == 1.0 and self.mlp == 1.0 and self.other == 1.0


def mod_routed_capacities(config) -> Dict[int, float]:
    """Map layer_idx -> routing capacity for every MoD-routed layer.

    Mirrors the routing plan ``UMichSLMModel.__init__`` actually builds:
      - block-atomic + ``stage_block_spec``: inserted blocks at
        ``mod_capacity_inserted``;
      - block-atomic flat: ``compute_mod_block_plan`` at ``mod_capacity``;
      - legacy per-layer: ``is_mod_layer`` minus always-used layers.
    Layers absent from the map are dense.
    """
    if not getattr(config, "use_mod", False):
        return {}
    num_layers = int(config.num_hidden_layers)
    out: Dict[int, float] = {}

    if getattr(config, "mod_atomic_block", False):
        block_size = int(getattr(config, "mod_block_size", 0) or 0)
        if block_size <= 0:
            hybrid = getattr(config, "hybrid_attention_ratio", None)
            every_n = int(getattr(config, "mod_every_n_layers", 0) or 0)
            block_size = len(hybrid) if hybrid else (every_n if every_n > 0 else 4)

        stage_spec = getattr(config, "stage_block_spec", None)
        if stage_spec:
            cap = float(getattr(config, "mod_capacity_inserted", 0.5))
            for b, entry in enumerate(stage_spec):
                if not bool(entry.get("is_inserted", False)):
                    continue
                for m in range(b * block_size, min((b + 1) * block_size, num_layers)):
                    out[m] = cap
        else:
            cap = float(getattr(config, "mod_capacity", 0.125))
            _, plan = compute_mod_block_plan(
                num_layers,
                block_size,
                io_blocks_dense=getattr(config, "mod_io_blocks_dense", True),
            )
            for members in plan:
                for m in members:
                    out[m] = cap
        return out

    # Legacy per-layer MoD.
    cap = float(getattr(config, "mod_capacity", 0.125))
    every_n = int(getattr(config, "mod_every_n_layers", 2) or 0)
    always_used = (
        config.get_always_used_layers() if hasattr(config, "get_always_used_layers") else set()
    )
    for idx in range(num_layers):
        if is_mod_layer(idx, every_n) and idx not in always_used:
            out[idx] = cap
    return out


def expected_recurrent_iters(config) -> float:
    """Mean recursion count of the recurrent core block (1.0 when disabled).

    Training samples the count per micro-batch; accounting uses the sampling
    distribution's expectation so token budgets reflect the average cost.
    """
    if not getattr(config, "use_recurrent_depth", False):
        return 1.0
    lo = int(getattr(config, "recurrent_min_iters", 1))
    hi = int(getattr(config, "recurrent_max_iters", 1))
    # "uniform" is the only sampling distribution implemented; its mean is the
    # midpoint. New distributions must extend this accounting too.
    return (lo + hi) / 2.0


def recurrent_block_size(config) -> int:
    bs = int(getattr(config, "recurrent_block_size", 0) or 0)
    if bs <= 0:
        hybrid = getattr(config, "hybrid_attention_ratio", None)
        bs = len(hybrid) if hybrid else 4
    return bs


def recurrent_core_layers(config) -> List[int]:
    """Layer indices of the looped core: everything between the first block
    (prelude) and the last block (coda). Empty when recurrent depth is off or
    the stack is too shallow to have a core.
    """
    if not getattr(config, "use_recurrent_depth", False):
        return []
    num_layers = int(config.num_hidden_layers)
    bs = recurrent_block_size(config)
    if num_layers < 2 * bs + 1:  # must match the config validator
        return []
    return list(range(bs, num_layers - bs))


def mat_ffn_routed_layers(config) -> Dict[int, float]:
    """{layer_idx: expected active MLP fraction} for ``mat_ffn`` routed layers.

    ADDED 2026-08-21. ``num_active_params`` had NO mat_ffn branch, so a mat_ffn
    model billed its routed layers at FULL nested width -- the whole point of
    the arm is that a token uses only a prefix of that width. Every mat_ffn MFU
    number computed before this date is against an inflated N, and any attempt
    to size the arm "at matched active params" using the plain parameter count
    was matching the wrong quantity (at d_ff=3584, quarters: 883.5M total but
    825.7M active).

    Mirrors the layer selection in ``modules/transformer_block.py`` exactly --
    including the depth CLAMP on dyn_start/dyn_end, which a shallow model needs
    or the dynamic band is empty and there are no routed layers at all. Only
    layers inside the band AND on the routing interval are routed; the ones
    between run a plain SwiGLU at the budget width (already the right count,
    since every token uses all of it) and the fixed prefix/suffix run full
    width, so neither is discounted here.
    """
    if not getattr(config, "use_mat_ffn", False):
        return {}
    L = int(config.num_hidden_layers)
    dff = int(config.intermediate_size)
    h = int(config.hidden_size)
    widths = list(getattr(config, "mat_ffn_widths", None) or [dff])
    budget = float(getattr(config, "mat_ffn_budget", 0.0) or 0.0)
    if budget <= 0:
        budget = sum(widths) / len(widths)
    budget = min(budget, float(dff))

    cap = max(0, (L - 2) // 3)
    lo = min(int(getattr(config, "mat_ffn_dyn_start", 0) or 0), cap)
    hi = L - min(int(getattr(config, "mat_ffn_dyn_end", 0) or 0), cap)
    interval = max(1, int(getattr(config, "mat_ffn_route_interval", 1) or 1))

    bias = bool(getattr(config, "mlp_bias", False))
    # The ROUTER runs on every token whatever width it picks, so it must not be
    # discounted. Fold it into both numerator and denominator rather than
    # scaling the whole mlp module by the width fraction.
    router = h * len(widths)

    def mlp_params(w):
        n = 3 * h * w + router
        if bias:
            n += 2 * w + h
        return float(n)

    frac = mlp_params(budget) / mlp_params(dff)
    return {i: frac for i in range(lo, hi) if (i - lo) % interval == 0}


def layer_compute_weights(config) -> List[LayerComputeWeights]:
    """Per-decoder-layer expected execution weights for the whole stack."""
    num_layers = int(config.num_hidden_layers)
    weights = [LayerComputeWeights() for _ in range(num_layers)]

    mlp_only = bool(getattr(config, "mod_mlp_only", False))
    for idx, cap in mod_routed_capacities(config).items():
        w = weights[idx]
        w.mlp = cap
        if not mlp_only:
            w.attn = cap
            w.attn_sq = cap * cap

    for idx, frac in mat_ffn_routed_layers(config).items():
        # mat_ffn touches the FFN only; attention is untouched by design, which
        # is the arm's whole cost argument.
        weights[idx].mlp *= frac

    iters = expected_recurrent_iters(config)
    if iters != 1.0:
        for idx in recurrent_core_layers(config):
            w = weights[idx]
            w.attn *= iters
            w.attn_sq *= iters
            w.mlp *= iters
            w.other *= iters
    return weights


# --------------------------------------------------------------- torch side
def split_layer_params(layer) -> Tuple[int, int, int]:
    """(attention, mlp, other) parameter counts of one decoder layer module.

    "other" is norms, routers, gates, MoDA projections — everything that runs
    for every token regardless of routing.
    """
    attn_mod = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    mlp_mod = getattr(layer, "mlp", None) or getattr(layer, "moe", None)
    attn = sum(p.numel() for p in attn_mod.parameters()) if attn_mod is not None else 0
    mlp = sum(p.numel() for p in mlp_mod.parameters()) if mlp_mod is not None else 0
    total = sum(p.numel() for p in layer.parameters())
    return attn, mlp, total - attn - mlp


def expected_trunk_iters(config) -> float:
    """Mean trunk recursion count for the GWT workspace update (1.0 if off)."""
    if not getattr(config, "use_workspace_trunk", False):
        return 1.0
    lo = int(getattr(config, "trunk_iters_min", 1))
    hi = int(getattr(config, "trunk_iters_max", 1))
    return (lo + hi) / 2.0


# Analytic (config-only) estimators — used by FlopsCounter, which never sees
# module objects. Projections only; norms are negligible at these widths.
def trunk_read_params(config) -> int:
    """Workspace read cross-attention: q/k/v/o at hidden x hidden."""
    h = int(config.hidden_size)
    return 4 * h * h


def trunk_block_params(config) -> int:
    """One trunk block: write cross-attn (4h^2) + slot self-attn (4h^2) +
    SwiGLU MLP (3 * h * intermediate)."""
    h = int(config.hidden_size)
    f = int(config.intermediate_size)
    return 8 * h * h + 3 * h * f


def trunk_compute_params(config) -> float:
    """Per-token compute-param contribution of the whole GWT mechanism.

    The read runs once per token; the trunk runs ``trunk_layers`` blocks
    ``E[T]`` times on ``slots`` states per ``trunk_chunk_size`` tokens.
    """
    if not getattr(config, "use_workspace_trunk", False):
        return 0.0
    from .modules.workspace_trunk import trunk_workspace_slots

    slots = trunk_workspace_slots(config)
    chunk = int(getattr(config, "trunk_chunk_size", 512))
    blocks = int(getattr(config, "trunk_layers", 2))
    return trunk_read_params(config) + (
        trunk_block_params(config) * blocks * expected_trunk_iters(config) * slots / chunk
    )


def pathway_lane_block_params(config) -> int:
    """One MoP lane block: cross-attn (q/o at h^2; k/v at h * kv_dim unless
    ``pathway_share_kv`` moved them to the once-per-token shared head) +
    SwiGLU MLP (3 * h * intermediate)."""
    h = int(config.hidden_size)
    f = int(config.intermediate_size)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", None) or heads)
    kv_dim = (h // heads) * kv_heads
    kv = 0 if getattr(config, "pathway_share_kv", False) else 2 * h * kv_dim
    return 2 * h * h + kv + 3 * h * f


def pathway_expected_lane_blocks(config, lane_fractions=None) -> float:
    """Expected executed lane blocks per token.

    Defaults to ``pathway_budget`` (the aux-loss target — the planning
    expectation). Pass measured ``lane_fractions`` (one per lane, summing to
    ~1) to bill realized routing instead.
    """
    if not getattr(config, "use_pathways", False):
        return 0.0
    depths = list(getattr(config, "pathway_depths", None) or [0, 2, 4])
    stages = int(getattr(config, "pathway_stages", 1) or 1)
    if lane_fractions is not None:
        if len(lane_fractions) != len(depths):
            raise ValueError(
                f"lane_fractions has {len(lane_fractions)} entries for {len(depths)} lanes."
            )
        return stages * float(sum(f * d for f, d in zip(lane_fractions, depths)))
    budget = float(getattr(config, "pathway_budget", 0.0) or 0.0)
    per_stage = budget if budget > 0 else sum(depths) / len(depths)
    return stages * per_stage


# --------------------------------------------------------------- torch side
def workspace_trunk_param_delta(trunk, config) -> float:
    """Signed unique-N -> compute-N adjustment for a WorkspaceTrunk module.

    The read is per-token (weight 1.0, no adjustment); trunk blocks bill per
    execution amortized over the chunk; the learned initial state (a lookup,
    no matmul) bills zero compute.
    """
    if trunk is None:
        return 0.0
    unique = float(sum(p.numel() for p in trunk.parameters()))
    read = float(sum(p.numel() for p in trunk.read.parameters()))
    # Sum over list slots so weight-shared blocks bill once per execution.
    block_exec = float(sum(sum(p.numel() for p in block.parameters()) for block in trunk.blocks))
    per_token = read + (
        block_exec * expected_trunk_iters(config) * trunk.num_slots / trunk.chunk_size
    )
    return per_token - unique


def pathways_param_delta(pathways, config) -> float:
    """Signed unique-N -> compute-N adjustment for a Pathways module (or a
    list of stacked stage modules).

    Router / export norm / shared-K/V head / merge norms+scales run for
    every token (weight 1.0); lane block slots bill at the expected lane cost
    (budget by default). Disjoint lanes subtract more unique params than they
    bill (delta < 0); the weight-tied recursive variant bills passes of one
    block (delta > 0); nested sits between.
    """
    if pathways is None:
        return 0.0
    stages = pathways if isinstance(pathways, (list, tuple)) else [pathways]
    per_block = 0.0
    unique_total = 0.0
    seen: set = set()
    # One entry per executable SLOT per stage; weight-tied blocks (recursive
    # variant) and cross-stage shared lanes (pathway_stage_share_lanes) yield
    # repeated objects — billed per pass, deduped for unique-N.
    for stage in stages:
        for block in stage.iter_lane_blocks():
            n = float(sum(p.numel() for p in block.parameters()))
            per_block = per_block or n
            if id(block) not in seen:
                seen.add(id(block))
                unique_total += n
    executed = pathway_expected_lane_blocks(config)  # already x pathway_stages
    return executed * per_block - unique_total


def model_compute_param_delta(model, config) -> float:
    """Signed adjustment for the whole bare model (``UMichSLMModel``):
    decoder-layer routing/recursion plus any model-level dynamic-compute
    modules (GWT trunk, MoP lanes). Single entry point for
    ``num_active_params`` / scaling_laws / diagnostics.
    """
    delta = layer_compute_param_delta(model.layers, config)
    delta += recurrent_mixer_param_delta(getattr(model, "recurrent_mixer", None), config)
    delta += workspace_trunk_param_delta(getattr(model, "workspace_trunk", None), config)
    stage_list = getattr(model, "pathway_stages", None)
    if stage_list is not None:
        delta += pathways_param_delta(list(stage_list), config)
    else:
        delta += pathways_param_delta(getattr(model, "pathways", None), config)
    return delta


def recurrent_mixer_param_delta(mixer, config) -> float:
    """Loop-boundary mixers (modules/loop_mixers.py) run once per ITERATION:
    the shared injection weights execute E[T] times, and E[T] of the
    ``max_iters`` stored hyper-connection sets execute per forward. Their matmuls
    are the dense injection B (2 d^2 FLOPs/token/iter) and the gate GEMM
    (6 n^2 d); the remaining elementwise work is not billed, like norms.
    """
    if mixer is None:
        return 0.0
    return float(mixer.compute_param_delta(expected_recurrent_iters(config)))


def layer_compute_param_delta(layers, config) -> float:
    """Signed adjustment turning a deduped param count into a per-token compute
    count: subtract each *unique* layer's params once, add back each layer
    *slot* at its expected execution weight. Weight-shared slots (MobileLLM-LS)
    therefore bill per execution, and MoD-routed slots bill at capacity.

    Returns 0.0 for fully dense stacks, so existing archs are unaffected.
    """
    weights = layer_compute_weights(config)
    layers = list(layers)
    if len(layers) != len(weights):
        return 0.0
    shared = bool(getattr(config, "use_block_weight_sharing", False))
    if all(w.is_dense for w in weights) and not shared:
        return 0.0

    delta = 0.0
    seen: set = set()
    for layer in layers:
        if id(layer) in seen:
            continue
        seen.add(id(layer))
        a, m, o = split_layer_params(layer)
        delta -= a + m + o
    for layer, w in zip(layers, weights):
        a, m, o = split_layer_params(layer)
        delta += w.attn * a + w.mlp * m + w.other * o
    return delta
