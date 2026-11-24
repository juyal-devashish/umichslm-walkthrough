"""Grow a trained SMALLER model of our own into a larger same-family target.

The rest of this package shrinks a larger external donor. This module goes the
other way: an ``ar18m`` cell donating to ``ar10m``, ``ar25m`` to ``ar50m``. Same
tokenizer, same block types, so every target tensor has a donor tensor of the
same role whose shape is no larger in any dimension. See
``docs/parameter_warm_start.md`` §7b for the literature and the experiment.

Two operators, applied at step 0 on top of the target's OWN fresh initialisation
(so the "noise" is exactly the init a cold run of the same seed starts from):

``szp``  shrink-zero-perturb (Mallik et al., arXiv 2605.13405; Ash & Adams):
         ``W = W_init + shrink * pad(W_donor)``. Norm gains are copied, not
         shrunk; new norm channels keep their init of 1. Not function-preserving
         and not meant to be: the paper's no-noise ablation is 0.30 nats worse.
``fp``   function-preserving (LEMON-style). The grown model computes exactly the
         donor's function at step 0: new residual channels stay zero (every
         residual writer's new rows/cols are zero), residual RMSNorm gains are
         rescaled by sqrt(h_s/h_t) because the zero channels shrink the RMS, the
         MLA latent norm likewise by sqrt(r_s/r_t), new heads/FFN neurons read
         from init but write through zeroed output columns, and inserted layers
         are copies of a neighbour with zeroed ``o_proj``/``down_proj`` (identity
         blocks under pre-norm).

Depth is grown by INTERPOSITION, by layer type: donor layers are placed in order
into same-type target slots, spread evenly (an order-preserving alignment), and
every remaining slot duplicates its nearest same-type donor layer. A DeltaNet
slot is never filled from an MLA layer.

Width growth places each donor tensor in the leading corner of the target
tensor. That is correct only because every per-head layout here is head-major
(``view(B, S, H, d)``): new heads, FFN neurons and latent dims are appended at
the end of their axis. A GQA target whose kv-head count changes would remap
q-heads to groups and break that, so it is refused.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

OPERATORS = ("szp", "fp")

# Config fields a growth step is ALLOWED to change: the size knobs. Anything else
# that differs between donor and target (a window, rope_theta, a router mode) is
# a different architecture and must be named explicitly by the caller.
GROW_SIZE_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "griffin_num_heads",
    "mla_kv_lora_rank",
    "hybrid_attention_ratio",
)

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

# Function-preserving roles, by parameter-name suffix.
_RESIDUAL_WRITERS = ("self_attn.o_proj.weight", "mlp.down_proj.weight")
_RESIDUAL_NORMS = ("input_layernorm.weight", "post_attention_layernorm.weight")


def split_layers(sd: Dict[str, torch.Tensor]) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """``(per-layer dicts keyed by suffix, non-layer tensors)``."""
    layers: Dict[int, Dict[str, torch.Tensor]] = {}
    rest: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        m = _LAYER_RE.match(k)
        if m:
            layers.setdefault(int(m.group(1)), {})[m.group(2)] = v
        else:
            rest[k] = v
    n = len(layers)
    if sorted(layers) != list(range(n)):
        raise ValueError(f"non-contiguous layer indices: {sorted(layers)}")
    return [layers[i] for i in range(n)], rest


def layer_signature(layer: Dict[str, Any]) -> Tuple[str, ...]:
    """The layer's TYPE: the sorted set of its parameter names."""
    return tuple(sorted(layer))


def align_layers(src: Sequence[Any], dst: Sequence[Any]) -> List[Tuple[int, bool]]:
    """Map each target slot to ``(donor layer, is_primary)``.

    Donor layers are placed, in order, into target slots of the same type,
    minimising the squared distance from the proportional position
    ``i * (n-1)/(m-1)`` (so the first and last donor layers land near the first
    and last slots). That is an order-preserving embedding, so with the
    non-primary slots made identity the target computes the donor's function.
    Every other slot duplicates the nearest same-type primary, the preceding one
    on a tie: interposition, not stacking.
    """
    m, n = len(src), len(dst)
    if m == 0 or n < m:
        raise ValueError(f"cannot grow {m} layers into {n}")
    missing = set(dst) - set(src)
    if missing:
        raise ValueError(f"target layer types absent from the donor: {missing}")
    ideal = [i * (n - 1) / max(m - 1, 1) for i in range(m)]
    inf = float("inf")
    cost = [[inf] * n for _ in range(m)]
    back = [[-1] * n for _ in range(m)]
    for j in range(n):
        if dst[j] == src[0]:
            cost[0][j] = (j - ideal[0]) ** 2
    for i in range(1, m):
        best, arg = inf, -1
        for j in range(n):
            # best over j' < j for row i-1, maintained incrementally
            if j > 0 and cost[i - 1][j - 1] < best:
                best, arg = cost[i - 1][j - 1], j - 1
            if dst[j] == src[i] and best < inf:
                cost[i][j] = best + (j - ideal[i]) ** 2
                back[i][j] = arg
    j = min(range(n), key=lambda c: cost[m - 1][c])
    if cost[m - 1][j] == inf:
        raise ValueError("no order-preserving, type-matched placement exists")
    pos = [0] * m
    for i in range(m - 1, -1, -1):
        pos[i] = j
        j = back[i][j]
    primary = {p: i for i, p in enumerate(pos)}
    out: List[Tuple[int, bool]] = []
    for j in range(n):
        if j in primary:
            out.append((primary[j], True))
            continue
        cands = [p for p in pos if dst[p] == dst[j]]
        p = min(cands, key=lambda c: (abs(c - j), c > j))
        out.append((primary[p], False))
    return out


def _corner(shape: Sequence[int]) -> Tuple[slice, ...]:
    return tuple(slice(0, d) for d in shape)


def _check_fits(name: str, s: torch.Tensor, t: torch.Tensor) -> None:
    if s.dim() != t.dim() or any(a > b for a, b in zip(s.shape, t.shape)):
        raise ValueError(f"{name}: donor {tuple(s.shape)} does not fit target {tuple(t.shape)}")


def _grow_szp(name: str, s: torch.Tensor, t: torch.Tensor, shrink: float) -> torch.Tensor:
    out = t.clone()
    sl = _corner(s.shape)
    if s.dim() == 1 and "norm" in name:
        out[sl] = s
    else:
        out[sl] += shrink * s
    return out


def _rescaled_gain(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """RMSNorm gain over a dim widened with ZERO channels: those shrink the RMS
    by sqrt(d_s/d_t), so the surviving channels need gain * sqrt(d_s/d_t). The
    new channels see zero input; give them the mean rescaled gain."""
    c = math.sqrt(s.numel() / t.numel())
    out = torch.full_like(t, float(s.float().mean()) * c)
    out[: s.numel()] = s * c
    return out


def _grow_fp(name: str, s: torch.Tensor, t: torch.Tensor, primary: bool) -> torch.Tensor:
    sl = _corner(s.shape)
    if name.endswith(_RESIDUAL_WRITERS):
        out = torch.zeros_like(t)
        if primary:
            out[sl] = s
        return out
    if name == "model.embed_tokens.weight":
        out = torch.zeros_like(t)
        out[sl] = s
        return out
    if name.endswith(_RESIDUAL_NORMS) or name == "model.norm.weight":
        return _rescaled_gain(s, t)
    if name.endswith("self_attn.kv_layernorm.weight"):
        return _rescaled_gain(s, t)
    out = t.clone()
    if name.endswith("self_attn.kv_down_proj.weight"):
        out[s.shape[0]:] = 0  # new latent dims must stay zero inside the latent RMSNorm
    elif name.endswith("self_attn.conv.bias"):
        out[s.shape[0]:] = 0  # depthwise conv on the normed residual: zero in, bias out
    elif s.dim() == 1:
        raise ValueError(f"{name}: no function-preserving rule for a 1-D tensor")
    out[sl] = s
    return out


def grow_state_dict(
    donor: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    *,
    operator: str = "szp",
    shrink: float = 0.4,
) -> Tuple[Dict[str, torch.Tensor], List[Tuple[int, bool]]]:
    """Grow ``donor`` into ``target`` (the target's fresh init). Returns the new
    state dict and the layer map (one ``(donor_layer, is_primary)`` per slot)."""
    if operator not in OPERATORS:
        raise ValueError(f"grow operator {operator!r} not in {OPERATORS}")
    d_layers, d_rest = split_layers(donor)
    t_layers, t_rest = split_layers(target)
    if set(d_rest) != set(t_rest):
        raise ValueError(f"non-layer tensors differ: {set(d_rest) ^ set(t_rest)}")
    lmap = align_layers([layer_signature(x) for x in d_layers],
                        [layer_signature(x) for x in t_layers])

    def grow(name: str, s: torch.Tensor, t: torch.Tensor, primary: bool) -> torch.Tensor:
        _check_fits(name, s, t)
        s = s.to(device=t.device, dtype=t.dtype)
        if operator == "szp":
            return _grow_szp(name, s, t, shrink)
        return _grow_fp(name, s, t, primary)

    out: Dict[str, torch.Tensor] = {}
    for k, t in t_rest.items():
        out[k] = grow(k, d_rest[k], t, True)
    for j, (i, primary) in enumerate(lmap):
        for suffix, t in t_layers[j].items():
            out[f"model.layers.{j}.{suffix}"] = grow(
                f"model.layers.{j}.{suffix}", d_layers[i][suffix], t, primary
            )
    return out, lmap


def load_grown_weights(
    path: Path,
    model: torch.nn.Module,
    *,
    operator: str = "szp",
    shrink: float = 0.4,
    allow_config_diff: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Grow the checkpoint at ``path`` into the live (freshly initialised) model.

    Same contract as ``titan.checkpoint.load_init_weights``: tensors only, the
    caller starts at step 0 with a fresh optimizer, and the donor's config.json
    must agree with the live model on everything except the size fields (plus
    ``allow_config_diff``). Returns a report for the log.
    """
    from safetensors.torch import load_file

    from ..titan.checkpoint import init_config_mismatches
    from ..titan.parallelize import unwrap_ddp

    path = Path(path)
    inner = unwrap_ddp(model)
    hf_model = getattr(inner, "inner", inner)
    ckpt_cfg = json.loads((path / "config.json").read_text())
    live_cfg = hf_model.config.to_dict()
    allowed = set(GROW_SIZE_FIELDS) | set(allow_config_diff or ())
    bad = init_config_mismatches(ckpt_cfg, live_cfg, allowed=allowed)
    if bad:
        lines = "\n".join(f"  {k}: donor={a!r} live={b!r}" for k, (a, b) in bad.items())
        raise ValueError(f"grow_from={path}: donor config differs on non-size fields:\n{lines}")
    kv_s, kv_t = ckpt_cfg.get("num_key_value_heads"), live_cfg.get("num_key_value_heads")
    if operator == "fp" and kv_s != kv_t:
        raise ValueError(f"fp growth cannot change num_key_value_heads ({kv_s} -> {kv_t})")

    donor = load_file(str(path / "model.safetensors"), device="cpu")
    live = {k: v.detach() for k, v in hf_model.state_dict().items()}
    with torch.no_grad():
        grown, lmap = grow_state_dict(donor, live, operator=operator, shrink=shrink)
        hf_model.load_state_dict(grown, strict=True)
    return {
        "operator": operator,
        "shrink": shrink if operator == "szp" else None,
        "layers": f"{len(split_layers(donor)[0])} -> {len(lmap)}",
        "layer_map": [f"{i}{'' if p else '*'}" for i, p in lmap],
        "size_changes": {k: v for k, v in init_config_mismatches(ckpt_cfg, live_cfg).items()},
    }
