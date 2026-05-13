"""Migrate AdamW optimizer state across a stack-time stage transition.

When a stage-N+1 model is built from a stage-N checkpoint, every parameter in
the new model needs an entry in the new optimizer state. For *inherited*
parameters (block weights copied from a source block) we deep-copy the source
parameter's ``(step, exp_avg, exp_avg_sq)``. For *fresh* parameters
(freshly-initialized inserted-block MoD routers, or any parameter that didn't
exist in the previous stage) we initialize zero momentum and a step value
that matches the rest of the model.

The migration operates on raw state-dict-shaped Python dicts so it can be
called without instantiating an Optimizer (cheaper for tests) and so it can be
used in-process or via the standalone CLI.

Adam optimizer state shape (from ``optim.state_dict()``):

    {
      "state": {
        param_id_int: {
          "step": int_or_tensor,
          "exp_avg": Tensor of param shape,
          "exp_avg_sq": Tensor of param shape,
          ...  # may include 'max_exp_avg_sq' for amsgrad
        },
        ...
      },
      "param_groups": [{"params": [param_id, ...], "lr": ..., ...}, ...],
    }

Parameter identity is matched by name. For block-internal parameters we
resolve the source name via the block-level ``layer_mapping`` returned by
:func:`build_stage_n_model`.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

LayerMapping = Dict[int, Union[int, str]]

_LAYER_NAME_RE = re.compile(r"^model\.layers\.(\d+)\.(.*)$")


def _name_to_id(model: nn.Module) -> Dict[str, int]:
    return {n: i for i, (n, _) in enumerate(model.named_parameters())}


def _resolve_source_name(
    new_param_name: str,
    layer_mapping: LayerMapping,
    block_size: int,
) -> Optional[str]:
    """Map a new-model parameter name back to its source-model name.

    For block-internal params, ``model.layers.{new_layer_idx}.{tail}`` ->
    ``model.layers.{src_layer_idx}.{tail}`` where
    ``src_layer_idx = src_block_idx * block_size + (new_layer_idx % block_size)``.

    Returns ``None`` for fresh parameters (router on a fresh inserted block,
    parameters outside ``model.layers.``, or new layers that have no source).
    """
    m = _LAYER_NAME_RE.match(new_param_name)
    if not m:
        # Non-block parameter — caller resolves these by 1:1 name match.
        return new_param_name

    new_layer_idx = int(m.group(1))
    tail = m.group(2)

    new_block_idx = new_layer_idx // block_size
    layer_in_block = new_layer_idx % block_size

    src = layer_mapping.get(new_block_idx)
    if src is None or src == "fresh":
        return None

    src_layer_idx = int(src) * block_size + layer_in_block
    return f"model.layers.{src_layer_idx}.{tail}"


def _zero_state(param: torch.Tensor, step_value: Any) -> Dict[str, Any]:
    """Zero AdamW state matching the parameter's shape."""
    if isinstance(step_value, torch.Tensor):
        step = step_value.detach().clone()
    else:
        step = step_value
    return {
        "step": step,
        "exp_avg": torch.zeros_like(param, memory_format=torch.preserve_format),
        "exp_avg_sq": torch.zeros_like(param, memory_format=torch.preserve_format),
    }


def _representative_step(prev_state: Dict[str, Any]) -> Any:
    """Return any step value from the previous optimizer state, for fresh params."""
    state = prev_state.get("state", {})
    if not state:
        return 0
    first = next(iter(state.values()))
    return first.get("step", 0)


def migrate_optimizer_state(
    prev_optim_state: Dict[str, Any],
    prev_model: nn.Module,
    new_model: nn.Module,
    layer_mapping: LayerMapping,
    *,
    block_size: int,
) -> Dict[str, Any]:
    """Build a state_dict for an AdamW on ``new_model`` matching the source moments.

    Args:
        prev_optim_state: ``optimizer.state_dict()`` from the stage-N optimizer.
        prev_model: stage-N model (used to resolve names -> param IDs).
        new_model: stage-N+1 model.
        layer_mapping: mapping returned by :func:`build_stage_n_model`.
        block_size: number of layers per block in the source/target models.

    Returns:
        A dict in ``optimizer.state_dict()`` shape, suitable for
        ``optimizer.load_state_dict(...)`` on a freshly-built AdamW over
        ``new_model``.
    """
    prev_id_by_name = _name_to_id(prev_model)
    prev_state = prev_optim_state.get("state", {})
    rep_step = _representative_step(prev_optim_state)

    new_state: Dict[int, Dict[str, Any]] = {}
    new_params_in_order = list(new_model.named_parameters())

    for new_id, (new_name, new_param) in enumerate(new_params_in_order):
        source_name = _resolve_source_name(new_name, layer_mapping, block_size)

        if source_name is None:
            # Fresh parameter (e.g. router on a fresh inserted block, or a new
            # parameter that has no analogue in the source model).
            new_state[new_id] = _zero_state(new_param, rep_step)
            continue

        # Look up the source param ID by name. If the source name doesn't
        # exist in the previous model (e.g. a non-block parameter that's new
        # in this stage), fall back to a fresh zero state.
        prev_id = prev_id_by_name.get(source_name)
        if prev_id is None or prev_id not in prev_state:
            new_state[new_id] = _zero_state(new_param, rep_step)
            continue

        # Inherited parameter: deep-copy the source moments. Validate shape
        # match — if shapes differ, fall back to fresh state (this should not
        # happen for the supported block-doubling pattern).
        src = prev_state[prev_id]
        src_exp_avg = src.get("exp_avg")
        if src_exp_avg is None or src_exp_avg.shape != new_param.shape:
            new_state[new_id] = _zero_state(new_param, rep_step)
            continue

        new_state[new_id] = copy.deepcopy(src)

    # Build param_groups by copying the previous group structure but with the
    # new param ID list. We only support a single parameter group here (which
    # is the current AdamW configuration in the codebase). For multi-group
    # optimizers, generalize this if needed.
    prev_groups = prev_optim_state.get("param_groups", [{}])
    new_groups = []
    for group in prev_groups:
        new_group = copy.deepcopy(group)
        new_group["params"] = list(range(len(new_params_in_order)))
        new_groups.append(new_group)
    if not new_groups:
        new_groups = [{"params": list(range(len(new_params_in_order)))}]

    return {"state": new_state, "param_groups": new_groups}


def validate_migrated_state(
    new_optim_state: Dict[str, Any], new_model: nn.Module
) -> None:
    """Sanity check that every parameter in ``new_model`` has a state entry."""
    n_params = len(list(new_model.named_parameters()))
    state = new_optim_state.get("state", {})
    missing = [pid for pid in range(n_params) if pid not in state]
    if missing:
        raise RuntimeError(
            f"Migrated optimizer state is missing entries for {len(missing)} "
            f"parameter IDs (e.g. {missing[:5]})."
        )
