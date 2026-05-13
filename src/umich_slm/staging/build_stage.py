"""Build the stage-N model from a previous-stage checkpoint.

Given:
  * a ``prev_config: UMichSLMConfig``,
  * the previous-stage state dict (``prev_state_dict: dict[str, torch.Tensor]``),
  * a :class:`StageSpec` describing the target block list,

this module constructs a stage-N :class:`UMichSLMForCausalLM`, copies inherited
block weights and non-block parameters from the previous stage, initializes
fresh MoD routers on inserted blocks, and returns:
  ``(new_config, new_model, layer_mapping)``

``layer_mapping`` is a ``dict[int, Union[int, str]]`` keyed by new block index;
each entry is the source block index (int) or ``"fresh"`` for a freshly
initialized block. ``layer_mapping`` is consumed by
:func:`migrate_optimizer_state` to rebuild Adam moments.

This module does **not** touch optimizer state — that is the job of
``optimizer_migration.py``. The split is intentional so the staging package
can be exercised in unit tests without an optimizer.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple, Union

import torch

from ..configuration_umichslm import UMichSLMConfig
from ..modeling_umichslm import UMichSLMForCausalLM
from ..modules.transformer_block import UMichSLMMoDDecoderLayer
from .block_spec import StageSpec


LayerMapping = Dict[int, Union[int, str]]


def _block_param_prefix(layer_idx: int, layers_per_block: int, block_idx: int) -> List[str]:
    """Prefixes ``model.layers.{layer_idx}.`` for the layers in a block.

    block ``block_idx`` covers layer indices
    ``block_idx * layers_per_block .. (block_idx + 1) * layers_per_block - 1``.
    """
    start = block_idx * layers_per_block
    return [f"model.layers.{i}." for i in range(start, start + layers_per_block)]


def _copy_inherited_block_weights(
    prev_state_dict: Dict[str, torch.Tensor],
    new_state_dict: Dict[str, torch.Tensor],
    src_block: int,
    dst_block: int,
    layers_per_block: int,
) -> None:
    """Copy the ``layers_per_block`` source-block layers to the destination block.

    Modifies ``new_state_dict`` in-place. Only keys that exist in BOTH state
    dicts are copied — this is intentional because the destination layer's
    architecture is determined by ``hybrid_attention_ratio`` and may not match
    the source layer 1:1 (though for the supported in-place-doubling pattern
    they always do).
    """
    src_prefixes = _block_param_prefix(0, layers_per_block, src_block)
    dst_prefixes = _block_param_prefix(0, layers_per_block, dst_block)

    for src_pref, dst_pref in zip(src_prefixes, dst_prefixes):
        # Find every key in prev that begins with src_pref; rewrite to dst_pref.
        for k, v in prev_state_dict.items():
            if not k.startswith(src_pref):
                continue
            tail = k[len(src_pref) :]
            new_key = dst_pref + tail
            if new_key in new_state_dict:
                new_state_dict[new_key] = v.clone()


def _copy_non_block_params(
    prev_state_dict: Dict[str, torch.Tensor],
    new_state_dict: Dict[str, torch.Tensor],
) -> None:
    """Copy parameters that live outside the per-block layer stack.

    These include ``model.embed_tokens.*``, ``model.norm.*``,
    ``model.global_tokens.*``, ``model.meta_tokens.*``, ``lm_head.*``,
    and the MTP heads (``mtp.*``). Anything *not* starting with
    ``model.layers.`` is treated as a non-block parameter.
    """
    for k, v in prev_state_dict.items():
        if k.startswith("model.layers."):
            continue
        if k in new_state_dict:
            new_state_dict[k] = v.clone()


def _zero_inserted_block_for_identity_init(
    new_model: UMichSLMForCausalLM,
    block_idx: int,
    layers_per_block: int,
) -> None:
    """LLaMA-Pro style identity init: zero the output projections of the block.

    For an inserted (newly-routed) block, zero ``self_attn.o_proj.weight`` and
    ``mlp.down_proj.weight`` so the block is initially a residual identity. The
    MoD gate then has very little effect at first, easing transition stability.
    """
    start = block_idx * layers_per_block
    layers = new_model.model.layers
    for i in range(start, start + layers_per_block):
        layer = layers[i]
        # Zero the attention output projection (not all attention classes have
        # the same name; defensively try a few common ones).
        for proj_name in ("o_proj", "wo", "out_proj"):
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, proj_name):
                with torch.no_grad():
                    getattr(attn, proj_name).weight.zero_()
                    if getattr(attn, proj_name).bias is not None:
                        getattr(attn, proj_name).bias.zero_()
                break
        # Zero the MLP down-projection.
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "down_proj"):
            with torch.no_grad():
                mlp.down_proj.weight.zero_()
                if getattr(mlp.down_proj, "bias", None) is not None:
                    mlp.down_proj.bias.zero_()


def _perturb_router(layer: torch.nn.Module, std: float) -> None:
    """Add small Gaussian noise to a copied MoD router so adjacent duplicated
    blocks at stage 3 don't end up with identical routing decisions."""
    if std <= 0:
        return
    if not hasattr(layer, "router") or not hasattr(layer.router, "router"):
        return
    with torch.no_grad():
        layer.router.router.weight.add_(
            torch.randn_like(layer.router.router.weight) * std
        )


def build_stage_n_model(
    prev_config: UMichSLMConfig,
    prev_state_dict: Dict[str, torch.Tensor],
    stage_spec: StageSpec,
    *,
    init_strategy: str = "exact_copy",
) -> Tuple[UMichSLMConfig, UMichSLMForCausalLM, LayerMapping]:
    """Build a stage-N model from a previous-stage state dict and a StageSpec.

    Args:
        prev_config: ``UMichSLMConfig`` of the source (previous-stage) model.
        prev_state_dict: state_dict of the source model.
        stage_spec: target layout — block-by-block source mapping and layouts.
        init_strategy: ``"exact_copy"`` (default — duplicate weights as-is) or
            ``"llama_pro_identity"`` (zero ``o_proj`` / ``mlp.down_proj`` on
            inserted blocks so they start as a residual identity).

    Returns:
        Tuple of ``(new_config, new_model, layer_mapping)``.
    """
    if init_strategy not in ("exact_copy", "llama_pro_identity"):
        raise ValueError(
            f"init_strategy must be 'exact_copy' or 'llama_pro_identity'; "
            f"got {init_strategy!r}."
        )

    block_size = stage_spec.block_size
    if block_size <= 0:
        raise ValueError("StageSpec must have at least one block with a non-empty layout.")

    # Build the new config by copying the previous one and overriding the
    # stack-shape fields. We deepcopy to avoid mutating the caller's config.
    new_config_dict = prev_config.to_diff_dict() if hasattr(prev_config, "to_diff_dict") else {}
    # ``to_diff_dict`` returns only fields that differ from defaults — to be
    # safe we instead copy the source config's full attributes via deepcopy.
    new_config = copy.deepcopy(prev_config)
    new_config.num_hidden_layers = stage_spec.num_hidden_layers
    new_config.attention_mode = stage_spec.attention_mode
    new_config.hybrid_attention_ratio = list(stage_spec.hybrid_attention_ratio)
    new_config.stage_block_spec = [bs.to_dict() for bs in stage_spec.block_specs]
    # In SWA mode the globals must already be present from stage 1 — see
    # _validate_attention_mode. The validator will throw if config drifts.
    # Re-run validation after the mutation.
    new_config._validate_attention_mode()
    new_config._validate_stage_block_spec()

    # Construct the new model. This initializes everything fresh; we then
    # selectively overwrite weights from the previous state dict.
    new_model = UMichSLMForCausalLM(new_config)
    new_state_dict = {k: v.clone() for k, v in new_model.state_dict().items()}

    # Copy non-block parameters first (embeddings, norms, lm_head, MTP, globals).
    _copy_non_block_params(prev_state_dict, new_state_dict)

    # Copy block weights and build the layer_mapping.
    layer_mapping: LayerMapping = {}
    for new_block_idx, bs in enumerate(stage_spec.block_specs):
        if bs.source is None:
            layer_mapping[new_block_idx] = "fresh"
            continue
        _copy_inherited_block_weights(
            prev_state_dict, new_state_dict,
            src_block=bs.source, dst_block=new_block_idx,
            layers_per_block=block_size,
        )
        layer_mapping[new_block_idx] = bs.source

    # Load the assembled state dict back onto the new model.
    missing, unexpected = new_model.load_state_dict(new_state_dict, strict=False)
    # ``strict=False`` because parameters that don't exist in the prev state
    # dict (e.g. fresh inserted-block routers) keep their fresh init — that's
    # the desired behavior.

    # Mark inserted blocks via set_inserted so MoD capacity flips to
    # mod_capacity_inserted, optionally apply identity init and router
    # perturbation.
    for new_block_idx, bs in enumerate(stage_spec.block_specs):
        start = new_block_idx * block_size
        for i in range(start, start + block_size):
            layer = new_model.model.layers[i]
            if isinstance(layer, UMichSLMMoDDecoderLayer):
                layer.set_inserted(bs.is_inserted)
                if bs.is_inserted and not bs.fresh_router:
                    _perturb_router(layer, bs.router_perturb_std)

        if bs.is_inserted and init_strategy == "llama_pro_identity":
            _zero_inserted_block_for_identity_init(
                new_model, new_block_idx, block_size
            )

    return new_config, new_model, layer_mapping
