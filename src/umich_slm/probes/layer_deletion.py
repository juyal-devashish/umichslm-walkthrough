"""Layer-deletion damage probe.

Per layer, replace the layer's contribution to the residual stream with a
no-op (residual pass-through) and measure the increase in validation loss.
A small increase means the layer is approximately a no-op already — a
candidate for skipping.

Implemented via a forward hook that overwrites the layer's output with its
input. The hook is removed after each measurement so the probe does not
interfere with subsequent calls.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def _replace_layer_with_passthrough(layer: torch.nn.Module):
    """Install a hook on ``layer`` that overwrites its output with its input."""

    def hook(module, args, kwargs, output):
        if not args:
            return output
        x = args[0]
        if isinstance(output, tuple):
            return (x,) + tuple(output[1:])
        return x

    return layer.register_forward_hook(hook, with_kwargs=True)


@torch.no_grad()
def layer_deletion_damage(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    base_loss: float | None = None,
) -> Dict[str, Any]:
    """Run the layer-deletion damage probe.

    Args:
        model: ``UMichSLMForCausalLM`` instance.
        input_ids: Tensor of shape ``[batch, seq_len]``.
        base_loss: Optional precomputed full-model CE; if None, computed here.

    Returns:
        Dict with key ``layer_deletion_ppl_increase`` — list[float] of length L
        giving the per-layer increase in (cross-entropy) loss.
    """
    if base_loss is None:
        base_loss = float(model(input_ids=input_ids, labels=input_ids).loss)

    increases: List[float] = []
    for i, layer in enumerate(model.model.layers):
        handle = _replace_layer_with_passthrough(layer)
        try:
            out = model(input_ids=input_ids, labels=input_ids)
            l = float(out.loss) if torch.isfinite(out.loss) else float("inf")
        finally:
            handle.remove()
        increases.append(max(0.0, l - base_loss))
    return {
        "layer_deletion_ppl_increase": increases,
        "base_loss": float(base_loss),
    }
