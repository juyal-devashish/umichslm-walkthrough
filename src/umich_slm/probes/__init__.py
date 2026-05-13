"""Probes for stack-transition decisions and post-stage analysis.

The plan calls for three probes (the user's chosen subset):
  * Block Influence (Men et al. 2024) — per-block normalized residual deltas.
  * Linearity (Saunshi et al. 2024) — how well a per-layer affine map predicts
    the residual update.
  * Layer-deletion damage — perplexity increase when a layer's contribution is
    replaced by a residual pass-through.

The :mod:`runner` module loads a checkpoint, runs all three probes on a small
held-out batch, and emits the JSON consumed by ``staging.io_boundary``.
"""

from .block_influence import block_influence
from .layer_deletion import layer_deletion_damage
from .linearity import linearity_r2

__all__ = ["block_influence", "linearity_r2", "layer_deletion_damage"]
