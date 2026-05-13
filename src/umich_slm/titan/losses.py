"""Loss bookkeeping helpers for the titan trainer.

Most of the heavy lifting (lm + mtp + moe_aux + mod_aux fusion, chunked
cross-entropy) happens inside ``UMichSLMForCausalLM.forward`` when ``labels``
is passed (see ``modeling_umichslm.py:1131-1175``). This module just gathers
the per-component scalars from a ``TitanModelOutput`` for logging.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .wrapper import TitanModelOutput


def detach_scalar(t: Optional[torch.Tensor]) -> Optional[float]:
    if t is None:
        return None
    return float(t.detach().float().item())


def loss_components(output: TitanModelOutput) -> Dict[str, Optional[float]]:
    """Return a dict of per-component loss scalars for logging."""
    return {
        "loss": detach_scalar(output.loss),
        "lm_loss": detach_scalar(output.lm_loss),
        "mtp_loss": detach_scalar(output.mtp_loss),
        "aux_loss": detach_scalar(output.aux_loss),
    }
