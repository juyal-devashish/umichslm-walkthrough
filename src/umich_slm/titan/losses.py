"""Loss bookkeeping helpers for the titan trainer.

Most of the heavy lifting (lm + mtp + moe_aux + mod_aux fusion, chunked
cross-entropy) happens inside ``UMichSLMForCausalLM.forward`` when ``labels``
is passed (see ``modeling_umichslm.py:1131-1175``). This module just gathers
the per-component scalars from a ``TitanModelOutput`` for logging.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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


# ---------------------------------------------------------------------------
# Per-token loss signals ([token_loss]; titan/harmony_annotate.py +
# modules/agentic_heads.py).
# ---------------------------------------------------------------------------

TOKEN_TARGET_KEYS = ("loss_weights", "token_role", "errhead_labels", "unlik_targets")


def token_targets_from_batch(
    batch: Dict[str, Any], device, token_loss_cfg, seq_len: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """The ``token_targets`` dict for ``model(...)``, or None when the batch has none.

    ``seq_len`` truncates (TST recovery slices its windows back to ``seq_len``).
    """
    tt: Dict[str, Any] = {}
    for k in TOKEN_TARGET_KEYS:
        v = batch.get(k)
        if v is not None:
            v = v[:, :seq_len] if seq_len is not None else v
            tt[k] = v.to(device, non_blocking=True)
    if not tt:
        return None
    tt["errhead_weight"] = float(token_loss_cfg.errhead_weight)
    tt["unlik_weight"] = float(token_loss_cfg.unlik_weight)
    return tt


class TokenStatAccumulator:
    """Accumulates ``TitanModelOutput.token_stats`` over microbatches / eval batches.

    Assistant-only CE is token-weighted (sum / count), never a mean of means:
    the assistant fraction swings several-fold between windows. Head and
    unlikelihood losses are weighted by their position counts for the same
    reason. ``collect_scores`` keeps (score, label) pairs at labelled ``<|call|>``
    positions for the validation AUROC.
    """

    def __init__(self, collect_scores: bool = False):
        self.sums: Dict[str, float] = {}
        self.collect_scores = collect_scores
        self.scores: List[float] = []
        self.labels: List[int] = []

    def _add(self, k: str, v: float) -> None:
        self.sums[k] = self.sums.get(k, 0.0) + v

    def add(self, stats: Optional[Dict[str, torch.Tensor]], errhead_labels=None) -> None:
        if not stats:
            return
        f = {k: float(v.float().item()) for k, v in stats.items() if v.dim() == 0}
        if "asst_count" in f:
            self._add("asst_sum", f["asst_loss_sum"])
            self._add("asst_n", f["asst_count"])
        if "errhead_count" in f:
            self._add("err_sum", f["errhead_loss"] * f["errhead_count"])
            self._add("err_n", f["errhead_count"])
        if "unlik_count" in f:
            self._add("ul_sum", f["unlik_loss"] * f["unlik_count"])
            self._add("ul_n", f["unlik_count"])
        if "lm_obj" in f:
            self._add("obj_sum", f["lm_obj"])
            self._add("obj_n", 1.0)
        if self.collect_scores and errhead_labels is not None and "errhead_scores" in stats:
            lab = errhead_labels.to(stats["errhead_scores"].device)
            m = lab >= 0
            if bool(m.any()):
                self.scores.extend(stats["errhead_scores"][m].float().cpu().tolist())
                self.labels.extend(lab[m].cpu().tolist())

    def result(self, world_size: int = 1, device=None) -> Dict[str, float]:
        """Means over everything added; all-reduced (and AUROC gathered) under DDP."""
        import torch.distributed as dist

        keys = ("asst_sum", "asst_n", "err_sum", "err_n", "ul_sum", "ul_n", "obj_sum", "obj_n")
        vals = [self.sums.get(k, 0.0) for k in keys]
        scores, labels = self.scores, self.labels
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            t = torch.tensor(vals, dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            vals = t.tolist()
            if self.collect_scores:
                gathered: List[Any] = [None] * world_size
                dist.all_gather_object(gathered, (scores, labels))
                scores = [s for g in gathered for s in g[0]]
                labels = [y for g in gathered for y in g[1]]
        s = dict(zip(keys, vals))
        out: Dict[str, float] = {}
        if s["asst_n"] > 0:
            out["asst_loss"] = s["asst_sum"] / s["asst_n"]
            out["asst_tokens"] = s["asst_n"]
        if s["err_n"] > 0:
            out["errhead_loss"] = s["err_sum"] / s["err_n"]
            out["errhead_calls"] = s["err_n"]
        if s["ul_n"] > 0:
            out["unlik_loss"] = s["ul_sum"] / s["ul_n"]
            out["unlik_positions"] = s["ul_n"]
        if s["obj_n"] > 0:
            out["lm_obj"] = s["obj_sum"] / s["obj_n"]
        if self.collect_scores and labels:
            from ..modules.agentic_heads import auroc

            a = auroc(scores, labels)
            if a is not None:
                out["errhead_auroc"] = a
            out["errhead_pos_frac"] = sum(labels) / len(labels)
        return out
