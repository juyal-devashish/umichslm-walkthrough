"""Per-token loss machinery for agentic training signals (P6 + P3).

``token_level_lm_losses`` replaces the plain chunked LM loss when the batch
carries per-token annotations (``titan/harmony_annotate.py``). One chunked pass
over the LM head computes, per token, the next-token cross-entropy and from it:

* ``lm_loss`` -- the UNWEIGHTED mean CE over valid labels, identical to
  ``UMichSLMForCausalLM._compute_shifted_lm_loss_chunked``. It is what
  ``val_lm_loss`` reports, so the ranking metric does not move when weights do.
* the weighted LM objective (P6) -- ``sum(w * CE) / sum(w)`` over valid labels,
  with ``w = loss_weights`` (``w_tool`` on tool-result tokens). Normalised by the
  weight sum, so the loss scale -- and hence the LR rule -- stays comparable.
* the assistant-only CE (``token_role == 1``), as a (sum, count) pair so callers
  can token-weight it across microbatches and ranks.
* unlikelihood (P3c, Welleck et al. 2019) -- at each label position ``t`` with
  ``unlik_targets[t] = f``, the term ``-log(1 - p(f | x_<t))``, averaged over
  such positions. ``f`` is the failing call's token at the same offset of the
  corrected call (see ``annotate_messages``).

and, outside the chunk loop, the failed-call head (P3b): ``ErrorHead`` reads the
final hidden state at every position (one ``hidden -> 1`` GEMV: negligible, and
it avoids a host sync to gather), and BCE is averaged over positions where
``errhead_labels >= 0`` (the ``<|call|>`` tokens).

Everything is computed on device without ``.item()``; the caller syncs only on
logging steps.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_INDEX = -100
ROLE_ASSISTANT = 1


class ErrorHead(nn.Module):
    """Scalar logit ``P(the tool result following this call is an error)``.

    Reads the final (post-norm) hidden state at the ``<|call|>`` token. The
    weight is zero-initialised (``_zero_init_weight`` survives ``post_init`` and
    post-load re-init), so the head starts at logit 0 and its first gradients
    into the trunk are pure BCE signal, not noise from a random projection.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=True)
        self.proj._zero_init_weight = True
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states).squeeze(-1)


def _chunk_size(config, bsz: int, seq_len: int) -> int:
    # Same policy as _compute_shifted_lm_loss_chunked: cap the fp32 logits slab.
    vocab = config.vocab_size
    chunk = getattr(config, "lm_loss_chunk_size", 512)
    chunk = max(1, min(chunk, seq_len - 1))
    cap_bytes = int(getattr(config, "lm_loss_chunk_cap_bytes", 1_200_000_000))
    per_pos = bsz * vocab * 4
    if per_pos > 0:
        chunk = min(chunk, max(32, cap_bytes // per_pos))
    return max(1, min(chunk, seq_len - 1))


def token_level_lm_losses(
    model,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    targets: Dict[str, object],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Return ``(lm_loss, objective, stats)``.

    ``targets`` keys (all optional, ``(B, S)`` aligned with ``labels``):
    ``loss_weights`` (float), ``token_role`` (int), ``unlik_targets`` (long,
    label-aligned), ``errhead_labels`` (int, aligned with INPUT positions --
    the head reads the hidden state AT the ``<|call|>`` token); and the python
    floats ``unlik_weight`` / ``errhead_weight``.

    ``objective`` is the term to train on: the weighted LM loss (or ``lm_loss``
    when no weights are given) plus ``unlik_weight * unlik`` plus
    ``errhead_weight * bce``. ``stats`` holds detached tensors for logging.
    """
    bsz, seq_len, _ = hidden_states.shape
    dev = hidden_states.device
    zero = hidden_states.new_zeros((), dtype=torch.float32)
    weights = targets.get("loss_weights")
    role = targets.get("token_role")
    unlik = targets.get("unlik_targets")
    errlab = targets.get("errhead_labels")
    unlik_w = float(targets.get("unlik_weight", 0.0) or 0.0)
    err_w = float(targets.get("errhead_weight", 0.0) or 0.0)

    ce_sum, n_valid = zero, zero
    wce_sum, w_sum = zero, zero
    asst_sum, asst_n = zero, zero
    ul_sum, ul_n = zero, zero
    if seq_len >= 2:
        chunk = _chunk_size(model.config, bsz, seq_len)
        vocab = model.config.vocab_size
        for start in range(0, seq_len - 1, chunk):
            end = min(start + chunk, seq_len - 1)
            logits = model.lm_head(hidden_states[:, start:end, :]).float()
            tgt = labels[:, start + 1 : end + 1].to(dev)
            valid = tgt != IGNORE_INDEX
            validf = valid.to(torch.float32)
            ce = F.cross_entropy(
                logits.reshape(-1, vocab),
                tgt.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view_as(validf)
            ce_sum = ce_sum + ce.sum()
            n_valid = n_valid + validf.sum()
            if weights is not None:
                w = weights[:, start + 1 : end + 1].to(dev, torch.float32) * validf
                wce_sum = wce_sum + (ce * w).sum()
                w_sum = w_sum + w.sum()
            if role is not None:
                a = (role[:, start + 1 : end + 1].to(dev) == ROLE_ASSISTANT).to(
                    torch.float32
                ) * validf
                asst_sum = asst_sum + (ce.detach() * a).sum()
                asst_n = asst_n + a.sum()
            if unlik is not None:
                f = unlik[:, start + 1 : end + 1].to(dev)
                um = ((f != IGNORE_INDEX) & valid).to(torch.float32)
                # log p(f) = logit_f - lse, and lse = ce + logit_label at valid positions.
                lt = logits.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                lf = logits.gather(-1, f.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                logp_f = (lf - (ce + lt)).clamp(max=0.0)
                p_f = torch.exp(logp_f).clamp(max=1.0 - 1e-6)
                ul = -torch.log1p(-p_f)
                ul_sum = ul_sum + (ul * um).sum()
                ul_n = ul_n + um.sum()

    lm_loss = ce_sum / n_valid.clamp(min=1.0)
    lm_obj = wce_sum / w_sum.clamp(min=1e-6) if weights is not None else lm_loss
    unlik_loss = ul_sum / ul_n.clamp(min=1.0)
    objective = lm_obj
    if unlik is not None and unlik_w > 0:
        objective = objective + unlik_w * unlik_loss

    stats: Dict[str, torch.Tensor] = {}
    head: Optional[nn.Module] = getattr(model, "errhead", None)
    if head is not None:
        scores = head(hidden_states).float()
        if errlab is not None:
            lab = errlab.to(dev)
            m = (lab >= 0).to(torch.float32)
            bce = F.binary_cross_entropy_with_logits(
                scores, lab.clamp(min=0).to(torch.float32), reduction="none"
            )
            err_n = m.sum()
            err_loss = (bce * m).sum() / err_n.clamp(min=1.0)
            stats["errhead_loss"] = err_loss.detach()
            stats["errhead_count"] = err_n.detach()
            stats["errhead_scores"] = scores.detach()
            objective = objective + err_w * err_loss
        # Keep the head in the graph on batches with no labelled call (and when
        # its weight is 0): DDP without find_unused_parameters requires every
        # parameter to receive a gradient on every step.
        objective = objective + 0.0 * scores.sum()

    stats["lm_obj"] = lm_obj.detach()
    if role is not None:
        stats["asst_loss_sum"] = asst_sum.detach()
        stats["asst_count"] = asst_n.detach()
    if unlik is not None:
        stats["unlik_loss"] = unlik_loss.detach()
        stats["unlik_count"] = ul_n.detach()
    return lm_loss, objective, stats


def auroc(scores, labels) -> Optional[float]:
    """Mann-Whitney AUROC with average ranks for ties; None if one class is empty."""
    import numpy as np
    from scipy.stats import rankdata

    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
