"""Bag-of-next-K lookahead: an order-free auxiliary planning objective.

MTP asks the model to name token i+2 exactly. Most of that target is arbitrary —
which synonym, which word order — so a large part of the gradient is spent on
detail the model cannot know and that does not reflect planning. Bag-of-next-K
asks the strictly easier, more semantic question:

    which tokens appear anywhere in the next K positions?

as a multi-label problem, order-free. A model that has "decided where the
sentence is going" can answer it; a model doing pure local continuation cannot.
The hypothesis is that this is a better-posed lookahead signal per unit compute
than exact multi-token prediction, especially on the reasoning/agentic ~40% of
the mix where the *content* of the next clause is far more predictable than its
exact surface form.

Cost. The head is the dominant term, because any vocab-sized output costs about
what the LM head costs (at 25m, one vocab projection is ~84% of the body's
per-token FLOPs). Two levers:
  * ``bag_share_head`` (default True) reuses ``lm_head``'s weights behind a
    small learned [d, d] transform, so the objective adds ~d^2 params, not d*V.
  * ``bag_vocab`` (default 0 = full vocab) folds every token id >= bag_vocab
    into a single OTHER class, cutting the output width. Order-free bag targets
    tolerate a coarse label space far better than exact next-token prediction
    does, which is what makes collapsing defensible here and not for MTP.

Memory. The [tokens x vocab] logits are exactly what OOMed the MTP cells
(13.7 GiB at B16/S4096/V56000). Same remedy: slice the positions, wrap each
slice in a checkpoint so its logits are recomputed in the backward, and build
the multi-hot target inside the slice so it is never materialised full-size.

Causality/safety: targets come from ``input_ids`` shifted forward, are masked at
``ignore_index``, and are clipped at document boundaries when ``doc_ids`` is
supplied, so the objective can never leak across documents.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .rmsnorm import RMSNorm


class BagLookaheadHead(nn.Module):
    """Predicts the SET of tokens occurring in the next K positions."""

    def __init__(self, config, lm_head: Optional[nn.Module] = None):
        super().__init__()
        h = config.hidden_size
        self.k = int(getattr(config, "bag_lookahead_k", 16))
        self.weight = float(getattr(config, "bag_lookahead_weight", 0.1))
        self.chunk_tokens = int(getattr(config, "bag_loss_chunk_tokens", 4096))
        self.vocab_size = int(config.vocab_size)
        # 0 => full vocab; otherwise ids >= bag_vocab collapse into one OTHER
        # class at index `bag_vocab` (so the output width is bag_vocab + 1).
        self.bag_vocab = int(getattr(config, "bag_vocab", 0) or 0)
        self.share_head = bool(getattr(config, "bag_share_head", True))

        self.transform = nn.Linear(h, h, bias=False)
        self.norm = RMSNorm(h, eps=getattr(config, "rms_norm_eps", 1e-6))
        if self.share_head and self.bag_vocab == 0 and lm_head is not None:
            # Weight-tied to the LM head: no new [d, V] matrix.
            self._shared_lm_head = lm_head
            self.out = None
        else:
            self._shared_lm_head = None
            self.out = nn.Linear(h, self.out_width, bias=False)

    @property
    def out_width(self) -> int:
        return self.vocab_size if self.bag_vocab == 0 else self.bag_vocab + 1

    def _project(self, hidden: torch.Tensor) -> torch.Tensor:
        z = self.norm(self.transform(hidden))
        if self.out is not None:
            return self.out(z)
        return self._shared_lm_head(z)

    def _collapse(self, ids: torch.Tensor) -> torch.Tensor:
        if self.bag_vocab == 0:
            return ids
        return torch.where(ids < self.bag_vocab, ids, ids.new_full((), self.bag_vocab))

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        doc_ids: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """hidden_states [B, S, d]; input_ids [B, S]; doc_ids [B, S] or None."""
        b, s, _ = hidden_states.shape
        k = min(self.k, max(s - 1, 0))
        if k <= 0:
            return hidden_states.new_zeros((), dtype=torch.float32)

        # targets[:, i, j] = token at position i + 1 + j, or ignore past the end
        # / across a document boundary.
        idx = (
            torch.arange(s, device=input_ids.device).unsqueeze(-1)
            + 1
            + torch.arange(k, device=input_ids.device)
        )
        valid = idx < s
        gather_idx = idx.clamp_max(s - 1).unsqueeze(0).expand(b, -1, -1)  # [B, S, k]
        tgt = input_ids.gather(1, gather_idx.reshape(b, -1)).view(b, s, k)
        mask = valid.unsqueeze(0).expand(b, -1, -1)
        if doc_ids is not None:
            same_doc = doc_ids.gather(1, gather_idx.reshape(b, -1)).view(b, s, k)
            mask = mask & (same_doc == doc_ids.unsqueeze(-1))
        tgt = self._collapse(tgt)

        flat_h = hidden_states.reshape(b * s, -1)
        flat_t = tgt.reshape(b * s, k)
        flat_m = mask.reshape(b * s, k)
        # A position supervises only if at least one of its K targets is real.
        rows = flat_m.any(-1)
        n_rows = int(rows.sum())
        if n_rows == 0:
            return hidden_states.new_zeros((), dtype=torch.float32)

        width = self.out_width

        # Rebalance: only <=K of `width` classes are positive, so unweighted BCE
        # is ~entirely negatives and the positives never move the logits.
        pos_weight = torch.tensor(max(float(width - k) / max(k, 1), 1.0))

        def _slice_loss(hh, tt, mm):
            logits = self._project(hh).float()  # [n, width]
            # One extra DUMP column absorbs masked slots. Without it a masked
            # slot (index 0) would scatter over a genuine positive at index 0 —
            # scatter_ resolves duplicate indices by last-write-wins.
            target = logits.new_zeros((logits.shape[0], width + 1))
            safe_t = torch.where(mm, tt, tt.new_full((), width))
            target.scatter_(1, safe_t, 1.0)
            target = target[:, :width]
            return F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pos_weight.to(logits.device), reduction="sum"
            )

        total = None
        sel_h, sel_t, sel_m = flat_h[rows], flat_t[rows], flat_m[rows]
        for i in range(0, n_rows, self.chunk_tokens):
            hh = sel_h[i : i + self.chunk_tokens]
            tt = sel_t[i : i + self.chunk_tokens]
            mm = sel_m[i : i + self.chunk_tokens]
            if torch.is_grad_enabled() and hh.requires_grad:
                part = checkpoint(_slice_loss, hh, tt, mm, use_reentrant=False)
            else:
                part = _slice_loss(hh, tt, mm)
            total = part if total is None else total + part
        # Mean over BOTH axes. Summing over the vocab axis would put the loss at
        # ~width*log2 (=38,800 at V=56000) and swamp the LM term at any sane
        # weight; per-class mean starts at ~log2 and stays commensurate with CE.
        return total / (n_rows * width)
