"""Token-Superposition Training (TST) primitives.

Implements the two core operations from Peng et al., "Efficient Pre-Training with
Token Superposition" (arXiv:2605.06546):

- ``bag_embeddings``: average non-overlapping contiguous s-grams of token
  embeddings along the sequence dim — produces L/s "s-tokens".
- ``shift_labels_for_tst``: shift labels left by ``s - 1`` so that each s-token
  at positions [t, t+s-1] predicts the bag at [t+s, t+2s-1].
- ``multihot_ce``: average of per-position causal CE losses across the bag of
  ``s`` next-token targets (Eq. 3 in the paper). Optional power-law weighting
  per Appendix D for ``s >= 8``.

These are referenced from ``modeling_umichslm.py`` when ``config.tst_active``
is true; phase (ii) "recovery" simply turns the flag off.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def bag_embeddings(embeds: torch.Tensor, s: int) -> torch.Tensor:
    """Average non-overlapping s-grams along the sequence dim.

    Args:
        embeds: ``(B, L, d)`` token embeddings.
        s: bag size. ``L`` must be divisible by ``s``.

    Returns:
        ``(B, L // s, d)`` superposed embeddings.
    """
    if s <= 1:
        return embeds
    b, l, d = embeds.shape
    if l % s != 0:
        raise ValueError(
            f"bag_embeddings: sequence length {l} not divisible by bag size {s}"
        )
    return embeds.view(b, l // s, s, d).mean(dim=2)


def shift_labels_for_tst(labels: torch.Tensor, s: int, pad_id: int = -100) -> torch.Tensor:
    """Left-shift raw input-id labels by ``s`` so each s-token bag predicts the next bag.

    The paper specifies the labels-as-already-shifted (NTP) convention and asks
    for an additional left shift of ``s - 1``. In this codebase ``labels`` is
    the raw input_id tensor and the trainer-side loss adds the +1 itself, so
    the net shift here is ``s``. After shifting, for s-token bag ``j`` at input
    positions ``[j*s, j*s+s-1]``, the bag of targets is at positions
    ``[j*s+s, j*s+2s-1]`` — exactly the paper's Eq.-3 alignment.

    Args:
        labels: ``(B, L)`` int64 labels (raw input_ids).
        s: bag size.
        pad_id: ignore-index used in the padded tail (default -100).

    Returns:
        ``(B, L)`` shifted labels with the last ``s`` positions set to ``pad_id``.
    """
    if s <= 1:
        return labels
    shifted = torch.full_like(labels, pad_id)
    if labels.shape[1] > s:
        shifted[:, : labels.shape[1] - s] = labels[:, s:]
    return shifted


def _bag_weights(s: int, weighting: str, device, dtype) -> torch.Tensor:
    if weighting == "uniform":
        return torch.full((s,), 1.0 / s, device=device, dtype=dtype)
    if weighting == "power_law":
        # Appendix D: power-law decay over within-bag offset.
        # w_k ∝ (k+1)^-1 for k in [0, s-1], normalized to sum to 1.
        idx = torch.arange(1, s + 1, device=device, dtype=dtype)
        w = idx.reciprocal()
        return w / w.sum()
    raise ValueError(f"unknown TST loss weighting: {weighting!r}")


def multihot_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    s: int,
    weighting: str = "uniform",
    ignore_index: int = -100,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Average of per-target causal CE losses over a bag of ``s`` future tokens.

    The model emits one prediction per s-token (``logits`` has length ``L/s``).
    Each prediction is scored against ``s`` consecutive ground-truth tokens
    forming the next bag.

    Args:
        logits: ``(B, L/s, V)`` logits — one per s-token.
        labels: ``(B, L)`` int64 token ids ALREADY shifted via
            ``shift_labels_for_tst``. The targets for s-token j are
            ``labels[:, j*s : (j+1)*s]``.
        s: bag size.
        weighting: "uniform" (Eq. 3) or "power_law" (Appendix D).
        ignore_index: label value that is masked out of the loss.
        chunk_size: if set, process logits in chunks along the s-token dim to
            bound peak fp32 memory (mirrors the standard CE-chunking pattern).

    Returns:
        scalar loss.
    """
    b, l_s, v = logits.shape
    expected_label_len = l_s * s
    if labels.shape[1] < expected_label_len:
        raise ValueError(
            f"multihot_ce: labels seq len {labels.shape[1]} < l_s*s = {expected_label_len}"
        )
    # Reshape labels into (B, L/s, s) bags aligned to logits.
    targets = labels[:, :expected_label_len].view(b, l_s, s)

    weights = _bag_weights(s, weighting, device=logits.device, dtype=logits.dtype)

    if chunk_size is None or chunk_size >= l_s:
        chunks = [(0, l_s)]
    else:
        chunks = [(i, min(i + chunk_size, l_s)) for i in range(0, l_s, chunk_size)]

    total = logits.new_zeros(())
    total_weight = logits.new_zeros(())
    for start, end in chunks:
        z = logits[:, start:end, :]  # (B, c, V)
        t = targets[:, start:end, :]  # (B, c, s)
        # Per-offset CE: reuse F.cross_entropy on the same logits with each of the s targets.
        flat_z = z.reshape(-1, v)  # (B*c, V)
        for k in range(s):
            tk = t[:, :, k].reshape(-1)  # (B*c,)
            per_tok = F.cross_entropy(
                flat_z, tk, reduction="none", ignore_index=ignore_index
            )  # (B*c,)
            valid = (tk != ignore_index).to(per_tok.dtype)
            wk = weights[k]
            total = total + wk * (per_tok * valid).sum()
            total_weight = total_weight + wk * valid.sum()
    return total / total_weight.clamp_min(1.0)
