"""Compose embeddings for SuperBPE tokens out of a donor model's embeddings.

~73% of our token *mass* has no verbatim donor token (see
:mod:`umich_slm.warmstart.vocab_align`), so those rows have to be built from the
donor embeddings of the pieces the token decomposes into. The rules here differ
only in how they weight and rescale those pieces; :func:`fit_positional` learns
its weights from the donor's own merge table rather than assuming them.

Two facts drive the design:

* **Mean-pooling shrinks the norm.** Averaging k near-orthogonal vectors gives a
  result with ~1/sqrt(k) of their norm. Left uncorrected, every composed row is
  systematically shorter than every exact row, which is a scale bug on ~73% of
  the table, not a subtle semantic one. Every rule here is norm-matched by
  default (``renorm=True``).
* **Input and output embeddings want different rules.** An input embedding is
  read as "what is this token"; averaging the pieces is defensible. An output
  embedding is a logit direction competing in a softmax, and the supertoken
  ``'yĠtheĠ'`` is predicted where the donor would have predicted its *first*
  piece. ``first`` is therefore the natural output-side rule. Both are exposed so
  the choice can be measured rather than argued.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from .vocab_align import Alignment

RULES = ("mean", "first", "last", "bytes", "positional")


def _weights_for(
    rule: str,
    ids: Sequence[int],
    nbytes: Sequence[int],
    pos_w: Optional[torch.Tensor],
) -> torch.Tensor:
    k = len(ids)
    if rule == "mean":
        w = torch.ones(k)
    elif rule == "first":
        w = torch.zeros(k)
        w[0] = 1.0
    elif rule == "last":
        w = torch.zeros(k)
        w[-1] = 1.0
    elif rule == "bytes":
        w = torch.tensor([float(max(b, 1)) for b in nbytes])
    elif rule == "positional":
        assert pos_w is not None, "positional rule needs fitted weights"
        # pos_w[0, 0] holds the fitted last/first ratio, so any arity -- including
        # the ~450 tokens that are whole memorised boilerplate sentences and
        # decompose into 50-132 donor pieces -- gets a well-defined row.
        r = float(pos_w[0, 0])
        w = torch.tensor([r**j for j in range(k)], dtype=torch.float32)
    else:
        raise ValueError(f"unknown rule {rule!r}")
    s = w.sum()
    return w / s if s.abs() > 1e-8 else torch.full((k,), 1.0 / k)


@torch.no_grad()
def compose_table(
    donor_emb: torch.Tensor,
    align: Alignment,
    *,
    rule: str = "mean",
    renorm: bool = True,
    pos_w: Optional[torch.Tensor] = None,
    noise_std: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Build an ``[our_vocab, d_donor]`` table from ``donor_emb``.

    Exact matches are copied. Composed rows are a weighted combination of their
    pieces, rescaled (when ``renorm``) to the mean norm of the exact rows so the
    two populations are on the same scale. ``noise_std`` adds a small symmetry-
    breaking jitter, in units of the per-row norm -- several of our tokens
    decompose into the *same* donor pieces in a different order, and identical
    rows in an embedding table are a degenerate starting point.
    """
    donor_emb = donor_emb.float()
    d = donor_emb.shape[1]
    out = torch.zeros(align.our_size, d)
    exact_norms: List[float] = []

    for i, ids in enumerate(align.parts):
        if not ids:
            continue
        if align.exact[i]:
            out[i] = donor_emb[ids[0]]
            exact_norms.append(float(out[i].norm()))
            continue
        w = _weights_for(rule, ids, align.part_bytes[i], pos_w)
        out[i] = (w.unsqueeze(1) * donor_emb[list(ids)]).sum(0)

    target = (
        sum(exact_norms) / len(exact_norms) if exact_norms else float(donor_emb.norm(dim=1).mean())
    )
    if renorm:
        composed = torch.tensor([not e and bool(p) for e, p in zip(align.exact, align.parts)])
        n = out[composed].norm(dim=1, keepdim=True).clamp_min(1e-6)
        out[composed] = out[composed] / n * target

    # Unmapped rows (partial UTF-8 with no donor character) get small random
    # vectors at the table's scale rather than zeros, which would be dead rows.
    missing = torch.tensor([not p for p in align.parts])
    if missing.any():
        r = torch.randn(int(missing.sum()), d, generator=generator)
        out[missing] = r / r.norm(dim=1, keepdim=True) * target

    if noise_std > 0:
        jitter = torch.randn(out.shape, generator=generator) * (noise_std * target / (d**0.5))
        out = out + jitter
    return out


@torch.no_grad()
def fit_positional(
    donor_emb: torch.Tensor,
    triples: Sequence[Sequence[int]],
    *,
    max_parts: int = 8,
    sample: int = 60000,
    seed: int = 0,
) -> torch.Tensor:
    """Least-squares fit of per-position composition weights on donor merges.

    Each ``(parent, left, right)`` triple from the donor's merge table is a known
    composite whose true embedding we have, so ``e[parent] ~= a*e[left] +
    b*e[right]`` is an ordinary least-squares problem with ~100k rows and two
    unknowns. Uniform mean-pooling assumes ``a = b = 0.5``; the fit says what the
    donor's own geometry actually implies, and it is usually not 0.5.

    Returns ``[max_parts+1, max_parts]`` weights, row ``k`` holding the weights
    for a ``k``-part decomposition. Rows beyond 2 are extrapolated from the
    fitted first/last asymmetry, since the merge table only ever supervises
    pairs.
    """
    g = torch.Generator().manual_seed(seed)
    trip = torch.tensor(list(triples)[: max(sample, 1) * 4], dtype=torch.long)
    if trip.numel() == 0:
        return torch.ones(max_parts + 1, max_parts) / max(max_parts, 1)
    if trip.shape[0] > sample:
        sel = torch.randperm(trip.shape[0], generator=g)[:sample]
        trip = trip[sel]
    e = donor_emb.float()
    y = e[trip[:, 0]]
    x1 = e[trip[:, 1]]
    x2 = e[trip[:, 2]]
    # Normal equations for min ||y - a*x1 - b*x2||^2 over the whole sample.
    g11 = float((x1 * x1).sum())
    g22 = float((x2 * x2).sum())
    g12 = float((x1 * x2).sum())
    b1 = float((x1 * y).sum())
    b2 = float((x2 * y).sum())
    det = g11 * g22 - g12 * g12
    if abs(det) < 1e-8:
        a = b = 0.5
    else:
        a = (b1 * g22 - b2 * g12) / det
        b = (g11 * b2 - g12 * b1) / det
    s = a + b
    if abs(s) < 1e-8:
        a = b = 0.5
        s = 1.0
    a, b = a / s, b / s

    w = torch.zeros(max_parts + 1, max_parts)
    w[0, 0] = (b / a) if a > 1e-6 else 1.0  # the fitted per-step ratio
    w[1, 0] = 1.0
    for k in range(2, max_parts + 1):
        # Geometric interpolation of the fitted first/last asymmetry across k
        # positions: ratio r = b/a per step, normalised to sum 1.
        row = torch.tensor([float(w[0, 0]) ** j for j in range(k)], dtype=torch.float32)
        w[k, :k] = row / row.sum()
    return w


def summarize(table: torch.Tensor, align: Alignment) -> Dict[str, float]:
    ex = torch.tensor(align.exact)
    co = torch.tensor([not e and bool(p) for e, p in zip(align.exact, align.parts)])
    n = table.norm(dim=1)
    return {
        "exact_norm_mean": float(n[ex].mean()) if ex.any() else 0.0,
        "composed_norm_mean": float(n[co].mean()) if co.any() else 0.0,
        "all_norm_std": float(n.std()),
        "dup_rows": int(len(table) - len(torch.unique(table, dim=0))),
    }


@torch.no_grad()
def fit_linear_composer(
    donor_emb: torch.Tensor,
    triples: Sequence[Sequence[int]],
    *,
    first_w: float = 0.5,
    ridge: float = 1e-2,
    shrink: float = 0.0,
    n_fit: int = 80000,
    seed: int = 0,
) -> torch.Tensor:
    """Learn a ``d x d`` correction on top of weighted-mean composition.

    Weighted means land in roughly the right neighbourhood but never on the
    target (median retrieval rank ~46 of 151k on Qwen3-0.6B), because the pooled
    vector lives in the *average* of the pieces' directions while a real
    composite embedding has its own. A single linear operator fitted on the
    donor's own merge table -- ``W @ pooled ~= parent`` by ridge regression over
    ~80k composites -- is the cheapest thing that can learn that systematic
    offset. It is a one-layer stand-in for ZeTT's hypernetwork and costs a
    minute of CPU rather than a training run.

    The fit is in the donor's embedding space, so it applies before any width
    selection and transfers to any composition arity.
    """
    g = torch.Generator().manual_seed(seed)
    trip = torch.tensor(list(triples), dtype=torch.long)
    if trip.shape[0] > n_fit:
        trip = trip[torch.randperm(trip.shape[0], generator=g)[:n_fit]]
    e = donor_emb.float()
    x = first_w * e[trip[:, 1]] + (1.0 - first_w) * e[trip[:, 2]]
    y = e[trip[:, 0]]
    d = x.shape[1]
    xtx = x.T @ x
    lam = ridge * float(xtx.diagonal().mean())
    eye = torch.eye(d)
    xtx = xtx + lam * eye
    w = torch.linalg.solve(xtx, x.T @ y + lam * eye).T  # [d, d], applied as W @ pooled
    if shrink > 0:
        w = (1.0 - shrink) * w + shrink * eye
    return w
