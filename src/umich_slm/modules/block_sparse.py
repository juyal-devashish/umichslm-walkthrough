"""Trainable block-sparse long-range attention: a SUBSTITUTIVE alternative to scribes.

WHY THIS EXISTS (2026-08-20). The scribe bank buys long-range content addressing
ADDITIVELY: every carrying layer pays the window branch *plus* C x W bank keys,
and C = S / scribe_chunk_size grows with the sequence. Measured, that is 3.60x
dense at 32K against 1.15x for the minimal-bank sinkdiet. Block-sparse selection
is SUBSTITUTIVE instead -- each query block attends to k selected key blocks of
real history, so the far-field cost is k * bsa_block_size keys REGARDLESS of S.
The two mechanisms answer the same question ("what happened outside my window?")
with opposite cost curves in S.

TWO ARMS, one core. They differ only in how a block is scored and whether the
selector gets a gradient:

  bsa_mode="moba"   Parameter-less gate (MoonshotAI 2502.13189). A block's key
                    is the MEAN of its keys; the score is q_bar . k_bar. Hard
                    top-k, and the score never enters the softmax, so the
                    selector is trained only through which blocks it happens to
                    show the model -- there is no direct gradient to the gate.

  bsa_mode="hils"   Learned compression + retrieval scores IN the forward
                    (Tencent 2607.02980). The block key is a learned projection
                    of the pooled block, and the winning block's score is ADDED
                    to every logit in that block, so the joint softmax sees it
                    and the LM loss differentiates through the selector. Costs
                    one d x d projection per layer over S/bsa_block_size rows.

CAUSALITY, and why the LSE merge stays exact. A query at position p has its SWA
window over [p-W+1, p]. Block-sparse selection draws ONLY from [0, p-W] -- key
blocks that end strictly before the window begins. Two consequences:
  1. strictly causal by construction, no per-token bookkeeping;
  2. the far branch and the window branch are DISJOINT key sets, which is the
     precondition merge_lse_branches() needs to reconstruct the exact joint
     softmax (it is a regrouping of one sum, not a blend of two).
Queries in the same query block share a selection -- that is what makes the
pattern block-sparse and, just as importantly, amortises the routing decision
over bsa_block_size tokens. Our own measurements price a routing point at
~30 ms/stage with a large fixed term, so a per-TOKEN selector would repeat the
mistake that made mopr expensive.

Sharing a selection is only causal if the shared decision is itself causal, and
that takes TWO things, not one. Eligibility uses the first query in the block
(the strictest member) -- that part was right from the start. The SCORE must use
the first query too. The first implementation scored with the block MEAN of the
queries, which silently made the block's first token's selection depend on its
own future; a multi-position probe caught it leaking at 4 of 6 positions
(0.02-0.08 in logits) while a single-position probe at 40 or 64 passed. Both
halves now key off q[block_start].

Documents are gated per key, not per block: a block is eligible if it is early
enough, and cross-document keys inside it are masked at attention time. A block
that ends up fully masked for a query contributes lse = -inf, i.e. weight zero;
the window branch always has >= 1 key so no row is -inf across all branches.

IMPLEMENTATION. The flex path builds a [B,H,nq,nk] boolean selection and hands
it to FlexAttention as a BlockMask, so no keys are ever copied -- which matters,
because the naive gather materialises k * bsa_block_size keys PER QUERY BLOCK and
at 32K that is several times larger than just keeping the whole sequence. The
gather path is kept as the correctness reference and is what the tests compare
against; it is also the fallback when flex is unavailable.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e30


def _pool_blocks(x: torch.Tensor, block: int) -> Tuple[torch.Tensor, int, int]:
    """[B,H,S,D] -> ([B,H,nk,D] block means, nk, pad). Pads the tail with zeros."""
    b, h, s, d = x.shape
    nk = (s + block - 1) // block
    pad = nk * block - s
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    xb = x.view(b, h, nk, block, d)
    if pad:
        # Don't let zero-padding drag the tail block's mean toward 0.
        w = torch.ones(nk * block, device=x.device, dtype=x.dtype)
        w[s:] = 0.0
        w = w.view(1, 1, nk, block, 1)
        return (xb * w).sum(3) / w.sum(3).clamp_min(1.0), nk, pad
    return xb.mean(3), nk, pad


class BlockSparseSelector(nn.Module):
    """Scores key blocks for each query block and returns a top-k selection.

    Owns the only parameters the mechanism adds: for "hils", one d x d key
    projection and one d x d query projection over POOLED rows (S/block of
    them), which is why the parameter and FLOP cost is negligible next to the
    layer it rides on. "moba" adds nothing at all.
    """

    def __init__(self, config, head_dim: int):
        super().__init__()
        self.mode = str(getattr(config, "bsa_mode", "moba"))
        self.block = int(getattr(config, "bsa_block_size", 128))
        self.top_k = int(getattr(config, "bsa_top_k", 4))
        self.window = int(getattr(config, "swa_window", 1024))
        self.head_dim = head_dim
        if self.mode not in ("moba", "hils"):
            raise ValueError(f"bsa_mode must be 'moba' or 'hils', got {self.mode!r}")
        if self.mode == "hils":
            # Zero-init the OUTPUT of the score path so the retrieval bias starts
            # at 0 and the arm begins as plain top-k selection, then learns to
            # bias the softmax. Same Flamingo-safe discipline as the bank reads.
            self.k_pool = nn.Linear(head_dim, head_dim, bias=False)
            self.q_pool = nn.Linear(head_dim, head_dim, bias=False)
            self.score_gate = nn.Parameter(torch.zeros(1))
            nn.init.normal_(self.k_pool.weight, std=0.02)
            nn.init.normal_(self.q_pool.weight, std=0.02)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, doc_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
        """q,k: [B,H,S,D]. Returns (sel[B,H,nq,nk] bool, bias[B,H,nq,nk] or None, nq, nk)."""
        b, h, s, d = q.shape
        blk = self.block
        kbar, nk, _ = _pool_blocks(k, blk)  # [B,H,nk,D]
        # The query side is represented by the block's FIRST query, NOT its mean.
        # Pooling the queries leaks: the mean over a query block includes queries
        # at LATER positions, so the selection shown to the block's first token
        # would depend on its own future. Measured -- with the mean, a
        # perturbation after position p changed logits at <= p at 4 of 6 probe
        # positions (0.02-0.08). The first query is the same rule chunk_mop uses
        # for its lane commitment, and it is what the eligibility test below
        # already assumes. Trade-off: a weaker selection signal than the mean,
        # bought for exact causality.
        nq = (s + blk - 1) // blk
        qpad = nq * blk - s
        qf = F.pad(q, (0, 0, 0, qpad)) if qpad else q
        qbar = qf.view(b, h, nq, blk, d)[:, :, :, 0]  # [B,H,nq,D]
        # AUTOCAST MUST BE OFF FOR THE WHOLE SCORE PATH (2026-08-21).
        #
        # `torch.einsum` and `nn.Linear` are autocast-eligible: under autocast
        # they return bf16 *even when handed fp32 inputs*, so the dtype of
        # everything below is a property of the surrounding autocast state, not
        # of this code. Under gradient checkpointing that made `bias` -- the one
        # float tensor this function hands back -- a tensor whose dtype could
        # differ between the forward and the recompute, which torch reports as
        #     CheckpointError: Recomputed values ... have different metadata
        # and which took down every swa_hils / swa_hils_deltamem cell at batch
        # 2/4/8 in gate T (it survived b16 only by luck of which frames
        # compiled). `bsa_mode="moba"` was unaffected for the revealing reason
        # that it returns bias=None and a bool `sel`: bool dtype cannot drift,
        # so the same latent instability was invisible there.
        #
        # Forcing fp32 here also makes the top-k reproducible run to run, which
        # is what the selection has to be for the recompute to agree at all.
        # Cost is negligible: these are [B,H,S/blk,D] pooled rows, 1/blk of the
        # tokens.
        with torch.autocast(device_type=q.device.type, enabled=False):
            if self.mode == "hils":
                # F.linear with BOTH sides cast, not self.k_pool(x.float()): under
                # a bf16 model the module's weight is bf16, so casting only the
                # input raises "expected mat1 and mat2 to have the same dtype".
                # Caught by tests/test_masking_flags_matrix.py the same day the
                # autocast fix landed -- it would have taken down every hils and
                # bsa_switch cell in the 32K gate.
                kbar = F.linear(kbar.float(), self.k_pool.weight.float())
                qbar = F.linear(qbar.float(), self.q_pool.weight.float())
            scores = torch.einsum("bhqd,bhkd->bhqk", qbar.float(), kbar.float()) * (d**-0.5)

        # Eligibility: key block must END strictly before the window of the
        # FIRST query in the query block. first_q = qi*blk, window starts at
        # first_q - W + 1, so we need (ki+1)*blk - 1 < first_q - W + 1.
        qi = torch.arange(nq, device=q.device).view(nq, 1)
        ki = torch.arange(nk, device=q.device).view(1, nk)
        eligible = ((ki + 1) * blk) <= (qi * blk - self.window + 1)  # [nq,nk]
        eligible = eligible.view(1, 1, nq, nk)

        if doc_ids is not None:
            # A block is worth selecting only if it shares a document with the
            # query block. Per-KEY doc masking still happens at attention time;
            # this only avoids spending a slot on a block that would be fully
            # masked. Block doc-set is approximated by its first and last token.
            dpad = F.pad(doc_ids, (0, nk * blk - s), value=-1) if nk * blk > s else doc_ids
            dblk = dpad.view(b, nk, blk)
            qd = F.pad(doc_ids, (0, nq * blk - s), value=-1) if nq * blk > s else doc_ids
            qd_first = qd.view(b, nq, blk)[:, :, 0]  # [B,nq]
            same = (dblk.unsqueeze(1) == qd_first.view(b, nq, 1, 1)).any(-1)  # [B,nq,nk]
            eligible = eligible & same.unsqueeze(1)

        scores = scores.masked_fill(~eligible, NEG)
        kk = min(self.top_k, nk)
        idx = scores.topk(kk, dim=-1).indices  # [B,H,nq,kk]
        sel = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, idx, True)
        sel = sel & eligible  # a block padded to NEG can still be picked when k > eligible count

        bias = None
        if self.mode == "hils":
            # The retrieval score enters the softmax, so the LM loss trains the
            # selector directly instead of only through which blocks it revealed.
            # Kept in fp32 for the same reason as the scores above; the flex
            # score_mod adds it to fp32 attention logits anyway.
            with torch.autocast(device_type=q.device.type, enabled=False):
                bias = (scores.float() * self.score_gate.float()).masked_fill(~sel, 0.0)
        return sel, bias, nq, nk


def block_sparse_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sel: torch.Tensor,
    bias: Optional[torch.Tensor],
    block: int,
    doc_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dense-mask reference for the far branch. Returns (out[B,S,H,D], lse[B,H,S]).

    O(S^2) and used for tests / small S only -- the flex path is the real one.
    Rows with no selected key return lse = -inf and an all-zero output, which
    merge_lse_branches() turns into weight zero.
    """
    b, h, s, d = q.shape
    nq, nk = sel.shape[-2], sel.shape[-1]
    # [B,H,nq,nk] -> [B,H,S,S'] then crop to S
    m = sel.repeat_interleave(block, dim=2).repeat_interleave(block, dim=3)
    m = m[:, :, :s, :s]
    add = None
    if bias is not None:
        add = bias.repeat_interleave(block, dim=2).repeat_interleave(block, dim=3)[:, :, :s, :s]
    if doc_ids is not None:
        m = m & (doc_ids.view(b, 1, s, 1) == doc_ids.view(b, 1, 1, s))
    # Strict causality is already implied by eligibility, but assert it cheaply:
    causal = torch.arange(s, device=q.device).view(s, 1) >= torch.arange(s, device=q.device).view(
        1, s
    )
    m = m & causal.view(1, 1, s, s)

    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * (d**-0.5)
    if add is not None:
        scores = scores + add
    scores = scores.masked_fill(~m, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # [B,H,S]; -inf where nothing selected
    p = torch.softmax(scores, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)  # all -inf row -> uniform NaN -> 0
    out = torch.einsum("bhqk,bhkd->bhqd", p, v.float())
    return out.permute(0, 2, 1, 3).to(v.dtype), lse


def block_sparse_flex(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sel: torch.Tensor,
    bias: Optional[torch.Tensor],
    block: int,
    doc_ids: Optional[torch.Tensor] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """FlexAttention far branch: no key copies. Returns (out[B,S,H,D], lse[B,H,S]) or None.

    Returns None if flex is unavailable or the shapes do not satisfy its block
    constraints, so the caller can fall back to the reference path rather than
    silently changing the model.
    """
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    except Exception:
        return None
    b, h, s, d = q.shape
    if s % block != 0:
        return None

    sel_i = sel  # [B,H,nq,nk] bool, indexed by BLOCK
    docs = doc_ids

    def mask_mod(bi, hi, qi, ki):  # noqa: ANN001 - flex passes int tensors
        ok = sel_i[bi, hi, qi // block, ki // block]
        ok = ok & (qi >= ki)
        if docs is not None:
            ok = ok & (docs[bi, qi] == docs[bi, ki])
        return ok

    def score_mod(score, bi, hi, qi, ki):  # noqa: ANN001
        return score + bias[bi, hi, qi // block, ki // block]

    try:
        bm = create_block_mask(mask_mod, b, h, s, s, device=q.device, BLOCK_SIZE=block)
        out, lse = flex_attention(
            q,
            k,
            v,
            block_mask=bm,
            score_mod=(score_mod if bias is not None else None),
            return_lse=True,
        )
    except Exception:
        return None
    return out.permute(0, 2, 1, 3), lse
