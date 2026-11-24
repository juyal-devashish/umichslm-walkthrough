"""Intra-document masking primitives (W2, LONG_CONTEXT_PLAN.md §4).

When several documents are packed into one fixed-length training window, a token in document
B must not attend to document A, and RoPE positions must **reset** at each document boundary.
Otherwise the model learns spurious cross-document dependencies and — critically for
long-context — never sees genuine long-range structure (the far positions are polluted by
unrelated docs). This is a correctness fix that also underpins the long-context work.

The signal is a per-token ``document_ids`` tensor ``[batch, seq_len]``: tokens sharing an id
belong to the same packed document. A sentinel id ``PREFIX_ID = -1`` marks per-sequence
prefix tokens (persistent globals / meta / attention sinks) that must stay visible to and from
every document (globals are per-sequence by default — see the plan).

These builders are pure and CPU-testable. The model folds the additive mask into its 4D
attention mask (SDPA/eager) and resets ``position_ids`` per document; DeltaNet consumes the
reset signal to zero its recurrent state at boundaries. A FlashAttention **varlen** path can
be built from ``cu_seqlens_from_doc_ids`` for throughput at 32K (GPU-only; not exercised here).
"""

from __future__ import annotations

import os

import torch

# Sentinel document id for per-sequence prefix tokens (globals / meta): visible everywhere.
PREFIX_ID = -1

# torch.compile guard hygiene for the varlen path (2026-08-24).
#
# ``cu_seqlens`` is handed to the attention layers as a MODULE ATTRIBUTE, so Dynamo
# turns it into a guarded graph input: ``tensor '..._intra_doc_cu_seqlens' size
# mismatch at index 0``. Its length is (#documents in the batch + 1), which changes
# EVERY step under packed data -- so every new document count is a fresh recompile of
# ``TransformerBlock.forward``. All blocks share that one code object, so the cache
# holds (block variants x doc counts) entries; past the limit Dynamo abandons the
# frame and runs it EAGER, silently (``suppress_errors``). Measured cost on the ar10m
# cells: dense ~16% of steps eager, swa_sinkdiet ~52%, mat_ffn a majority -- and the
# penalty scales with how many block variants an arch has (qwen3_like 1 graph,
# swa_sinkdiet 2, mat_ffn 5, think_suffix 11), i.e. it reads as an ARCHITECTURE result.
#
# ``max_seqlen`` is a python int on the same path, so its VALUE is specialized too --
# a second, unbounded axis. It is only an upper bound for FA2's tile grid (tiles past
# the true extent retire early), so rounding it UP is numerically inert and collapses
# the axis to ceil(S / bucket) values.
#
# The bucket is FA2's row-block size, which makes the rounding free as well as inert:
# the varlen grid is ceil(max_seqlen / BLOCK_M) row-blocks, so rounding max_seqlen up
# to a multiple of BLOCK_M leaves that count IDENTICAL -- no extra blocks are launched,
# and at S=4096 the axis drops from unbounded to at most 32 values.
_MAX_SEQLEN_BUCKET = 128

# ...but ``ceil(S / 128)`` is 32 values only at S=4096. It scales LINEARLY with the
# sequence, so the same reasoning that fixed 4096 leaves 32768 with 256 values -- past
# the 128-entry limit on the max_seqlen axis ALONE, before multiplying by an arch's
# block variants. That is exactly what happened on Blackwell (job 59164519): both
# qwen3_like and mat_ffn_thresh_gqa printed the recompile-limit warning at S=32768 and
# neither did at S=4096. Dense survived it -- one block variant degrades to eager and
# still computes the same function -- while mat_ffn_thresh_gqa (5 variants) died in the
# gradient-checkpoint recompute: once Dynamo abandons the frame, the activation pass and
# the recompute can run DIFFERENT code, and the saved-tensor list came back with
# positions 0 and 1 swapped ([896] norm weight against [2,32768,896] hidden states).
#
# So the axis has to be bounded by a CONSTANT, not by the sequence. The bucket now
# scales with S to hold it at `_MAX_SEQLEN_AXIS` values at any context.
#
# This is the one place the rounding stops being free. At bucket 128 it was exactly
# inert -- ceil(max_seqlen / BLOCK_M) was unchanged -- but a coarser bucket rounds
# max_seqlen up past the next row-block boundary, launching tile rows that retire
# immediately. Bounded and small: at most `bucket / BLOCK_M - 1` extra row-blocks
# against a grid of ceil(max_seqlen / BLOCK_M), and they do no work. Weigh that against
# the alternative, which is not "a slower kernel" but a MAJORITY OF STEPS IN EAGER
# (measured at ar10m: dense ~16%, swa_sinkdiet ~52%, mat_ffn a majority) plus a
# recompute that can crash the run outright.
#
# S=4096 is UNCHANGED by construction (4096 <= 128*32 keeps bucket == 128), which
# matters: every sweep cell to date ran at 4096 and none of them may move.
_MAX_SEQLEN_AXIS = 32


def _max_seqlen_bucket(seq_len: int) -> int:
    """Bucket width holding the ``max_seqlen`` guard to ``_MAX_SEQLEN_AXIS`` values."""
    span = _MAX_SEQLEN_BUCKET * _MAX_SEQLEN_AXIS
    mult = max(1, -(-int(seq_len) // span))
    return _MAX_SEQLEN_BUCKET * mult


def build_intra_doc_additive_mask(
    document_ids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Block-diagonal additive attention mask from per-token document ids.

    Args:
        document_ids: LongTensor ``[batch, seq_len]``. Tokens with equal ids may attend to
            each other; ``PREFIX_ID`` (-1) tokens attend to / are attended by everyone.
        dtype: floating dtype of the attention scores (mask uses ``finfo(dtype).min``).

    Returns:
        Additive mask ``[batch, 1, seq_len, seq_len]``: 0.0 where attention is allowed by the
        document structure, ``finfo(dtype).min`` where it crosses a document boundary. This is
        *added* to the causal/padding 4D mask (same convention as the SWA/globals masks), so it
        composes with them; causality is enforced by the mask it is added to.
    """
    if document_ids.dim() != 2:
        raise ValueError(f"document_ids must be [batch, seq_len]; got shape {document_ids.shape}")
    q = document_ids.unsqueeze(2)  # [B, S, 1]
    k = document_ids.unsqueeze(1)  # [B, 1, S]
    is_prefix = (q == PREFIX_ID) | (k == PREFIX_ID)
    same_doc = q == k
    allowed = same_doc | is_prefix  # [B, S, S]
    min_val = torch.finfo(dtype).min
    mask = torch.zeros_like(allowed, dtype=dtype)
    mask = mask.masked_fill(~allowed, min_val)
    return mask.unsqueeze(1)  # [B, 1, S, S]


def build_intra_doc_position_ids(document_ids: torch.Tensor) -> torch.Tensor:
    """Per-document position ids: RoPE positions reset to 0 at each document boundary.

    Assumes packed documents are contiguous runs of equal id (the packing invariant). Prefix
    (``PREFIX_ID``) tokens are treated as their own leading run and keep positions ``0..P-1``.

    Args:
        document_ids: LongTensor ``[batch, seq_len]`` of contiguous per-doc id runs.

    Returns:
        LongTensor ``[batch, seq_len]`` where each document's tokens are numbered from 0.
    """
    if document_ids.dim() != 2:
        raise ValueError(f"document_ids must be [batch, seq_len]; got shape {document_ids.shape}")
    B, S = document_ids.shape
    idx = torch.arange(S, device=document_ids.device).unsqueeze(0).expand(B, S)
    is_start = torch.ones_like(document_ids, dtype=torch.bool)
    is_start[:, 1:] = document_ids[:, 1:] != document_ids[:, :-1]
    # index of the most recent run start at or before each position
    run_start = torch.where(is_start, idx, torch.zeros_like(idx))
    run_start = torch.cummax(run_start, dim=1).values
    return (idx - run_start).long()


def build_doc_start_mask(document_ids: torch.Tensor) -> torch.Tensor:
    """Boolean ``[batch, seq_len]`` that is True at the first token of each document.

    DeltaNet (and any chunk-recurrent linear-attention layer) must **reset** its recurrent
    state at these positions so state does not leak across packed documents.
    """
    if document_ids.dim() != 2:
        raise ValueError(f"document_ids must be [batch, seq_len]; got shape {document_ids.shape}")
    is_start = torch.ones_like(document_ids, dtype=torch.bool)
    is_start[:, 1:] = document_ids[:, 1:] != document_ids[:, :-1]
    return is_start


def _compile_friendly(cu: torch.Tensor, max_seqlen: int, seq_len: int) -> tuple:
    """Strip the two per-step ``torch.compile`` specialization axes off a varlen pair.

    Marks ``cu``'s only dimension dynamic so Dynamo emits ONE graph for every document
    count instead of one per count, and rounds ``max_seqlen`` up to ``_MAX_SEQLEN_BUCKET``
    so its guarded python value takes at most ``ceil(seq_len / bucket)`` values. Both are
    numerically inert: ``cu`` is unchanged, and ``max_seqlen`` is an upper bound on the
    FA2 tile grid, never a mask extent (the segments in ``cu`` set that).

    See ``_MAX_SEQLEN_BUCKET`` for why this matters -- it is the difference between a
    compiled step and an eager one, for the majority of steps on multi-variant archs.
    """
    if os.environ.get("UMICHSLM_DISABLE_COMPILE_HYGIENE") == "1":
        # Escape hatch for IN-JOB A/B only (the repo requires paired benchmarks):
        # reproduces the pre-2026-08-24 behaviour so the recovered time can be
        # measured against the same GPU, process settings and data in one run.
        return cu, int(max_seqlen)
    bucket = _max_seqlen_bucket(seq_len)
    rounded = min(int(seq_len), -(-int(max_seqlen) // bucket) * bucket)
    max_seqlen = max(int(max_seqlen), rounded)
    if cu.dim() == 1 and cu.shape[0] > 1:
        try:
            torch._dynamo.mark_dynamic(cu, 0)
        except Exception:  # pragma: no cover - never let guard hygiene break a forward
            pass
    return cu, max_seqlen


def build_batched_cu_seqlens(document_ids: torch.Tensor) -> tuple:
    """Batched ``cu_seqlens`` + ``max_seqlen`` for ``flash_attn_varlen_func``.

    Flattens ``[batch, seq_len]`` document ids to the ``[batch*seq_len]`` layout FA2 varlen
    expects and returns the cumulative document boundaries across the whole (flattened) batch,
    **always splitting at row boundaries** so two rows never merge into one segment even if
    they reuse the same id value. This is the varlen alternative to the O(S²) additive mask:
    each ``[cu[i], cu[i+1])`` segment is attended causally and independently.

    Args:
        document_ids: LongTensor ``[batch, seq_len]`` of contiguous per-doc id runs. Must NOT
            contain ``PREFIX_ID`` — a per-sequence shared prefix cannot be a varlen segment
            (the caller routes prefix-bearing sequences to the additive-mask path instead).

    Returns:
        ``(cu_seqlens, max_seqlen)`` — int32 tensor ``[num_segments + 1]`` and a python int.
    """
    if document_ids.dim() != 2:
        raise ValueError(f"document_ids must be [batch, seq_len]; got shape {document_ids.shape}")
    B, S = document_ids.shape
    device = document_ids.device
    flat = document_ids.reshape(-1)  # [B*S]
    idx = torch.arange(B * S, device=device)
    row_start = (idx % S) == 0  # force a boundary at every row start
    change = torch.zeros(B * S, dtype=torch.bool, device=device)
    change[1:] = flat[1:] != flat[:-1]
    boundary = row_start | change
    starts = torch.nonzero(boundary, as_tuple=False).flatten()
    cu = torch.cat([starts, torch.tensor([B * S], device=device)]).to(torch.int32)
    max_seqlen = int((cu[1:] - cu[:-1]).max().item())
    return _compile_friendly(cu, max_seqlen, S)


def cu_seqlens_from_doc_ids(document_ids: torch.Tensor) -> torch.Tensor:
    """Cumulative sequence lengths for FlashAttention varlen, from a single-row doc-id vector.

    Args:
        document_ids: LongTensor ``[seq_len]`` (one packed sequence) of contiguous id runs.

    Returns:
        Int32 tensor ``[num_docs + 1]`` of run boundaries, e.g. ``[0, len_a, len_a+len_b, ...]``
        — the ``cu_seqlens`` argument of ``flash_attn_varlen_func``. (GPU varlen path; the
        SDPA/eager paths use the additive mask instead.)
    """
    if document_ids.dim() != 1:
        raise ValueError(
            f"cu_seqlens expects a 1-D [seq_len] doc-id vector; got {document_ids.shape}"
        )
    S = document_ids.shape[0]
    if S == 0:
        return torch.zeros(1, dtype=torch.int32, device=document_ids.device)
    change = document_ids[1:] != document_ids[:-1]
    starts = torch.nonzero(change, as_tuple=False).flatten() + 1
    bounds = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=document_ids.device),
            starts,
            torch.tensor([S], dtype=torch.long, device=document_ids.device),
        ]
    )
    return bounds.to(torch.int32)


def jitter_position_ids(position_ids: torch.Tensor, max_gap: int, frac: float = 0.5) -> torch.Tensor:
    """Randomized position ids for training (``mla_position_jitter``; arXiv 2606.23687).

    A random ``frac`` of the rows get ids rebuilt as a running sum of gaps drawn from
    ``[1, max_gap]``, restarting at 0 wherever the input id is 0 (a document start under
    intra-doc masking, else the row start). Order within a document is preserved, so
    causality and relative order are unchanged; only the distances the RoPE layers see
    are stretched, up to ~``max_gap`` x the training length. The other rows keep their
    contiguous ids, so the model still sees the eval-time geometry.
    """
    B, S = position_ids.shape
    dev = position_ids.device
    start = position_ids == 0
    gaps = torch.randint(1, max_gap + 1, (B, S), device=dev).masked_fill(start, 0)
    run = gaps.cumsum(dim=1)
    idx = torch.arange(S, device=dev).unsqueeze(0).expand(B, S)
    last_start = torch.cummax(torch.where(start, idx, torch.zeros_like(idx)), dim=1).values
    jittered = run - run.gather(1, last_start)
    rows = torch.rand(B, 1, device=dev) < frac
    return torch.where(rows, jittered, position_ids)
