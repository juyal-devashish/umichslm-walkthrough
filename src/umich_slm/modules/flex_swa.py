"""FlexAttention backend for sliding-window attention with persistent globals.

This realises the *globals + sliding-window* attention pattern (see
``swa_with_globals.py`` and ``GlobalTokens.build_attention_mask``) as a
:func:`torch.nn.attention.flex_attention.flex_attention` call driven by a
``BlockMask``. FlexAttention compiles a fused (Triton) kernel and, via the
block mask, **skips the fully-masked score blocks** — so the work is
O(seq · (window + globals)) on a flash-style kernel instead of the O(seq²)
dense-mask SDPA path the module falls back to today.

Why this and not a hand-written Triton kernel: FlexAttention *is* the
compiler that emits an optimised Triton kernel for an arbitrary ``mask_mod``;
for a globals+window mask it produces essentially the kernel one would hand-roll
(block-sparse flash), with the block-skipping and backward pass handled for us.
The module docstring in ``swa_with_globals.py`` anticipated exactly this switch.

Availability is probed at import; when FlexAttention (or CUDA/Triton) is absent
the caller keeps its SDPA fallback, so CPU / older-torch paths are unaffected.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

_FLEX_OK = False
try:  # torch >= 2.5
    from torch.nn.attention.flex_attention import (
        create_block_mask as _create_block_mask,
        flex_attention as _flex_attention,
    )

    _FLEX_OK = True
except Exception:  # pragma: no cover - depends on torch build
    _create_block_mask = None
    _flex_attention = None


def flex_available() -> bool:
    """True when FlexAttention is importable in this torch build."""
    return _FLEX_OK


# Compile the attention op once. Compiling *only* this callable yields the fused
# kernel while leaving the surrounding model eager (torch_compile is off on the
# titan path to match the HF path). ``dynamic=False`` since our shapes are fixed.
_compiled_flex: Optional[Callable] = None
_compiled_flex_dynamic: Optional[bool] = False


def set_flex_compile_dynamic(dynamic: Optional[bool]) -> None:
    """Choose the ``dynamic=`` of the standalone compiled flex op (default False).

    Training feeds fixed shapes, so static specialization is right there. Eval is
    not: the capability probes run sequences of many lengths, and a bank arm's
    pathway lanes gather a data-dependent width on top. Each new shape is a
    recompile, and past Dynamo's recompile limit (8) the frame runs EAGER for the
    rest of the process -- which for flex means the unfused path that materializes
    the full scores matrix. At 32K that is a 14.7 GiB allocation, and it OOMed every
    bank arm's long probes on a 44 GB L40S (GL 63934335). ``None`` lets Dynamo mark
    the changing dims dynamic after the first recompile, so one kernel serves every
    length. Call before the first flex call; it drops any cached compile.
    """
    global _compiled_flex, _compiled_flex_dynamic
    _compiled_flex_dynamic = dynamic
    _compiled_flex = None


def _get_compiled_flex() -> Callable:
    global _compiled_flex
    if _compiled_flex is None:
        _compiled_flex = torch.compile(_flex_attention, dynamic=_compiled_flex_dynamic)
    return _compiled_flex


def _flex_inline_enabled() -> bool:
    """Whether flex_sdpa lets an ENCLOSING torch.compile frame capture the raw
    flex op (no graph break) instead of always calling the separately-compiled
    wrapper. Env-gated for A/B: UMICH_FLEX_INLINE=0 restores the break."""
    import os

    return os.environ.get("UMICH_FLEX_INLINE", "1") != "0"


def make_swa_globals_mask_mod(num_globals: int, window: int) -> Callable:
    """Return a FlexAttention ``mask_mod`` for globals + sliding window.

    Absolute-index convention (matches ``GlobalTokens.build_attention_mask``):
    positions ``0..num_globals-1`` are globals, ``num_globals..`` are content.

      * a global query (row < N) attends causally to earlier positions
        (which are all globals, since globals lead the sequence);
      * a content query attends to (a) every global key and (b) content keys
        within ``window`` positions back (causal).
    """
    n = int(num_globals)
    w = int(window)

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ANN001 - flex passes int tensors
        is_global_q = q_idx < n
        is_global_k = kv_idx < n
        causal = kv_idx <= q_idx
        # Global query: causal among all (only earlier globals satisfy kv<=q<N).
        global_rule = is_global_q & causal
        # Content query: any global key, or a content key inside the window.
        dist = q_idx - kv_idx
        in_window = (~is_global_k) & (dist >= 0) & (dist < w)
        content_rule = (~is_global_q) & (is_global_k | in_window)
        return global_rule | content_rule

    return mask_mod


def make_swa_globals_doc_mask_mod(num_globals: int, window: int, doc_ids: torch.Tensor) -> Callable:
    """Doc-aware variant of :func:`make_swa_globals_mask_mod`.

    ``doc_ids`` is the model's ``full_doc_ids`` tensor ``[B, S]`` — document ids
    over the FULL sequence including the global prefix (prefix rows carry
    ``PREFIX_ID``; their value is irrelevant here because global keys are
    admitted by the ``is_global_k`` rule regardless of document).

    Semantics (matches the SDPA additive path: globals+window ∧ intra-doc):
      * globals are per-sequence state, visible to content in EVERY document;
      * content→content additionally requires ``same document`` — the sliding
        window never crosses a packed-document boundary.
    """
    n = int(num_globals)
    w = int(window)
    docs = doc_ids

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ANN001 - flex passes int tensors
        is_global_q = q_idx < n
        is_global_k = kv_idx < n
        causal = kv_idx <= q_idx
        global_rule = is_global_q & causal
        dist = q_idx - kv_idx
        same_doc = docs[b, q_idx] == docs[b, kv_idx]
        in_window = (~is_global_k) & (dist >= 0) & (dist < w) & same_doc
        content_rule = (~is_global_q) & (is_global_k | in_window)
        return global_rule | content_rule

    return mask_mod


# BlockMask depends only on (q_len, kv_len, num_globals, window, device); for a
# fixed-shape training run it is identical across every SWA layer and every step,
# so build it once and share. Keyed cache amortises the (non-trivial) build.
_BLOCK_MASK_CACHE: Dict[Tuple, object] = {}


def make_causal_doc_mask_mod(
    doc_ids: Optional[torch.Tensor],
    q_pos: Optional[torch.Tensor] = None,
    window: int = 0,
) -> Callable:
    """``mask_mod`` for plain causal attention, optionally ∧ intra-document.

    Used by the MoP lane blocks (``modules/pathways.py``), which cross-attend to
    the full shared-stream export: keys span the whole sequence, so the only
    constraints are causality and (when packing documents) same-document.

    ``q_pos`` remaps the query axis for the GATHERED lane path: row ``q_idx`` of
    the compact query tensor carries the token whose ORIGINAL position is
    ``q_pos[b, q_idx]``, so causality and the document lookup must both be
    evaluated at that original position rather than at the compact index. Pass
    ``None`` when queries are the full sequence (index == position).

    ``window`` > 0 additionally bands attention to that many positions back,
    matching the backbone's SWA band; 0 leaves the lanes full-context.

    Padding rows of a ragged gather point at some real (unselected) position, so
    every query row admits at least its own key and no row is fully masked —
    flex would emit NaN for an empty row. The caller discards those rows.
    """
    docs = doc_ids
    qp = q_pos
    w = int(window)

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ANN001 - flex passes int tensors
        pos_q = q_idx if qp is None else qp[b, q_idx]
        allowed = kv_idx <= pos_q
        if w > 0:
            allowed = allowed & ((pos_q - kv_idx) < w)
        if docs is not None:
            allowed = allowed & (docs[b, pos_q] == docs[b, kv_idx])
        return allowed

    return mask_mod


def make_interval_mask_mod(
    lo: torch.Tensor, hi: torch.Tensor, bank_start: Optional[int] = None
) -> Callable:
    """``mask_mod`` admitting kv columns inside a per-query-row interval.

    Under packed documents the MoP-lane predicate ``causal ∧ same-document ∧
    window`` collapses to a CONTIGUOUS kv interval per query row:

        kv ∈ [ max(doc_start(pos_q), pos_q - w + 1), pos_q ]

    because documents occupy contiguous position runs (the global prefix
    included — its rows lead the sequence). ``lo``/``hi`` are ``[B, k]``
    precomputed per (possibly gathered) query row. Two coalesced gathers per
    evaluation, versus the general predicate's THREE including a
    double-indirect document lookup (``docs[b, qp[b, q_idx]]``) — mask_mod
    runs inside the fwd AND bwd flex kernels on every partial block, so its
    cost is kernel inner-loop cost.
    """
    lo_t = lo
    hi_t = hi

    if bank_start is None:

        def mask_mod(b, h, q_idx, kv_idx):  # noqa: ANN001 - flex passes int tensors
            return (kv_idx >= lo_t[b, q_idx]) & (kv_idx <= hi_t[b, q_idx])

    else:
        # pathway_bank_export: kv columns >= bank_start are appended bank rows
        # — position-free and visible to every query row.
        bank_t = int(bank_start)

        def mask_mod(b, h, q_idx, kv_idx):  # noqa: ANN001
            return (kv_idx >= bank_t) | (
                (kv_idx >= lo_t[b, q_idx]) & (kv_idx <= hi_t[b, q_idx])
            )

    return mask_mod


def doc_segment_starts(doc_ids: torch.Tensor) -> torch.Tensor:
    """``[B, S]`` start position of each token's document segment.

    Requires only that equal doc ids form contiguous runs per row (packed
    layout) — id VALUES may be arbitrary. Position 0 always starts a segment.
    """
    b, s = doc_ids.shape
    idx = torch.arange(s, device=doc_ids.device)
    boundary = torch.ones(b, s, dtype=torch.bool, device=doc_ids.device)
    boundary[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
    marker = torch.where(boundary, idx.expand(b, s), idx.new_zeros(1))
    return torch.cummax(marker, dim=1).values


def get_interval_block_mask(
    lo: torch.Tensor,
    hi: torch.Tensor,
    kv_len: int,
    step_cache: Optional[dict] = None,
    cache_key: Optional[Tuple] = None,
    bank_start: Optional[int] = None,
):
    """BlockMask from per-row [lo, hi] kv intervals (see make_interval_mask_mod).

    Depends on this step's routing/document layout, so it goes through the
    caller's per-forward ``step_cache`` (gradient-checkpoint recompute reuses
    the identical object)."""
    if step_cache is not None and cache_key is not None:
        bm = step_cache.get(cache_key)
        if bm is not None:
            return bm
    bm = _create_block_mask(
        make_interval_mask_mod(lo, hi, bank_start=bank_start),
        B=int(lo.shape[0]),
        H=None,
        Q_LEN=int(lo.shape[1]),
        KV_LEN=int(kv_len),
        device=lo.device,
        _compile=True,
    )
    if step_cache is not None and cache_key is not None:
        step_cache[cache_key] = bm
    return bm


def make_mod_compact_mask_mod(
    positions: torch.Tensor,
    doc_ids: Optional[torch.Tensor] = None,
    num_globals: int = 0,
    window: int = 0,
    valid: Optional[torch.Tensor] = None,
    extra_kv: int = 0,
) -> Callable:
    """``mask_mod`` for a block-atomic MoD block running on a GATHERED sequence.

    Unlike :func:`make_causal_doc_mask_mod` (MoP lanes, where keys span the full
    sequence so ``kv_idx`` is already an original position), the MoD compact path
    is SELF-attention among the routed tokens: **both** axes are compact, so both
    must be remapped through ``positions`` before any position test.

    ``positions`` is ``[B, k]`` -- the original position of each gathered token,
    ascending per row. ``doc_ids`` is the matching ``[B, k]`` gather of the
    document ids (``PREFIX_ID`` on prefix rows), or ``None`` when documents are
    not packed.

    Visibility, matching the dense (non-MoD) path for the same layer:
      * causal in ORIGINAL position space;
      * ``window > 0`` bands content to that many positions back, with prefix
        globals (original position < ``num_globals``) exempt -- they are
        per-sequence state, attendable from anywhere. ``window == 0`` means no
        band, i.e. the full-context MLA member of the block;
      * with ``doc_ids``, content may only see its own document; ``PREFIX_ID``
        rows are visible across all documents (same rule as ``build_doc_mask``).

    No row can be fully masked: ``kv_idx == q_idx`` always yields the token's own
    position, which is causal, at distance 0, and trivially same-document -- so
    flex never sees an empty row (which would produce NaN).

    ``extra_kv`` covers out-of-sequence bank keys the attention layer appends
    AFTER the compact rows (swa_scribe's sink bank via ``_extend_bank_kv``):
    trailing ``extra_kv`` key columns are visible to every query, exactly like
    the zero (fully-open) mask columns the SDPA path appends. Sink tokens are
    learned parameters — content-free, so cross-document visibility is by
    design, matching the dense path.
    """
    from .doc_masking import PREFIX_ID

    pos = positions
    docs = doc_ids
    vld = valid
    n = int(num_globals)
    w = int(window)
    k_real = int(positions.shape[1])
    has_bank = int(extra_kv) > 0

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ANN001 - flex passes int tensors
        # Bank columns sit past the compact rows; clamp so the gathers below
        # stay in-bounds (their result is unused for bank columns).
        is_bank = kv_idx >= k_real if has_bank else None
        kv_idx = kv_idx.clamp(max=k_real - 1) if has_bank else kv_idx
        pq = pos[b, q_idx]
        pk = pos[b, kv_idx]
        causal = pk <= pq
        if docs is not None:
            # Position < num_globals does NOT identify a global once
            # build_intra_doc_position_ids has reset each document to 0 -- the
            # first n tokens of every document would qualify. Use the id.
            dq = docs[b, q_idx]
            dk = docs[b, kv_idx]
            is_global_q = dq == PREFIX_ID
            is_global_k = dk == PREFIX_ID
            same_doc = dq == dk
        else:
            is_global_q = pq < n
            is_global_k = pk < n
            same_doc = causal | ~causal  # all-True, same dtype/shape as the others
        # Global query sees only globals (see build_mod_compact_masks); content
        # sees every global plus its own document inside the band.
        in_band = causal & same_doc & ((pq - pk) < w) if w > 0 else causal & same_doc
        allowed = (is_global_q & is_global_k & causal) | (
            (~is_global_q) & (is_global_k | in_band)
        )
        if vld is not None:
            # Padding slots of a ragged gather are invisible as keys; the diagonal
            # is re-admitted so no row is empty (an empty row is NaN under flex).
            allowed = (allowed & vld[b, kv_idx]) | (q_idx == kv_idx)
        if is_bank is not None:
            allowed = allowed | is_bank
        return allowed

    return mask_mod


def get_mod_compact_block_mask(
    positions: torch.Tensor,
    doc_ids: Optional[torch.Tensor],
    num_globals: int,
    window: int,
    step_cache: Optional[dict] = None,
    cache_key: Optional[Tuple] = None,
    valid: Optional[torch.Tensor] = None,
    extra_kv: int = 0,
):
    """BlockMask for a gathered MoD block. Depends on this step's routing.

    ``positions`` changes every step (the router picks different tokens), so this
    can never use the process-wide shape cache -- it goes through the caller's
    per-forward ``step_cache``, which also makes gradient-checkpoint recompute
    reuse the identical object instead of rebuilding it.
    """
    if step_cache is not None and cache_key is not None:
        bm = step_cache.get(cache_key)
        if bm is not None:
            return bm
    k = int(positions.shape[1])
    bm = _create_block_mask(
        make_mod_compact_mask_mod(positions, doc_ids, num_globals, window, valid, extra_kv),
        B=int(positions.shape[0]),
        H=None,
        Q_LEN=k,
        KV_LEN=k + int(extra_kv),
        device=positions.device,
        _compile=True,
    )
    if step_cache is not None and cache_key is not None:
        step_cache[cache_key] = bm
    return bm


def get_causal_doc_block_mask(
    doc_ids: Optional[torch.Tensor],
    q_len: int,
    kv_len: int,
    device: torch.device,
    q_pos: Optional[torch.Tensor] = None,
    batch: Optional[int] = None,
    step_cache: Optional[dict] = None,
    cache_key: Optional[Tuple] = None,
    window: int = 0,
):
    """BlockMask for causal ∧ intra-doc attention over a full or gathered query axis.

    With ``doc_ids=None`` and ``q_pos=None`` the pattern depends only on shape, so
    it lands in the process-wide shape cache. Otherwise it depends on this step's
    document layout (and, for the gathered path, this step's routing), so it must
    go through the caller's per-forward ``step_cache`` — which is also what makes
    gradient-checkpoint recompute reuse the identical object instead of rebuilding.
    """
    if doc_ids is None and q_pos is None:
        key = ("causal", int(q_len), int(kv_len), int(window), str(device))
        bm = _BLOCK_MASK_CACHE.get(key)
        if bm is None:
            bm = _create_block_mask(
                make_causal_doc_mask_mod(None, None, window),
                B=None,
                H=None,
                Q_LEN=int(q_len),
                KV_LEN=int(kv_len),
                device=device,
                _compile=True,
            )
            _BLOCK_MASK_CACHE[key] = bm
        return bm

    if step_cache is not None and cache_key is not None:
        bm = step_cache.get(cache_key)
        if bm is not None:
            return bm
    bm = _create_block_mask(
        make_causal_doc_mask_mod(doc_ids, q_pos, window),
        B=int(batch) if batch is not None else None,
        H=None,
        Q_LEN=int(q_len),
        KV_LEN=int(kv_len),
        device=device,
        _compile=True,
    )
    if step_cache is not None and cache_key is not None:
        step_cache[cache_key] = bm
    return bm


def get_block_mask(
    q_len: int,
    kv_len: int,
    num_globals: int,
    window: int,
    device: torch.device,
):
    """Return a cached ``BlockMask`` for the globals+window pattern."""
    key = (int(q_len), int(kv_len), int(num_globals), int(window), str(device))
    bm = _BLOCK_MASK_CACHE.get(key)
    if bm is None:
        mask_mod = make_swa_globals_mask_mod(num_globals, window)
        # B=None, H=None -> the mask broadcasts across batch and heads.
        bm = _create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=int(q_len),
            KV_LEN=int(kv_len),
            device=device,
            _compile=True,
        )
        _BLOCK_MASK_CACHE[key] = bm
    return bm


def get_doc_block_mask(
    doc_ids: torch.Tensor,
    num_globals: int,
    window: int,
    step_cache: Optional[dict] = None,
):
    """Return a BlockMask for globals+window ∧ intra-doc masking.

    Unlike :func:`get_block_mask` this depends on the batch's document layout,
    so it CANNOT live in the global shape-keyed cache — doc boundaries change
    every step. ``step_cache`` is a per-forward dict created by the model and
    shared across all SWA layers, so the (non-trivial) build runs once per
    training step instead of once per layer; gradient-checkpoint recompute hits
    the same dict and reuses the mask.
    """
    q_len = kv_len = int(doc_ids.shape[1])
    key = (q_len, kv_len, int(num_globals), int(window))
    if step_cache is not None:
        bm = step_cache.get(key)
        if bm is not None:
            return bm
    mask_mod = make_swa_globals_doc_mask_mod(num_globals, window, doc_ids)
    bm = _create_block_mask(
        mask_mod,
        B=int(doc_ids.shape[0]),
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=doc_ids.device,
        _compile=True,
    )
    if step_cache is not None:
        step_cache[key] = bm
    return bm


def flex_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Run FlexAttention over [B, H, S, D] q/k/v with a prebuilt ``block_mask``.

    With ``enable_gqa=False`` (default), ``key``/``value`` are already
    GQA-expanded to the query head count by the caller. ``enable_gqa=True``
    accepts native kv-head-count K/V and groups q heads in-kernel (no
    repeat_interleave materialization). Default scale is ``1/sqrt(head_dim)``,
    matching the SDPA path.

    Inside a ``torch.compile`` frame the RAW op is used so the enclosing graph
    captures and fuses it — calling the separately-compiled wrapper from a
    compiled frame forces a graph break at every flex call site.
    (``UMICH_FLEX_INLINE=0`` restores the always-break behavior for A/B.)
    """
    inline = _flex_inline_enabled() and torch.compiler.is_compiling()
    fn = _flex_attention if inline else _get_compiled_flex()
    return fn(query, key, value, block_mask=block_mask, enable_gqa=enable_gqa)
