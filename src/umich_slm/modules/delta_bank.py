"""Cumulative chunk-writes into the register bank (``use_delta_bank``).

The swa_deltabank arch is swa_hybrid32 plus exactly ONE mechanism: the register
bank becomes *stateful*. At one write layer the sequence is split into
``delta_chunk_size``-token chunks and scanned left to right; each chunk writes
into the bank through a zero-init cross-attention (bank slots are the queries,
chunk tokens the keys/values), producing a cumulative per-chunk bank state

    D_0 = 0,  D_{c+1} = D_c + write(norm(S_c), norm(chunk_c))
    S_c = B0 + budget * tanh(D_c / budget)

where ``budget = |dev_scale| * ||B0||_rms`` (``dev_scale`` learnable, init
``delta_state_clamp``).

BOUNDING THE STATE IS LOAD-BEARING, not a nicety. The first version accumulated
raw (``S_{c+1} = S_c + write(...)``) and the 2026-07-20 25m cell came in at
val_lm_loss 4.1045 vs 3.1855 for its own zero-init control — a 0.92-nat loss.
Measured on that checkpoint, ||S_c||/||B0|| was 737x after ONE chunk write and
3090x by chunk 7. The reason that is fatal rather than merely untidy: the delta
read reuses the SAME ``k_proj``/``v_proj`` as the static bank read
(``swa_registers._delta_read_merge``), so a 700x-scale state produces 700x
attention logits, the softmax over bank slots saturates into a hard argmax, and
the read degenerates. Nothing in the loss opposes the blow-up — ``state_norm``
makes the write scale-invariant in its own input, so the state's magnitude is a
free parameter. ``tanh`` is the identity near 0, so the zero-init contract
survives exactly (at init ``write`` outputs 0 => D_c == 0 => S_c == B0, bitwise
swa_hybrid32) and insertion is still gradual, but |S_c - B0| can never exceed
``budget``.

and a token in chunk c reads S_c — the state accumulated from chunks < c only.
That is the causality contract: the read can carry content *forward* across
chunks (the out-of-band channel every probed arch lacks, 2026-07-20 findings)
but never backward, and never within the writing chunk.

Zero-init contract: ``write`` has a zero-init output projection, so at init
S_c == B0 for every c and the model is bitwise a swa_hybrid32 model — the
Flamingo-safe insertion used by every bank read in this family.

Intra-doc masking: writes follow the ScribeBank ownership rule (the chunk's
last valid token's doc owns the chunk; only its tokens are written), and the
scan RESETS to B0 when chunk ownership changes doc, so state never crosses a
document boundary. The read side gates on the recorded per-chunk state doc and
falls back to B0 on mismatch (see ``RegisterReadMixin._register_read``).

The scan is sequential over chunks (C = ceil(T/512) ≈ 8 at 4096) with batched
attention inside each step — O(C · N · K) work, negligible next to the window.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .workspace_trunk import TrunkCrossAttention, get_rmsnorm_class


class DeltaBank(nn.Module):
    """Chunk-scan writer producing per-chunk register-bank states."""

    def __init__(self, config):
        super().__init__()
        h = config.hidden_size
        self.chunk_size = int(config.delta_chunk_size)
        self.top_k = int(getattr(config, "delta_write_topk", 0)) or None
        norm_class = get_rmsnorm_class(
            zero_centered=getattr(config, "use_zero_centered_rmsnorm", False)
        )
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.state_norm = norm_class(h, eps=eps)
        self.src_norm = norm_class(h, eps=eps)
        # Learnable deviation budget, in units of ||B0||_rms (see forward).
        self.dev_scale = nn.Parameter(
            torch.tensor(float(getattr(config, "delta_state_clamp", 1.0)))
        )
        # Zero-init write output: at init every state == B0 (see module doc).
        self.write = TrunkCrossAttention(config, zero_init_output=True)

    def forward(
        self, bank0: torch.Tensor, content: torch.Tensor, doc_ids: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """bank0: [N, d] (RegisterBank tokens, post-refine if any);
        content: [B, T, d] content-only hidden states at the write layer;
        doc_ids: [B, T] or None.

        Returns (states [B, C, N, d], state_docs [B, C] or None) where
        states[:, c] is what chunk-c readers see and state_docs[:, c] is the
        document whose writes it accumulates (gate + B0-fallback at read time).
        """
        b, t, h = content.shape
        k = self.chunk_size
        n_chunks = (t + k - 1) // k
        pad = n_chunks * k - t
        padded = F.pad(content, (0, 0, 0, pad)) if pad else content
        chunks = padded.view(b, n_chunks, k, h)
        valid = torch.ones(b, n_chunks * k, dtype=torch.bool, device=content.device)
        if pad:
            valid[:, t:] = False
        valid = valid.view(b, n_chunks, k)

        owner = None
        key_mask = valid
        if doc_ids is not None:
            padded_docs = F.pad(doc_ids, (0, pad), value=-1) if pad else doc_ids
            cdocs = padded_docs.view(b, n_chunks, k)
            # Last VALID token's doc owns the chunk; write only its tokens
            # (same rule as ScribeBank / WorkspaceTrunk._doc_mask_for_chunk).
            last_idx = valid.float().cumsum(-1).argmax(-1)  # [b, C]
            owner = cdocs.gather(-1, last_idx.unsqueeze(-1)).squeeze(-1)  # [b, C]
            key_mask = valid & (cdocs == owner.unsqueeze(-1))

        base = bank0.to(content.dtype).unsqueeze(0).expand(b, -1, -1)  # [b, N, d]
        # Deviation budget, expressed in units of the bank's OWN rms so the
        # written state can never leave the scale the shared k_proj/v_proj were
        # trained for (see the class docstring for the measured failure).
        b0_rms = bank0.detach().float().pow(2).mean().sqrt().clamp_min(1e-6)
        budget = (self.dev_scale.float().abs() * b0_rms).clamp_min(1e-6).to(content.dtype)
        dev = torch.zeros_like(base)
        state = base
        # Doc whose writes `state` currently accumulates. Pristine state (== B0)
        # is doc-neutral; recording the current chunk's owner lets its own-doc
        # tokens pass the read gate onto what is anyway B0.
        state_doc = owner[:, 0] if owner is not None else None
        states, state_docs = [], []
        for c in range(n_chunks):
            states.append(state)
            if owner is not None:
                state_docs.append(state_doc)
                # Reset BEFORE writing chunk c: accumulated state must never
                # cross a document boundary. (Readers of chunk c already got
                # the pre-reset state above — its doc gate blocks new-doc
                # tokens, while old-doc stragglers inside chunk c may still
                # legitimately read their own doc's accumulated state.)
                reset = (owner[:, c] != state_doc).view(b, 1, 1)
                dev = torch.where(reset, torch.zeros_like(dev), dev)
                state = base + budget * torch.tanh(dev / budget)
                state_doc = owner[:, c]
            update = self.write(
                self.state_norm(state),
                self.src_norm(chunks[:, c]),
                key_mask=key_mask[:, c],
                top_k=self.top_k,
            )
            # Accumulate in the UNBOUNDED deviation, then squash. tanh is the
            # identity near 0, so the zero-init contract is exact (dev == 0 =>
            # state == B0) and insertion stays gradual, while |state - B0| is
            # hard-bounded by `budget` no matter how many chunks are scanned.
            dev = dev + update
            state = base + budget * torch.tanh(dev / budget)
        out_states = torch.stack(states, dim=1)  # [B, C, N, d]
        out_docs = torch.stack(state_docs, dim=1) if state_docs else None
        return out_states, out_docs
