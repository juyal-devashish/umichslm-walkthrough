"""Bank-reconstruction loss: supervise the out-of-band channel directly.

WHY. The 2026-07-20 probe sweep found that NO arch in the bank family carries
content out of band: beyond window+bank reach every arm sat at floor/chance, and
the register deficit turned out to be a missing-sink effect rather than a recall
effect. The banks tie on val_lm_loss while not actually moving information
across chunks. Next-token prediction never forces them to — a local window plus
a sink explains almost every token, so there is no gradient pressure to write
anything durable into the bank.

This loss creates that pressure. For chunk c, take positions from an EARLIER
chunk and require them to be recoverable from the bank state S_c *alone* — no
window, no attention over the raw tokens:

    q_j   = query(learned slot + position embedding of j)
    ctx_j = CrossAttention(q_j, keys/values = S_c)
    loss  = CE(head(ctx_j), token_j)

S_c accumulates writes from chunks strictly before c (the DeltaBank causality
contract), so this is a legitimate objective and not a future leak. Because the
reconstruction path cannot see the local window at all, the only way to drive
the loss down is to store recoverable content in the bank — exactly the channel
the probes say is missing.

APPLICABILITY. This needs a bank that is actually WRITTEN. It is meaningful for
``use_delta_bank`` (per-chunk states) and for the scribe/workspace arms, and
meaningless for a static register bank such as swa_hybrid32, where S_c == B0 for
every c and the target is unpredictable by construction. So it is an arch-paired
ablation (deltabank vs deltabank+recon), not a universal aux task, and it must
never be reported as a like-for-like comparison against a static-bank arm.

COST. ``bank_recon_samples`` queries per chunk (default 32) against N bank slots,
plus one head application each: at 4096/chunk-512 that is 32*8 = 256 queries per
sequence against 4096 tokens, ~6% overhead.

DOC SAFETY. A sampled position is supervised only when its document matches the
document the state accumulated (``state_docs``), mirroring the read-side gate,
so reconstruction never crosses a document boundary.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rmsnorm import RMSNorm


class BankReconstructionHead(nn.Module):
    """Reconstruct earlier-chunk tokens from the bank state alone."""

    def __init__(self, config, lm_head: Optional[nn.Module] = None):
        super().__init__()
        h = config.hidden_size
        self.hidden_size = h
        self.n_samples = int(getattr(config, "bank_recon_samples", 32))
        self.weight = float(getattr(config, "bank_recon_weight", 0.1))
        self.chunk_size = int(getattr(config, "delta_chunk_size", 512))
        self.n_heads = max(1, int(getattr(config, "num_attention_heads", 8)))
        while h % self.n_heads != 0 and self.n_heads > 1:
            self.n_heads -= 1
        self.head_dim = h // self.n_heads

        # Query = learned slot vector + embedding of the target's offset inside
        # its chunk. The offset is what makes the query point at a POSITION
        # rather than asking "what was in this chunk generally".
        self.slot = nn.Parameter(torch.zeros(h))
        self.pos_embed = nn.Embedding(self.chunk_size, h)
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)
        self.norm = RMSNorm(h, eps=getattr(config, "rms_norm_eps", 1e-6))
        self._shared_lm_head = lm_head
        self.out = None if lm_head is not None else nn.Linear(h, config.vocab_size, bias=False)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def _head(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x) if self.out is not None else self._shared_lm_head(x)

    def forward(
        self,
        states: torch.Tensor,
        input_ids: torch.Tensor,
        state_docs: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.Tensor] = None,
        prefix_len: int = 0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """states [B, C, N, d] (state for chunk c = writes from chunks < c);
        input_ids [B, S] the FULL sequence; doc_ids [B, S_content] or None.

        Returns a scalar CE over sampled earlier-chunk positions.
        """
        b, c, n_slots, d = states.shape
        if c < 2:
            # With one chunk there is nothing written before it to recover.
            return states.new_zeros((), dtype=torch.float32)
        device = states.device
        ck = self.chunk_size
        content = input_ids[:, prefix_len:]
        t_content = content.shape[1]
        m = self.n_samples

        # For each chunk c >= 1, sample m positions uniformly from chunk c-1 —
        # the most recently written chunk, i.e. the shortest carry that still
        # crosses a chunk boundary. Chunk 0 has no predecessor, so it is skipped.
        tgt_chunks = torch.arange(1, c, device=device)  # [c-1]
        offs = torch.randint(
            0, ck, (b, c - 1, m), device=device, generator=generator
        )  # position inside chunk c-1
        pos = (tgt_chunks - 1).view(1, -1, 1) * ck + offs  # [B, c-1, m] content pos
        in_range = pos < t_content
        pos_safe = pos.clamp_max(max(t_content - 1, 0))

        target = content.gather(1, pos_safe.reshape(b, -1)).view(b, c - 1, m)

        valid = in_range
        if state_docs is not None and doc_ids is not None:
            tok_doc = doc_ids.gather(1, pos_safe.reshape(b, -1)).view(b, c - 1, m)
            # The state a chunk-c reader sees accumulated state_docs[:, c].
            sdoc = state_docs[:, 1:c].unsqueeze(-1)  # [B, c-1, 1]
            valid = valid & (tok_doc == sdoc)
        if not bool(valid.any()):
            return states.new_zeros((), dtype=torch.float32)

        # Queries: learned slot + in-chunk offset embedding.
        q = self.slot.view(1, 1, 1, d) + self.pos_embed(offs)  # [B, c-1, m, d]
        q = self.q_proj(q).view(b, (c - 1) * m, self.n_heads, self.head_dim).transpose(1, 2)

        # Keys/values: the state visible to chunk c, one per sampled query.
        st = states[:, 1:c]  # [B, c-1, N, d] — state for chunks 1..c-1
        st = st.unsqueeze(2).expand(-1, -1, m, -1, -1).reshape(b, (c - 1) * m * n_slots, d)
        kv_shape = (b, (c - 1) * m, n_slots, self.n_heads, self.head_dim)
        k = self.k_proj(st).view(*kv_shape)
        v = self.v_proj(st).view(*kv_shape)
        # -> [B*(c-1)*m, H, N, dh] for a batched, unmasked cross-attention.
        nq = (c - 1) * m
        k = k.permute(0, 1, 3, 2, 4).reshape(b * nq, self.n_heads, n_slots, self.head_dim)
        v = v.permute(0, 1, 3, 2, 4).reshape(b * nq, self.n_heads, n_slots, self.head_dim)
        qh = q.transpose(1, 2).reshape(b * nq, self.n_heads, 1, self.head_dim)

        ctx = F.scaled_dot_product_attention(
            qh.contiguous(), k.contiguous(), v.contiguous()
        )  # [B*nq, H, 1, dh]
        ctx = ctx.reshape(b, nq, self.n_heads * self.head_dim)
        ctx = self.norm(self.o_proj(ctx))

        logits = self._head(ctx).float().reshape(b * nq, -1)
        tgt = torch.where(valid, target, target.new_full((), -100)).reshape(b * nq)
        return F.cross_entropy(logits, tgt, ignore_index=-100)
