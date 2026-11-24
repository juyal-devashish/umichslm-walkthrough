"""swa_scribe: FA2-native multi-bank SWA attention via exact LSE merging.

Four banks, no prepended prefix, window kernel stays FlashAttention-native:
  * sink bank (``swa_sink_count``, default 16): out-of-sequence learned keys
    joined to the window branch's softmax by an EXACT log-sum-exp merge — the
    joint softmax a prefix provides, without the prefix kernel path. Live from
    step 0 (no gate), content-free (Gu et al. 2410.10781: sinks must be).
  * scribe bank (``use_scribe_bank``): W learned queries summarize each K-token
    chunk at ONE write layer (optionally top-k sparse); committed scribe states
    are read by later SWA layers as a third joint-softmax branch under a
    block-causal chunk mask (+ doc gating). Causality: a token in chunk c reads
    only scribes of chunks < c, whose receptive field ends before c starts.
  * register bank: inherited gated additive read (swa_registers), optionally
    refined once per forward by a zero-init bidirectional pass over the slots
    (``swa_register_refine``; the bank is outside the sequence, so slot<->slot
    attention is legal and mask-free).
  * workspace trunk: composed at the model level as in the swa_*_ws arms, with
    ``trunk_write_topk`` for GWT-style write competition.

FA2 path: attn_sink.flash_attn_with_lse (differentiable LSE) yields the window
branch's LSE ([b,h,s] batched; [h,total_q] varlen); sink/scribe branches are
tiny manual attentions over the SAME post-RoPE queries with the same scale, so
merge_lse_branches() reconstructs the exact joint softmax. SDPA/eager/MoD-ext
paths instead CONCATENATE bank keys/values + mask columns before dispatch —
natively joint, used as the correctness reference. Requires attention_dropout
== 0 (FA2's LSE is pre-dropout). Scribe reads are skipped on the MoD compact
path (chunk structure is defined on original positions; the write layer sits
at a dense block boundary so compact members never read scribes anyway).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swa_registers import SWARegistersAttention, SWARegistersFlashAttention2
from .workspace_trunk import TrunkCrossAttention, get_rmsnorm_class


def merge_lse_branches(outs: List[torch.Tensor], lses: List[torch.Tensor]) -> torch.Tensor:
    """Exact joint softmax from per-branch (output, LSE) pairs.

    outs: [B,S,H,D] each; lses: [B,H,S] each (natural log). A branch whose row
    is fully masked must pass lse = -inf there; the window branch always has
    >= 1 key, so no row is -inf across all branches. Computed in fp32.
    """
    stack_lse = torch.stack([l.float() for l in lses], dim=0)  # [K,B,H,S]
    m = stack_lse.max(dim=0).values
    w = torch.exp(stack_lse - m.unsqueeze(0))  # [K,B,H,S]
    w = w / w.sum(dim=0, keepdim=True)
    out = torch.zeros_like(outs[0], dtype=torch.float32)
    for k, o in enumerate(outs):
        out = out + o.float() * w[k].permute(0, 2, 1).unsqueeze(-1)  # [B,S,H,1]
    return out.to(outs[0].dtype)


class BankRefiner(nn.Module):
    """One bidirectional self-attention pass over a bank; zero-init residual."""

    def __init__(self, config):
        super().__init__()
        norm_class = get_rmsnorm_class(
            zero_centered=getattr(config, "use_zero_centered_rmsnorm", False)
        )
        self.norm = norm_class(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.attn = TrunkCrossAttention(config, zero_init_output=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        t = self.norm(tokens.unsqueeze(0))
        return tokens + self.attn(t, t).squeeze(0)


class ScribeBank(nn.Module):
    """Chunk-committed scribe writes: W learned queries per K-token chunk."""

    def __init__(self, config):
        super().__init__()
        h = config.hidden_size
        self.slots = int(config.scribe_slots_per_chunk)
        self.chunk_size = int(config.scribe_chunk_size)
        self.top_k = int(getattr(config, "scribe_write_topk", 0)) or None
        init_std = float(getattr(config, "initializer_range", 0.02))
        self.tokens = nn.Parameter(torch.randn(self.slots, h) * init_std)  # AdamH group conv.
        norm_class = get_rmsnorm_class(
            zero_centered=getattr(config, "use_zero_centered_rmsnorm", False)
        )
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.q_norm = norm_class(h, eps=eps)
        self.src_norm = norm_class(h, eps=eps)
        # Zero-init write output: at init every scribe state == its learned
        # query (content-free), Flamingo-safe like the trunk/register reads.
        self.write = TrunkCrossAttention(config, zero_init_output=True)

    def forward(
        self, content: torch.Tensor, doc_ids: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t, h = content.shape
        k = self.chunk_size
        n_chunks = (t + k - 1) // k
        pad = n_chunks * k - t
        padded = F.pad(content, (0, 0, 0, pad)) if pad else content
        chunks = padded.view(b * n_chunks, k, h)
        valid = torch.ones(b, n_chunks * k, dtype=torch.bool, device=content.device)
        if pad:
            valid[:, t:] = False
        scribe_doc = None
        key_mask = valid.view(b * n_chunks, k)
        if doc_ids is not None:
            padded_docs = F.pad(doc_ids, (0, pad), value=-1) if pad else doc_ids
            cdocs = padded_docs.view(b, n_chunks, k)
            # Last VALID token's doc owns the chunk; write only its tokens
            # (same rule as WorkspaceTrunk._doc_mask_for_chunk).
            last_idx = valid.view(b, n_chunks, k).float().cumsum(-1).argmax(-1)  # [b,C]
            scribe_doc = cdocs.gather(-1, last_idx.unsqueeze(-1)).squeeze(-1)  # [b,C]
            key_mask = key_mask & (cdocs == scribe_doc.unsqueeze(-1)).view(b * n_chunks, k)
        q = self.q_norm(self.tokens).unsqueeze(0).expand(b * n_chunks, -1, -1)
        states = self.tokens.unsqueeze(0) + self.write(
            q, self.src_norm(chunks), key_mask=key_mask, top_k=self.top_k
        )
        return states.view(b, n_chunks, self.slots, h), scribe_doc


class ScribeSinkMixin:
    """Adds the sink joint-softmax branch and the scribe read to a registers-SWA class."""

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)
        # use_logit_sinks (P2 `sink_diet`, 2026-08-20): a learned PER-HEAD sink
        # logit instead of a sink *bank*. The head gets an extra softmax
        # competitor with an all-zero value vector, so it drains attention mass
        # exactly like a sink token but costs no K/V, no sequence position and
        # no k_proj/v_proj work -- and, unlike a prefix, it never pushes the
        # layer off the FA2 varlen path. Mechanism per gpt-oss / StreamingLLM
        # (2309.17453, 2410.10781). Init 0.0 (the standard choice): the sink
        # starts at logit 0 and learns its own strength, so this arm is NOT
        # bitwise inert at init -- it is a distinct architecture, not a flag on
        # an existing one.
        self.use_logit_sinks = bool(getattr(config, "use_logit_sinks", False))
        if self.use_logit_sinks:
            self.logit_sink = nn.Parameter(torch.zeros(self.num_heads))
        # Block-sparse far branch (block_sparse.py). Disjoint from the window by
        # construction -- it draws only from key blocks that END before the
        # window BEGINS -- which is what lets merge_lse_branches() reconstruct
        # the exact joint softmax rather than blend two overlapping softmaxes.
        self.use_bsa = bool(getattr(config, "use_block_sparse_attn", False))
        self._bsa_min_seq = int(getattr(config, "bsa_min_seq", 0))
        self._bsa_on_layer = self.use_bsa and (
            layer_idx is None
            or int(getattr(config, "bsa_layer_interval", 1)) <= 1
            or (layer_idx % int(getattr(config, "bsa_layer_interval", 1))) == 0
        )
        if self._bsa_on_layer:
            from .block_sparse import BlockSparseSelector

            self.bsa_selector = BlockSparseSelector(config, self.head_dim)
        self._wants_window_lse = (
            int(getattr(config, "swa_sink_count", 0)) > 0
            or bool(getattr(config, "use_scribe_bank", False))
            or self.use_logit_sinks
            or self._bsa_on_layer
        )
        object.__setattr__(self, "_sink_bank", None)
        self._scribe_states = None  # [B,C,W,d], set by the model above the write layer
        self._scribe_doc = None  # [B,C]
        self._scribe_token_docs = None  # [B,S]
        self._scribe_chunk_size = int(getattr(config, "scribe_chunk_size", 512))

    def bind_sink_bank(self, bank) -> None:
        object.__setattr__(self, "_sink_bank", bank)

    # -- shared branch builders ------------------------------------------------
    def _bank_kv(self, tokens: torch.Tensor, bsz: int, dtype):
        n = tokens.shape[0]
        k = (
            self.k_proj(tokens.to(dtype))
            .view(1, n, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(tokens.to(dtype))
            .view(1, n, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = self._repeat_kv(k, self.num_key_value_groups).expand(bsz, -1, -1, -1)
        v = self._repeat_kv(v, self.num_key_value_groups).expand(bsz, -1, -1, -1)
        return k, v  # [B,H,n,D]

    def _scribe_read_mask(self, q_len: int, device) -> Optional[torch.Tensor]:
        """Additive [B,1,q_len,C*W] mask: block-causal over chunks + doc gate."""
        states, sdoc = self._scribe_states, self._scribe_doc
        if states is None:
            return None
        b, c, w, _ = states.shape
        pos = torch.arange(q_len, device=device)
        tok_chunk = pos // self._scribe_chunk_size  # [S]
        chunk_ids = torch.arange(c, device=device)
        allow = chunk_ids.view(1, 1, c) < tok_chunk.view(1, q_len, 1)  # [1,S,C]
        allow = allow.expand(b, -1, -1)
        if sdoc is not None and self._scribe_token_docs is not None:
            allow = allow & (sdoc.unsqueeze(1) == self._scribe_token_docs.unsqueeze(-1))
        mask = torch.zeros(b, 1, q_len, c * w, device=device)
        mask.masked_fill_(
            ~allow.repeat_interleave(w, dim=-1).unsqueeze(1), torch.finfo(torch.float32).min
        )
        return mask

    # -- FA2 path: LSE merge -----------------------------------------------------
    def _bsa_branch(self, q_bhsd, key_states, value_states):
        """Far block-sparse branch. q: [B,H,S,D]; key/value: [B,S,Hkv,D] (FA2 layout).

        Returns (out[B,S,H,D], lse[B,H,S]) or None when the branch is off for
        this call -- short sequence (bsa_min_seq), MoD compact rows, or a flex
        path that declined the shapes.
        """
        from .block_sparse import block_sparse_flex, block_sparse_reference

        s = q_bhsd.shape[2]
        if s < self._bsa_min_seq:
            return None
        # The MoD compact path renumbers rows, so block indices no longer mean
        # positions -- same reason scribe reads are skipped there.
        if getattr(self, "_mod_external_mask", None) is not None:
            return None
        k = self._repeat_kv(key_states.transpose(1, 2), self.num_key_value_groups).float()
        v = self._repeat_kv(value_states.transpose(1, 2), self.num_key_value_groups).float()
        docs = getattr(self, "_flex_doc_ids", None)
        if docs is not None and docs.shape[-1] != s:
            docs = None
        sel, bias, _, _ = self.bsa_selector(q_bhsd, k, docs)
        if not bool(sel.any()):
            return None
        res = block_sparse_flex(q_bhsd, k, v, sel, bias, self.bsa_selector.block, docs)
        if res is None:
            res = block_sparse_reference(q_bhsd, k, v, sel, bias, self.bsa_selector.block, docs)
        return res

    def _merge_bank_branches(
        self,
        query_states: torch.Tensor,
        out_w: torch.Tensor,
        lse_w: torch.Tensor,
        key_states: torch.Tensor = None,
        value_states: torch.Tensor = None,
    ) -> torch.Tensor:
        """query_states/out_w: [B,S,H,D] (FA2 layout); lse_w: [B,H,S]."""
        bsz, s, h, d = query_states.shape
        q = query_states.permute(0, 2, 1, 3).float()  # [B,H,S,D]
        scale = d**-0.5
        outs, lses = [out_w], [lse_w]
        if self._bsa_on_layer and key_states is not None and self._bank_read_on():
            far = self._bsa_branch(q, key_states, value_states)
            if far is not None:
                outs.append(far[0].to(out_w.dtype))
                lses.append(far[1])
        if self._sink_bank is not None:
            k, v = self._bank_kv(self._sink_bank.tokens, bsz, query_states.dtype)
            logits = torch.matmul(q, k.float().transpose(-1, -2)) * scale  # [B,H,S,n]
            lses.append(torch.logsumexp(logits, dim=-1))
            outs.append(torch.matmul(F.softmax(logits, dim=-1), v.float()).permute(0, 2, 1, 3))
        if (
            self._scribe_states is not None
            and getattr(self, "_mod_external_mask", None) is None
            and self._bank_read_on()
        ):
            st = self._scribe_states
            b2, c, w, hd = st.shape
            flat = st.reshape(b2, c * w, hd).to(query_states.dtype)
            k = (
                self.k_proj(flat)
                .view(b2, c * w, self.num_key_value_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                self.v_proj(flat)
                .view(b2, c * w, self.num_key_value_heads, self.head_dim)
                .transpose(1, 2)
            )
            k = self._repeat_kv(k, self.num_key_value_groups)
            v = self._repeat_kv(v, self.num_key_value_groups)
            logits = torch.matmul(q, k.float().transpose(-1, -2)) * scale  # [B,H,S,C*W]
            rmask = self._scribe_read_mask(s, query_states.device)
            logits = logits + rmask
            lse = torch.logsumexp(logits, dim=-1)
            outs.append(
                torch.nan_to_num(torch.matmul(F.softmax(logits, dim=-1), v.float())).permute(
                    0, 2, 1, 3
                )
            )
            lses.append(lse)  # rows with no visible scribe give lse=-inf -> weight 0
        if self.use_logit_sinks:
            # Exact: adding a competitor with logit s_h and value 0 scales the
            # window output by denom/(denom + exp(s_h)). Expressed as a branch
            # with zero output and lse = s_h, merge_lse_branches computes that
            # for free -- no extra matmul, no key, no value.
            outs.append(torch.zeros_like(out_w))
            lses.append(
                self.logit_sink.to(torch.float32).view(1, h, 1).expand(bsz, h, s)
            )
        if len(outs) == 1:
            return out_w
        return merge_lse_branches(outs, lses)

    def _bsa_unmask(self, attention_mask, query_states, key_states, q_len):
        """SDPA/eager twin of the block-sparse branch: UN-mask the selected blocks.

        The far keys are ordinary sequence keys, so there is nothing to
        concatenate -- the window mask has simply already set them to -inf. This
        clears the selected, causal, same-document entries back to 0, giving one
        native joint softmax over window + selected blocks. That is the same
        function the FA2 path reconstructs with merge_lse_branches, which is why
        it is the correctness reference the tests compare against.

        Returns the mask unchanged when the branch is off, so an arm without
        use_block_sparse_attn is bitwise unaffected.
        """
        if not self._bsa_on_layer or attention_mask is None:
            return attention_mask
        if q_len < self._bsa_min_seq or getattr(self, "_mod_external_mask", None) is not None:
            return attention_mask
        if not self._bank_read_on() or key_states.shape[2] != q_len:
            return attention_mask
        docs = getattr(self, "_flex_doc_ids", None)
        if docs is not None and docs.shape[-1] != q_len:
            docs = None
        q = query_states.float()
        sel, bias, _, _ = self.bsa_selector(q, key_states.float(), docs)
        blk = self.bsa_selector.block
        m = sel.repeat_interleave(blk, 2).repeat_interleave(blk, 3)[:, :, :q_len, :q_len]
        pos = torch.arange(q_len, device=key_states.device)
        m = m & (pos.view(q_len, 1) >= pos.view(1, q_len)).view(1, 1, q_len, q_len)
        if docs is not None:
            m = m & (docs.view(-1, 1, q_len, 1) == docs.view(-1, 1, 1, q_len))
        out = attention_mask.masked_fill(m, 0.0)
        if bias is not None:
            add = bias.repeat_interleave(blk, 2).repeat_interleave(blk, 3)[:, :, :q_len, :q_len]
            out = out + torch.where(m, add.to(out.dtype), torch.zeros_like(out))
        return out

    # -- SDPA/eager/ext-mask path: concat keys + mask columns --------------------
    def _extend_bank_kv(
        self, hidden_states, query_states, key_states, value_states, attention_mask, q_len
    ):
        bsz = key_states.shape[0]
        extra_k, extra_v, cols = [], [], []
        dtype = hidden_states.dtype
        kv_len = key_states.shape[2]
        attention_mask = self._bsa_unmask(attention_mask, query_states, key_states, q_len)
        if self._sink_bank is not None:
            k, v = self._bank_kv(self._sink_bank.tokens, bsz, dtype)
            extra_k.append(k.to(key_states.dtype))
            extra_v.append(v.to(value_states.dtype))
            cols.append(torch.zeros(bsz, 1, q_len, k.shape[2], device=key_states.device))
        if (
            self._scribe_states is not None
            and getattr(self, "_mod_external_mask", None) is None
            and self._bank_read_on()
        ):
            st = self._scribe_states
            b2, c, w, hd = st.shape
            flat = st.reshape(b2, c * w, hd)
            k = (
                self.k_proj(flat.to(dtype))
                .view(b2, c * w, self.num_key_value_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                self.v_proj(flat.to(dtype))
                .view(b2, c * w, self.num_key_value_heads, self.head_dim)
                .transpose(1, 2)
            )
            extra_k.append(self._repeat_kv(k, self.num_key_value_groups).to(key_states.dtype))
            extra_v.append(self._repeat_kv(v, self.num_key_value_groups).to(value_states.dtype))
            cols.append(self._scribe_read_mask(q_len, key_states.device))
        if self.use_logit_sinks:
            # SDPA/eager twin of the FA2 branch above: one ZERO key and ZERO
            # value column, biased by the per-head sink logit. q . 0 = 0, so the
            # column's logit is exactly s_h, and a zero value adds nothing to
            # the output while still entering the denominator -- algebraically
            # identical to the LSE-merge form, just not free here.
            nh = key_states.shape[1]
            zk = torch.zeros(
                bsz, nh, 1, key_states.shape[-1],
                device=key_states.device, dtype=key_states.dtype,
            )
            extra_k.append(zk)
            extra_v.append(
                torch.zeros(
                    bsz, nh, 1, value_states.shape[-1],
                    device=value_states.device, dtype=value_states.dtype,
                )
            )
            cols.append(
                self.logit_sink.to(torch.float32)
                .view(1, nh, 1, 1)
                .expand(bsz, nh, q_len, 1)
            )
        if not extra_k:
            return key_states, value_states, attention_mask
        if attention_mask is None:
            # Materialize the plain causal mask so bank columns can be appended.
            base = torch.zeros(1, 1, q_len, kv_len, device=key_states.device)
            causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=key_states.device).tril(
                kv_len - q_len
            )
            base.masked_fill_(~causal, torch.finfo(torch.float32).min)
            attention_mask = base
        attention_mask = attention_mask.to(torch.float32)
        if attention_mask.shape[0] == 1 and bsz > 1:
            attention_mask = attention_mask.expand(bsz, -1, -1, -1)
        cols = [c.expand(bsz, -1, -1, -1) for c in cols]
        # torch.cat needs every non-cat dim to agree. The per-head sink column
        # is the only one with head dim > 1, so broadcast the base mask and the
        # head-agnostic columns up to it when it is present.
        nheads = max([c.shape[1] for c in cols] + [attention_mask.shape[1]])
        if nheads > 1:
            cols = [c if c.shape[1] == nheads else c.expand(-1, nheads, -1, -1) for c in cols]
            if attention_mask.shape[1] == 1:
                attention_mask = attention_mask.expand(-1, nheads, -1, -1)
        attention_mask = torch.cat([attention_mask, *cols], dim=-1).to(hidden_states.dtype)
        key_states = torch.cat([key_states, *extra_k], dim=2)
        value_states = torch.cat([value_states, *extra_v], dim=2)
        return key_states, value_states, attention_mask


class SWAScribeAttention(ScribeSinkMixin, SWARegistersAttention):
    """Eager/SDPA scribe layer: banks joined by k/v + mask-column extension."""


class SWAScribeFlashAttention2(ScribeSinkMixin, SWARegistersFlashAttention2):
    """FA2 scribe layer: window branch native FA2, banks via exact LSE merge."""


SWA_SCRIBE_CLASSES = {
    "eager": SWAScribeAttention,
    "sdpa": SWAScribeAttention,
    "flash_attention_2": SWAScribeFlashAttention2,
}
