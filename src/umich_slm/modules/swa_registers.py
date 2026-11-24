"""Sliding-window attention with a register-bank read (``swa_use_registers``).

Kernel-friendly twin of ``swa_with_globals.py``: the 256 persistent globals move
OUT of the token sequence. Self-attention becomes a plain sliding-window band —
FlashAttention-2 native (``window_size``), including the varlen intra-doc path —
and each SWA layer additionally reads a model-level learned register bank
through a position-free cross-attention gated to zero at init.

Why this exists: the prefix-globals pattern ("all globals + drifting band") is
inexpressible in FA2 and forces the SDPA additive-mask path (or FlexAttention).
With registers there is no prefix, so
  * FA2 runs natively on every SWA layer (train and varlen intra-doc);
  * intra-doc masking costs O(S) via ``flash_attn_varlen_func`` — no
    ``[B,1,S,S]`` additive masks, which is what made 32K infeasible for the
    prefix path (see CLAUDE.md);
  * generation needs no prepended prefix in the KV cache.

Design contract (vs. prefix globals):
  * Register read: ``q = q_proj(x)`` (un-rotated — the read is position-free),
    ``k/v = k_proj/v_proj(bank)``; GQA-expanded; softmax over the N registers
    only; a per-head gate (zero-init) scales the read; the result goes through
    ``o_proj``'s weight (no bias) and is ADDED to the self-attention output.
    Because ``o_proj`` is linear with no bias here, ``o_proj(swa) + o_proj(g·read)``
    is exactly the pre-``o_proj`` merge — two softmaxes instead of the joint
    prefix softmax is the one intentional math difference from swa_globals.
  * Zero-init gates: at init the model is EXACTLY a pure-SWA model
    (Flamingo-style safe insertion); gates open first, then the bank trains.
  * The bank is model-level state (one ``RegisterBank`` on ``UMichSLMModel``),
    bound to each layer as a plain attribute (``bind_register_bank``) so its
    parameters are counted/checkpointed once, on the model.
  * Under block-atomic MoD the runner installs ``_mod_external_mask`` (window
    over ORIGINAL positions of the gathered sequence); self-attention then runs
    the SDPA path under that mask, and the register read is unaffected — it is
    per-token and position-free, so gather/scatter commutes with it.
  * Costs: the read recomputes ``q_proj`` (one extra d×d matmul per SWA layer)
    plus an S×N attention — ~N/S of a full attention. Both are small next to
    what the prefix path pays in mask handling; noted here because
    ``titan/throughput.py`` models neither (its swa MFU stays "slightly
    optimistic" exactly as for swa_globals).

The band is ``config.swa_window`` — same knob as swa_with_globals, same
"``sliding_window`` is NOT the band" trap (that key belongs to the periodic
global MLA layers). This module pins ``self.sliding_window = swa_window`` on
its own layers so the base class's native FA2/SDPA window machinery enforces
the correct band.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import UMichSLMAttention, UMichSLMFlashAttention2
from .swa_with_globals import SWAWithGlobalsAttention


class RegisterReadMixin:
    """Adds the register-bank cross-read to an attention base class.

    Must precede the base class in the MRO. The mixin overrides ``forward`` to
    (1) honor the MoD compact-path contract (``_mod_external_mask``) by routing
    self-attention through the SDPA path under the runner's mask, and
    (2) add the gated register read to the attention output.
    """

    _full_attention_kind = False  # SWA band; never carries an attn_logit_sink

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)
        self.swa_window = int(getattr(config, "swa_window", 1024))
        if self.swa_window <= 0:
            raise ValueError(f"swa_window must be > 0, got {self.swa_window}.")
        # Override the base-class init: every registers-SWA layer is a local
        # layer with band = swa_window, independent of is_global_attention_layer
        # (that schedule belongs to the periodic global MLA layers).
        self.sliding_window = self.swa_window
        self.use_sliding_window = True
        # Per-head zero-init gate: the register read starts OFF, so the layer is
        # bitwise a pure-SWA layer at init.
        self.register_gate = nn.Parameter(torch.zeros(self.num_heads))
        # swa_bank_read_interval: register/scribe reads run only on layers with
        # layer_idx % interval == 0 (per-layer bank reads scale with carried
        # depth — the deep-model wall-clock toll; sink is exempt, it is the
        # softmax stabilizer). interval 1 == historical every-layer behavior.
        self._bank_read_interval = int(getattr(config, "swa_bank_read_interval", 1))
        # swa_fuse_register_proj: share q_proj/o_proj between the window branch
        # and the bank read (exactly equivalent; see RegisterReadMixin.forward).
        self._fuse_register_proj = bool(getattr(config, "swa_fuse_register_proj", False))
        # Installed by the MoD block runner on the gathered compact sequence.
        self._mod_external_mask = None
        # Plain (non-module) reference to the model-level bank; set via
        # bind_register_bank so the parameters live on the model exactly once.
        object.__setattr__(self, "_register_bank", None)
        # swa_deltabank stashes (set by the model above the write layer, cleared
        # at forward start): per-chunk cumulative bank states + doc bookkeeping.
        self._delta_states = None  # [B, C, N, d]
        self._delta_state_docs = None  # [B, C]
        self._delta_token_docs = None  # [B, T_content]
        self._delta_prefix_len = 0
        self._delta_chunk_size = int(getattr(config, "delta_chunk_size", 512))

    def bind_register_bank(self, bank) -> None:
        object.__setattr__(self, "_register_bank", bank)

    def _bank_read_on(self) -> bool:
        """Interval gate shared by the register read and the scribe joint keys."""
        iv = getattr(self, "_bank_read_interval", 1)
        if iv <= 1:
            return True
        li = getattr(self, "layer_idx", None)
        return li is None or (li % iv == 0)

    def _register_read(
        self,
        hidden_states: torch.Tensor,
        q_raw: Optional[torch.Tensor] = None,
        pre_o: bool = False,
    ) -> Optional[torch.Tensor]:
        """Bank read for this layer.

        ``q_raw``: the un-rotated ``q_proj(hidden_states)`` when the caller has
        already computed it for the window branch. The read has always used the
        raw projection (position-free by design), so reusing it is exact -- it
        just stops the layer paying for the identical GEMM twice.

        ``pre_o``: return the read BEFORE ``o_proj``. ``o_proj`` is bias-free
        (see the note at the tail of this method), so the caller can fold the
        read into the window branch's pre-projection sum and run one output
        GEMM for both. Also exact.
        """
        bank = self._register_bank
        if bank is None or not self._bank_read_on():
            return None
        bsz, q_len, _ = hidden_states.shape
        # Per-forward refined tensor (swa_register_refine) takes priority; the
        # model stashes it each forward and clears it at the next forward start.
        override = getattr(self, "_register_tokens_override", None)
        tokens = bank.tokens if override is None else override
        reg = tokens.to(dtype=hidden_states.dtype)  # [N, d]
        n_reg = reg.shape[0]

        # Un-rotated query: the read is position-free by design.
        qp = self.q_proj(hidden_states) if q_raw is None else q_raw
        q = qp.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(reg).view(1, n_reg, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(reg).view(1, n_reg, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        k = self._repeat_kv(k, self.num_key_value_groups).expand(bsz, -1, -1, -1)
        v = self._repeat_kv(v, self.num_key_value_groups).expand(bsz, -1, -1, -1)

        # Every token sees every register — no mask, not causal.
        # Contiguous q/k/v: k/v are .view(...).transpose(1,2) + _repeat_kv +
        # expand, so the head dim carries the SMALLER stride and the batch dim is
        # stride-0. Inductor's meta kernel for
        # _scaled_dot_product_flash_attention_backward asserts the contiguous
        # layout and hard-fails ("expected size 7==7, stride 48==6144"), which
        # killed the compiled backward on the register-reading arms
        # (swa_hybrid32, 2026-07-20). Shape-dependent — the 50m cell compiled
        # fine — so it must be forced, not left to inductor's layout choice.
        # The materialised copy is tiny (bsz x H x n_reg x dh).
        read = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())

        # swa_deltabank: replace the static read with the per-chunk cumulative
        # state read where available. Skipped on the MoD compact path (chunk
        # membership is defined on ORIGINAL positions, which the gathered
        # sequence no longer carries) — compact-path tokens read the static
        # bank, mirroring the scribe-read rule.
        states = getattr(self, "_delta_states", None)
        if states is not None and getattr(self, "_mod_external_mask", None) is None:
            read = self._delta_read_merge(q, read, states)

        gate = self.register_gate.to(read.dtype).view(1, -1, 1, 1)
        read = (gate * read).transpose(1, 2).reshape(bsz, q_len, self.num_heads * self.head_dim)
        # Weight-only o_proj: with a bias it would be added twice across the two
        # branches; without bias this equals the pre-o_proj merge exactly -- which
        # is what ``pre_o`` hands back, letting the caller do the merge and skip
        # this second full-size projection entirely.
        if pre_o:
            return read
        return F.linear(read, self.o_proj.weight)

    def _delta_read_merge(
        self, q: torch.Tensor, static_read: torch.Tensor, states: torch.Tensor
    ) -> torch.Tensor:
        """Per-chunk register read against cumulative delta-bank states.

        q / static_read: [B, H, S, dh]; states: [B, C, N, d]. Content position
        p reads states[:, (p - prefix) // chunk] — the bank as written by
        chunks strictly before its own. Prefix positions and (under intra-doc
        masking) tokens whose doc does not match their chunk-state's doc fall
        back to the static read, so at init (zero-init writes, states == B0)
        this is exactly the static read.
        """
        bsz, n_heads, q_len, _ = q.shape
        p = int(getattr(self, "_delta_prefix_len", 0))
        tc = q_len - p
        b, c, n_reg, d_model = states.shape
        ck = self._delta_chunk_size
        if tc <= 0 or b != bsz or tc > c * ck:
            # Shape mismatch (e.g. cached generation feeding a partial
            # sequence): the stashes don't describe this forward — static read.
            return static_read

        flat = states.reshape(b * c, n_reg, d_model).to(q.dtype)
        k = (
            self.k_proj(flat)
            .view(b * c, n_reg, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(flat)
            .view(b * c, n_reg, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = self._repeat_kv(k, self.num_key_value_groups)
        v = self._repeat_kv(v, self.num_key_value_groups)

        qc = q[:, :, p:, :]  # [B, H, Tc, dh]
        pad = c * ck - tc
        if pad:
            qc = F.pad(qc, (0, 0, 0, pad))
        qc = (
            qc.view(b, n_heads, c, ck, self.head_dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(b * c, n_heads, ck, self.head_dim)
        )
        # Contiguous q/k/v: k/v come from .view(...).transpose(1,2) + _repeat_kv,
        # which leaves the head dim with the SMALLER stride (head=dh, slots=H*dh).
        # Inductor's meta kernel for _scaled_dot_product_flash_attention_backward
        # asserts the contiguous layout and hard-fails on that transposed one
        # ("expected size 7==7, stride 48==6144"), killing the compiled backward —
        # this is what crashed swa_scribe_mopd_delta under torch_compile=true
        # (2026-07-20). Eager tolerates the strides; compile does not.
        delta = F.scaled_dot_product_attention(
            qc.contiguous(), k.contiguous(), v.contiguous()
        )  # [B*C, H, ck, dh]
        delta = (
            delta.view(b, c, n_heads, ck, self.head_dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(b, n_heads, c * ck, self.head_dim)
        )
        if pad:
            delta = delta[:, :, :tc, :]

        content_static = static_read[:, :, p:, :]
        sdocs, tdocs = self._delta_state_docs, self._delta_token_docs
        if sdocs is not None and tdocs is not None:
            # Broadcast each chunk's doc id over its ck slots by expand+reshape
            # rather than gathering with a computed `arange(tc) // ck` index.
            # Both give token i the id of chunk i // ck (the clamp was dead: the
            # guard above forces tc <= c*ck, hence (tc-1)//ck <= c-1), but the
            # gather form bakes Min(FloorDiv(idx, ck), c-1) into the load index.
            # Inductor's loop merger calls stride_vars on that, which substitutes
            # concrete offsets and hands sympy's Min a FloorDiv(-head_dim, ck)
            # it refuses to order -- "The argument '(((-32))//512)' is not
            # comparable" -- failing the compile of this function and the two
            # frames enclosing it. suppress_errors then silently drops the whole
            # register-read path to eager for the run (swa_deltabank, 2026-07-22).
            # The expand+reshape form is affine, so no Min/FloorDiv appears.
            state_docs = sdocs.unsqueeze(-1).expand(b, c, ck).reshape(b, c * ck)
            match = tdocs == state_docs[:, :tc]  # [B, Tc]
            delta = torch.where(match.view(b, 1, tc, 1), delta, content_static)
        if p:
            return torch.cat([static_read[:, :, :p, :], delta], dim=2)
        return delta

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        depth_kv_cache=None,
        precomputed_q: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # ``precomputed_q``: the caller supplies the un-rotated query projection
        # instead of this layer computing q_proj(hidden_states). Used by the
        # pause stream (modules/pause_stream.py), where the queries come from
        # the PREDICTION stream while K/V must come from the STATE stream, so
        # the two cannot both be derived from one `hidden_states`. The bank read
        # already uses the raw projection, so a supplied q flows into both the
        # window branch and the register read -- i.e. the prediction stream
        # reads the bank too, which is what we want.
        #
        # Fused projection sharing: one q_proj and one o_proj for the layer
        # instead of one per branch. Exactly equivalent (see _register_read), so
        # it is an optimisation, not an arm -- but it is gated so the A/B can be
        # measured and so a regression has a single switch to fall back on.
        fuse = self._fuse_register_proj and self._register_bank is not None and self._bank_read_on()
        q_raw = None
        pre_o_add = None
        if fuse:
            q_raw = self.q_proj(hidden_states) if precomputed_q is None else precomputed_q
            pre_o_add = self._register_read(hidden_states, q_raw=q_raw, pre_o=True)
            if pre_o_add is None:  # gate said no after all -- fall back cleanly
                fuse, q_raw = False, None
        if q_raw is None:
            # Unfused, but a caller-supplied q still has to reach the window
            # branch; otherwise the prediction stream would silently attend with
            # the STATE stream's queries and the arm would be a no-op.
            q_raw = precomputed_q

        ext_mask = getattr(self, "_mod_external_mask", None)
        if ext_mask is not None:
            # Block-atomic MoD compact path: the runner's mask already encodes
            # causal + window over ORIGINAL positions of the gathered sequence.
            # FA2 cannot express it and the base window logic must not stack a
            # second (compact-position) band on top, so run the base SDPA path
            # under the external mask.
            prev_use_sw = self.use_sliding_window
            self.use_sliding_window = False
            prev_impl = getattr(self.config, "_attn_implementation", "eager")
            if prev_impl == "flash_attention_2":
                self.config._attn_implementation = "sdpa"
            try:
                out, weights, present = UMichSLMAttention.forward(
                    self,
                    hidden_states=hidden_states,
                    attention_mask=ext_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    depth_kv_cache=depth_kv_cache,
                    precomputed_q=q_raw,
                    pre_o_add=pre_o_add,
                )
            finally:
                self.config._attn_implementation = prev_impl
                self.use_sliding_window = prev_use_sw
        else:
            out, weights, present = super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                depth_kv_cache=depth_kv_cache,
                precomputed_q=q_raw,
                pre_o_add=pre_o_add,
            )

        if fuse:
            return out, weights, present  # read already folded in before o_proj
        # q_raw is None in the ordinary unfused case, so the read projects from
        # hidden_states exactly as before; it is non-None only when a caller
        # supplied the query (the pause stream), where projecting from
        # hidden_states would read the bank with the wrong stream's query.
        read = self._register_read(hidden_states, q_raw=q_raw)
        if read is not None:
            out = out + read
        return out, weights, present


class SWARegistersAttention(RegisterReadMixin, UMichSLMAttention):
    """Eager/SDPA registers-SWA layer (base forward dispatches on
    ``config._attn_implementation``, building the dense window mask itself)."""


class SWARegistersFlashAttention2(RegisterReadMixin, UMichSLMFlashAttention2):
    """FA2 registers-SWA layer: native ``window_size`` band, varlen intra-doc.

    With no sequence prefix, ``_intra_doc_force_sdpa`` is never set for this
    arch, so packed intra-doc batches ride ``flash_attn_varlen_func`` at O(S)
    memory — the whole point of the registers design.
    """


SWA_REGISTERS_CLASSES = {
    "eager": SWARegistersAttention,
    "sdpa": SWARegistersAttention,
    "flash_attention_2": SWARegistersFlashAttention2,
}


class FullRegisterReadMixin(RegisterReadMixin):
    """The register read on a FULL-attention layer (``attn_register_count``).

    Same zero-init gated cross-read of the model-level bank as the SWA twins, but
    the layer keeps its unbounded causal span: RegisterReadMixin installs the SWA
    band, and this undoes it. Built for the DeltaNet hybrids, whose attention
    slots are the only softmax layers, so the bank has no SWA layer to serve.
    Takes an attn_logit_sink like any full-attention layer.
    """

    _full_attention_kind = True

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)
        self.sliding_window = None
        self.use_sliding_window = False


class FullRegistersAttention(FullRegisterReadMixin, UMichSLMAttention):
    """Eager/SDPA full-attention layer with the register read."""


class FullRegistersFlashAttention2(FullRegisterReadMixin, UMichSLMFlashAttention2):
    """FA2 full-attention layer with the register read (varlen intra-doc)."""


FULL_REGISTERS_CLASSES = {
    "eager": FullRegistersAttention,
    "sdpa": FullRegistersAttention,
    "flash_attention_2": FullRegistersFlashAttention2,
}


class SWAHybridAttention(SWAWithGlobalsAttention):
    """Half-prefix / half-registers SWA layer (``swa_register_count`` split).

    Inherits ALL prefix machinery from :class:`SWAWithGlobalsAttention` — the
    (smaller) global prefix keeps the joint-softmax sink, the doc-aware
    FlexAttention fast path, the MoD external-mask contract, and visibility to
    the MLA full-attention layers via the residual stream. On top of that, the
    layer adds the same gated position-free register read as the registers
    twin. At init (gates zero) the layer is bitwise a swa_globals layer with
    the smaller prefix.

    Note the hybrid stays on the prefix kernel path (flex/SDPA, never FA2
    varlen) — the split's value is functional (sink + MLA visibility from the
    prefix half, cheap gated capacity + a post-training attachment point from
    the register half), not kernel efficiency.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)
        self.register_gate = nn.Parameter(torch.zeros(self.num_heads))
        object.__setattr__(self, "_register_bank", None)
        # Delta-bank stashes (see RegisterReadMixin.__init__): the hybrid class
        # does not run the mixin's __init__, so mirror them here.
        self._delta_states = None
        self._delta_state_docs = None
        self._delta_token_docs = None
        self._delta_prefix_len = 0
        self._delta_chunk_size = int(getattr(config, "delta_chunk_size", 512))
        # swa_bank_read_interval gate (see RegisterReadMixin.__init__). Also
        # mirrored here: _register_read below is the mixin's, and it calls
        # _bank_read_on(), which reads this attribute. Omitting it raised
        # AttributeError on every swa_hybrid forward.
        self._bank_read_interval = int(getattr(config, "swa_bank_read_interval", 1))
        # swa_fuse_register_proj is deliberately NOT honoured here. This class has
        # its own forward (below) that adds the read AFTER o_proj, so setting the
        # flag would be a silent no-op -- the exact kind of knob-that-does-nothing
        # that has already invalidated two ablations. Pin it False so an inspection
        # of the live module reports the truth.
        self._fuse_register_proj = False

    bind_register_bank = RegisterReadMixin.bind_register_bank
    _bank_read_on = RegisterReadMixin._bank_read_on
    _register_read = RegisterReadMixin._register_read
    _delta_read_merge = RegisterReadMixin._delta_read_merge

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        depth_kv_cache=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        out, weights, present = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            depth_kv_cache=depth_kv_cache,
        )
        # The read covers every position, including the prepended globals —
        # position-free and gated, so it commutes with the MoD gather/scatter
        # exactly as in the registers twin.
        read = self._register_read(hidden_states)
        if read is not None:
            out = out + read
        return out, weights, present


SWA_HYBRID_CLASSES = {
    "eager": SWAHybridAttention,
    "sdpa": SWAHybridAttention,
    "flash_attention_2": SWAHybridAttention,
}
