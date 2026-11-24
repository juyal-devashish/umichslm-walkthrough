"""Sliding-window attention with persistent learnable global tokens.

This is the attention layer used by ``attention_mode == "swa_globals_mla"``. Each
content token attends to:
  - all ``N`` persistent global tokens (always);
  - a sliding window of ``swa_window`` previous content tokens (causal).

Each global token attends to all earlier positions (causal among globals only,
since globals are emitted before content positionally).

Which config key sets the band
------------------------------
This layer reads **``config.swa_window``** (default 1024) and nothing else.
``config.sliding_window`` is a *different* knob, consumed only by
``attention.py`` / ``mla.py`` — i.e. by the periodic **global** MLA layers
(``global_attention_every_n_layers``), not by the SWA band here. Setting
``sliding_window`` to narrow "the window" is a silent no-op for this module:
a 2026-07 ablation did exactly that and trained three supposedly-different
w256/w512/w1024 models that were all identical 1024-window runs (they landed
0.003 apart, inside the 0.0127 seed noise, which is what exposed it).

Implementation notes
--------------------
We subclass :class:`UMichSLMAttention` to reuse Q/K/V/o projections, RoPE init,
GQA setup, and the SDPA/eager forward path. Only the attention mask is different.

The naive ``flash_attn_func(window_size=(W+N, 0))`` trick is **not** equivalent to
"globals + window": the band drifts as content positions grow, so content tokens
beyond ``W`` from the prefix lose access to the globals. We therefore build an
explicit boolean mask via :func:`GlobalTokens.build_attention_mask` and run it
through SDPA. If the SDPA path is too slow at the 100M tier we will switch to
``torch.nn.attention.flex_attention`` — see plan §A risks.

Globals are model-level state (one ``GlobalTokens`` instance on
``UMichSLMModel``). Before each forward, ``UMichSLMModel.forward`` sets
``self_attn.num_protected_prefix = num_globals`` on every SWAWithGlobalsAttention
in the stack so the mask builder knows where the globals end and the content
begins.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .attention import UMichSLMAttention
from .global_tokens import GlobalTokens


class SWAWithGlobalsAttention(UMichSLMAttention):
    """Sliding-window attention with N learnable globals always attended.

    Reuses everything from :class:`UMichSLMAttention` except the mask:
    instead of relying on ``self.use_sliding_window`` (which is gated on
    ``is_global_attention_layer``), we always build the globals + window mask.
    """

    _full_attention_kind = False  # SWA band; never carries an attn_logit_sink

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)
        # The model stashes the current number of prepended globals here at the
        # start of every forward pass. Default 0 means "no globals" (same as a
        # plain SWA layer); the configured swa_window is still used as the band.
        self.num_protected_prefix: int = 0
        self.swa_window: int = int(getattr(config, "swa_window", 1024))
        if self.swa_window <= 0:
            raise ValueError(f"swa_window must be > 0, got {self.swa_window}.")
        # Force eager/SDPA for the SWA-globals *fallback*; FA2's window_size band
        # cannot natively express "all-globals + sliding window" once content
        # tokens are more than window away from the prefix. The fast path uses
        # FlexAttention (see below), which expresses it exactly as a mask_mod.
        impl = getattr(config, "_attn_implementation", "eager")
        if impl == "flash_attention_2":
            self._effective_attn_impl = "sdpa"
        else:
            self._effective_attn_impl = impl
        # Set per-forward with a FlexAttention BlockMask when the fast path runs;
        # the parent dispatch picks it up. None => parent uses SDPA/flash/eager.
        self._flex_block_mask = None
        # Model-driven flex controls, refreshed every forward by UMichSLMModel:
        #   _flex_disable: True when the incoming attention_mask carries padding —
        #     the BlockMask cannot encode it, so flex must yield to the SDPA
        #     additive path (which composes padding + globals + window + docs).
        #   _flex_doc_ids / _flex_doc_cache: set when intra-doc masking is active;
        #     the flex branch then builds a doc-aware BlockMask (globals visible
        #     across docs, content window confined to its document) instead of
        #     the shape-cached one. Without these attrs a doc-masked batch would
        #     silently LEAK cross-document attention on the flex path.
        self._flex_disable = False
        self._flex_doc_ids = None
        self._flex_doc_cache = None

    def _can_use_flex(
        self, past_key_value, output_attentions, depth_kv_cache, hidden_states
    ) -> bool:
        """Whether the FlexAttention fast path applies to this call.

        Restricted to the no-cache (training / full-prefill) CUDA path with no
        attention dropout, no per-weight output, and no MoDA depth KV — the SDPA
        fallback covers generation/cache and CPU. PoPE (doubled head dim) also
        stays on eager.
        """
        from .flex_swa import flex_available

        return bool(
            flex_available()
            and past_key_value is None
            and not output_attentions
            and depth_kv_cache is None
            and getattr(hidden_states, "is_cuda", False)
            and float(getattr(self, "attention_dropout", 0.0) or 0.0) == 0.0
            and not getattr(self, "_use_pope", False)
            and not getattr(self, "_flex_disable", False)
        )

    def _build_globals_window_mask(
        self,
        q_len: int,
        kv_len: int,
        num_globals: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return an SDPA-style additive mask of shape [1, 1, q_len, kv_len].

        Positions 0..num_globals-1 (in the kv axis) are globals; the rest are
        content. The query axis is the trailing q_len positions of the same
        sequence (so position k in q corresponds to position kv_len - q_len + k
        in the global indexing).
        """
        # Build the full [kv_len, kv_len] bool "may attend" mask, then slice the
        # query rows we need.
        content_len = max(0, kv_len - num_globals)
        bool_mask = GlobalTokens.build_attention_mask(
            content_len=content_len,
            window=self.swa_window,
            num_globals=num_globals,
            device=device,
        )
        # Take the q_len rows corresponding to the *last* q_len positions of kv.
        bool_mask = bool_mask[-q_len:, :]
        # Convert to SDPA additive mask: 0 where allowed, -inf where forbidden.
        additive = torch.zeros(q_len, kv_len, device=device, dtype=dtype)
        additive.masked_fill_(~bool_mask, float("-inf"))
        return additive.unsqueeze(0).unsqueeze(0)

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
        # ---- Block-atomic MoD compact path -----------------------------------
        # When the MoD block runner drives this layer over a gathered (compact)
        # sequence, it supplies a complete additive mask built from the tokens'
        # ORIGINAL positions (so the sliding window is honoured despite the
        # sequence being non-contiguous). Bypass the flex/globals-window build
        # entirely and let the parent attend under that mask. RoPE still uses the
        # gathered position_ids passed through.
        ext_mask = getattr(self, "_mod_external_mask", None)
        if ext_mask is not None:
            prev_use_sw = self.use_sliding_window
            self.use_sliding_window = False
            try:
                return super().forward(
                    hidden_states=hidden_states,
                    attention_mask=ext_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    depth_kv_cache=depth_kv_cache,
                )
            finally:
                self.use_sliding_window = prev_use_sw

        # Build the globals + window mask once and pass it through the parent
        # forward. The parent's mask-extension logic is bypassed because
        # use_sliding_window=False on this class (we don't go through
        # is_global_attention_layer); we hand it a complete mask.
        bsz, q_len, _ = hidden_states.size()
        # Compute kv_len: q_len plus past cache (if any).
        past_len = 0
        if past_key_value is not None:
            if hasattr(past_key_value, "get_seq_length"):
                past_len = past_key_value.get_seq_length(self.layer_idx)
            elif isinstance(past_key_value, tuple) and len(past_key_value) >= 1:
                past_len = past_key_value[0].shape[2]
        kv_len = q_len + past_len

        n_globals = int(self.num_protected_prefix)

        # ---- Fast path: FlexAttention block-sparse kernel -------------------
        # Build (once, cached) a BlockMask for the globals+window pattern and let
        # the parent dispatch run the fused kernel. Skips the O(seq^2) dense-mask
        # SDPA path. The mask_mod is verified identical to build_attention_mask.
        if self._can_use_flex(past_key_value, output_attentions, depth_kv_cache, hidden_states):
            from .flex_swa import get_block_mask, get_doc_block_mask

            doc_ids = getattr(self, "_flex_doc_ids", None)
            if doc_ids is not None:
                # Intra-doc masking active: the BlockMask must also encode the
                # document blocks or cross-doc attention leaks. Built once per
                # step via the model-shared per-forward cache.
                block_mask = get_doc_block_mask(
                    doc_ids=doc_ids,
                    num_globals=n_globals,
                    window=self.swa_window,
                    step_cache=getattr(self, "_flex_doc_cache", None),
                )
            else:
                block_mask = get_block_mask(
                    q_len=q_len,
                    kv_len=kv_len,
                    num_globals=n_globals,
                    window=self.swa_window,
                    device=hidden_states.device,
                )
            prev_use_sw = self.use_sliding_window
            self.use_sliding_window = False
            self._flex_block_mask = block_mask
            try:
                return super().forward(
                    hidden_states=hidden_states,
                    attention_mask=None,  # the BlockMask fully encodes the pattern
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    depth_kv_cache=depth_kv_cache,
                )
            finally:
                self._flex_block_mask = None
                self.use_sliding_window = prev_use_sw

        # ---- Fallback: SDPA/eager over a materialized globals+window mask ----
        if n_globals > 0:
            globals_mask = self._build_globals_window_mask(
                q_len=q_len,
                kv_len=kv_len,
                num_globals=n_globals,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            # Compose with any externally-supplied attention_mask
            if attention_mask is not None:
                # Broadcast to [bsz, 1, q_len, kv_len] if needed
                if attention_mask.shape[-1] != kv_len:
                    # External mask shape mismatch — drop it; our globals mask is
                    # already a complete causal+window+globals mask.
                    attention_mask = globals_mask
                else:
                    attention_mask = attention_mask + globals_mask
            else:
                attention_mask = globals_mask
        # Force the parent's SWA-extension code to NOT add another window mask:
        # our mask already encodes the window. We achieve this by clearing the
        # parent's flag for the duration of this call.
        prev_use_sw = self.use_sliding_window
        self.use_sliding_window = False
        try:
            # Force the configured implementation (flash_attention_2 -> sdpa).
            prev_impl = getattr(self.config, "_attn_implementation", "eager")
            self.config._attn_implementation = self._effective_attn_impl
            try:
                out = super().forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    depth_kv_cache=depth_kv_cache,
                )
            finally:
                self.config._attn_implementation = prev_impl
        finally:
            self.use_sliding_window = prev_use_sw
        return out


# Mirror the ATTENTION_CLASSES dict from attention.py so the dispatcher can
# select the right implementation. Both keys point to the same SDPA-backed
# implementation today; FA2 fast path is intentionally absent (see module
# docstring). The dispatcher in transformer_block.py imports this dict lazily.
SWA_GLOBALS_CLASSES = {
    "eager": SWAWithGlobalsAttention,
    "sdpa": SWAWithGlobalsAttention,
    "flash_attention_2": SWAWithGlobalsAttention,
}
