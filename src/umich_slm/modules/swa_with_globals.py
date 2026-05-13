"""Sliding-window attention with persistent learnable global tokens.

This is the attention layer used by ``attention_mode == "swa_globals_mla"``. Each
content token attends to:
  - all ``N`` persistent global tokens (always);
  - a sliding window of ``swa_window`` previous content tokens (causal).

Each global token attends to all earlier positions (causal among globals only,
since globals are emitted before content positionally).

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

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)
        # The model stashes the current number of prepended globals here at the
        # start of every forward pass. Default 0 means "no globals" (same as a
        # plain SWA layer); the configured swa_window is still used as the band.
        self.num_protected_prefix: int = 0
        self.swa_window: int = int(getattr(config, "swa_window", 1024))
        if self.swa_window <= 0:
            raise ValueError(f"swa_window must be > 0, got {self.swa_window}.")
        # Force eager/SDPA for the SWA-globals pattern; FA2's window_size band
        # cannot natively express "all-globals + sliding window" once content
        # tokens are more than window away from the prefix.
        impl = getattr(config, "_attn_implementation", "eager")
        if impl == "flash_attention_2":
            self._effective_attn_impl = "sdpa"
        else:
            self._effective_attn_impl = impl

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
