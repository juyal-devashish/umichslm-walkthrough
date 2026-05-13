"""Hymba hybrid attention-SSM module with learnable meta tokens.

Hymba (NVIDIA, ICLR 2025) combines attention heads and Mamba2 SSM heads in parallel
within each transformer layer. Key innovations include learnable meta tokens that
mitigate attention sinks, per-channel fusion scaling, and cross-layer KV cache sharing.

The HymbaHybridHead module conforms to the same interface as UMichSLMAttention
(returns (hidden_states, attn_weights, present_key_value)), so it can be used
as a drop-in replacement via _get_attention_class() in transformer_block.py.

References:
- Hymba: https://arxiv.org/abs/2411.13676
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .mamba2 import Mamba2SSM, Mamba2Cache
from .hymba_cache import HymbaDynamicCache
from .rmsnorm import RMSNorm

logger = logging.getLogger(__name__)


class MetaTokens(nn.Module):
    """Learnable tokens prepended to every input sequence.

    Meta tokens serve as attention sinks, preventing the model from wasting
    capacity on the first few real tokens. They are prepended before the
    transformer layer loop and stripped after.

    Args:
        num_tokens: Number of meta tokens.
        hidden_size: Hidden dimension size.
    """

    def __init__(self, num_tokens: int, hidden_size: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.tokens = nn.Parameter(torch.randn(num_tokens, hidden_size) * 0.02)

    def forward(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Expand meta tokens for a batch.

        Args:
            batch_size: Batch size.
            dtype: Target dtype.
            device: Target device.

        Returns:
            Meta tokens tensor [batch_size, num_tokens, hidden_size].
        """
        return self.tokens.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=dtype, device=device)


class HymbaHybridHead(nn.Module):
    """Hybrid attention + SSM head for Hymba.

    Combines an inner attention module and a Mamba2 SSM in parallel with learned
    per-channel fusion scaling. Conforms to the same interface as UMichSLMAttention.

    Architecture:
        hidden_states -> [attention path] -> attn_out
        hidden_states -> [SSM path]       -> ssm_out
        output = beta1 * RMSNorm(attn_out) + beta2 * RMSNorm(ssm_out)

    The inner attention class is selected by calling _get_attention_class with
    _skip_hymba=True to prevent infinite recursion.

    Args:
        config: Model configuration.
        layer_idx: Layer index.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Inner attention (GQA, MLA, or standard)
        # Use _skip_hymba=True to avoid recursion
        from .transformer_block import _get_attention_class

        inner_attention_class = _get_attention_class(config, layer_idx, _skip_hymba=True)
        self.attention = inner_attention_class(config=config, layer_idx=layer_idx)

        # SSM component
        self.ssm = Mamba2SSM(
            d_model=config.hidden_size,
            d_state=getattr(config, "hymba_ssm_state_size", 16),
            d_conv=getattr(config, "hymba_ssm_conv_size", 4),
            expand=getattr(config, "hymba_ssm_expand", 2),
        )

        # Fusion: beta1 * RMSNorm(attn_out) + beta2 * RMSNorm(ssm_out)
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.attn_norm = RMSNorm(self.hidden_size, eps=eps)
        self.ssm_norm = RMSNorm(self.hidden_size, eps=eps)
        self.attn_scale = nn.Parameter(torch.ones(self.hidden_size))  # beta1
        self.ssm_scale = nn.Parameter(torch.ones(self.hidden_size))  # beta2

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        depth_kv_cache=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[object]]:
        """Forward pass through hybrid attention + SSM.

        Args:
            hidden_states: Input [batch, seq_len, hidden_size].
            attention_mask: Attention mask.
            position_ids: Position IDs.
            past_key_value: Cache (HymbaDynamicCache or legacy).
            output_attentions: Whether to return attention weights.
            use_cache: Whether to return updated cache.

        Returns:
            Tuple of (output, attn_weights, present_key_value).
        """
        # Split hybrid cache into attention and SSM components
        attn_cache = None
        ssm_cache = None

        if past_key_value is not None:
            if isinstance(past_key_value, HymbaDynamicCache):
                # Extract attention cache as a tuple for the inner attention module
                attn_kv = past_key_value.get_attn_cache(self.layer_idx)
                # Inner attention expects (key, value) tuple or Cache object
                attn_cache = attn_kv
                ssm_cache = past_key_value.get_ssm_cache(self.layer_idx)
            elif isinstance(past_key_value, (list, tuple)):
                # Legacy tuple format
                attn_cache = past_key_value

        # Attention path
        attn_out, attn_weights, attn_present = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=attn_cache,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # SSM path
        ssm_out, ssm_new_cache = self.ssm(
            hidden_states,
            cache=ssm_cache,
            use_cache=use_cache,
        )

        # Fuse: per-channel scaling of normalized outputs
        output = self.attn_scale * self.attn_norm(attn_out) + self.ssm_scale * self.ssm_norm(
            ssm_out
        )

        # Build hybrid cache
        present_key_value = None
        if use_cache:
            if isinstance(past_key_value, HymbaDynamicCache):
                # Store the full attention KV (inner attention already concatenated)
                if attn_present is not None:
                    if isinstance(attn_present, (list, tuple)) and len(attn_present) == 2:
                        past_key_value.set_attn_cache(
                            attn_present[0], attn_present[1], self.layer_idx
                        )
                # Update SSM state
                if ssm_new_cache is not None:
                    past_key_value.update_ssm(ssm_new_cache, self.layer_idx)
                present_key_value = past_key_value
            else:
                # Create a new HymbaDynamicCache
                hymba_cache = HymbaDynamicCache()
                if attn_present is not None:
                    if isinstance(attn_present, (list, tuple)) and len(attn_present) == 2:
                        hymba_cache.set_attn_cache(attn_present[0], attn_present[1], self.layer_idx)
                if ssm_new_cache is not None:
                    hymba_cache.update_ssm(ssm_new_cache, self.layer_idx)
                present_key_value = hymba_cache

        return output, attn_weights, present_key_value
