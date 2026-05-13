"""Multi-head attention with Flash Attention 2 and RoPE/PoPE support.

This module implements the attention mechanism used in modern transformer
architectures with support for:
- Flash Attention 2 for efficient attention computation
- Grouped-Query Attention (GQA) for memory efficiency
- Rotary Position Embeddings (RoPE) or Polar Position Embeddings (PoPE)
- KV caching for efficient autoregressive generation
- Sliding Window Attention (SWA) with Global Refresh

Note on PoPE compatibility:
    PoPE (Polar Position Embeddings) doubles the head dimension for Q/K tensors
    by interleaving x/y components. This is incompatible with Flash Attention
    and SDPA which require Q/K/V to have matching head dimensions. When PoPE
    is enabled, attention automatically falls back to eager implementation.
"""

import logging
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from .rotary import RotaryEmbedding, PolarPositionEmbedding
from .repo import RePoPositionPredictor, RePoPositionPredictorWithBase
from .sliding_window import (
    create_sliding_window_mask,
    is_global_attention_layer,
)

# Check for Flash Attention 2 availability
FLASH_ATTENTION_AVAILABLE = False
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    pass


def _get_unpad_data(attention_mask: torch.Tensor):
    """Get indices and other data needed for unpadding sequences.

    Args:
        attention_mask: Attention mask of shape (batch_size, seq_len).

    Returns:
        Tuple of (indices, cu_seqlens, max_seqlen_in_batch).
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


class UMichSLMAttention(nn.Module):
    """Multi-head attention with RoPE/PoPE and Flash Attention 2 support.

    This implements multi-head self-attention with:
    - Grouped-Query Attention (GQA) via num_key_value_heads
    - Flash Attention 2 as primary implementation (with fallbacks)
    - RoPE or PoPE for position encoding
    - KV cache support for efficient generation

    Args:
        config: Model configuration object.
        layer_idx: Index of this layer (used for KV cache).

    Attention Implementations (selected via config._attn_implementation):
        - "flash_attention_2": Flash Attention 2 (requires flash-attn package)
        - "sdpa": PyTorch's scaled_dot_product_attention
        - "eager": Standard PyTorch attention (fallback)

    Note:
        When using PoPE (Polar Position Embeddings), Flash Attention and SDPA
        are automatically disabled because PoPE doubles Q/K head dimensions
        which is incompatible with these optimized kernels. Eager attention
        is used instead.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        # Sliding Window Attention configuration
        self.sliding_window = getattr(config, "sliding_window", None)
        self.global_attention_every_n_layers = getattr(config, "global_attention_every_n_layers", 4)

        # Determine if this layer uses global or local attention
        if self.sliding_window is not None and layer_idx is not None:
            self.use_sliding_window = not is_global_attention_layer(
                layer_idx, self.global_attention_every_n_layers
            )
        else:
            self.use_sliding_window = False

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads "
                f"(got hidden_size={self.hidden_size} and num_heads={self.num_heads})"
            )

        # Q, K, V projections
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=getattr(config, "attention_bias", False),
        )

        # Position embeddings (RoPE or PoPE)
        pos_type = getattr(config, "position_embedding_type", "rope")
        if pos_type == "pope":
            self.rotary_emb = PolarPositionEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=getattr(config, "rope_theta", 10000.0),
            )
        else:
            rope_scaling = getattr(config, "rope_scaling", None) or {}
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=getattr(config, "rope_theta", 10000.0),
                scaling_factor=rope_scaling.get("factor", 1.0),
                rope_type=rope_scaling.get("type", "default"),
                yarn_beta_fast=rope_scaling.get("beta_fast", 32.0),
                yarn_beta_slow=rope_scaling.get("beta_slow", 1.0),
                yarn_mscale=rope_scaling.get("mscale", 1.0),
            )

        # Cache PoPE check to avoid isinstance() on every forward pass
        # PoPE is incompatible with Flash Attention/SDPA due to head_dim doubling
        self._use_pope = pos_type == "pope"
        self._pope_warning_logged = False

        # DRoPE (Dropped Rotary Position Embeddings) support
        # When drope_enabled=True, position embeddings are skipped entirely
        # This is toggled during training after drope_drop_fraction of steps
        self.drope_enabled = False

        # RePo (Re-Positioning) support - learned continuous positions
        self.use_repo = getattr(config, "use_repo", False)
        repo_start_layer = getattr(config, "repo_start_layer", 5)
        self._repo_active = self.use_repo and (
            layer_idx is not None and layer_idx >= repo_start_layer
        )

        if self._repo_active:
            repo_dim = getattr(config, "repo_dim", None) or (self.hidden_size // 8)
            use_base_positions = getattr(config, "repo_use_base_positions", True)

            if use_base_positions:
                self.repo_predictor = RePoPositionPredictorWithBase(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    repo_dim=repo_dim,
                )
            else:
                self.repo_predictor = RePoPositionPredictor(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    repo_dim=repo_dim,
                )
        else:
            self.repo_predictor = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        depth_kv_cache=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for attention.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).
            attention_mask: Mask of shape (batch, 1, query_len, key_len) or None.
            position_ids: Position indices of shape (batch, seq_len).
            past_key_value: Cached (key, value) tensors for incremental decoding.
            output_attentions: Whether to return attention weights.
            use_cache: Whether to return updated cache.
            depth_kv_cache: Optional MoDADepthKVCache for depth attention.

        Returns:
            Tuple of:
                - Output tensor of shape (batch, seq_len, hidden_size)
                - Attention weights (if output_attentions=True, else None)
                - Updated cache (if use_cache=True, else None)
        """
        bsz, q_len, _ = hidden_states.size()

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Apply position embeddings (RoPE or PoPE) unless DRoPE is active
        if not self.drope_enabled:
            if position_ids is None:
                # Determine past length from cache
                if past_key_value is not None:
                    if hasattr(past_key_value, "get_seq_length"):
                        # Cache object
                        past_len = past_key_value.get_seq_length(self.layer_idx)
                    else:
                        # Legacy tuple format
                        past_len = past_key_value[0].shape[2]
                else:
                    past_len = 0
                position_ids = (
                    torch.arange(past_len, past_len + q_len, device=hidden_states.device)
                    .unsqueeze(0)
                    .expand(bsz, -1)
                )

            # Use RePo (learned continuous positions) or standard RoPE
            if self._repo_active and self.repo_predictor is not None:
                # Predict continuous positions from hidden states
                # hidden_states here is normalized (from input_layernorm in transformer block)
                if hasattr(self.repo_predictor, "forward") and hasattr(
                    self.repo_predictor, "delta_proj"
                ):
                    # RePoPositionPredictorWithBase needs base positions
                    positions = self.repo_predictor(hidden_states, position_ids)
                else:
                    positions = self.repo_predictor(hidden_states)
                query_states, key_states = self.rotary_emb.forward_continuous(
                    query_states, key_states, positions
                )
            else:
                query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)
        # When drope_enabled=True, query_states and key_states are used without position encoding

        # Handle KV cache
        if past_key_value is not None:
            if hasattr(past_key_value, "update"):
                # New Cache object interface
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx
                )
            else:
                # Legacy tuple format
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # Build cache output
        if use_cache:
            if past_key_value is not None and hasattr(past_key_value, "update"):
                # Cache object was already updated
                present_key_value = past_key_value
            else:
                # Return tuple format
                present_key_value = (key_states, value_states)
        else:
            present_key_value = None

        # MoDA: extend KV with depth cache if available
        if depth_kv_cache is not None:
            attn_output, attn_weights = self._moda_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
                depth_kv_cache,
            )
        else:
            # Standard attention path
            # Repeat KV heads for grouped-query attention
            key_states_expanded = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states_expanded = self._repeat_kv(value_states, self.num_key_value_groups)

            # Apply sliding window attention mask if needed
            kv_len = key_states_expanded.shape[2]
            if (
                self.use_sliding_window
                and self.sliding_window is not None
                and kv_len > self.sliding_window
            ):
                swa_mask = create_sliding_window_mask(
                    seq_len=kv_len,
                    window_size=self.sliding_window,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                if q_len < kv_len:
                    swa_mask = swa_mask[:, :, -q_len:, :]
                if attention_mask is not None:
                    attention_mask = attention_mask + swa_mask
                else:
                    attention_mask = swa_mask

            # Choose attention implementation
            attn_implementation = getattr(self.config, "_attn_implementation", "eager")

            if self._use_pope and attn_implementation in ("flash_attention_2", "sdpa"):
                if not self._pope_warning_logged:
                    logger.warning(
                        "PoPE position embeddings are incompatible with %s due to head dimension "
                        "doubling. Falling back to eager attention. This may impact performance.",
                        attn_implementation,
                    )
                    self._pope_warning_logged = True
                attn_implementation = "eager"

            if attn_implementation == "flash_attention_2" and FLASH_ATTENTION_AVAILABLE:
                attn_output = self._flash_attention_forward(
                    query_states, key_states_expanded, value_states_expanded, attention_mask, q_len
                )
                attn_weights = None
            elif attn_implementation == "sdpa":
                attn_output = self._sdpa_attention_forward(
                    query_states, key_states_expanded, value_states_expanded, attention_mask
                )
                attn_weights = None
            else:
                attn_output, attn_weights = self._eager_attention_forward(
                    query_states,
                    key_states_expanded,
                    value_states_expanded,
                    attention_mask,
                    output_attentions,
                )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, present_key_value

    def _moda_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
        depth_kv_cache,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Attention forward with MoDA depth KV extension.

        Extends standard KV with depth KV from preceding layers and computes
        attention under a single unified softmax. Uses SDPA or eager attention
        (FA2 cannot handle arbitrary masks).

        Args:
            query_states: Query tensor [batch, heads, seq_len, head_dim].
            key_states: Key tensor [batch, kv_heads, kv_len, head_dim] (before GQA expansion).
            value_states: Value tensor [batch, kv_heads, kv_len, head_dim].
            attention_mask: Optional attention mask.
            output_attentions: Whether to return attention weights.
            depth_kv_cache: MoDADepthKVCache with depth KV pairs.

        Returns:
            Tuple of (output, attention_weights or None).
        """
        from .moda import build_moda_combined_mask

        depth_kv = depth_kv_cache.get_depth_kv(self.layer_idx)

        if depth_kv is None:
            # No depth KV available yet (first layer), run standard attention
            key_expanded = self._repeat_kv(key_states, self.num_key_value_groups)
            value_expanded = self._repeat_kv(value_states, self.num_key_value_groups)
            return self._eager_attention_forward(
                query_states, key_expanded, value_expanded, attention_mask, output_attentions
            )

        k_depth, v_depth = depth_kv
        # k_depth, v_depth: (batch, num_kv_heads, depth_seq_len, head_dim)

        # Expand sequence KV for GQA
        key_expanded = self._repeat_kv(key_states, self.num_key_value_groups)
        value_expanded = self._repeat_kv(value_states, self.num_key_value_groups)

        # Expand depth KV for GQA
        k_depth_expanded = self._repeat_kv(k_depth, self.num_key_value_groups)
        v_depth_expanded = self._repeat_kv(v_depth, self.num_key_value_groups)

        # Concatenate sequence + depth KV along sequence dimension
        key_extended = torch.cat([key_expanded, k_depth_expanded], dim=2)
        value_extended = torch.cat([value_expanded, v_depth_expanded], dim=2)

        q_len = query_states.shape[2]
        seq_kv_len = key_expanded.shape[2]
        depth_seq_len = k_depth_expanded.shape[2]

        # Determine number of depth layers for mask construction
        # Each depth entry covers the original seq_len
        orig_seq_len = key_states.shape[2]  # before GQA expansion, seq dimension
        num_depth_layers = depth_seq_len // orig_seq_len if orig_seq_len > 0 else 0

        # Build combined mask: causal for seq + block-diagonal for depth
        past_len = seq_kv_len - q_len
        combined_mask = build_moda_combined_mask(
            seq_len=q_len,
            depth_seq_len=depth_seq_len,
            num_depth_layers=num_depth_layers,
            device=query_states.device,
            dtype=query_states.dtype,
            past_len=past_len,
        )

        # Use SDPA with explicit mask (FA2 can't handle arbitrary masks)
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_extended,
            value_extended,
            attn_mask=combined_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attn_weights = None

        if output_attentions:
            # Fall back to eager for attention weights
            actual_head_dim = query_states.shape[-1]
            attn_weights = torch.matmul(query_states, key_extended.transpose(2, 3)) / math.sqrt(
                actual_head_dim
            )
            attn_weights = attn_weights + combined_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )

        return attn_output, attn_weights

    def _flash_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        query_length: int,
    ) -> torch.Tensor:
        """Flash Attention 2 forward pass.

        Args:
            query_states: Query tensor [batch, heads, seq_len, head_dim].
            key_states: Key tensor [batch, heads, kv_len, head_dim].
            value_states: Value tensor [batch, heads, kv_len, head_dim].
            attention_mask: Optional attention mask.
            query_length: Length of query sequence.

        Returns:
            Attention output tensor.

        Raises:
            RuntimeError: If called with PoPE enabled (Q/K have doubled head_dim).
        """
        # Safety check: Flash Attention requires Q, K, V to have compatible head dimensions
        if self._use_pope:
            raise RuntimeError(
                "Flash Attention is incompatible with PoPE due to head dimension mismatch. "
                "This is a bug - PoPE should route to eager attention."
            )

        # Transpose for flash attention: [batch, seq_len, heads, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # FlashAttention only supports fp16 and bf16
        # Convert to bf16 if needed (FlashAttention is GPU-only, modern GPUs support bf16)
        original_dtype = query_states.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            # Convert to bf16 (preferred for training) or fp16 if bf16 not available
            target_dtype = torch.bfloat16
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Determine if causal
        causal = query_length != 1

        # Flash attention
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=self.attention_dropout if self.training else 0.0,
            causal=causal,
        )

        # Convert back to original dtype if needed
        if attn_output.dtype != original_dtype:
            attn_output = attn_output.to(original_dtype)

        return attn_output.transpose(1, 2)

    def _sdpa_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """PyTorch SDPA forward pass.

        Uses torch.nn.functional.scaled_dot_product_attention which can
        automatically use FlashAttention or Memory-Efficient Attention
        backends when available.

        Args:
            query_states: Query tensor [batch, heads, seq_len, head_dim].
            key_states: Key tensor [batch, heads, kv_len, head_dim].
            value_states: Value tensor [batch, heads, kv_len, head_dim].
            attention_mask: Optional attention mask.

        Returns:
            Attention output tensor.

        Raises:
            RuntimeError: If called with PoPE enabled (Q/K have doubled head_dim).
        """
        # Safety check: SDPA requires Q, K, V to have compatible head dimensions
        if self._use_pope:
            raise RuntimeError(
                "SDPA is incompatible with PoPE due to head dimension mismatch. "
                "This is a bug - PoPE should route to eager attention."
            )

        # Determine if we need causal masking
        is_causal = attention_mask is None and query_states.shape[2] > 1

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        return attn_output

    def _eager_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Standard PyTorch attention (fallback implementation).

        This is the reference implementation that works on all hardware
        but is slower than Flash Attention or SDPA.

        Args:
            query_states: Query tensor [batch, heads, seq_len, head_dim].
            key_states: Key tensor [batch, heads, kv_len, head_dim].
            value_states: Value tensor [batch, heads, kv_len, head_dim].
            attention_mask: Optional attention mask.
            output_attentions: Whether to return attention weights.

        Returns:
            Tuple of (output, attention_weights or None).
        """
        # Compute attention scores
        # Use actual head_dim from tensor (may differ from self.head_dim if using PoPE)
        # Note: PoPE doubles head_dim by interleaving (x, y) components. The dot product
        # sum(q_x*k_x + q_y*k_y) is mathematically equivalent to the polar form, and
        # scaling by sqrt(actual_head_dim) maintains numerical stability.
        actual_head_dim = query_states.shape[-1]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            actual_head_dim
        )

        # Apply causal mask
        seq_len = query_states.shape[2]
        kv_len = key_states.shape[2]

        # Check if we need to create a fresh causal mask
        # This happens when: (1) no mask provided, or (2) mask dimensions don't match
        # (e.g., when PoPE forces eager attention but mask was prepared for SDPA)
        needs_fresh_mask = attention_mask is None
        if attention_mask is not None and attention_mask.shape[-1] != kv_len:
            needs_fresh_mask = True
            if self._use_pope and not getattr(self, "_pope_mask_warning_logged", False):
                logger.warning(
                    "Attention mask dimensions don't match key_states (mask: %d, keys: %d). "
                    "This can happen when PoPE forces eager attention but mask was prepared "
                    "for SDPA. Creating fresh causal mask.",
                    attention_mask.shape[-1],
                    kv_len,
                )
                self._pope_mask_warning_logged = True

        if needs_fresh_mask:
            # Create causal mask from scratch
            causal_mask = torch.triu(
                torch.full((seq_len, kv_len), float("-inf"), device=query_states.device),
                diagonal=kv_len - seq_len + 1,
            )
            attn_weights = attn_weights + causal_mask
        else:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads for grouped-query attention.

        This expands the KV heads to match the number of query heads
        for efficient batched computation.

        Args:
            hidden_states: Tensor of shape [batch, num_kv_heads, seq_len, head_dim].
            n_rep: Number of times to repeat each KV head.

        Returns:
            Expanded tensor of shape [batch, num_kv_heads * n_rep, seq_len, head_dim].
        """
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class UMichSLMFlashAttention2(UMichSLMAttention):
    """Flash Attention 2 variant of UMichSLMAttention.

    This class forces the use of Flash Attention 2 and provides
    optimized handling for variable-length sequences.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not FLASH_ATTENTION_AVAILABLE:
            raise ImportError(
                "Flash Attention 2 is not available. Please install flash-attn: "
                "pip install flash-attn --no-build-isolation"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        depth_kv_cache=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass using Flash Attention 2.

        This override ensures Flash Attention 2 is always used.
        Falls back to base class for MoDA (needs arbitrary masks).
        """
        # Fall back to base class for unsupported cases:
        # - output_attentions: Flash Attention doesn't support attention weights output
        # - PoPE: head_dim doubling is incompatible with Flash Attention
        # - MoDA: FA2 can't handle arbitrary masks needed for depth attention
        if output_attentions or self._use_pope or depth_kv_cache is not None:
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                depth_kv_cache=depth_kv_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Apply position embeddings unless DRoPE is active
        if not self.drope_enabled:
            if position_ids is None:
                past_len = past_key_value[0].shape[2] if past_key_value is not None else 0
                position_ids = (
                    torch.arange(past_len, past_len + q_len, device=hidden_states.device)
                    .unsqueeze(0)
                    .expand(bsz, -1)
                )

            # Use RePo (learned continuous positions) or standard RoPE
            if self._repo_active and self.repo_predictor is not None:
                if hasattr(self.repo_predictor, "delta_proj"):
                    positions = self.repo_predictor(hidden_states, position_ids)
                else:
                    positions = self.repo_predictor(hidden_states)
                query_states, key_states = self.rotary_emb.forward_continuous(
                    query_states, key_states, positions
                )
            else:
                query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)
        # When drope_enabled=True, query_states and key_states are used without position encoding

        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV for GQA
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Transpose for flash attention: [batch, seq, heads, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # FlashAttention only supports fp16 and bf16
        # Convert to bf16 if needed (FlashAttention is GPU-only, modern GPUs support bf16)
        original_dtype = query_states.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            # Convert to bf16 (preferred for training) or fp16 if bf16 not available
            target_dtype = torch.bfloat16
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Determine sliding window for flash attention
        window_size_param = (-1, -1)  # No window (full attention)
        if self.use_sliding_window and self.sliding_window is not None:
            window_size_param = (self.sliding_window, 0)  # Left window only for causal

        # Flash attention with optional sliding window
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=self.attention_dropout if self.training else 0.0,
            causal=q_len != 1,
            window_size=window_size_param,
        )

        # Convert back to original dtype if needed
        if attn_output.dtype != original_dtype:
            attn_output = attn_output.to(original_dtype)

        # Reshape and project
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class UMichSLMSdpaAttention(UMichSLMAttention):
    """SDPA variant of UMichSLMAttention.

    This class forces the use of PyTorch's scaled_dot_product_attention
    which can leverage FlashAttention or Memory-Efficient backends.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        depth_kv_cache=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass using SDPA."""
        # Fall back to base class for unsupported cases:
        # - output_attentions: SDPA doesn't support attention weights output
        # - PoPE: head_dim doubling is incompatible with SDPA
        # - MoDA: base class handles depth KV extension with combined mask
        if output_attentions or self._use_pope or depth_kv_cache is not None:
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                depth_kv_cache=depth_kv_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Apply position embeddings unless DRoPE is active
        if not self.drope_enabled:
            if position_ids is None:
                # Determine past length from cache
                if past_key_value is not None:
                    if hasattr(past_key_value, "get_seq_length"):
                        past_len = past_key_value.get_seq_length(self.layer_idx)
                    else:
                        past_len = past_key_value[0].shape[2]
                else:
                    past_len = 0
                position_ids = (
                    torch.arange(past_len, past_len + q_len, device=hidden_states.device)
                    .unsqueeze(0)
                    .expand(bsz, -1)
                )

            # Use RePo (learned continuous positions) or standard RoPE
            if self._repo_active and self.repo_predictor is not None:
                if hasattr(self.repo_predictor, "delta_proj"):
                    positions = self.repo_predictor(hidden_states, position_ids)
                else:
                    positions = self.repo_predictor(hidden_states)
                query_states, key_states = self.rotary_emb.forward_continuous(
                    query_states, key_states, positions
                )
            else:
                query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)
        # When drope_enabled=True, query_states and key_states are used without position encoding

        # Handle KV cache
        if past_key_value is not None:
            if hasattr(past_key_value, "update"):
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx
                )
            else:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # Build cache output
        if use_cache:
            if past_key_value is not None and hasattr(past_key_value, "update"):
                present_key_value = past_key_value
            else:
                present_key_value = (key_states, value_states)
        else:
            present_key_value = None

        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Apply sliding window attention mask if needed
        kv_len = key_states.shape[2]
        if (
            self.use_sliding_window
            and self.sliding_window is not None
            and kv_len > self.sliding_window
        ):
            swa_mask = create_sliding_window_mask(
                seq_len=kv_len,
                window_size=self.sliding_window,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            if q_len < kv_len:
                swa_mask = swa_mask[:, :, -q_len:, :]

            if attention_mask is not None:
                attention_mask = attention_mask + swa_mask
            else:
                attention_mask = swa_mask

        # Use SDPA
        is_causal = attention_mask is None and q_len > 1

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, present_key_value


ATTENTION_CLASSES = {
    "eager": UMichSLMAttention,
    "sdpa": UMichSLMSdpaAttention,
    "flash_attention_2": UMichSLMFlashAttention2,
}


def get_attention_class(attn_implementation: str = "eager"):
    """Get the attention class for the given implementation.

    Args:
        attn_implementation: One of "eager", "sdpa", or "flash_attention_2".

    Returns:
        The attention class.
    """
    if attn_implementation not in ATTENTION_CLASSES:
        raise ValueError(
            f"Unknown attention implementation: {attn_implementation}. "
            f"Available: {list(ATTENTION_CLASSES.keys())}"
        )
    return ATTENTION_CLASSES[attn_implementation]
