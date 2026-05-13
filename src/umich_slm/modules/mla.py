"""Multi-head Latent Attention (MLA) implementation.

This module implements Multi-head Latent Attention as described in DeepSeek V2/V3:
https://arxiv.org/abs/2405.04434

MLA compresses Key-Value pairs into a low-rank latent representation, dramatically
reducing KV cache memory during inference. Key features:
- Low-rank KV compression via learned projections
- Decoupled RoPE for position encoding (separate from compressed KV)
- Optional Q compression for further memory savings
- Compatible with sliding window attention

The core idea is that instead of caching full K and V tensors, we cache:
1. A compressed latent vector c_kv (much smaller)
2. A small RoPE-encoded key component k_rope

During attention, K and V are reconstructed from the latent + RoPE keys.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rotary import RotaryEmbedding
from .sliding_window import create_sliding_window_mask, is_global_attention_layer
from .repo import RePoPositionPredictor, RePoPositionPredictorWithBase


class UMichSLMMLAAttention(nn.Module):
    """Multi-head Latent Attention with KV compression.

    This implements MLA as described in DeepSeek V2/V3, which compresses KV
    into a low-rank latent space for memory-efficient inference.

    Architecture:
        1. Compress hidden states to KV latent: c_kv = W_dkv @ x
        2. Decompress to get K (non-RoPE part) and V: k_nope, v = W_uk/uv @ c_kv
        3. Compute RoPE keys separately: k_rope = RoPE(W_kr @ x)
        4. Concatenate K: k = [k_nope, k_rope]
        5. Optionally compress Q similarly
        6. Compute attention with reconstructed K, V

    KV Cache stores only:
        - c_kv: [batch, seq, kv_lora_rank] - compressed latent
        - k_rope: [batch, seq, num_heads, qk_rope_head_dim] - RoPE keys

    Args:
        config: Model configuration with MLA parameters.
        layer_idx: Index of this layer (used for cache and SWA).
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        # MLA dimensions from config
        self.kv_lora_rank = getattr(config, "mla_kv_lora_rank", 512)
        self.q_lora_rank = getattr(config, "mla_q_lora_rank", 0)  # 0 = no Q compression
        self.qk_nope_head_dim = getattr(config, "mla_qk_nope_head_dim", 128)
        self.qk_rope_head_dim = getattr(config, "mla_qk_rope_head_dim", 64)
        self.v_head_dim = getattr(config, "mla_v_head_dim", 128)

        # Total Q/K head dimension
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # Sliding Window Attention configuration
        self.sliding_window = getattr(config, "sliding_window", None)
        self.global_attention_every_n_layers = getattr(config, "global_attention_every_n_layers", 4)

        # Determine if this layer uses sliding window
        if self.sliding_window is not None and layer_idx is not None:
            self.use_sliding_window = not is_global_attention_layer(
                layer_idx, self.global_attention_every_n_layers
            )
        else:
            self.use_sliding_window = False

        # KV Compression projections
        # Down-projection: hidden_size -> kv_lora_rank (with layernorm)
        self.kv_down_proj = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank,
            bias=False,
        )
        self.kv_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=getattr(config, "rms_norm_eps", 1e-6))

        # Up-projections: kv_lora_rank -> K (nope) and V
        # K nope: kv_lora_rank -> num_heads * qk_nope_head_dim
        self.k_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * self.qk_nope_head_dim,
            bias=False,
        )
        # V: kv_lora_rank -> num_heads * v_head_dim
        self.v_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * self.v_head_dim,
            bias=False,
        )

        # RoPE key projection: hidden_size -> num_heads * qk_rope_head_dim
        self.k_rope_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.qk_rope_head_dim,
            bias=False,
        )

        # Q projections (with optional compression)
        if self.q_lora_rank > 0:
            # Compressed Q path
            self.q_down_proj = nn.Linear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
            )
            self.q_layernorm = nn.RMSNorm(
                self.q_lora_rank, eps=getattr(config, "rms_norm_eps", 1e-6)
            )
            # Q nope: q_lora_rank -> num_heads * qk_nope_head_dim
            self.q_up_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.qk_nope_head_dim,
                bias=False,
            )
            # Q rope: q_lora_rank -> num_heads * qk_rope_head_dim
            self.q_rope_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.qk_rope_head_dim,
                bias=False,
            )
        else:
            # Standard Q projection (no compression)
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )

        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
        )

        # RoPE for position encoding (only applied to the rope dimensions)
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        self.rotary_emb = RotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000.0),
            scaling_factor=rope_scaling.get("factor", 1.0),
            rope_type=rope_scaling.get("type", "default"),
            yarn_beta_fast=rope_scaling.get("beta_fast", 32.0),
            yarn_beta_slow=rope_scaling.get("beta_slow", 1.0),
            yarn_mscale=rope_scaling.get("mscale", 1.0),
        )

        # RePo (learned position embeddings) support
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
                    position_scale=1.0,
                    bias=False,
                )
            else:
                self.repo_predictor = RePoPositionPredictor(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    repo_dim=repo_dim,
                    bias=False,
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
        """Forward pass for MLA.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            attention_mask: Attention mask [batch, 1, q_len, kv_len] or None.
            position_ids: Position indices [batch, seq_len].
            past_key_value: Cached (c_kv, k_rope) tensors for incremental decoding.
            output_attentions: Whether to return attention weights.
            use_cache: Whether to return updated cache.

        Returns:
            Tuple of:
                - Output tensor [batch, seq_len, hidden_size]
                - Attention weights (if output_attentions=True, else None)
                - Updated cache tuple (c_kv, k_rope) or None
        """
        bsz, q_len, _ = hidden_states.size()

        # =============== Compute Q ===============
        if self.q_lora_rank > 0:
            # Compressed Q path
            q_compressed = self.q_down_proj(hidden_states)
            q_compressed = self.q_layernorm(q_compressed)
            q_nope = self.q_up_proj(q_compressed)  # [batch, seq, num_heads * qk_nope_head_dim]
            q_rope = self.q_rope_proj(q_compressed)  # [batch, seq, num_heads * qk_rope_head_dim]

            # Reshape
            q_nope = q_nope.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim)
            q_rope = q_rope.view(bsz, q_len, self.num_heads, self.qk_rope_head_dim)
        else:
            # Standard Q (no compression)
            query_states = self.q_proj(hidden_states)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.qk_head_dim)

            # Split into nope and rope parts
            q_nope = query_states[..., : self.qk_nope_head_dim]
            q_rope = query_states[..., self.qk_nope_head_dim :]

        # =============== Compute KV ===============
        # Compress to latent
        c_kv = self.kv_down_proj(hidden_states)  # [batch, seq, kv_lora_rank]
        c_kv = self.kv_layernorm(c_kv)

        # Decompress K (non-RoPE part) and V
        k_nope = self.k_up_proj(c_kv)  # [batch, seq, num_heads * qk_nope_head_dim]
        v_states = self.v_up_proj(c_kv)  # [batch, seq, num_heads * v_head_dim]

        # Compute RoPE keys
        k_rope = self.k_rope_proj(hidden_states)  # [batch, seq, num_heads * qk_rope_head_dim]

        # Reshape
        k_nope = k_nope.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim)
        v_states = v_states.view(bsz, q_len, self.num_heads, self.v_head_dim)
        k_rope = k_rope.view(bsz, q_len, self.num_heads, self.qk_rope_head_dim)

        # =============== Handle KV Cache - Get past length ===============
        past_len = 0
        past_c_kv = None
        past_k_rope = None

        if past_key_value is not None:
            # Handle both HuggingFace Cache object and tuple formats
            if hasattr(past_key_value, "get_seq_length"):
                # HuggingFace Cache object - extract our MLA cache from it
                # The cache stores our (c_kv, k_rope) tuple at layer_idx
                try:
                    layer_cache = (
                        past_key_value[self.layer_idx]
                        if self.layer_idx is not None
                        else past_key_value[0]
                    )
                    if layer_cache is not None:
                        if isinstance(layer_cache, tuple) and len(layer_cache) == 2:
                            c_kv_candidate, k_rope_candidate = layer_cache
                            if c_kv_candidate is not None and k_rope_candidate is not None:
                                past_c_kv = c_kv_candidate
                                past_k_rope = k_rope_candidate
                                past_len = past_c_kv.shape[1]
                except (IndexError, KeyError, TypeError):
                    # Cache doesn't have this layer yet
                    pass
            elif isinstance(past_key_value, tuple) and len(past_key_value) == 2:
                # Direct tuple format: (c_kv, k_rope)
                c_kv_candidate, k_rope_candidate = past_key_value
                if c_kv_candidate is not None and k_rope_candidate is not None:
                    # Make sure these are tensors, not tuples from multi-layer cache
                    if torch.is_tensor(c_kv_candidate):
                        past_c_kv = c_kv_candidate
                        past_k_rope = k_rope_candidate
                        past_len = past_c_kv.shape[1]

        # =============== Apply RoPE ===============
        if position_ids is None:
            position_ids = (
                torch.arange(past_len, past_len + q_len, device=hidden_states.device)
                .unsqueeze(0)
                .expand(bsz, -1)
            )

        # Transpose for RoPE: [batch, num_heads, seq, head_dim]
        q_rope = q_rope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)

        # Apply RoPE to the rope parts only (with RePo if active)
        if self._repo_active and self.repo_predictor is not None:
            # Use learned continuous positions
            if hasattr(self.repo_predictor, "delta_proj"):
                positions = self.repo_predictor(hidden_states, position_ids)
            else:
                positions = self.repo_predictor(hidden_states)
            q_rope, k_rope = self.rotary_emb.forward_continuous(q_rope, k_rope, positions)
        else:
            q_rope, k_rope = self.rotary_emb(q_rope, k_rope, position_ids)

        # Transpose back: [batch, seq, num_heads, head_dim]
        q_rope = q_rope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)

        # =============== Handle KV Cache - Concatenate ===============
        if past_c_kv is not None and past_k_rope is not None:
            c_kv = torch.cat([past_c_kv, c_kv], dim=1)
            k_rope = torch.cat([past_k_rope, k_rope], dim=1)

            # Recompute k_nope and v from full c_kv
            k_nope = self.k_up_proj(c_kv)
            v_states = self.v_up_proj(c_kv)
            k_nope = k_nope.view(bsz, -1, self.num_heads, self.qk_nope_head_dim)
            v_states = v_states.view(bsz, -1, self.num_heads, self.v_head_dim)

        # Build cache output
        if use_cache:
            present_key_value = (c_kv, k_rope)
        else:
            present_key_value = None

        # =============== Concatenate K parts ===============
        # k_full: [batch, kv_len, num_heads, qk_head_dim]
        k_states = torch.cat([k_nope, k_rope], dim=-1)

        # Also concatenate Q parts
        # q_full: [batch, q_len, num_heads, qk_head_dim]
        q_states = torch.cat([q_nope, q_rope], dim=-1)

        # =============== Attention computation ===============
        # Transpose for attention: [batch, num_heads, seq, head_dim]
        q_states = q_states.transpose(1, 2)
        k_states = k_states.transpose(1, 2)
        v_states = v_states.transpose(1, 2)

        kv_len = k_states.shape[2]

        # Apply sliding window mask if needed
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

        # Compute attention scores
        # Q: [batch, heads, q_len, qk_head_dim]
        # K: [batch, heads, kv_len, qk_head_dim]
        attn_weights = torch.matmul(q_states, k_states.transpose(2, 3)) / math.sqrt(
            self.qk_head_dim
        )

        # Apply causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.full((q_len, kv_len), float("-inf"), device=q_states.device),
                diagonal=kv_len - q_len + 1,
            )
            attn_weights = attn_weights + causal_mask
        else:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Apply to values
        # V: [batch, heads, kv_len, v_head_dim]
        attn_output = torch.matmul(attn_weights, v_states)

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, present_key_value

    def get_cache_size_per_token(self) -> int:
        """Return the cache size per token in elements.

        For MLA, we cache:
        - c_kv: kv_lora_rank elements
        - k_rope: num_heads * qk_rope_head_dim elements

        This is much smaller than standard attention which caches:
        - K: num_heads * head_dim
        - V: num_heads * head_dim

        Returns:
            Number of elements per token in the cache.
        """
        return self.kv_lora_rank + self.num_heads * self.qk_rope_head_dim

    def get_standard_cache_size_per_token(self) -> int:
        """Return what the cache size would be for standard MHA.

        Returns:
            Number of elements per token for standard K+V cache.
        """
        return 2 * self.num_heads * self.qk_head_dim  # K + V


class UMichSLMMLASdpaAttention(UMichSLMMLAAttention):
    """MLA with SDPA backend.

    Uses PyTorch's scaled_dot_product_attention for potentially better
    performance on supported hardware.
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
        if output_attentions:
            # SDPA doesn't support attention weights
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        # =============== Compute Q ===============
        if self.q_lora_rank > 0:
            q_compressed = self.q_down_proj(hidden_states)
            q_compressed = self.q_layernorm(q_compressed)
            q_nope = self.q_up_proj(q_compressed)
            q_rope = self.q_rope_proj(q_compressed)
            q_nope = q_nope.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim)
            q_rope = q_rope.view(bsz, q_len, self.num_heads, self.qk_rope_head_dim)
        else:
            query_states = self.q_proj(hidden_states)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.qk_head_dim)
            q_nope = query_states[..., : self.qk_nope_head_dim]
            q_rope = query_states[..., self.qk_nope_head_dim :]

        # =============== Compute KV ===============
        c_kv = self.kv_down_proj(hidden_states)
        c_kv = self.kv_layernorm(c_kv)
        k_nope = self.k_up_proj(c_kv)
        v_states = self.v_up_proj(c_kv)
        k_rope = self.k_rope_proj(hidden_states)

        k_nope = k_nope.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim)
        v_states = v_states.view(bsz, q_len, self.num_heads, self.v_head_dim)
        k_rope = k_rope.view(bsz, q_len, self.num_heads, self.qk_rope_head_dim)

        # =============== Handle KV Cache - Get past length ===============
        past_len = 0
        past_c_kv = None
        past_k_rope = None

        if past_key_value is not None:
            if hasattr(past_key_value, "get_seq_length"):
                try:
                    layer_cache = (
                        past_key_value[self.layer_idx]
                        if self.layer_idx is not None
                        else past_key_value[0]
                    )
                    if layer_cache is not None:
                        if isinstance(layer_cache, tuple) and len(layer_cache) == 2:
                            c_kv_candidate, k_rope_candidate = layer_cache
                            if c_kv_candidate is not None and k_rope_candidate is not None:
                                past_c_kv = c_kv_candidate
                                past_k_rope = k_rope_candidate
                                past_len = past_c_kv.shape[1]
                except (IndexError, KeyError, TypeError):
                    pass
            elif isinstance(past_key_value, tuple) and len(past_key_value) == 2:
                c_kv_candidate, k_rope_candidate = past_key_value
                if c_kv_candidate is not None and k_rope_candidate is not None:
                    if torch.is_tensor(c_kv_candidate):
                        past_c_kv = c_kv_candidate
                        past_k_rope = k_rope_candidate
                        past_len = past_c_kv.shape[1]

        # =============== Apply RoPE ===============
        if position_ids is None:
            position_ids = (
                torch.arange(past_len, past_len + q_len, device=hidden_states.device)
                .unsqueeze(0)
                .expand(bsz, -1)
            )

        q_rope = q_rope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)

        # Apply RoPE (with RePo if active)
        if self._repo_active and self.repo_predictor is not None:
            # Use learned continuous positions
            if hasattr(self.repo_predictor, "delta_proj"):
                positions = self.repo_predictor(hidden_states, position_ids)
            else:
                positions = self.repo_predictor(hidden_states)
            q_rope, k_rope = self.rotary_emb.forward_continuous(q_rope, k_rope, positions)
        else:
            q_rope, k_rope = self.rotary_emb(q_rope, k_rope, position_ids)

        q_rope = q_rope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)

        # =============== Handle KV Cache - Concatenate ===============
        if past_c_kv is not None and past_k_rope is not None:
            c_kv = torch.cat([past_c_kv, c_kv], dim=1)
            k_rope = torch.cat([past_k_rope, k_rope], dim=1)
            k_nope = self.k_up_proj(c_kv)
            v_states = self.v_up_proj(c_kv)
            k_nope = k_nope.view(bsz, -1, self.num_heads, self.qk_nope_head_dim)
            v_states = v_states.view(bsz, -1, self.num_heads, self.v_head_dim)

        present_key_value = (c_kv, k_rope) if use_cache else None

        # =============== Concatenate and transpose ===============
        k_states = torch.cat([k_nope, k_rope], dim=-1)
        q_states = torch.cat([q_nope, q_rope], dim=-1)

        q_states = q_states.transpose(1, 2)
        k_states = k_states.transpose(1, 2)
        v_states = v_states.transpose(1, 2)

        kv_len = k_states.shape[2]

        # Apply sliding window mask
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
            q_states,
            k_states,
            v_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, present_key_value


MLA_ATTENTION_CLASSES = {
    "eager": UMichSLMMLAAttention,
    "sdpa": UMichSLMMLASdpaAttention,
}


def get_mla_attention_class(attn_implementation: str = "eager"):
    """Get the MLA attention class for the given implementation.

    Args:
        attn_implementation: One of "eager" or "sdpa".

    Returns:
        The MLA attention class.
    """
    if attn_implementation not in MLA_ATTENTION_CLASSES:
        raise ValueError(
            f"Unknown attention implementation for MLA: {attn_implementation}. "
            f"Available: {list(MLA_ATTENTION_CLASSES.keys())}"
        )
    return MLA_ATTENTION_CLASSES[attn_implementation]
