"""Multi-Token Prediction (MTP) Module.

This module implements Multi-Token Prediction as described in:
- "Better & Faster Large Language Models via Multi-Token Prediction" (Gloeckle et al., 2024)
- DeepSeek V3 Technical Report (2024)

Architecture Overview
---------------------
Multi-Token Prediction trains the model to predict multiple future tokens simultaneously,
providing denser training signals and improving sample efficiency. Unlike standard
next-token prediction which only supervises on position i+1, MTP adds supervision for
positions i+2, i+3, ..., i+D+1 where D is the number of prediction heads.

The DeepSeek V3 approach uses a sequential (causal) chain of MTP modules:

    Main Model: h_i predicts t_{i+1} (standard next-token prediction)

    MTP Depth 0: h^1_i = MTP_0(h_i, embed(t_{i+1})) predicts t_{i+2}
    MTP Depth 1: h^2_i = MTP_1(h^1_i, embed(t_{i+2})) predicts t_{i+3}
    ...
    MTP Depth D-1: h^D_i = MTP_{D-1}(h^{D-1}_i, embed(t_{i+D})) predicts t_{i+D+1}

Each MTP module at depth k:
1. Takes representation h^k_i from previous depth (or h_i from main model if k=0)
2. Incorporates the embedding of the "known" token t_{i+k+1}
3. Projects the combined representation through lightweight transformer layers
4. Directly predicts the token t_{i+k+2} using the LM head (no additional shift!)

Key Design Decisions
--------------------
1. **Input tokens, not labels**: We use actual input token IDs for embeddings, not labels.
   Labels may contain -100 (ignore_index) which cannot be used for embedding lookup.

2. **No padding in hidden states**: We avoid padding with zeros in the sequential chain,
   as zeros corrupt the learned representations. Instead, we work with variable-length
   sequences that naturally shrink with depth.

3. **Direct prediction (no shift)**: Unlike the main model which shifts logits/labels,
   MTP modules directly predict the target. The "shift" is implicit in the architecture:
   combining h_i with embed(t_{i+1}) produces a representation that directly targets t_{i+2}.

4. **Shared loss function**: CrossEntropyLoss is instantiated once in __init__ to avoid
   overhead of creating it in each forward pass.

5. **Dropout for regularization**: Optional dropout between MTP layers via `mtp_dropout`.

Training Dynamics
-----------------
- MTP loss is weighted (typically 0.1-0.3) and added to the main next-token loss
- Deeper predictions are harder but provide complementary gradients
- The sequential structure enables speculative decoding at inference time

References
----------
- Meta Multi-Token Prediction: https://arxiv.org/abs/2404.19737
- DeepSeek V3: https://arxiv.org/abs/2412.19437
"""

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput

from .rmsnorm import RMSNorm
from .swiglu import SwiGLUMLP


@dataclass
class MTPOutput(ModelOutput):
    """Output container for Multi-Token Prediction.

    This dataclass follows HuggingFace conventions for model outputs,
    allowing attribute and dict-style access.

    Attributes:
        loss: Total MTP loss (weighted average across all prediction depths).
            Shape: scalar tensor. Returns 0.0 if sequence is too short for MTP.
        losses_per_depth: List of individual loss values for each prediction depth.
            Length equals the number of active depths (may be less than num_heads
            for short sequences). Each element is a scalar tensor.
        logits_per_depth: List of prediction logits for each depth.
            Each element has shape [batch, valid_seq_len, vocab_size] where
            valid_seq_len decreases with depth.
        num_active_depths: Number of depths that had valid predictions.
            Useful for monitoring training dynamics across depths.

    Example:
        >>> output = mtp_module(hidden_states, input_ids, labels)
        >>> print(f"Total MTP loss: {output.loss.item():.4f}")
        >>> for d, loss in enumerate(output.losses_per_depth):
        ...     print(f"  Depth {d}: {loss.item():.4f}")
    """

    loss: Optional[torch.FloatTensor] = None
    losses_per_depth: Optional[List[torch.FloatTensor]] = None
    logits_per_depth: Optional[List[torch.FloatTensor]] = None
    num_active_depths: Optional[int] = None


class MTPProjection(nn.Module):
    """Projection layer for combining hidden representations with token embeddings.

    This layer implements the core fusion mechanism of MTP: combining the
    hidden representation from the previous depth with the embedding of the
    next token to predict. The concatenation followed by linear projection
    allows the model to learn how to integrate token identity with context.

    Architecture:
        [hidden_state || token_embedding] -> Linear(2H, H) -> output

    Args:
        hidden_size: Dimension of hidden representations (H).

    Example:
        >>> proj = MTPProjection(hidden_size=768)
        >>> h = torch.randn(2, 100, 768)  # [batch, seq, hidden]
        >>> e = torch.randn(2, 100, 768)  # [batch, seq, hidden]
        >>> out = proj(h, e)  # [2, 100, 768]
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # Project concatenated [hidden_state, embedding] from 2*hidden_size to hidden_size
        # No bias following LLaMA/DeepSeek convention
        self.proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Combine hidden states with token embeddings via concatenation and projection.

        Args:
            hidden_states: Hidden states from previous depth.
                Shape: [batch_size, seq_len, hidden_size]
            token_embeddings: Embeddings of tokens to incorporate.
                Shape: [batch_size, seq_len, hidden_size]

        Returns:
            Combined representation.
                Shape: [batch_size, seq_len, hidden_size]

        Note:
            Both inputs must have the same shape. If they differ, the caller
            is responsible for truncating/aligning them.
        """
        # Concatenate along the hidden dimension: [batch, seq, 2*hidden]
        combined = torch.cat([hidden_states, token_embeddings], dim=-1)
        return self.proj(combined)


class MTPTransformerBlock(nn.Module):
    """Lightweight transformer block for MTP modules.

    This is a simplified transformer layer designed to be parameter-efficient
    while still providing sufficient representational capacity for MTP.
    Following DeepSeek V3's approach, attention is disabled by default to
    save memory and compute.

    Architecture (use_attention=False, default):
        x -> LayerNorm -> MLP -> + x (residual)

    Architecture (use_attention=True):
        x -> LayerNorm -> SelfAttention -> + x -> LayerNorm -> MLP -> + x

    Args:
        hidden_size: Dimension of hidden representations.
        intermediate_size: Dimension of MLP hidden layer (typically 8/3 * hidden_size).
        rms_norm_eps: Epsilon for RMSNorm numerical stability.
        hidden_act: Activation function for MLP (default: "silu" for SwiGLU).
        use_attention: Whether to include self-attention. Disabled by default
            as it significantly increases memory usage with minimal benefit for MTP.
        dropout: Dropout probability for regularization.

    Note:
        When use_attention=True, a simple scaled dot-product attention without
        positional encoding is used. This is intentional: MTP modules process
        already-positioned representations from the main model.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
        hidden_act: str = "silu",
        use_attention: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        self.dropout_prob = dropout

        # Pre-MLP layer norm (or pre-attention if attention enabled)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Post-attention layer norm (only used when attention is enabled)
        if use_attention:
            self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
            # Simple self-attention without RoPE for MTP
            # Positions are already encoded in the main model's hidden states
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.post_attention_layernorm = None

        # MLP (SwiGLU)
        self.mlp = SwiGLUMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            bias=False,
        )

        # Dropout (only create if needed)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the MTP transformer block.

        Args:
            hidden_states: Input tensor.
                Shape: [batch_size, seq_len, hidden_size]
            attention_mask: Optional additive attention mask for self-attention.
                Shape: [batch_size, 1, seq_len, seq_len] or broadcastable.
                Values should be 0 for positions to attend to and -inf for masked positions.
                Only used when use_attention=True.

        Returns:
            Output tensor with same shape as input.
                Shape: [batch_size, seq_len, hidden_size]
        """
        residual = hidden_states

        if self.use_attention:
            # Pre-norm + Self-attention
            hidden_states = self.input_layernorm(hidden_states)

            batch_size, seq_len, _ = hidden_states.shape

            # Compute Q, K, V projections
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

            # Use F.scaled_dot_product_attention for efficiency (handles causal mask)
            hidden_states = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout_prob if self.training else 0.0,
                is_causal=True,
            )

            hidden_states = self.o_proj(hidden_states)
            hidden_states = residual + hidden_states
            residual = hidden_states

            # Use post-attention norm before MLP
            hidden_states = self.post_attention_layernorm(hidden_states)
        else:
            # No attention: just apply norm before MLP
            hidden_states = self.input_layernorm(hidden_states)

        # MLP with residual connection
        hidden_states = self.mlp(hidden_states)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MTPModule(nn.Module):
    """Single Multi-Token Prediction module for one prediction depth.

    Each MTP module represents one step in the prediction chain. Given:
    - The representation h^{k}_i from the previous depth (or h_i from main model for k=0)
    - The embedding of the "known" next token t_{i+k+1}

    It produces:
    - A representation h^{k+1}_i that directly predicts token t_{i+k+2}
    - This representation is also passed to the next MTP module (depth k+1)

    Position Indexing Example (for depth=0, position i=0):
        - Input: h_0 (hidden at position 0 from main model)
        - Incorporate: embed(t_1) (embedding of token at position 1)
        - Output: h^1_0 which directly predicts t_2 (token at position 2)

    Args:
        hidden_size: Dimension of hidden representations.
        intermediate_size: Dimension of MLP hidden layer.
        rms_norm_eps: Epsilon for RMSNorm.
        hidden_act: Activation function.
        num_layers: Number of transformer layers in this MTP module.
            More layers = more capacity but more compute.
        dropout: Dropout probability for regularization.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
        hidden_act: str = "silu",
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Projection to combine hidden states with embeddings
        self.projection = MTPProjection(hidden_size)

        # Transformer layers (lightweight, typically without attention)
        self.layers = nn.ModuleList([
            MTPTransformerBlock(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
                hidden_act=hidden_act,
                use_attention=False,  # Memory-efficient: no attention by default
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output normalization (prepares for LM head projection)
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the MTP module.

        Args:
            hidden_states: Representations from previous depth.
                Shape: [batch_size, seq_len, hidden_size]
            token_embeddings: Embeddings of tokens to incorporate.
                Shape: [batch_size, seq_len, hidden_size]
                Must match hidden_states in batch and sequence dimensions.
            attention_mask: Optional attention mask (passed to transformer layers).
                Only used if layers have use_attention=True.

        Returns:
            Output representations ready for LM head projection.
                Shape: [batch_size, seq_len, hidden_size]
        """
        # Combine hidden states with token embeddings
        hidden_states = self.projection(hidden_states, token_embeddings)

        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        return hidden_states


class MultiTokenPrediction(nn.Module):
    """Multi-Token Prediction head for training.

    This module implements the complete MTP system with D sequential prediction
    modules. During training, it predicts D additional future tokens beyond the
    standard next-token prediction, providing denser supervision signals.

    Architecture:
        The main model produces hidden states h_i for each position i. These are
        passed through a chain of D MTP modules. At each depth, the output
        DIRECTLY predicts the target (no additional shift needed):

        Depth 0: h^1_i = MTP_0(h_i, embed(t_{i+1})) -> logits directly predict t_{i+2}
        Depth 1: h^2_i = MTP_1(h^1_i, embed(t_{i+2})) -> logits directly predict t_{i+3}
        ...

    Position Alignment (for depth d, original position i):
        - Hidden input: h^d_i (from previous depth or main model)
        - Incorporate: embed(t_{i+d+1}) - the token d+1 positions ahead
        - Target: t_{i+d+2} - the token d+2 positions ahead

    Sequence Length Reduction:
        For a sequence of length L, at depth d:
        - Valid positions: 0 to L-d-2 (shrinks by 1 per depth)
        - This is because we need tokens up to position L-1 for embeddings

    Loss Computation:
        Each depth contributes equally to the total MTP loss:
        total_loss = mtp_weight * mean(loss_d for d in range(D))

    Args:
        config: Model configuration containing:
            - mtp_num_heads: Number of prediction depths (D)
            - mtp_loss_weight: Weight for MTP loss (typical: 0.1-0.3)
            - mtp_share_head: Whether to share LM head across depths
            - mtp_num_layers: Number of transformer layers per depth
            - mtp_dropout: Dropout probability (default 0.0)
            - hidden_size, intermediate_size, etc.
        embed_tokens: Shared embedding layer from the main model.
            Used to get token embeddings for the tokens to incorporate.
        lm_head: Optional shared LM head (used if mtp_share_head=True).
            Sharing reduces parameters but may limit depth-specific learning.

    Example:
        >>> mtp = MultiTokenPrediction(config, model.embed_tokens, model.lm_head)
        >>> hidden = model.model(input_ids).last_hidden_state
        >>> mtp_output = mtp(hidden, input_ids, labels)
        >>> total_loss = ntp_loss + mtp_output.loss
    """

    def __init__(
        self,
        config,
        embed_tokens: nn.Embedding,
        lm_head: Optional[nn.Linear] = None,
    ):
        super().__init__()

        # Extract configuration
        self.num_heads = config.mtp_num_heads
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.loss_weight = config.mtp_loss_weight
        self.share_head = config.mtp_share_head

        # Dropout from config (default 0.0 if not specified)
        self.dropout = getattr(config, "mtp_dropout", 0.0)

        # Shared embedding layer (from main model) - used read-only
        self.embed_tokens = embed_tokens

        # MTP modules for each prediction depth
        self.mtp_modules = nn.ModuleList([
            MTPModule(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                rms_norm_eps=config.rms_norm_eps,
                hidden_act=config.hidden_act,
                num_layers=config.mtp_num_layers,
                dropout=self.dropout,
            )
            for _ in range(self.num_heads)
        ])

        # Output heads - either shared or per-depth
        if self.share_head and lm_head is not None:
            self.lm_heads = None
            self._shared_lm_head = lm_head
        else:
            # Independent heads for each depth
            self.lm_heads = nn.ModuleList([
                nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                for _ in range(self.num_heads)
            ])
            self._shared_lm_head = None

        # Shared loss function (instantiate once, not in forward loop)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

    def get_lm_head(self, depth: int) -> nn.Linear:
        """Get the LM head for a specific prediction depth.

        Args:
            depth: The prediction depth (0-indexed).

        Returns:
            The linear layer for projecting hidden states to vocabulary logits.
        """
        if self._shared_lm_head is not None:
            return self._shared_lm_head
        return self.lm_heads[depth]

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids_or_labels: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_logits: bool = False,
    ) -> MTPOutput:
        """Compute MTP loss during training.

        This method processes the hidden states through all MTP depths,
        computing predictions and losses at each depth.

        API Compatibility:
            Supports both old and new signatures:
            - New: forward(hidden_states, input_ids, labels, attention_mask)
            - Old: forward(hidden_states, labels, attention_mask) [deprecated]

        Args:
            hidden_states: Output from main model backbone.
                Shape: [batch_size, seq_len, hidden_size]
            input_ids_or_labels: Either input_ids (new API) or labels (old API).
                Shape: [batch_size, seq_len]
            labels: Target token IDs for loss computation (new API only).
                Shape: [batch_size, seq_len]
                May contain -100 for positions to ignore in loss.
            attention_mask: Optional attention mask.
                Shape: [batch_size, seq_len]
                Currently unused unless MTP uses attention (disabled by default).
            output_logits: Whether to return logits_per_depth. Default False to
                save memory during training. Set True only for debugging/analysis.

        Returns:
            MTPOutput containing:
                - loss: Weighted average loss across all depths
                - losses_per_depth: Individual losses for each depth
                - logits_per_depth: Prediction logits for each depth
                - num_active_depths: How many depths had valid predictions

        Note:
            If sequence is too short for any MTP prediction, returns zero loss.
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Handle backward compatibility: old API passed (hidden_states, labels)
        # New API passes (hidden_states, input_ids, labels)
        if labels is None:
            # Old API: input_ids_or_labels is actually labels
            warnings.warn(
                "Calling MTP with (hidden_states, labels) is deprecated. "
                "Use (hidden_states, input_ids, labels) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            labels = input_ids_or_labels
            # For old API, we have to use labels for embeddings (may have -100)
            # Replace -100 with 0 to avoid embedding lookup errors
            input_ids = labels.clone()
            input_ids[input_ids < 0] = 0
        else:
            input_ids = input_ids_or_labels

        # Input validation
        if input_ids.shape != (batch_size, seq_len):
            raise ValueError(
                f"input_ids shape {input_ids.shape} doesn't match "
                f"hidden_states batch/seq dims ({batch_size}, {seq_len})"
            )
        if labels.shape != (batch_size, seq_len):
            raise ValueError(
                f"labels shape {labels.shape} doesn't match "
                f"hidden_states batch/seq dims ({batch_size}, {seq_len})"
            )

        # Minimum sequence length: need at least 3 tokens for depth 0
        # (position 0, token to incorporate at position 1, target at position 2)
        min_seq_len = 3
        if seq_len < min_seq_len:
            # Return zero loss that stays on computation graph
            zero_loss = hidden_states.new_zeros((), requires_grad=True) * 0
            return MTPOutput(
                loss=zero_loss,
                losses_per_depth=[],
                logits_per_depth=[],
                num_active_depths=0,
            )

        losses_per_depth = []
        logits_per_depth = [] if output_logits else None

        # Current representation starts from main model output
        # We'll update this as we go through depths
        current_hidden = hidden_states

        for depth in range(self.num_heads):
            # At depth d, position i:
            # - We use h^d_i (current_hidden at position i)
            # - We incorporate embed(t_{i+d+1}) = embed(input_ids[i+d+1])
            # - We predict t_{i+d+2} = labels[i+d+2]
            #
            # For this to work for position i=0, we need:
            # - input_ids[d+1] to exist -> d+1 < seq_len -> d < seq_len-1
            # - labels[d+2] to exist -> d+2 < seq_len -> d < seq_len-2
            #
            # Valid length = number of positions we can process = seq_len - d - 2

            incorporate_offset = depth + 1  # Offset for tokens to incorporate
            target_offset = depth + 2       # Offset for target tokens

            # Check if we have enough tokens for this depth
            if target_offset >= seq_len:
                break

            # Number of valid positions at this depth
            valid_len = seq_len - target_offset

            if valid_len <= 0:
                break

            # Get token embeddings to incorporate
            # For position i, we incorporate token at position (i + incorporate_offset)
            # So we take input_ids[incorporate_offset : incorporate_offset + valid_len]
            incorporate_token_ids = input_ids[:, incorporate_offset:incorporate_offset + valid_len]

            # Get embeddings (no gradient - these are inputs, not learned here)
            with torch.no_grad():
                incorporate_embeddings = self.embed_tokens(incorporate_token_ids)

            # Get the corresponding hidden states from previous depth
            # current_hidden[i] represents original position i
            # We only process positions 0 to valid_len-1
            h_input = current_hidden[:, :valid_len]

            # Forward through MTP module
            mtp_hidden = self.mtp_modules[depth](
                hidden_states=h_input,
                token_embeddings=incorporate_embeddings,
                attention_mask=None,  # Attention mask not needed for MLP-only MTP
            )

            # Compute logits for prediction
            lm_head = self.get_lm_head(depth)
            # Cast to float32 only if not already float32 (avoid unnecessary copy)
            mtp_logits = lm_head(mtp_hidden)
            if mtp_logits.dtype != torch.float32:
                mtp_logits = mtp_logits.float()

            # Only store logits if requested (saves significant memory during training)
            if output_logits:
                logits_per_depth.append(mtp_logits)

            # Get target labels for this depth
            # For position i, target is at position (i + target_offset)
            # So we take labels[target_offset : target_offset + valid_len]
            target_labels = labels[:, target_offset:target_offset + valid_len]

            # IMPORTANT: No shift needed! The MTP output at position i directly
            # predicts the target at position i. The "shift" is implicit in how
            # we constructed the targets (offset by target_offset from original).
            #
            # mtp_logits[i] predicts target_labels[i] = labels[target_offset + i]

            # Compute cross-entropy loss
            flat_logits = mtp_logits.reshape(-1, self.vocab_size)
            flat_labels = target_labels.reshape(-1)

            depth_loss = self.loss_fct(flat_logits, flat_labels)
            losses_per_depth.append(depth_loss)

            # Update hidden states for next depth
            # mtp_hidden[i] now represents "position i having seen tokens up to i+d+1"
            current_hidden = mtp_hidden

        # Compute total loss
        num_active = len(losses_per_depth)
        if num_active > 0:
            # Average across depths, then apply weight
            total_loss = torch.stack(losses_per_depth).mean() * self.loss_weight
        else:
            # Return zero loss that stays on computation graph
            total_loss = hidden_states.new_zeros((), requires_grad=True) * 0

        return MTPOutput(
            loss=total_loss,
            losses_per_depth=losses_per_depth,
            logits_per_depth=logits_per_depth,
            num_active_depths=num_active,
        )


def compute_mtp_loss(
    hidden_states: torch.Tensor,
    input_ids_or_labels: torch.Tensor,
    labels_or_mtp_module: Union[torch.Tensor, "MultiTokenPrediction"],
    mtp_module_or_attention_mask: Union["MultiTokenPrediction", torch.Tensor, None] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, MTPOutput]:
    """Convenience function to compute MTP loss.

    This is a simple wrapper around MultiTokenPrediction.forward() that
    returns the loss and full output as a tuple for easy integration.

    API Compatibility:
        Supports both old and new signatures:
        - New: compute_mtp_loss(hidden_states, input_ids, labels, mtp_module, attention_mask)
        - Old: compute_mtp_loss(hidden_states, labels, mtp_module, attention_mask) [deprecated]

    Args:
        hidden_states: Output from main model backbone.
            Shape: [batch_size, seq_len, hidden_size]
        input_ids_or_labels: Either input_ids (new API) or labels (old API).
            Shape: [batch_size, seq_len]
        labels_or_mtp_module: Either labels (new API) or mtp_module (old API).
        mtp_module_or_attention_mask: Either mtp_module (new API) or attention_mask (old API).
        attention_mask: Optional attention mask (new API only).

    Returns:
        Tuple of (mtp_loss, mtp_output):
            - mtp_loss: Scalar tensor with weighted MTP loss
            - mtp_output: Full MTPOutput with per-depth information

    Example:
        >>> mtp_loss, mtp_output = compute_mtp_loss(hidden, input_ids, labels, mtp)
        >>> total_loss = ntp_loss + mtp_loss
        >>> print(f"MTP loss: {mtp_loss.item():.4f}")
        >>> print(f"Active depths: {mtp_output.num_active_depths}")
    """
    # Detect which API is being used
    if isinstance(labels_or_mtp_module, MultiTokenPrediction):
        # Old API: compute_mtp_loss(hidden_states, labels, mtp_module, attention_mask)
        warnings.warn(
            "compute_mtp_loss(hidden_states, labels, mtp_module) is deprecated. "
            "Use compute_mtp_loss(hidden_states, input_ids, labels, mtp_module) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        labels = input_ids_or_labels
        mtp_module = labels_or_mtp_module
        attn_mask = mtp_module_or_attention_mask if isinstance(mtp_module_or_attention_mask, torch.Tensor) else None

        mtp_output = mtp_module(
            hidden_states=hidden_states,
            input_ids_or_labels=labels,  # Will trigger old API path
            labels=None,
            attention_mask=attn_mask,
        )
    else:
        # New API: compute_mtp_loss(hidden_states, input_ids, labels, mtp_module, attention_mask)
        input_ids = input_ids_or_labels
        labels = labels_or_mtp_module
        mtp_module = mtp_module_or_attention_mask
        attn_mask = attention_mask

        if not isinstance(mtp_module, (MultiTokenPrediction, GloeckleMultiTokenPrediction)):
            raise TypeError(
                f"mtp_module must be MultiTokenPrediction or GloeckleMultiTokenPrediction, "
                f"got {type(mtp_module)}"
            )

        mtp_output = mtp_module(
            hidden_states=hidden_states,
            input_ids_or_labels=input_ids,
            labels=labels,
            attention_mask=attn_mask,
        )

    return mtp_output.loss, mtp_output


# ---------------------------------------------------------------------------
# Gloeckle parallel MTP variant
# ---------------------------------------------------------------------------
#
# Gloeckle et al. 2024 ("Better & Faster Large Language Models via Multi-Token
# Prediction", arXiv:2404.19737) describes a *parallel* MTP design distinct from
# DeepSeek-V3's sequential chain implemented above:
#
#   For each head i in 0..D-1, head i predicts t_{i+offset} from h_t directly.
#   Each head has a small per-head adapter (typ. d×d linear) followed by an
#   unembedding to vocab logits. The adapters and unembedding heads are trained
#   together with the main lm-loss; the heads are *discarded at inference*.
#
# In this codebase we add Gloeckle as an alternative MTP variant rather than a
# replacement, so the existing sequential chain (DeepSeek style) stays available
# under ``mtp_variant="deepseek"``.
#
# Stacking-pretraining choice:
#   The user's plan calls for *independent* unembedding heads (no sharing with
#   the model's main ``lm_head``). This is enforced here — Gloeckle MTP always
#   builds 4 fresh ``nn.Linear(d, vocab_size)`` modules and ignores any
#   ``lm_head`` argument passed to the constructor.
#
# Globals masking (SWA path):
#   When ``attention_mode == "swa_globals_mla"``, the MTP loss must skip the
#   first ``num_globals`` positions of the sequence (the persistent global
#   tokens are not real tokens). Pass ``globals_mask`` (a 1D bool tensor of
#   length ``seq_len``, True for content) into ``forward`` and the loss will
#   be masked accordingly.


class GloeckleMultiTokenPrediction(nn.Module):
    """Parallel multi-token prediction heads (Gloeckle et al. 2024).

    Architecture:
      For each head i: logits_i = W_unemb_i @ Adapter_i(hidden_states[:, :T-o_i]).
      Loss_i is CE(logits_i, labels[:, o_i:]); o_i is ``predict_offsets[i]``.
      Total = ``loss_weight`` * mean(Loss_0, ..., Loss_{D-1}).

    Args:
        config: Model configuration. Reads ``mtp_num_heads``, ``mtp_loss_weight``,
            ``mtp_predict_offsets`` (default ``[2, 3, 4, 5]``), ``hidden_size``,
            ``vocab_size``, ``initializer_range`` (for adapter init).
        embed_tokens: Unused; kept for API compatibility with the factory.
        lm_head: Ignored — Gloeckle heads always carry their own unembedding.

    Notes:
      * MTP heads are dropped at inference: ``del model.mtp`` removes
        ``num_heads * (d*d + d*vocab_size)`` parameters from the model.
      * Each head's adapter is initialized with std
        ``initializer_range / sqrt(num_heads)`` so that early MTP gradients
        do not dominate the main loss path.
    """

    def __init__(
        self,
        config,
        embed_tokens: Optional[nn.Embedding] = None,
        lm_head: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.num_heads = int(config.mtp_num_heads)
        if self.num_heads <= 0:
            raise ValueError(
                f"GloeckleMultiTokenPrediction requires mtp_num_heads > 0, "
                f"got {self.num_heads}."
            )
        self.hidden_size = int(config.hidden_size)
        self.vocab_size = int(config.vocab_size)
        self.loss_weight = float(config.mtp_loss_weight)
        self.predict_offsets = list(
            getattr(config, "mtp_predict_offsets", None) or [i + 2 for i in range(self.num_heads)]
        )
        if len(self.predict_offsets) != self.num_heads:
            raise ValueError(
                f"mtp_predict_offsets length ({len(self.predict_offsets)}) must equal "
                f"mtp_num_heads ({self.num_heads})."
            )
        for o in self.predict_offsets:
            if int(o) <= 0:
                raise ValueError(f"mtp_predict_offsets entries must be > 0; got {o}.")

        # Per-head d×d adapter and an INDEPENDENT d->vocab unembedding head.
        # Adapters init at smaller std to keep MTP gradients from dominating
        # the main lm_head early.
        init_std = float(getattr(config, "initializer_range", 0.02))
        adapter_std = init_std / max(1.0, float(self.num_heads) ** 0.5)
        self.adapters = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(self.num_heads)]
        )
        self.lm_heads = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.vocab_size, bias=False) for _ in range(self.num_heads)]
        )
        for adapter in self.adapters:
            nn.init.normal_(adapter.weight, mean=0.0, std=adapter_std)
        for head in self.lm_heads:
            nn.init.normal_(head.weight, mean=0.0, std=adapter_std)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids_or_labels: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_logits: bool = False,
        globals_mask: Optional[torch.Tensor] = None,
    ) -> "MTPOutput":
        """Compute parallel MTP loss.

        Args:
            hidden_states: Backbone outputs, shape [B, T, d].
            input_ids_or_labels: For Gloeckle we only need labels; this argument
                is accepted to match the sequential MTP API but we ignore it
                if ``labels`` is given separately.
            labels: Target token IDs of shape [B, T]; -100 marks ignore.
            attention_mask: Optional [B, T] bool mask. If provided, positions
                where the mask is False are masked from the per-head losses.
            output_logits: If True, returns logits per head (memory-heavy).
            globals_mask: Optional 1D bool tensor of length T. False positions
                are excluded from the loss. Used for the SWA + globals path so
                the prepended-globals positions never contribute to MTP loss.

        Returns:
            MTPOutput with weighted total loss and per-head losses.
        """
        if labels is None:
            # Match the legacy old-API behavior of MultiTokenPrediction.
            labels = input_ids_or_labels

        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must be [B, T, d]; got shape {tuple(hidden_states.shape)}."
            )
        bsz, seq_len, _ = hidden_states.shape

        # If sequence is too short for even the smallest offset, return zero loss.
        max_offset = max(self.predict_offsets)
        if seq_len <= max_offset:
            zero = hidden_states.new_zeros((), requires_grad=True) * 0
            return MTPOutput(
                loss=zero,
                losses_per_depth=[],
                logits_per_depth=[],
                num_active_depths=0,
            )

        # Build a single keep-mask combining attention_mask and globals_mask.
        keep = None
        if attention_mask is not None:
            keep = attention_mask.bool()
        if globals_mask is not None:
            gm = globals_mask.bool()
            if gm.dim() == 1:
                gm = gm.unsqueeze(0).expand(bsz, -1)
            keep = gm if keep is None else (keep & gm)

        per_depth_losses: List[torch.Tensor] = []
        per_depth_logits: List[torch.Tensor] = []
        active = 0
        for head_idx, offset in enumerate(self.predict_offsets):
            if seq_len <= offset:
                continue
            valid = seq_len - offset

            h = hidden_states[:, :valid, :]
            h = self.adapters[head_idx](h)
            logits = self.lm_heads[head_idx](h)  # [B, valid, V]

            tgt = labels[:, offset : offset + valid]  # [B, valid]
            tgt_for_loss = tgt
            if keep is not None:
                # Keep is on input positions; the *target* at offset corresponds
                # to position (i + offset). Mask the loss using whether the
                # source position i is "kept".
                src_keep = keep[:, :valid]
                tgt_for_loss = tgt.masked_fill(~src_keep, -100)

            head_loss = self.loss_fct(
                logits.reshape(-1, self.vocab_size),
                tgt_for_loss.reshape(-1),
            )
            per_depth_losses.append(head_loss)
            if output_logits:
                per_depth_logits.append(logits)
            active += 1

        if active == 0:
            zero = hidden_states.new_zeros((), requires_grad=True) * 0
            return MTPOutput(
                loss=zero,
                losses_per_depth=[],
                logits_per_depth=[],
                num_active_depths=0,
            )

        avg = torch.stack(per_depth_losses).mean()
        total = self.loss_weight * avg

        return MTPOutput(
            loss=total,
            losses_per_depth=per_depth_losses,
            logits_per_depth=per_depth_logits if output_logits else [],
            num_active_depths=active,
        )


def build_mtp_module(
    config,
    embed_tokens: nn.Embedding,
    lm_head: Optional[nn.Linear] = None,
) -> nn.Module:
    """Construct the configured MTP variant.

    ``mtp_variant`` selects between:
      * ``"deepseek"`` (default) — sequential causal chain :class:`MultiTokenPrediction`.
      * ``"gloeckle"`` — parallel heads :class:`GloeckleMultiTokenPrediction` with
        independent unembedding heads (``lm_head`` argument is ignored).
    """
    variant = getattr(config, "mtp_variant", "deepseek")
    if variant == "deepseek":
        return MultiTokenPrediction(config=config, embed_tokens=embed_tokens, lm_head=lm_head)
    if variant == "gloeckle":
        return GloeckleMultiTokenPrediction(
            config=config, embed_tokens=embed_tokens, lm_head=None
        )
    raise ValueError(
        f"Unknown mtp_variant={variant!r}. Expected one of: 'deepseek', 'gloeckle'."
    )
