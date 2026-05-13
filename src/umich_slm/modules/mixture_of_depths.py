"""Mixture of Depths (MoD) Module.

This module implements Mixture of Depths as described in:
"Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"
(Raposo et al., 2024) - https://arxiv.org/abs/2404.02258

Key concepts:
- Dynamic compute allocation: tokens can skip layers based on learned routing
- Top-k routing: only k tokens (based on capacity) are processed per layer
- Auxiliary loss: trains router for autoregressive inference
- Significant FLOPs reduction with minimal performance impact

The router assigns a scalar weight to each token. Tokens with top-k weights
are processed through the layer; others pass through via residual connection.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput


@dataclass
class MoDOutput(ModelOutput):
    """Output type for Mixture of Depths layers.

    Attributes:
        hidden_states: Output hidden states.
        router_weights: Router weights for each token.
        selected_mask: Boolean mask indicating which tokens were selected.
        aux_loss: Auxiliary loss for training router for inference.
    """

    hidden_states: torch.FloatTensor = None
    router_weights: Optional[torch.FloatTensor] = None
    selected_mask: Optional[torch.BoolTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None


class MoDRouter(nn.Module):
    """Router for Mixture of Depths.

    Computes a scalar routing weight for each token via linear projection.
    During training, uses top-k selection; during inference, can use
    learned threshold via auxiliary loss or auxiliary predictor.

    Args:
        hidden_size: Dimension of hidden representations.
        capacity: Fraction of tokens to process (0.0-1.0).
        aux_loss_weight: Weight for auxiliary loss.
        use_aux_predictor: Use auxiliary MLP for inference routing.
    """

    def __init__(
        self,
        hidden_size: int,
        capacity: float = 0.125,
        aux_loss_weight: float = 0.01,
        use_aux_predictor: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.capacity = capacity
        self.aux_loss_weight = aux_loss_weight
        self.use_aux_predictor = use_aux_predictor

        # Router projection: hidden_size -> 1 scalar weight per token
        self.router = nn.Linear(hidden_size, 1, bias=False)

        # Optional auxiliary predictor for inference
        if use_aux_predictor:
            self.aux_predictor = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, 1),
            )
        else:
            self.aux_predictor = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        use_cache: bool = False,
        force_route_prefix: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute routing decisions.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            use_cache: If True, use inference mode (no top-k, use threshold).
            force_route_prefix: Number of leading positions that must be
                force-routed (always selected). Used by the SWA + persistent
                globals path so the prepended global tokens always traverse
                every block. The aux loss is computed only over the remaining
                content positions so the router is supervised only on tokens
                where it actually has a meaningful choice.

        Returns:
            Tuple of (router_weights, selected_mask, aux_loss).
            - router_weights: [batch, seq_len] scalar weights
            - selected_mask: [batch, seq_len] bool mask of selected tokens
            - aux_loss: Auxiliary loss for training (None during inference)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute router weights: [batch, seq_len, 1] -> [batch, seq_len]
        router_weights = self.router(hidden_states).squeeze(-1)

        prefix = max(0, int(force_route_prefix))
        if prefix > seq_len:
            prefix = seq_len
        content_len = seq_len - prefix

        # Calculate k based on capacity, computed over the *content* portion
        # so capacity describes "fraction of content tokens that get processed".
        k = max(1, int(content_len * self.capacity)) if content_len > 0 else 0

        aux_loss = None

        if self.training:
            # Training mode: use top-k selection on the content slice; the
            # forced prefix is set to True unconditionally.
            selected_mask = torch.zeros(
                batch_size, seq_len, dtype=torch.bool, device=hidden_states.device
            )
            if prefix > 0:
                selected_mask[:, :prefix] = True
            if content_len > 0 and k > 0:
                content_weights = router_weights[:, prefix:]
                _, top_k_indices = torch.topk(content_weights, k, dim=-1)
                # Scatter into the content slice (offset by prefix).
                selected_mask[:, prefix:].scatter_(1, top_k_indices, True)

            # Compute auxiliary loss only over the content slice — the prefix
            # has no learnable choice so there is nothing to supervise there.
            if content_len > 0:
                targets = selected_mask[:, prefix:].float()
                aux_loss = (
                    F.binary_cross_entropy_with_logits(
                        router_weights[:, prefix:], targets, reduction="mean"
                    )
                    * self.aux_loss_weight
                )
            else:
                aux_loss = router_weights.new_zeros(())

        else:
            # Inference mode
            if use_cache and seq_len == 1:
                # Autoregressive generation: single token. The single token
                # must not be a "global" (we should never decode at a global
                # position), so prefix logic doesn't apply here.
                if self.aux_predictor is not None:
                    pred_logits = self.aux_predictor(hidden_states).squeeze(-1)
                    selected_mask = pred_logits > 0
                else:
                    selected_mask = torch.sigmoid(router_weights) > 0.5
            else:
                # Batch inference or prefill: use top-k on the content slice.
                selected_mask = torch.zeros(
                    batch_size, seq_len, dtype=torch.bool, device=hidden_states.device
                )
                if prefix > 0:
                    selected_mask[:, :prefix] = True
                if content_len > 0 and k > 0:
                    content_weights = router_weights[:, prefix:]
                    _, top_k_indices = torch.topk(content_weights, k, dim=-1)
                    selected_mask[:, prefix:].scatter_(1, top_k_indices, True)

        return router_weights, selected_mask, aux_loss


class MoDGate(nn.Module):
    """Gating mechanism for MoD that applies router weights to outputs.

    The gate multiplies the block output by the router weight, allowing
    gradients to flow through the routing decision.

    Args:
        hidden_size: Dimension of hidden representations.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        router_weights: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply gating to combine block output with residual.

        For selected tokens: output = router_weight * block_output + hidden_states
        For skipped tokens: output = hidden_states

        Args:
            hidden_states: Original input [batch, seq_len, hidden_size].
            block_output: Output from attention+MLP [batch, seq_len, hidden_size].
            router_weights: Router weights [batch, seq_len].
            selected_mask: Selection mask [batch, seq_len].

        Returns:
            Gated output [batch, seq_len, hidden_size].
        """
        # Expand router weights for broadcasting
        router_weights = router_weights.unsqueeze(-1)  # [batch, seq_len, 1]
        selected_mask = selected_mask.unsqueeze(-1)  # [batch, seq_len, 1]

        # Apply gating: selected tokens get weighted block output
        # router_weight * block_output + residual for selected
        # just residual for skipped
        gated_output = torch.where(
            selected_mask,
            router_weights * block_output + hidden_states,
            hidden_states,
        )

        return gated_output


def is_mod_layer(layer_idx: int, mod_every_n_layers: int) -> bool:
    """Determine if a layer should use MoD routing.

    Args:
        layer_idx: Index of the layer (0-based).
        mod_every_n_layers: Apply MoD every N layers. 0 disables.

    Returns:
        True if this layer should use MoD routing.
    """
    if mod_every_n_layers <= 0:
        return False
    # Apply MoD to layers where (layer_idx % mod_every_n_layers) != 0
    # This means layer 0 is always full, layer 1 is MoD (if every_n=2), etc.
    # Actually per paper: "routing every other block" suggests 1, 3, 5, etc.
    return (layer_idx % mod_every_n_layers) == 1


def compute_mod_aux_loss(
    mod_outputs: Tuple[MoDOutput, ...],
) -> Optional[torch.Tensor]:
    """Compute total MoD auxiliary loss from all layers.

    Args:
        mod_outputs: Tuple of MoDOutput from each MoD layer.

    Returns:
        Total auxiliary loss, or None if no losses.
    """
    total_loss = None
    for output in mod_outputs:
        if output is not None and output.aux_loss is not None:
            if total_loss is None:
                total_loss = output.aux_loss
            else:
                total_loss = total_loss + output.aux_loss
    return total_loss


class MoDProcessor:
    """Utility class for processing tokens through MoD layers efficiently.

    This handles the token selection and recombination logic for MoD,
    allowing efficient processing where only selected tokens go through
    the expensive attention and MLP computations.
    """

    @staticmethod
    def select_tokens(
        hidden_states: torch.Tensor,
        selected_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Select tokens for processing based on mask.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            selected_mask: Boolean mask [batch, seq_len].
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.

        Returns:
            Tuple of (selected_hidden, selected_attn_mask, selected_pos_ids, indices).
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Get indices of selected tokens per batch
        # For simplicity, we'll use a padded approach
        num_selected = selected_mask.sum(dim=1).max().item()

        # Gather selected tokens
        # Create index tensor for gathering
        selected_indices = []
        for b in range(batch_size):
            batch_indices = selected_mask[b].nonzero(as_tuple=True)[0]
            # Pad to max selected
            if len(batch_indices) < num_selected:
                pad_size = num_selected - len(batch_indices)
                batch_indices = F.pad(batch_indices, (0, pad_size), value=0)
            selected_indices.append(batch_indices[:num_selected])

        indices = torch.stack(selected_indices)  # [batch, num_selected]

        # Gather hidden states
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, hidden_size)
        selected_hidden = torch.gather(hidden_states, 1, indices_expanded)

        # Handle attention mask if provided
        selected_attn_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                selected_attn_mask = torch.gather(attention_mask, 1, indices)
            # For 4D attention mask, more complex handling needed

        # Handle position IDs if provided
        selected_pos_ids = None
        if position_ids is not None:
            selected_pos_ids = torch.gather(position_ids, 1, indices)

        return selected_hidden, selected_attn_mask, selected_pos_ids, indices

    @staticmethod
    def scatter_tokens(
        full_hidden: torch.Tensor,
        processed_hidden: torch.Tensor,
        indices: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter processed tokens back to full sequence.

        Args:
            full_hidden: Original hidden states [batch, seq_len, hidden_size].
            processed_hidden: Processed tokens [batch, num_selected, hidden_size].
            indices: Indices of selected tokens [batch, num_selected].
            selected_mask: Selection mask [batch, seq_len].

        Returns:
            Full hidden states with processed tokens scattered back.
        """
        batch_size, seq_len, hidden_size = full_hidden.shape

        # Create output tensor (copy of input for non-selected positions)
        output = full_hidden.clone()

        # Scatter processed tokens back
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, hidden_size)
        output.scatter_(1, indices_expanded, processed_hidden)

        return output
