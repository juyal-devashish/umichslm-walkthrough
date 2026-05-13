"""Mixture of Experts (MoE) implementation.

This module implements sparse MoE layers with load balancing losses to ensure
experts are utilized uniformly across tokens.

References:
    - Switch Transformers: https://arxiv.org/abs/2101.03961
    - Mixtral of Experts: https://arxiv.org/abs/2401.04088
    - GShard: https://arxiv.org/abs/2006.16668
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swiglu import SwiGLUMLP


@dataclass
class MoEOutput:
    """Output from MoE layer including auxiliary losses.

    Attributes:
        hidden_states: Output tensor of shape (batch, seq_len, hidden_size).
        router_logits: Router logits for each token of shape (batch * seq_len, num_experts).
        aux_loss: Load balancing auxiliary loss.
        router_z_loss: Router z-loss to prevent large logits.
    """

    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    aux_loss: torch.Tensor
    router_z_loss: torch.Tensor


class MoERouter(nn.Module):
    """Router network for Mixture of Experts.

    The router computes gating scores that determine which experts process each token.
    Supports both top-k routing and auxiliary losses for load balancing.

    Args:
        hidden_size: Dimension of input hidden states.
        num_experts: Number of expert networks.
        top_k: Number of experts to route each token to.
        router_jitter: Add noise to router during training for exploration.
        router_dtype: Dtype for router computation (default: float32 for stability).
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        router_jitter: float = 0.0,
        router_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_jitter = router_jitter
        self.router_dtype = router_dtype

        # Linear layer to compute expert scores
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Tuple of (router_weights, selected_experts, router_logits):
                - router_weights: Softmax weights for selected experts (batch*seq, top_k)
                - selected_experts: Indices of selected experts (batch*seq, top_k)
                - router_logits: Raw router logits (batch*seq, num_experts)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Reshape for routing: [batch * seq_len, hidden_size]
        hidden_states_flat = hidden_states.view(-1, hidden_size)

        # Cast to router dtype for stability
        hidden_states_flat = hidden_states_flat.to(self.router_dtype)

        # Add jitter noise during training for exploration
        if self.training and self.router_jitter > 0:
            noise = torch.empty_like(hidden_states_flat).uniform_(
                -self.router_jitter, self.router_jitter
            )
            hidden_states_flat = hidden_states_flat * (1 + noise)

        # Compute router logits: [batch * seq_len, num_experts]
        router_logits = self.gate(hidden_states_flat)

        # Compute router probabilities
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        router_weights, selected_experts = torch.topk(
            router_probs, self.top_k, dim=-1
        )

        # Normalize weights to sum to 1
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)

        return router_weights, selected_experts, router_logits


class MoELayer(nn.Module):
    """Mixture of Experts layer with load balancing losses.

    Replaces the standard MLP with a sparse mixture of expert MLPs.
    Each token is routed to top-k experts, and their outputs are combined
    using the router weights.

    Args:
        hidden_size: Dimension of input/output hidden states.
        intermediate_size: Dimension of expert intermediate layers.
        num_experts: Number of expert networks.
        num_experts_per_tok: Number of experts each token is routed to (top-k).
        hidden_act: Activation function for experts ("silu" or "gelu").
        mlp_bias: Whether to use bias in expert MLPs.
        router_jitter: Jitter noise for router during training.
        router_aux_loss_coef: Coefficient for load balancing auxiliary loss.
        router_z_loss_coef: Coefficient for router z-loss.

    Shape:
        - Input: (batch, seq_len, hidden_size)
        - Output: MoEOutput with hidden_states of shape (batch, seq_len, hidden_size)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int = 2,
        hidden_act: str = "silu",
        mlp_bias: bool = False,
        router_jitter: float = 0.0,
        router_aux_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.001,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef

        # Router for expert selection
        self.router = MoERouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=num_experts_per_tok,
            router_jitter=router_jitter,
        )

        # Expert networks (using SwiGLU MLPs)
        self.experts = nn.ModuleList([
            SwiGLUMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                bias=mlp_bias,
            )
            for _ in range(num_experts)
        ])

    def compute_aux_loss(
        self,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Compute load balancing auxiliary loss.

        This loss encourages uniform expert utilization across tokens.
        Based on Switch Transformers (Fedus et al., 2021).

        The loss is: num_experts * sum_i(f_i * P_i)
        where:
            - f_i is the fraction of tokens routed to expert i
            - P_i is the mean router probability for expert i

        Args:
            router_logits: Router logits of shape (num_tokens, num_experts).
            selected_experts: Selected expert indices of shape (num_tokens, top_k).

        Returns:
            Scalar auxiliary loss tensor.
        """
        num_tokens = router_logits.shape[0]

        # Compute router probabilities
        router_probs = F.softmax(router_logits, dim=-1)

        # Compute fraction of tokens routed to each expert (f_i)
        # Count how many times each expert is selected
        expert_mask = F.one_hot(
            selected_experts, num_classes=self.num_experts
        ).float()  # [num_tokens, top_k, num_experts]
        expert_mask = expert_mask.sum(dim=1)  # [num_tokens, num_experts]
        tokens_per_expert = expert_mask.sum(dim=0)  # [num_experts]
        fraction_tokens = tokens_per_expert / (num_tokens * self.num_experts_per_tok)

        # Compute mean router probability for each expert (P_i)
        mean_router_prob = router_probs.mean(dim=0)  # [num_experts]

        # Auxiliary loss: encourages f_i and P_i to be uniform (1/num_experts each)
        aux_loss = self.num_experts * (fraction_tokens * mean_router_prob).sum()

        return aux_loss

    def compute_router_z_loss(
        self,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute router z-loss to prevent large router logits.

        Large router logits can lead to numerical instability and poor
        generalization. This loss penalizes the log-sum-exp of router logits.

        Based on ST-MoE (Zoph et al., 2022).

        Args:
            router_logits: Router logits of shape (num_tokens, num_experts).

        Returns:
            Scalar z-loss tensor.
        """
        # z_loss = mean(log(sum(exp(logits)))^2)
        log_z = torch.logsumexp(router_logits, dim=-1)  # [num_tokens]
        z_loss = (log_z ** 2).mean()
        return z_loss

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> MoEOutput:
        """Apply MoE layer with routing and load balancing.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            MoEOutput containing output hidden states and auxiliary losses.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        input_dtype = hidden_states.dtype

        # Route tokens to experts
        router_weights, selected_experts, router_logits = self.router(hidden_states)

        # Compute auxiliary losses
        aux_loss = self.compute_aux_loss(router_logits, selected_experts)
        router_z_loss = self.compute_router_z_loss(router_logits)

        # Reshape for expert processing
        hidden_states_flat = hidden_states.view(-1, hidden_size)  # [batch * seq, hidden]

        # Initialize output tensor
        output = torch.zeros_like(hidden_states_flat)

        # Process each expert (could be parallelized on multiple GPUs)
        for expert_idx, expert in enumerate(self.experts):
            # Find which tokens are routed to this expert
            # selected_experts has shape [batch * seq, top_k]
            expert_mask = (selected_experts == expert_idx).any(dim=-1)  # [batch * seq]

            if expert_mask.sum() == 0:
                continue

            # Get tokens for this expert
            expert_input = hidden_states_flat[expert_mask]  # [num_tokens_for_expert, hidden]

            # Process through expert
            expert_output = expert(expert_input)  # [num_tokens_for_expert, hidden]

            # Get weights for this expert
            # For each token, find the weight corresponding to this expert
            expert_weights_mask = (selected_experts == expert_idx)  # [batch * seq, top_k]
            # Get the position in top_k where this expert appears
            expert_weight_idx = expert_weights_mask.float().argmax(dim=-1)  # [batch * seq]
            # Gather the corresponding weight
            expert_weights = router_weights.gather(1, expert_weight_idx.unsqueeze(-1))  # [batch * seq, 1]
            expert_weights = expert_weights[expert_mask]  # [num_tokens_for_expert, 1]

            # Add weighted expert output
            output[expert_mask] += expert_weights * expert_output

        # Reshape back
        output = output.view(batch_size, seq_len, hidden_size).to(input_dtype)

        return MoEOutput(
            hidden_states=output,
            router_logits=router_logits,
            aux_loss=aux_loss * self.router_aux_loss_coef,
            router_z_loss=router_z_loss * self.router_z_loss_coef,
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}, "
            f"num_experts={self.num_experts}, num_experts_per_tok={self.num_experts_per_tok}"
        )


class SparseMoE(nn.Module):
    """Sparse Mixture of Experts with shared expert (optional).

    This variant includes an optional "shared expert" that processes all tokens,
    combined with sparse routing to specialized experts. This can improve
    training stability and baseline performance.

    Args:
        hidden_size: Dimension of input/output hidden states.
        intermediate_size: Dimension of expert intermediate layers.
        num_experts: Number of routed expert networks.
        num_experts_per_tok: Number of experts each token is routed to.
        num_shared_experts: Number of shared experts (process all tokens).
        hidden_act: Activation function for experts.
        mlp_bias: Whether to use bias in expert MLPs.
        router_aux_loss_coef: Coefficient for load balancing auxiliary loss.
        router_z_loss_coef: Coefficient for router z-loss.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int = 2,
        num_shared_experts: int = 0,
        hidden_act: str = "silu",
        mlp_bias: bool = False,
        router_aux_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.001,
    ):
        super().__init__()
        self.num_shared_experts = num_shared_experts

        # Sparse MoE layer (routed experts)
        self.moe = MoELayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_act=hidden_act,
            mlp_bias=mlp_bias,
            router_aux_loss_coef=router_aux_loss_coef,
            router_z_loss_coef=router_z_loss_coef,
        )

        # Shared experts (if any)
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                SwiGLUMLP(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    hidden_act=hidden_act,
                    bias=mlp_bias,
                )
                for _ in range(num_shared_experts)
            ])
        else:
            self.shared_experts = None

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> MoEOutput:
        """Apply sparse MoE with optional shared experts.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            MoEOutput containing output hidden states and auxiliary losses.
        """
        # Apply routed MoE
        moe_output = self.moe(hidden_states)
        output = moe_output.hidden_states

        # Add shared expert outputs
        if self.shared_experts is not None:
            for shared_expert in self.shared_experts:
                output = output + shared_expert(hidden_states)

        return MoEOutput(
            hidden_states=output,
            router_logits=moe_output.router_logits,
            aux_loss=moe_output.aux_loss,
            router_z_loss=moe_output.router_z_loss,
        )


def compute_moe_loss(
    moe_outputs: list,
    aux_loss_coef: float = 1.0,
    z_loss_coef: float = 1.0,
) -> torch.Tensor:
    """Compute total MoE auxiliary loss from all layers.

    Args:
        moe_outputs: List of MoEOutput from each MoE layer.
        aux_loss_coef: Global coefficient for auxiliary losses.
        z_loss_coef: Global coefficient for z-losses.

    Returns:
        Total auxiliary loss to add to the main loss.
    """
    total_aux_loss = sum(out.aux_loss for out in moe_outputs if out is not None)
    total_z_loss = sum(out.router_z_loss for out in moe_outputs if out is not None)

    return aux_loss_coef * total_aux_loss + z_loss_coef * total_z_loss
