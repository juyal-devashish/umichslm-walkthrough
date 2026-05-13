"""SwiGLU MLP implementation.

Reference: https://arxiv.org/abs/2002.05202
"""

import torch
import torch.nn as nn


class SwiGLUMLP(nn.Module):
    """SwiGLU-based MLP as used in modern transformer architectures (LLaMA, Mistral, etc.).

    SwiGLU(x) = (Swish(xW_gate) * xW_up) W_down

    Where Swish (also known as SiLU) is: swish(x) = x * sigmoid(x)

    This activation function has been shown to improve training dynamics compared
    to standard FFN with ReLU or GELU activations.

    Args:
        hidden_size: Input and output dimension.
        intermediate_size: Hidden dimension of the MLP.
            For parameter parity with standard FFN, use: intermediate_size = int(8/3 * hidden_size)
            rounded to a multiple of 256 for efficiency.
        hidden_act: Activation function (default "silu" for Swish).
        bias: Whether to use bias in linear projections.

    Shape:
        - Input: (*, hidden_size)
        - Output: (*, hidden_size)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Gate projection (for the gating mechanism)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        # Up projection (the value that gets gated)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        # Down projection (back to hidden_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

        # Activation function (SiLU = Swish with beta=1)
        self.act_fn = self._get_activation(hidden_act)

    def _get_activation(self, hidden_act: str) -> nn.Module:
        """Get the activation function module."""
        activations = {
            "silu": nn.SiLU(),
            "swish": nn.SiLU(),  # Swish is the same as SiLU
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
        }
        if hidden_act not in activations:
            raise ValueError(
                f"Unknown activation: {hidden_act}. "
                f"Supported: {list(activations.keys())}"
            )
        return activations[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SwiGLU MLP.

        Args:
            x: Input tensor of shape (*, hidden_size).

        Returns:
            Output tensor of shape (*, hidden_size).
        """
        # SwiGLU: (activation(gate(x)) * up(x)) @ down
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"bias={self.gate_proj.bias is not None}"
        )


def compute_intermediate_size(hidden_size: int, multiple_of: int = 256) -> int:
    """Compute the intermediate size for SwiGLU to match standard FFN parameter count.

    For a standard FFN with 4x expansion: params = 2 * hidden_size * 4 * hidden_size
    For SwiGLU with 3 projections: params = 3 * hidden_size * intermediate_size

    To match: intermediate_size = 8/3 * hidden_size

    Args:
        hidden_size: The hidden dimension of the model.
        multiple_of: Round up to this multiple for efficiency.

    Returns:
        The computed intermediate size.
    """
    intermediate_size = int(8 * hidden_size / 3)
    # Round up to multiple
    intermediate_size = ((intermediate_size + multiple_of - 1) // multiple_of) * multiple_of
    return intermediate_size
