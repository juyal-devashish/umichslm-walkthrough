"""RMSNorm (Root Mean Square Layer Normalization) implementation.

Reference: https://arxiv.org/abs/1910.07467

This module provides two variants:
- RMSNorm: Standard RMSNorm without mean centering
- ZeroCenteredRMSNorm: RMSNorm with mean subtraction (used in Gemma 2, etc.)
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm normalizes inputs by their root mean square, without the mean-centering
    of standard LayerNorm. This is more efficient and works well with modern transformers.

    Formula: output = x / sqrt(mean(x²) + eps) * weight

    Args:
        hidden_size: The size of the last dimension of the input tensor.
        eps: A small constant for numerical stability.

    Shape:
        - Input: (*, hidden_size)
        - Output: (*, hidden_size)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to the input.

        Args:
            hidden_states: Input tensor of shape (*, hidden_size).

        Returns:
            Normalized tensor of the same shape.
        """
        input_dtype = hidden_states.dtype
        # Cast to float32 for numerical stability during variance computation
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Cast back to input dtype, including the weight
        return (self.weight * hidden_states).to(input_dtype)

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.variance_epsilon}"


class ZeroCenteredRMSNorm(nn.Module):
    """Zero-Centered Root Mean Square Layer Normalization.

    This variant first subtracts the mean before computing the RMS normalization,
    combining the benefits of mean-centering (like LayerNorm) with the simplicity
    of RMSNorm. Used in architectures like Gemma 2.

    Formula: output = (x - mean(x)) / sqrt(mean((x - mean(x))²) + eps) * weight

    Args:
        hidden_size: The size of the last dimension of the input tensor.
        eps: A small constant for numerical stability.

    Shape:
        - Input: (*, hidden_size)
        - Output: (*, hidden_size)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply Zero-Centered RMSNorm to the input.

        Args:
            hidden_states: Input tensor of shape (*, hidden_size).

        Returns:
            Normalized tensor of the same shape.
        """
        input_dtype = hidden_states.dtype
        # Cast to float32 for numerical stability
        hidden_states = hidden_states.to(torch.float32)
        # Zero-center: subtract mean
        hidden_states = hidden_states - hidden_states.mean(-1, keepdim=True)
        # Compute RMS of centered values
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Cast back to input dtype, including the weight
        return (self.weight * hidden_states).to(input_dtype)

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.variance_epsilon}, zero_centered=True"


def get_rmsnorm_class(zero_centered: bool = False):
    """Get the appropriate RMSNorm implementation.

    Args:
        zero_centered: If True, return ZeroCenteredRMSNorm. If False, return
            the most efficient standard RMSNorm available.

    Returns:
        RMSNorm class (ZeroCenteredRMSNorm, torch.nn.RMSNorm, or custom RMSNorm).
    """
    if zero_centered:
        return ZeroCenteredRMSNorm

    # For standard RMSNorm, use PyTorch's if available (more optimized)
    if hasattr(torch.nn, "RMSNorm"):
        return torch.nn.RMSNorm
    return RMSNorm
