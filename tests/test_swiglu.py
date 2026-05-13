"""Tests for SwiGLU MLP module."""

import pytest
import torch

from umich_slm.modules.swiglu import SwiGLUMLP, compute_intermediate_size


class TestSwiGLUMLP:
    """Tests for SwiGLU MLP implementation."""

    def test_output_shape(self):
        """Output should have the same shape as input."""
        mlp = SwiGLUMLP(hidden_size=64, intermediate_size=172)
        x = torch.randn(2, 16, 64)
        out = mlp(x)
        assert out.shape == x.shape

    def test_output_shape_2d(self):
        """Should work with 2D input."""
        mlp = SwiGLUMLP(hidden_size=64, intermediate_size=172)
        x = torch.randn(2, 64)
        out = mlp(x)
        assert out.shape == x.shape

    def test_no_bias_by_default(self):
        """Bias should be disabled by default."""
        mlp = SwiGLUMLP(hidden_size=64, intermediate_size=172)
        assert mlp.gate_proj.bias is None
        assert mlp.up_proj.bias is None
        assert mlp.down_proj.bias is None

    def test_with_bias(self):
        """Should support bias when enabled."""
        mlp = SwiGLUMLP(hidden_size=64, intermediate_size=172, bias=True)
        assert mlp.gate_proj.bias is not None
        assert mlp.up_proj.bias is not None
        assert mlp.down_proj.bias is not None

    def test_gradient_flow(self):
        """Gradients should flow through the MLP."""
        mlp = SwiGLUMLP(hidden_size=64, intermediate_size=172)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert mlp.gate_proj.weight.grad is not None
        assert mlp.up_proj.weight.grad is not None
        assert mlp.down_proj.weight.grad is not None

    def test_silu_activation(self):
        """Default activation should be SiLU."""
        mlp = SwiGLUMLP(hidden_size=64, intermediate_size=172, hidden_act="silu")
        assert isinstance(mlp.act_fn, torch.nn.SiLU)

    def test_gelu_activation(self):
        """Should support GELU activation."""
        mlp = SwiGLUMLP(hidden_size=64, intermediate_size=172, hidden_act="gelu")
        assert isinstance(mlp.act_fn, torch.nn.GELU)

    def test_invalid_activation(self):
        """Should raise error for invalid activation."""
        with pytest.raises(ValueError):
            SwiGLUMLP(hidden_size=64, intermediate_size=172, hidden_act="invalid")

    def test_mixed_precision(self):
        """Should work with mixed precision."""
        mlp = SwiGLUMLP(hidden_size=64, intermediate_size=172)
        x = torch.randn(2, 16, 64).to(torch.float16)
        # Convert mlp to float16
        mlp = mlp.to(torch.float16)
        out = mlp(x)
        assert out.dtype == torch.float16

    def test_repr(self):
        """String representation should include key parameters."""
        mlp = SwiGLUMLP(hidden_size=64, intermediate_size=172, bias=False)
        repr_str = repr(mlp)
        assert "64" in repr_str
        assert "172" in repr_str


class TestComputeIntermediateSize:
    """Tests for compute_intermediate_size helper."""

    def test_basic_computation(self):
        """Should compute approximately 8/3 * hidden_size."""
        size = compute_intermediate_size(768)
        # 8/3 * 768 = 2048
        assert size == 2048

    def test_rounds_to_multiple(self):
        """Should round up to multiple of 256."""
        size = compute_intermediate_size(64)
        assert size % 256 == 0

    def test_custom_multiple(self):
        """Should support custom rounding multiple."""
        size = compute_intermediate_size(64, multiple_of=64)
        assert size % 64 == 0

    def test_known_values(self):
        """Test known good values."""
        # These are common configurations
        assert compute_intermediate_size(256) >= 682  # 8/3 * 256 = 682
        assert compute_intermediate_size(1024) >= 2730  # 8/3 * 1024 = 2730
