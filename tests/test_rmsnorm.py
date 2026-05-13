"""Tests for RMSNorm module."""

import pytest
import torch

from umich_slm.modules.rmsnorm import RMSNorm, ZeroCenteredRMSNorm, get_rmsnorm_class


class TestRMSNorm:
    """Tests for RMSNorm implementation."""

    def test_output_shape(self):
        """Output should have the same shape as input."""
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_output_shape_2d(self):
        """Should work with 2D input."""
        norm = RMSNorm(64)
        x = torch.randn(2, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization(self):
        """Output should have approximately unit RMS."""
        norm = RMSNorm(64)
        # Set weights to 1 for this test
        norm.weight.data.fill_(1.0)
        x = torch.randn(2, 16, 64)
        out = norm(x)

        # Compute RMS of output
        rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
        # Should be close to 1
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_mixed_precision_bf16(self):
        """Should handle bfloat16 inputs correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for bf16 test")

        norm = RMSNorm(64).cuda()
        x = torch.randn(2, 16, 64, device="cuda").to(torch.bfloat16)
        out = norm(x)
        assert out.dtype == torch.bfloat16
        assert out.shape == x.shape

    def test_mixed_precision_fp16(self):
        """Should handle float16 inputs correctly."""
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64).to(torch.float16)
        out = norm(x)
        assert out.dtype == torch.float16
        assert out.shape == x.shape

    def test_gradient_flow(self):
        """Gradients should flow through the layer."""
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert norm.weight.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(norm.weight.grad).any()

    def test_epsilon_prevents_division_by_zero(self):
        """Epsilon should prevent NaN when input is zero."""
        norm = RMSNorm(64, eps=1e-6)
        x = torch.zeros(2, 16, 64)
        out = norm(x)
        assert not torch.isnan(out).any()

    def test_learnable_weight(self):
        """Weight parameter should be learnable."""
        norm = RMSNorm(64)
        assert norm.weight.requires_grad
        assert norm.weight.shape == (64,)

    def test_repr(self):
        """String representation should include key parameters."""
        norm = RMSNorm(64, eps=1e-5)
        repr_str = repr(norm)
        assert "64" in repr_str
        assert "1e-05" in repr_str

    def test_get_rmsnorm_class(self):
        """get_rmsnorm_class should return a valid class."""
        cls = get_rmsnorm_class()
        assert cls is not None
        # Should be able to instantiate
        norm = cls(64)
        x = torch.randn(2, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_get_rmsnorm_class_zero_centered(self):
        """get_rmsnorm_class with zero_centered=True should return ZeroCenteredRMSNorm."""
        cls = get_rmsnorm_class(zero_centered=True)
        assert cls is ZeroCenteredRMSNorm


class TestZeroCenteredRMSNorm:
    """Tests for ZeroCenteredRMSNorm implementation."""

    def test_output_shape(self):
        """Output should have the same shape as input."""
        norm = ZeroCenteredRMSNorm(64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_output_shape_2d(self):
        """Should work with 2D input."""
        norm = ZeroCenteredRMSNorm(64)
        x = torch.randn(2, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_zero_mean_output(self):
        """Output should have approximately zero mean."""
        norm = ZeroCenteredRMSNorm(64)
        # Set weights to 1 for this test
        norm.weight.data.fill_(1.0)
        x = torch.randn(2, 16, 64) * 10 + 5  # Add offset to test centering
        out = norm(x)

        # Mean of output should be close to 0
        out_mean = out.mean(dim=-1)
        assert torch.allclose(out_mean, torch.zeros_like(out_mean), atol=1e-5)

    def test_normalization(self):
        """Output should have approximately unit RMS after centering."""
        norm = ZeroCenteredRMSNorm(64)
        # Set weights to 1 for this test
        norm.weight.data.fill_(1.0)
        x = torch.randn(2, 16, 64)
        out = norm(x)

        # Compute RMS of output (which is already zero-centered)
        rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
        # Should be close to 1
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_differs_from_standard_rmsnorm(self):
        """Zero-centered output should differ from standard RMSNorm."""
        std_norm = RMSNorm(64)
        zc_norm = ZeroCenteredRMSNorm(64)
        # Use same weights
        zc_norm.weight.data = std_norm.weight.data.clone()

        # Create input with non-zero mean
        x = torch.randn(2, 16, 64) + 5.0

        out_std = std_norm(x)
        out_zc = zc_norm(x)

        # Outputs should be different
        assert not torch.allclose(out_std, out_zc, atol=1e-3)

        # Zero-centered should have zero mean, standard should not
        assert out_zc.mean(dim=-1).abs().max() < 1e-5
        assert out_std.mean(dim=-1).abs().max() > 0.1  # Standard preserves offset

    def test_mixed_precision_bf16(self):
        """Should handle bfloat16 inputs correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for bf16 test")

        norm = ZeroCenteredRMSNorm(64).cuda()
        x = torch.randn(2, 16, 64, device="cuda").to(torch.bfloat16)
        out = norm(x)
        assert out.dtype == torch.bfloat16
        assert out.shape == x.shape

    def test_mixed_precision_fp16(self):
        """Should handle float16 inputs correctly."""
        norm = ZeroCenteredRMSNorm(64)
        x = torch.randn(2, 16, 64).to(torch.float16)
        out = norm(x)
        assert out.dtype == torch.float16
        assert out.shape == x.shape

    def test_gradient_flow(self):
        """Gradients should flow through the layer."""
        norm = ZeroCenteredRMSNorm(64)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert norm.weight.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(norm.weight.grad).any()

    def test_epsilon_prevents_division_by_zero(self):
        """Epsilon should prevent NaN when centered input is zero."""
        norm = ZeroCenteredRMSNorm(64, eps=1e-6)
        # All same values -> after centering, all zeros
        x = torch.ones(2, 16, 64) * 5.0
        out = norm(x)
        assert not torch.isnan(out).any()

    def test_learnable_weight(self):
        """Weight parameter should be learnable."""
        norm = ZeroCenteredRMSNorm(64)
        assert norm.weight.requires_grad
        assert norm.weight.shape == (64,)

    def test_repr(self):
        """String representation should include key parameters."""
        norm = ZeroCenteredRMSNorm(64, eps=1e-5)
        repr_str = repr(norm)
        assert "64" in repr_str
        assert "zero_centered=True" in repr_str
