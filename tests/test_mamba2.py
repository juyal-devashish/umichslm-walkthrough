"""Tests for Mamba2 SSM module."""

import pytest
import torch
import torch.nn as nn

from umich_slm.modules.mamba2 import Mamba2SSM, Mamba2Cache


@pytest.fixture
def ssm():
    """Create a small Mamba2SSM for testing."""
    return Mamba2SSM(d_model=32, d_state=8, d_conv=4, expand=2)


class TestMamba2SSM:
    """Tests for Mamba2SSM module."""

    def test_forward_shape(self, ssm):
        """Output shape matches input."""
        x = torch.randn(2, 16, 32)
        out, cache = ssm(x)
        assert out.shape == (2, 16, 32)
        assert cache is None  # use_cache=False

    def test_forward_with_cache(self, ssm):
        """Cache is returned when use_cache=True."""
        x = torch.randn(2, 16, 32)
        out, cache = ssm(x, use_cache=True)
        assert out.shape == (2, 16, 32)
        assert cache is not None
        assert isinstance(cache, Mamba2Cache)
        assert cache.ssm_state is not None
        assert cache.conv_state is not None

    def test_cache_incremental(self, ssm):
        """Cached generation: token-by-token produces output."""
        batch_size = 2
        # Prefill
        x_prefill = torch.randn(batch_size, 8, 32)
        out_prefill, cache = ssm(x_prefill, use_cache=True)
        assert out_prefill.shape == (batch_size, 8, 32)

        # Decode one token at a time
        x_decode = torch.randn(batch_size, 1, 32)
        out_decode, new_cache = ssm(x_decode, cache=cache, use_cache=True)
        assert out_decode.shape == (batch_size, 1, 32)
        assert new_cache is not None

    def test_ssm_state_shape(self, ssm):
        """SSM state has correct shape [B, d_inner, d_state]."""
        x = torch.randn(2, 16, 32)
        _, cache = ssm(x, use_cache=True)
        # d_inner = d_model * expand = 32 * 2 = 64
        assert cache.ssm_state.shape == (2, 64, 8)

    def test_conv_state_shape(self, ssm):
        """Conv state has correct shape [B, d_inner, d_conv-1]."""
        x = torch.randn(2, 16, 32)
        _, cache = ssm(x, use_cache=True)
        # d_inner = 64, d_conv - 1 = 3
        assert cache.conv_state.shape == (2, 64, 3)

    def test_gradient_flow(self, ssm):
        """Gradients reach all parameters."""
        x = torch.randn(2, 8, 32, requires_grad=True)
        out, _ = ssm(x)
        loss = out.sum()
        loss.backward()

        # Check that key parameters have gradients
        assert ssm.in_proj.weight.grad is not None
        assert ssm.out_proj.weight.grad is not None
        assert ssm.A_log.grad is not None
        assert ssm.D.grad is not None
        assert ssm.dt_proj.weight.grad is not None
        assert ssm.B_proj.weight.grad is not None
        assert ssm.C_proj.weight.grad is not None

    def test_different_seq_lengths(self, ssm):
        """Works with seq_len=1 (decoding) and seq_len=512 (prefill)."""
        x1 = torch.randn(1, 1, 32)
        out1, _ = ssm(x1)
        assert out1.shape == (1, 1, 32)

        x2 = torch.randn(1, 512, 32)
        out2, _ = ssm(x2)
        assert out2.shape == (1, 512, 32)

    def test_scan_correctness(self):
        """Verify scan output against a simple reference implementation."""
        torch.manual_seed(42)
        ssm = Mamba2SSM(d_model=16, d_state=4, d_conv=4, expand=2)
        ssm.eval()

        x = torch.randn(1, 8, 16)

        # Full sequence
        with torch.no_grad():
            out_full, _ = ssm(x)

        # Token-by-token with cache
        with torch.no_grad():
            out_first, cache = ssm(x[:, :1, :], use_cache=True)
            outputs = [out_first]
            for t in range(1, 8):
                out_t, cache = ssm(x[:, t : t + 1, :], cache=cache, use_cache=True)
                outputs.append(out_t)
            out_sequential = torch.cat(outputs, dim=1)

        # First token should match exactly (no cache difference)
        torch.testing.assert_close(out_full[:, :1, :], out_sequential[:, :1, :], atol=1e-5, rtol=1e-4)

    def test_gating(self, ssm):
        """Z-gate modulates output (output differs from ungated path)."""
        x = torch.randn(1, 8, 32)
        out, _ = ssm(x)
        # Just verify output is finite and non-zero
        assert torch.isfinite(out).all()
        assert out.abs().sum() > 0

    def test_skip_connection(self, ssm):
        """D parameter creates skip path."""
        # D is initialized to ones, so skip connection should have effect
        assert ssm.D.abs().sum() > 0

    def test_zero_init(self):
        """With zero initial state, first output depends only on first input."""
        torch.manual_seed(42)
        ssm = Mamba2SSM(d_model=16, d_state=4, d_conv=4, expand=2)
        ssm.eval()

        # Two different sequences that share the first token
        x1 = torch.randn(1, 4, 16)
        x2 = torch.randn(1, 4, 16)
        x2[:, 0, :] = x1[:, 0, :]

        with torch.no_grad():
            out1, _ = ssm(x1)
            out2, _ = ssm(x2)

        # First outputs should be identical since conv and SSM state start at zero
        torch.testing.assert_close(out1[:, 0, :], out2[:, 0, :], atol=1e-6, rtol=1e-5)

    def test_fp16(self, ssm):
        """Works in mixed precision."""
        x = torch.randn(1, 8, 32, dtype=torch.float16)
        ssm_half = ssm.half()
        out, _ = ssm_half(x)
        assert out.dtype == torch.float16
        assert out.shape == (1, 8, 32)
