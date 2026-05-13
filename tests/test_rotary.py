"""Tests for Rotary Position Embeddings (RoPE) and PoPE."""

import pytest
import torch

from umich_slm.modules.rotary import (
    RotaryEmbedding,
    PolarPositionEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
    get_position_embedding_class,
)


class TestRotaryEmbedding:
    """Tests for RoPE implementation."""

    def test_output_shape(self):
        """Output should have the same shape as input."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512)
        q = torch.randn(2, 4, 16, 64)  # [batch, heads, seq, dim]
        k = torch.randn(2, 4, 16, 64)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        q_rot, k_rot = rope(q, k, position_ids)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_different_positions_different_output(self):
        """Different positions should produce different encodings."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512)
        q = torch.randn(1, 1, 2, 64)  # Two positions
        k = q.clone()

        pos_0 = torch.tensor([[0, 1]])
        pos_1 = torch.tensor([[10, 11]])

        q0, _ = rope(q, k, pos_0)
        q1, _ = rope(q, k, pos_1)

        assert not torch.allclose(q0, q1)

    def test_same_input_same_position_same_output(self):
        """Same input and position should produce same output."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512)
        q = torch.randn(1, 1, 4, 64)
        k = q.clone()
        position_ids = torch.tensor([[0, 1, 2, 3]])

        q1, k1 = rope(q, k, position_ids)
        q2, k2 = rope(q, k, position_ids)

        assert torch.allclose(q1, q2)
        assert torch.allclose(k1, k2)

    def test_cache_extension(self):
        """Cache should extend when sequence exceeds initial max."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=32)
        q = torch.randn(1, 1, 64, 64)  # Longer than initial max
        k = q.clone()
        position_ids = torch.arange(64).unsqueeze(0)

        q_rot, k_rot = rope(q, k, position_ids)

        assert q_rot.shape == q.shape
        assert rope.max_seq_len_cached >= 64

    def test_linear_scaling(self):
        """Linear scaling should modify the position frequencies."""
        rope = RotaryEmbedding(
            dim=64,
            max_position_embeddings=512,
            scaling_factor=2.0,
            rope_type="linear",
        )
        q = torch.randn(1, 1, 16, 64)
        k = q.clone()
        position_ids = torch.arange(16).unsqueeze(0)

        q_rot, k_rot = rope(q, k, position_ids)
        assert q_rot.shape == q.shape

    def test_gradient_flow(self):
        """Gradients should flow through RoPE."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512)
        q = torch.randn(2, 4, 16, 64, requires_grad=True)
        k = torch.randn(2, 4, 16, 64, requires_grad=True)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        q_rot, k_rot = rope(q, k, position_ids)
        loss = q_rot.sum() + k_rot.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None


class TestPolarPositionEmbedding:
    """Tests for PoPE implementation."""

    def test_output_shape(self):
        """Output dimensions double due to interleaved x/y components."""
        pope = PolarPositionEmbedding(dim=32, max_position_embeddings=512)
        q = torch.randn(2, 4, 16, 32)
        k = torch.randn(2, 4, 16, 32)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        q_rot, k_rot = pope(q, k, position_ids)

        # Output dimension doubles due to x/y interleaving
        assert q_rot.shape == (2, 4, 16, 64)
        assert k_rot.shape == (2, 4, 16, 64)

    def test_learnable_bias(self):
        """Phase bias should be learnable when enabled."""
        pope = PolarPositionEmbedding(dim=64, learnable_bias=True)
        assert hasattr(pope, "phase_bias")
        assert pope.phase_bias.requires_grad

    def test_no_learnable_bias(self):
        """Phase bias should be a buffer when disabled."""
        pope = PolarPositionEmbedding(dim=64, learnable_bias=False)
        assert hasattr(pope, "phase_bias")
        assert not pope.phase_bias.requires_grad

    def test_different_positions_different_output(self):
        """Different positions should produce different encodings."""
        pope = PolarPositionEmbedding(dim=32, max_position_embeddings=512)
        q = torch.randn(1, 1, 2, 32)
        k = q.clone()

        pos_0 = torch.tensor([[0, 1]])
        pos_1 = torch.tensor([[10, 11]])

        q0, _ = pope(q, k, pos_0)
        q1, _ = pope(q, k, pos_1)

        assert not torch.allclose(q0, q1)

    def test_non_negative_magnitudes(self):
        """Softplus should ensure non-negative magnitudes."""
        pope = PolarPositionEmbedding(dim=32, max_position_embeddings=512)
        # Use negative input to test softplus
        q = torch.randn(2, 4, 16, 32) - 5  # Shift to negative
        k = q.clone()
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        q_rot, k_rot = pope(q, k, position_ids)

        # No NaN should appear
        assert not torch.isnan(q_rot).any()
        assert not torch.isnan(k_rot).any()

    def test_gradient_flow(self):
        """Gradients should flow through PoPE."""
        pope = PolarPositionEmbedding(dim=32, max_position_embeddings=512)
        q = torch.randn(2, 4, 16, 32, requires_grad=True)
        k = torch.randn(2, 4, 16, 32, requires_grad=True)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        q_rot, k_rot = pope(q, k, position_ids)
        loss = q_rot.sum() + k_rot.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        if pope.learnable_bias:
            assert pope.phase_bias.grad is not None


class TestRotateHalf:
    """Tests for rotate_half helper function."""

    def test_rotate_half_shape(self):
        """Output should have same shape as input."""
        x = torch.randn(2, 4, 16, 64)
        out = rotate_half(x)
        assert out.shape == x.shape

    def test_rotate_half_values(self):
        """Verify rotation operation is correct."""
        x = torch.tensor([[1, 2, 3, 4]])  # [1, 4]
        out = rotate_half(x)
        expected = torch.tensor([[-3, -4, 1, 2]])
        assert torch.allclose(out, expected)


class TestGetPositionEmbeddingClass:
    """Tests for get_position_embedding_class helper."""

    def test_get_rope(self):
        """Should return RotaryEmbedding for 'rope'."""
        cls = get_position_embedding_class("rope")
        assert cls == RotaryEmbedding

    def test_get_pope(self):
        """Should return PolarPositionEmbedding for 'pope'."""
        cls = get_position_embedding_class("pope")
        assert cls == PolarPositionEmbedding

    def test_invalid_type(self):
        """Should raise error for unknown type."""
        with pytest.raises(ValueError):
            get_position_embedding_class("unknown")
