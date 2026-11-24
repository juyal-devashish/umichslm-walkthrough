"""Tests for attention module."""

import pytest
import torch

from umich_slm import UMichSLMConfig
from umich_slm.modules.attention import (
    UMichSLMAttention,
    ATTENTION_CLASSES,
    get_attention_class,
    FLASH_ATTENTION_AVAILABLE,
)


class TestUMichSLMAttention:
    """Tests for attention implementation."""

    @pytest.fixture
    def attention_config(self):
        """Create a config for attention testing."""
        return UMichSLMConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            attention_dropout=0.0,
        )

    @pytest.fixture
    def attention_module(self, attention_config):
        """Create an attention module."""
        return UMichSLMAttention(attention_config, layer_idx=0)

    def test_output_shape(self, attention_module, attention_config):
        """Output should have the same shape as input."""
        hidden_states = torch.randn(2, 16, attention_config.hidden_size)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        output, _, _ = attention_module(hidden_states, position_ids=position_ids)

        assert output.shape == hidden_states.shape

    def test_gqa_kv_heads(self, attention_config):
        """GQA should use fewer KV heads than query heads."""
        attn = UMichSLMAttention(attention_config, layer_idx=0)

        assert attn.num_heads == 4
        assert attn.num_key_value_heads == 2
        assert attn.num_key_value_groups == 2

    def test_kv_cache(self, attention_module, attention_config):
        """KV cache should enable incremental decoding."""
        # First forward pass
        hidden_1 = torch.randn(1, 4, attention_config.hidden_size)
        pos_1 = torch.arange(4).unsqueeze(0)
        _, _, past_kv = attention_module(hidden_1, position_ids=pos_1, use_cache=True)

        assert past_kv is not None
        assert past_kv[0].shape[2] == 4  # KV cache has 4 positions

        # Second forward pass (next token)
        hidden_2 = torch.randn(1, 1, attention_config.hidden_size)
        pos_2 = torch.tensor([[4]])
        output, _, new_past_kv = attention_module(
            hidden_2, position_ids=pos_2, past_key_value=past_kv, use_cache=True
        )

        # Cache should have grown
        assert new_past_kv[0].shape[2] == 5  # 4 + 1

    def test_causal_masking(self, attention_module, attention_config):
        """Attention should be causal (no attending to future tokens)."""
        hidden_states = torch.randn(1, 8, attention_config.hidden_size)
        position_ids = torch.arange(8).unsqueeze(0)

        _, attn_weights, _ = attention_module(
            hidden_states, position_ids=position_ids, output_attentions=True
        )

        if attn_weights is not None:
            # Upper triangular should be zero after softmax
            for i in range(8):
                for j in range(i + 1, 8):
                    # Future positions should have ~0 attention
                    assert attn_weights[0, :, i, j].max() < 0.01

    def test_gradient_flow(self, attention_module, attention_config):
        """Gradients should flow through attention."""
        hidden_states = torch.randn(2, 16, attention_config.hidden_size, requires_grad=True)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        output, _, _ = attention_module(hidden_states, position_ids=position_ids)
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert attention_module.q_proj.weight.grad is not None

    def test_position_ids_default(self, attention_module, attention_config):
        """Should create default position IDs if not provided."""
        hidden_states = torch.randn(2, 16, attention_config.hidden_size)

        # Should not raise even without position_ids
        output, _, _ = attention_module(hidden_states)
        assert output.shape == hidden_states.shape

    def test_mixed_precision(self, attention_config):
        """Should work with mixed precision."""
        attn = UMichSLMAttention(attention_config, layer_idx=0).to(torch.float16)
        hidden_states = torch.randn(2, 16, attention_config.hidden_size, dtype=torch.float16)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        output, _, _ = attn(hidden_states, position_ids=position_ids)
        assert output.dtype == torch.float16


class TestSDPAAttention:
    """Tests for SDPA attention implementation."""

    @pytest.fixture
    def sdpa_config(self):
        """Create a config for SDPA testing."""
        config = UMichSLMConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
        )
        config._attn_implementation = "sdpa"
        return config

    def test_sdpa_output_shape(self, sdpa_config):
        """SDPA attention should produce correct output shape."""
        attn = get_attention_class("sdpa")(sdpa_config, layer_idx=0)
        hidden_states = torch.randn(2, 16, sdpa_config.hidden_size)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        output, weights, _ = attn(hidden_states, position_ids=position_ids)

        assert output.shape == hidden_states.shape
        # SDPA doesn't return attention weights
        assert weights is None


class TestFlashAttention:
    """Tests for Flash Attention 2 implementation."""

    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention not available")
    def test_flash_attention_available(self):
        """Flash Attention should be importable when available."""
        assert FLASH_ATTENTION_AVAILABLE

    def test_flash_attention_class_exists(self):
        """Flash Attention class should be in ATTENTION_CLASSES."""
        assert "flash_attention_2" in ATTENTION_CLASSES


class TestGetAttentionClass:
    """Tests for get_attention_class helper."""

    def test_get_eager(self):
        """Should return base class for 'eager'."""
        cls = get_attention_class("eager")
        assert cls == UMichSLMAttention

    def test_get_sdpa(self):
        """Should return SDPA class for 'sdpa'."""
        cls = get_attention_class("sdpa")
        assert cls.__name__ == "UMichSLMSdpaAttention"

    def test_invalid_implementation(self):
        """Should raise error for unknown implementation."""
        with pytest.raises(ValueError):
            get_attention_class("unknown")


class TestPoPEAttention:
    """Tests for PoPE attention implementation."""

    @pytest.fixture
    def pope_attention_config(self):
        """Create a config for PoPE attention testing."""
        return UMichSLMConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            attention_dropout=0.0,
            position_embedding_type="pope",
        )

    def test_pope_output_shape(self, pope_attention_config):
        """PoPE attention should produce correct output shape."""
        attn = UMichSLMAttention(pope_attention_config, layer_idx=0)
        hidden_states = torch.randn(2, 16, pope_attention_config.hidden_size)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        output, _, _ = attn(hidden_states, position_ids=position_ids)

        # Output should match input shape (hidden_size, not doubled)
        assert output.shape == hidden_states.shape

    def test_pope_kv_cache(self, pope_attention_config):
        """PoPE attention should work with KV cache."""
        attn = UMichSLMAttention(pope_attention_config, layer_idx=0)

        # First pass
        hidden_1 = torch.randn(1, 8, pope_attention_config.hidden_size)
        pos_1 = torch.arange(8).unsqueeze(0)
        _, _, past_kv = attn(hidden_1, position_ids=pos_1, use_cache=True)

        # Second pass (incremental)
        hidden_2 = torch.randn(1, 1, pope_attention_config.hidden_size)
        pos_2 = torch.tensor([[8]])
        output, _, new_past_kv = attn(
            hidden_2, position_ids=pos_2, past_key_value=past_kv, use_cache=True
        )

        # Output should still have correct shape
        assert output.shape == hidden_2.shape

    def test_pope_forces_eager_attention(self, pope_attention_config):
        """PoPE should use eager attention regardless of config."""
        # Even with SDPA config, PoPE should force eager
        pope_attention_config._attn_implementation = "sdpa"
        attn = UMichSLMAttention(pope_attention_config, layer_idx=0)

        assert attn._use_pope is True

        # Running forward should work (falls back to eager)
        hidden_states = torch.randn(2, 16, pope_attention_config.hidden_size)
        output, _, _ = attn(hidden_states)
        assert output.shape == hidden_states.shape

    def test_pope_multiple_incremental_steps(self, pope_attention_config):
        """PoPE attention should work across multiple incremental steps."""
        attn = UMichSLMAttention(pope_attention_config, layer_idx=0)

        # Initial context
        hidden = torch.randn(1, 4, pope_attention_config.hidden_size)
        pos = torch.arange(4).unsqueeze(0)
        _, _, past_kv = attn(hidden, position_ids=pos, use_cache=True)

        # Multiple incremental steps (simulating generation)
        for step in range(10):
            hidden = torch.randn(1, 1, pope_attention_config.hidden_size)
            pos = torch.tensor([[4 + step]])
            output, _, past_kv = attn(
                hidden, position_ids=pos, past_key_value=past_kv, use_cache=True
            )

        # Should not raise and output should have correct shape
        assert output.shape == (1, 1, pope_attention_config.hidden_size)
