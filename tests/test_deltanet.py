"""Tests for Gated DeltaNet and Kimi Delta Attention implementations.

Tests cover:
- Forward pass shapes and output correctness
- Causal property (no future information leakage)
- Recurrent state caching for incremental decoding
- Gradient flow through delta rule
- Channel-wise vs scalar gating
- Hybrid architecture layer selection
"""

import pytest
import torch
import torch.nn as nn
from transformers.cache_utils import Cache

from umich_slm.configuration_umichslm import UMichSLMConfig
from umich_slm.configs.presets import get_umichslm_config
from umich_slm.modules.deltanet import (
    GatedDeltaNet,
    KimiDeltaAttention,
    DeltaNetCache,
    DELTANET_CLASSES,
    get_deltanet_class,
    FLA_AVAILABLE,
)
from umich_slm.modules.kv_cache import DeltaNetDynamicCache
from umich_slm.modules.transformer_block import _should_use_deltanet


class TestGatedDeltaNet:
    """Tests for Gated DeltaNet (Qwen3-Next variant)."""

    @pytest.fixture
    def config(self):
        """Create a minimal config for testing."""
        return UMichSLMConfig(
            hidden_size=256,
            num_attention_heads=4,
            num_hidden_layers=4,
        )

    @pytest.fixture
    def deltanet(self, config):
        """Create a GatedDeltaNet module."""
        return GatedDeltaNet(config=config, layer_idx=0)

    def test_forward_shape(self, deltanet):
        """Test that output shape matches input shape."""
        batch_size, seq_len, hidden_size = 2, 16, 256
        x = torch.randn(batch_size, seq_len, hidden_size)

        output, attn_weights, cache = deltanet(x)

        assert output.shape == x.shape
        assert attn_weights is None  # Linear attention doesn't produce weights

    def test_forward_with_cache(self, deltanet):
        """Test incremental decoding with recurrent state caching."""
        batch_size, hidden_size = 2, 256

        # First pass - process initial sequence
        x1 = torch.randn(batch_size, 8, hidden_size)
        output1, _, cache1 = deltanet(x1, use_cache=True)

        assert output1.shape == x1.shape
        assert cache1 is not None  # Cache should be returned
        assert isinstance(cache1, DeltaNetCache)
        assert cache1.recurrent_state is not None
        # Conv state should be present since conv is enabled by default
        assert cache1.conv_state is not None
        assert cache1.conv_state.shape == (batch_size, hidden_size, deltanet.conv_kernel_size - 1)

        # Second pass - process single token with cache
        x2 = torch.randn(batch_size, 1, hidden_size)
        output2, _, cache2 = deltanet(x2, past_key_value=cache1, use_cache=True)

        assert output2.shape == x2.shape
        assert cache2 is not None
        assert isinstance(cache2, DeltaNetCache)

    def test_causal_no_future_leak(self, deltanet):
        """Test that outputs don't depend on future tokens."""
        batch_size, seq_len, hidden_size = 1, 8, 256
        x = torch.randn(batch_size, seq_len, hidden_size)

        # Get full sequence output
        output_full, _, _ = deltanet(x)

        # Get output for first half only
        output_half, _, cache = deltanet(x[:, :4], use_cache=True)

        # The first 4 outputs should be identical
        # (since linear attention with delta rule is causal)
        torch.testing.assert_close(
            output_full[:, :4],
            output_half,
            rtol=1e-4,
            atol=1e-4,
        )

    def test_gradient_flow(self, deltanet):
        """Test that gradients flow through the delta rule."""
        batch_size, seq_len, hidden_size = 2, 8, 256
        x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)

        output, _, _ = deltanet(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert x.grad.abs().sum() > 0  # Gradients should be non-zero

    def test_parameter_count(self, deltanet):
        """Test that the module has expected parameters."""
        # Should have: q_proj, k_proj, v_proj, gate_proj, beta_proj, o_proj, conv
        param_names = [name for name, _ in deltanet.named_parameters()]

        assert "q_proj.weight" in param_names
        assert "k_proj.weight" in param_names
        assert "v_proj.weight" in param_names
        assert "gate_proj.weight" in param_names
        assert "beta_proj.weight" in param_names
        assert "o_proj.weight" in param_names

    def test_gating_shape_scalar(self, deltanet):
        """Test that gating is scalar per head (not channel-wise)."""
        # GatedDeltaNet uses scalar gating per head
        # gate_proj outputs [hidden_size] -> [num_heads]
        assert deltanet.gate_proj.out_features == deltanet.num_heads
        assert deltanet.beta_proj.out_features == deltanet.num_heads


class TestKimiDeltaAttention:
    """Tests for Kimi Delta Attention (KDA, channel-wise gating)."""

    @pytest.fixture
    def config(self):
        """Create a minimal config for testing."""
        return UMichSLMConfig(
            hidden_size=256,
            num_attention_heads=4,
            num_hidden_layers=4,
            deltanet_variant="kda",
        )

    @pytest.fixture
    def kda(self, config):
        """Create a KimiDeltaAttention module."""
        return KimiDeltaAttention(config=config, layer_idx=0)

    def test_forward_shape(self, kda):
        """Test that output shape matches input shape."""
        batch_size, seq_len, hidden_size = 2, 16, 256
        x = torch.randn(batch_size, seq_len, hidden_size)

        output, attn_weights, cache = kda(x)

        assert output.shape == x.shape
        assert attn_weights is None

    def test_channel_wise_gating(self, kda):
        """Test that KDA uses channel-wise gating."""
        # KDA gate_proj outputs [hidden_size] -> [num_heads * head_dim * expand_v]
        expected_gate_dim = kda.num_heads * kda.head_dim * kda.expand_v
        assert kda.gate_proj.out_features == expected_gate_dim
        assert kda.beta_proj.out_features == expected_gate_dim

    def test_gradient_flow(self, kda):
        """Test that gradients flow through channel-wise gating."""
        batch_size, seq_len, hidden_size = 2, 8, 256
        x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)

        output, _, _ = kda(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestDeltaNetRegistry:
    """Tests for the DeltaNet class registry."""

    def test_registry_contents(self):
        """Test that registry contains expected classes."""
        assert "gated" in DELTANET_CLASSES
        assert "kda" in DELTANET_CLASSES
        assert DELTANET_CLASSES["gated"] == GatedDeltaNet
        assert DELTANET_CLASSES["kda"] == KimiDeltaAttention

    def test_get_deltanet_class(self):
        """Test get_deltanet_class helper function."""
        assert get_deltanet_class("gated") == GatedDeltaNet
        assert get_deltanet_class("kda") == KimiDeltaAttention

    def test_get_deltanet_class_invalid(self):
        """Test get_deltanet_class with invalid variant."""
        with pytest.raises(ValueError, match="Unknown DeltaNet variant"):
            get_deltanet_class("invalid")

    def test_deltanet_dynamic_cache_init_works_with_old_cache_signature(self, monkeypatch):
        """DeltaNetDynamicCache should tolerate older Cache.__init__ signatures."""
        original_init = Cache.__init__

        def old_cache_init(self):
            self.layers = []

        monkeypatch.setattr(Cache, "__init__", old_cache_init)
        try:
            cache = DeltaNetDynamicCache(num_layers=4, deltanet_layers=[True, False, True, False])
        finally:
            monkeypatch.setattr(Cache, "__init__", original_init)

        assert len(cache) == 4
        assert cache.layers == []


class TestHybridLayerSelection:
    """Tests for hybrid architecture layer selection."""

    def test_deltanet_disabled_by_default(self):
        """Test that DeltaNet is disabled by default."""
        config = UMichSLMConfig(
            hidden_size=256,
            num_hidden_layers=8,
            # use_deltanet defaults to False
        )

        for layer_idx in range(8):
            assert not _should_use_deltanet(config, layer_idx)

    def test_all_deltanet_when_enabled_default(self):
        """Test all layers use DeltaNet by default when use_deltanet=True.

        The default deltanet_every_n_layers=0 means all layers use DeltaNet.
        """
        config = UMichSLMConfig(
            hidden_size=256,
            num_hidden_layers=8,
            use_deltanet=True,
            # deltanet_every_n_layers defaults to 0 (all DeltaNet)
        )

        for layer_idx in range(8):
            assert _should_use_deltanet(
                config, layer_idx
            ), f"Layer {layer_idx} should use DeltaNet with default settings"

    def test_all_deltanet_explicit_zero(self):
        """Test all layers use DeltaNet when explicitly set to every_n=0."""
        config = UMichSLMConfig(
            hidden_size=256,
            num_hidden_layers=8,
            use_deltanet=True,
            deltanet_every_n_layers=0,  # All DeltaNet
        )

        for layer_idx in range(8):
            assert _should_use_deltanet(config, layer_idx)

    def test_hybrid_every_n_layers(self):
        """Test hybrid pattern with every_n_layers=4."""
        config = UMichSLMConfig(
            hidden_size=256,
            num_hidden_layers=8,
            use_deltanet=True,
            deltanet_every_n_layers=4,  # Every 4th layer is attention
        )

        # Layers 0,1,2 = DeltaNet, 3 = Attention, 4,5,6 = DeltaNet, 7 = Attention
        expected = [True, True, True, False, True, True, True, False]
        for layer_idx in range(8):
            result = _should_use_deltanet(config, layer_idx)
            assert (
                result == expected[layer_idx]
            ), f"Layer {layer_idx}: expected {expected[layer_idx]}, got {result}"

    def test_hybrid_attention_ratio(self):
        """Test explicit hybrid pattern with ratio list."""
        config = UMichSLMConfig(
            hidden_size=256,
            num_hidden_layers=12,
            use_deltanet=True,
            hybrid_attention_ratio=["delta", "delta", "delta", "attn"],  # 3:1 ratio
        )

        # Pattern repeats: delta, delta, delta, attn, delta, delta, delta, attn, ...
        for layer_idx in range(12):
            expected = (layer_idx % 4) != 3  # DeltaNet unless idx%4 == 3
            result = _should_use_deltanet(config, layer_idx)
            assert result == expected, f"Layer {layer_idx}: expected {expected}, got {result}"

    def test_first_n_layers_attention(self):
        """Test forcing first N layers to use attention."""
        config = UMichSLMConfig(
            hidden_size=256,
            num_hidden_layers=8,
            use_deltanet=True,
            deltanet_first_n_layers=2,  # First 2 layers are attention
            deltanet_every_n_layers=0,  # Rest are DeltaNet
        )

        # Layers 0,1 = Attention (forced), 2+ = DeltaNet
        assert not _should_use_deltanet(config, 0)
        assert not _should_use_deltanet(config, 1)
        assert _should_use_deltanet(config, 2)
        assert _should_use_deltanet(config, 7)

    def test_qwen3_5_like_tiny_preset_hybrid_pattern(self):
        """Qwen3.5-like tiny preset should follow the 3:1 hybrid pattern."""
        config = get_umichslm_config("qwen3_5_like_tiny")
        expected = [True, True, True, False, True, True, True, False, True]

        for layer_idx, expected_uses_delta in enumerate(expected):
            assert _should_use_deltanet(config, layer_idx) == expected_uses_delta


class TestDeltaNetConfig:
    """Tests for DeltaNet configuration."""

    def test_default_config_disabled(self):
        """Test that DeltaNet is disabled by default."""
        config = UMichSLMConfig(hidden_size=256)
        assert config.use_deltanet is False

    def test_deltanet_config_values(self):
        """Test DeltaNet config values are stored correctly."""
        config = UMichSLMConfig(
            hidden_size=256,
            use_deltanet=True,
            deltanet_variant="kda",
            deltanet_expand_k=2,
            deltanet_expand_v=2,
            deltanet_every_n_layers=4,
            hybrid_attention_ratio=["delta", "attn"],
        )

        assert config.use_deltanet is True
        assert config.deltanet_variant == "kda"
        assert config.deltanet_expand_k == 2
        assert config.deltanet_expand_v == 2
        assert config.deltanet_every_n_layers == 4
        assert config.hybrid_attention_ratio == ["delta", "attn"]

    def test_invalid_variant_raises(self):
        """Test that invalid variant raises ValueError."""
        with pytest.raises(ValueError, match="deltanet_variant must be one of"):
            UMichSLMConfig(
                hidden_size=256,
                use_deltanet=True,
                deltanet_variant="invalid",
            )

    def test_invalid_hybrid_ratio_raises(self):
        """Test that invalid hybrid ratio raises ValueError."""
        with pytest.raises(ValueError, match="hybrid_attention_ratio"):
            UMichSLMConfig(
                hidden_size=256,
                use_deltanet=True,
                hybrid_attention_ratio=["delta", "invalid"],
            )

    def test_mla_and_deltanet_conflict(self):
        """Test that MLA and DeltaNet cannot be used together."""
        with pytest.raises(ValueError, match="Cannot use both MLA and DeltaNet"):
            UMichSLMConfig(
                hidden_size=256,
                use_mla=True,
                use_deltanet=True,
            )


class TestDeltaNetWithExpansion:
    """Tests for DeltaNet with key/value expansion."""

    @pytest.fixture
    def config_expanded(self):
        """Create config with K/V expansion."""
        return UMichSLMConfig(
            hidden_size=256,
            num_attention_heads=4,
            deltanet_expand_k=2,
            deltanet_expand_v=2,
        )

    @pytest.fixture
    def deltanet_expanded(self, config_expanded):
        """Create GatedDeltaNet with expansion."""
        return GatedDeltaNet(config=config_expanded, layer_idx=0)

    def test_expanded_projections(self, deltanet_expanded):
        """Test that K/V projections are expanded."""
        # With expand_k=2, expand_v=2:
        # k_proj: hidden_size -> num_heads * head_dim * expand_k
        # v_proj: hidden_size -> num_heads * head_dim * expand_v
        head_dim = 256 // 4  # 64
        expected_k_dim = 4 * head_dim * 2  # 512
        expected_v_dim = 4 * head_dim * 2  # 512

        assert deltanet_expanded.k_proj.out_features == expected_k_dim
        assert deltanet_expanded.v_proj.out_features == expected_v_dim

    def test_expanded_forward(self, deltanet_expanded):
        """Test forward with expanded K/V dimensions."""
        batch_size, seq_len, hidden_size = 2, 8, 256
        x = torch.randn(batch_size, seq_len, hidden_size)

        output, _, cache = deltanet_expanded(x, use_cache=True)

        assert output.shape == x.shape
        # Cache should have shape [B, H, D_k, D_v] with expanded dimensions


class TestDeltaNetWithoutConv:
    """Tests for DeltaNet without short convolution."""

    @pytest.fixture
    def config_no_conv(self):
        """Create config without short conv."""
        return UMichSLMConfig(
            hidden_size=256,
            num_attention_heads=4,
            deltanet_use_short_conv=False,
        )

    @pytest.fixture
    def deltanet_no_conv(self, config_no_conv):
        """Create GatedDeltaNet without conv."""
        return GatedDeltaNet(config=config_no_conv, layer_idx=0)

    def test_no_conv_module(self, deltanet_no_conv):
        """Test that conv module is None."""
        assert deltanet_no_conv.conv is None

    def test_forward_without_conv(self, deltanet_no_conv):
        """Test forward without short convolution."""
        batch_size, seq_len, hidden_size = 2, 8, 256
        x = torch.randn(batch_size, seq_len, hidden_size)

        output, _, _ = deltanet_no_conv(x)
        assert output.shape == x.shape


class TestDeltaNetIncrementalDecoding:
    """Tests for incremental decoding with conv state caching.

    These tests verify that token-by-token generation produces the same
    outputs as processing the full sequence at once. This is critical for
    correct behavior during autoregressive generation.
    """

    @pytest.fixture
    def config(self):
        """Create a config with short conv enabled."""
        return UMichSLMConfig(
            hidden_size=256,
            num_attention_heads=4,
            num_hidden_layers=4,
            deltanet_use_short_conv=True,
            deltanet_conv_kernel_size=4,
        )

    @pytest.fixture
    def deltanet(self, config):
        """Create a GatedDeltaNet module."""
        return GatedDeltaNet(config=config, layer_idx=0)

    def test_incremental_matches_full_sequence(self, deltanet):
        """Test that incremental decoding produces same output as full sequence.

        This is the key test for conv state caching correctness.
        """
        torch.manual_seed(42)
        batch_size, seq_len, hidden_size = 1, 12, 256

        # Create a fixed input sequence
        x_full = torch.randn(batch_size, seq_len, hidden_size)

        # Process full sequence at once
        deltanet.eval()
        with torch.no_grad():
            output_full, _, _ = deltanet(x_full)

        # Process token by token with caching
        outputs_incremental = []
        cache = None
        for t in range(seq_len):
            x_t = x_full[:, t : t + 1, :]  # Single token
            with torch.no_grad():
                output_t, _, cache = deltanet(x_t, past_key_value=cache, use_cache=True)
            outputs_incremental.append(output_t)

        output_incremental = torch.cat(outputs_incremental, dim=1)

        # Outputs should match
        torch.testing.assert_close(
            output_full,
            output_incremental,
            rtol=1e-4,
            atol=1e-4,
            msg="Incremental decoding output differs from full sequence output",
        )

    def test_incremental_with_prefill(self, deltanet):
        """Test incremental decoding after processing a prefix (prefill).

        Common pattern: process prompt in one pass, then generate token-by-token.
        """
        torch.manual_seed(123)
        batch_size, prefill_len, gen_len, hidden_size = 1, 8, 4, 256

        x_full = torch.randn(batch_size, prefill_len + gen_len, hidden_size)

        # Process full sequence
        deltanet.eval()
        with torch.no_grad():
            output_full, _, _ = deltanet(x_full)

        # Process with prefill + incremental
        with torch.no_grad():
            # Prefill phase
            output_prefill, _, cache = deltanet(x_full[:, :prefill_len], use_cache=True)

            # Generation phase (token by token)
            outputs_gen = []
            for t in range(gen_len):
                x_t = x_full[:, prefill_len + t : prefill_len + t + 1]
                output_t, _, cache = deltanet(x_t, past_key_value=cache, use_cache=True)
                outputs_gen.append(output_t)

        output_combined = torch.cat([output_prefill] + outputs_gen, dim=1)

        torch.testing.assert_close(
            output_full,
            output_combined,
            rtol=1e-4,
            atol=1e-4,
            msg="Prefill + incremental output differs from full sequence",
        )

    def test_conv_state_shape(self, deltanet):
        """Test that conv state has correct shape throughout generation."""
        batch_size, hidden_size = 2, 256
        kernel_size = deltanet.conv_kernel_size

        # Process various sequence lengths
        for seq_len in [1, 2, 3, 8, 16]:
            x = torch.randn(batch_size, seq_len, hidden_size)
            _, _, cache = deltanet(x, use_cache=True)

            assert cache.conv_state is not None
            assert cache.conv_state.shape == (
                batch_size,
                hidden_size,
                kernel_size - 1,
            ), f"Conv state shape wrong for seq_len={seq_len}"

    def test_incremental_kda(self):
        """Test incremental decoding for KimiDeltaAttention (channel-wise gating)."""
        config = UMichSLMConfig(
            hidden_size=256,
            num_attention_heads=4,
            deltanet_variant="kda",
            deltanet_use_short_conv=True,
        )
        kda = KimiDeltaAttention(config=config, layer_idx=0)

        torch.manual_seed(42)
        batch_size, seq_len, hidden_size = 1, 8, 256
        x_full = torch.randn(batch_size, seq_len, hidden_size)

        # Full sequence
        kda.eval()
        with torch.no_grad():
            output_full, _, _ = kda(x_full)

        # Incremental
        outputs_inc = []
        cache = None
        for t in range(seq_len):
            with torch.no_grad():
                out_t, _, cache = kda(x_full[:, t : t + 1], past_key_value=cache, use_cache=True)
            outputs_inc.append(out_t)

        output_inc = torch.cat(outputs_inc, dim=1)

        torch.testing.assert_close(
            output_full,
            output_inc,
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.skipif(not FLA_AVAILABLE, reason="FLA library not available")
class TestDeltaNetWithFLA:
    """Tests for DeltaNet with FLA library (requires CUDA)."""

    @pytest.fixture
    def config(self):
        """Create config for FLA testing."""
        return UMichSLMConfig(
            hidden_size=256,
            num_attention_heads=4,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fla_forward(self, config):
        """Test forward with FLA Triton kernels."""
        deltanet = GatedDeltaNet(config=config, layer_idx=0).cuda()
        x = torch.randn(2, 16, 256, device="cuda")

        output, _, _ = deltanet(x)
        assert output.shape == x.shape
        assert output.device.type == "cuda"
