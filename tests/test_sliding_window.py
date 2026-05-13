"""Tests for Sliding Window Attention with Global Refresh."""

import pytest
import torch

from umich_slm import UMichSLMConfig, UMichSLMForCausalLM, UMichSLMModel
from umich_slm.modules.sliding_window import (
    create_sliding_window_mask,
    create_sliding_window_mask_for_sdpa,
    is_global_attention_layer,
    SlidingWindowCache,
    get_attention_pattern_for_layer,
)


class TestSlidingWindowMask:
    """Tests for sliding window mask creation."""

    def test_mask_shape(self):
        """Test mask has correct shape."""
        mask = create_sliding_window_mask(seq_len=10, window_size=4)
        assert mask.shape == (1, 1, 10, 10)

    def test_mask_is_causal(self):
        """Test that mask is causal (no future tokens)."""
        mask = create_sliding_window_mask(seq_len=5, window_size=10)
        # Upper triangle (future) should be -inf
        for i in range(5):
            for j in range(i + 1, 5):
                assert mask[0, 0, i, j] == float("-inf")

    def test_mask_respects_window_size(self):
        """Test that positions outside window are masked."""
        seq_len = 10
        window_size = 3
        mask = create_sliding_window_mask(seq_len=seq_len, window_size=window_size)

        for i in range(seq_len):
            for j in range(seq_len):
                if j <= i and (i - j) < window_size:
                    # Within window and causal - should be 0
                    assert mask[0, 0, i, j] == 0.0
                else:
                    # Outside window or future - should be -inf
                    assert mask[0, 0, i, j] == float("-inf")

    def test_mask_window_size_one(self):
        """Test with window size of 1 (only self-attention)."""
        mask = create_sliding_window_mask(seq_len=5, window_size=1)
        # Only diagonal should be attended to
        for i in range(5):
            assert mask[0, 0, i, i] == 0.0
            for j in range(5):
                if i != j:
                    assert mask[0, 0, i, j] == float("-inf")

    def test_mask_window_larger_than_seq(self):
        """Test when window size >= sequence length."""
        mask = create_sliding_window_mask(seq_len=5, window_size=10)
        # Should be equivalent to standard causal mask
        for i in range(5):
            for j in range(5):
                if j <= i:
                    assert mask[0, 0, i, j] == 0.0
                else:
                    assert mask[0, 0, i, j] == float("-inf")


class TestSdpaSlidingWindowMask:
    """Tests for SDPA-compatible sliding window mask."""

    def test_mask_shape(self):
        """Test mask has correct shape."""
        mask = create_sliding_window_mask_for_sdpa(seq_len=10, window_size=4)
        assert mask.shape == (1, 1, 10, 10)
        assert mask.dtype == torch.bool

    def test_mask_logic_inverted(self):
        """Test that True means 'do not attend' for SDPA."""
        mask = create_sliding_window_mask_for_sdpa(seq_len=5, window_size=3)
        regular_mask = create_sliding_window_mask(seq_len=5, window_size=3)

        for i in range(5):
            for j in range(5):
                # SDPA: True = don't attend, regular: -inf = don't attend
                if regular_mask[0, 0, i, j] == float("-inf"):
                    assert mask[0, 0, i, j] == True
                else:
                    assert mask[0, 0, i, j] == False


class TestGlobalAttentionLayers:
    """Tests for global attention layer determination."""

    def test_first_layer_is_global(self):
        """First layer should always use global attention."""
        assert is_global_attention_layer(0, 4) == True
        assert is_global_attention_layer(0, 8) == True

    def test_every_n_layers_is_global(self):
        """Every N-th layer should use global attention."""
        # With global_attention_every_n_layers=4
        assert is_global_attention_layer(0, 4) == True   # Layer 0
        assert is_global_attention_layer(1, 4) == False  # Layer 1
        assert is_global_attention_layer(2, 4) == False  # Layer 2
        assert is_global_attention_layer(3, 4) == False  # Layer 3
        assert is_global_attention_layer(4, 4) == True   # Layer 4
        assert is_global_attention_layer(5, 4) == False  # Layer 5
        assert is_global_attention_layer(8, 4) == True   # Layer 8

    def test_disabled_global_refresh(self):
        """Setting global_attention_every_n_layers <= 0 disables global refresh."""
        assert is_global_attention_layer(0, 0) == False
        assert is_global_attention_layer(4, 0) == False
        assert is_global_attention_layer(0, -1) == False

    def test_get_attention_pattern(self):
        """Test get_attention_pattern_for_layer function."""
        # Full attention pattern
        assert get_attention_pattern_for_layer(0, 8, 4096, 4, "full") == "global"
        assert get_attention_pattern_for_layer(3, 8, 4096, 4, "full") == "global"

        # Sliding window only
        assert get_attention_pattern_for_layer(0, 8, 4096, 4, "sliding_window") == "local"
        assert get_attention_pattern_for_layer(3, 8, 4096, 4, "sliding_window") == "local"

        # Sliding window with global refresh
        assert get_attention_pattern_for_layer(0, 8, 4096, 4, "sliding_window_global_refresh") == "global"
        assert get_attention_pattern_for_layer(1, 8, 4096, 4, "sliding_window_global_refresh") == "local"
        assert get_attention_pattern_for_layer(4, 8, 4096, 4, "sliding_window_global_refresh") == "global"


class TestSlidingWindowCache:
    """Tests for rolling KV cache."""

    @pytest.fixture
    def cache(self):
        return SlidingWindowCache(
            window_size=8,
            num_layers=2,
            num_heads=4,
            head_dim=16,
            batch_size=1,
        )

    def test_cache_initialization(self, cache):
        """Test cache is properly initialized."""
        assert cache.get_seq_length(0) == 0
        assert cache.get_seq_length(1) == 0

    def test_cache_update_within_window(self, cache):
        """Test updating cache when within window size."""
        key_states = torch.randn(1, 4, 3, 16)
        value_states = torch.randn(1, 4, 3, 16)

        cached_k, cached_v = cache.update(key_states, value_states, layer_idx=0)

        assert cache.get_seq_length(0) == 3
        assert cached_k.shape == (1, 4, 3, 16)
        assert cached_v.shape == (1, 4, 3, 16)
        assert torch.allclose(cached_k, key_states)

    def test_cache_rolling_when_overflow(self, cache):
        """Test cache rolls when exceeding window size."""
        # Add 6 tokens
        k1 = torch.randn(1, 4, 6, 16)
        v1 = torch.randn(1, 4, 6, 16)
        cache.update(k1, v1, layer_idx=0)
        assert cache.get_seq_length(0) == 6

        # Add 4 more tokens (total 10 > window_size=8)
        k2 = torch.randn(1, 4, 4, 16)
        v2 = torch.randn(1, 4, 4, 16)
        cached_k, cached_v = cache.update(k2, v2, layer_idx=0)

        # Should only keep 8 tokens
        assert cache.get_seq_length(0) == 8
        assert cached_k.shape == (1, 4, 8, 16)

    def test_cache_reset(self, cache):
        """Test cache reset."""
        k = torch.randn(1, 4, 5, 16)
        v = torch.randn(1, 4, 5, 16)
        cache.update(k, v, layer_idx=0)
        assert cache.get_seq_length(0) == 5

        cache.reset()
        assert cache.get_seq_length(0) == 0


class TestSlidingWindowModel:
    """Tests for full model with sliding window attention."""

    @pytest.fixture
    def swa_config(self):
        return UMichSLMConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            sliding_window=16,  # Small window for testing
            global_attention_every_n_layers=2,  # Layers 0, 2 are global
        )

    @pytest.fixture
    def swa_model(self, swa_config):
        return UMichSLMForCausalLM(swa_config)

    def test_swa_model_forward(self, swa_model):
        """Test forward pass with sliding window."""
        input_ids = torch.randint(0, 1000, (2, 32))  # Longer than window
        outputs = swa_model(input_ids)

        assert outputs.logits is not None
        assert outputs.logits.shape == (2, 32, 1000)

    def test_swa_layers_have_correct_pattern(self, swa_model):
        """Test that layers have correct attention pattern."""
        layers = swa_model.model.layers

        # Layer 0: global (first layer)
        assert layers[0].self_attn.use_sliding_window == False

        # Layer 1: local
        assert layers[1].self_attn.use_sliding_window == True

        # Layer 2: global (every 2nd)
        assert layers[2].self_attn.use_sliding_window == False

        # Layer 3: local
        assert layers[3].self_attn.use_sliding_window == True

    def test_swa_model_generation(self, swa_model):
        """Test generation with sliding window."""
        swa_model.eval()
        input_ids = torch.randint(0, 1000, (1, 10))

        with torch.no_grad():
            outputs = swa_model.generate(
                input_ids,
                max_new_tokens=20,  # Generate enough to exceed window
                do_sample=False,
                eos_token_id=None,
                pad_token_id=0,
            )

        assert outputs.shape[1] == 30  # 10 + 20

    def test_swa_gradient_flow(self, swa_model):
        """Test gradients flow through sliding window attention."""
        swa_model.train()
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = torch.randint(0, 1000, (2, 32))

        outputs = swa_model(input_ids, labels=labels)
        outputs.loss.backward()

        # Check that attention projections have gradients
        for layer in swa_model.model.layers:
            assert layer.self_attn.q_proj.weight.grad is not None

    def test_swa_config_preserved(self, swa_model, swa_config, tmp_path):
        """Test sliding window config is preserved on save/load."""
        save_path = tmp_path / "swa_model"
        swa_model.save_pretrained(save_path)

        loaded_model = UMichSLMForCausalLM.from_pretrained(save_path)

        assert loaded_model.config.sliding_window == swa_config.sliding_window
        assert loaded_model.config.global_attention_every_n_layers == swa_config.global_attention_every_n_layers


class TestSlidingWindowNoWindow:
    """Tests for model without sliding window (default behavior)."""

    @pytest.fixture
    def regular_config(self):
        return UMichSLMConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            # No sliding_window parameter
        )

    @pytest.fixture
    def regular_model(self, regular_config):
        return UMichSLMForCausalLM(regular_config)

    def test_no_sliding_window_by_default(self, regular_model):
        """Test that sliding window is disabled by default."""
        for layer in regular_model.model.layers:
            assert layer.self_attn.use_sliding_window == False
            assert layer.self_attn.sliding_window is None

    def test_regular_forward_works(self, regular_model):
        """Test that regular forward still works."""
        input_ids = torch.randint(0, 1000, (2, 32))
        outputs = regular_model(input_ids)
        assert outputs.logits.shape == (2, 32, 1000)
