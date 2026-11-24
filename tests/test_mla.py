"""Tests for Multi-head Latent Attention (MLA) implementation."""

import pytest
import torch

from umich_slm import UMichSLMConfig, UMichSLMModel, UMichSLMForCausalLM
from umich_slm.modules.mla import (
    UMichSLMMLAAttention,
    UMichSLMMLASdpaAttention,
    get_mla_attention_class,
)


class TestMLAAttention:
    """Tests for the MLA attention module."""

    @pytest.fixture
    def mla_config(self):
        """Create a config with MLA enabled."""
        return UMichSLMConfig(
            vocab_size=100,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=64,
            use_mla=True,
            mla_kv_lora_rank=32,  # Compress KV to 32-dim
            mla_q_lora_rank=0,  # No Q compression
            mla_qk_nope_head_dim=16,
            mla_qk_rope_head_dim=8,
            mla_v_head_dim=16,
        )

    @pytest.fixture
    def mla_config_q_compressed(self):
        """Create a config with MLA enabled and Q compression."""
        return UMichSLMConfig(
            vocab_size=100,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=64,
            use_mla=True,
            mla_kv_lora_rank=32,
            mla_q_lora_rank=48,  # Compress Q to 48-dim
            mla_qk_nope_head_dim=16,
            mla_qk_rope_head_dim=8,
            mla_v_head_dim=16,
        )

    @pytest.fixture
    def mla_config_with_swa(self):
        """Create a config with MLA and sliding window attention."""
        return UMichSLMConfig(
            vocab_size=100,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            max_position_embeddings=64,
            use_mla=True,
            mla_kv_lora_rank=32,
            mla_q_lora_rank=0,
            mla_qk_nope_head_dim=16,
            mla_qk_rope_head_dim=8,
            mla_v_head_dim=16,
            sliding_window=16,
            global_attention_every_n_layers=2,
        )

    def test_mla_output_shape(self, mla_config):
        """MLA output should have correct shape."""
        attn = UMichSLMMLAAttention(mla_config, layer_idx=0)
        hidden_states = torch.randn(2, 16, mla_config.hidden_size)

        output, attn_weights, cache = attn(hidden_states)

        assert output.shape == hidden_states.shape

    def test_mla_with_q_compression(self, mla_config_q_compressed):
        """MLA with Q compression should work correctly."""
        attn = UMichSLMMLAAttention(mla_config_q_compressed, layer_idx=0)
        hidden_states = torch.randn(2, 16, mla_config_q_compressed.hidden_size)

        output, attn_weights, cache = attn(hidden_states)

        assert output.shape == hidden_states.shape

    def test_mla_cache_format(self, mla_config):
        """MLA cache should contain (c_kv, k_rope)."""
        attn = UMichSLMMLAAttention(mla_config, layer_idx=0)
        hidden_states = torch.randn(2, 16, mla_config.hidden_size)

        output, attn_weights, cache = attn(hidden_states, use_cache=True)

        assert cache is not None
        assert isinstance(cache, tuple)
        assert len(cache) == 2

        c_kv, k_rope = cache
        # c_kv: [batch, seq, kv_lora_rank]
        assert c_kv.shape == (2, 16, mla_config.mla_kv_lora_rank)
        # k_rope: [batch, seq, num_heads, qk_rope_head_dim]
        assert k_rope.shape == (
            2,
            16,
            mla_config.num_attention_heads,
            mla_config.mla_qk_rope_head_dim,
        )

    def test_mla_cache_is_smaller(self, mla_config):
        """MLA cache should be smaller than standard attention cache."""
        attn = UMichSLMMLAAttention(mla_config, layer_idx=0)

        mla_cache_size = attn.get_cache_size_per_token()
        standard_cache_size = attn.get_standard_cache_size_per_token()

        # MLA cache should be smaller
        assert mla_cache_size < standard_cache_size

        # Verify the expected values
        # MLA: kv_lora_rank + num_heads * qk_rope_head_dim
        expected_mla = mla_config.mla_kv_lora_rank + (
            mla_config.num_attention_heads * mla_config.mla_qk_rope_head_dim
        )
        assert mla_cache_size == expected_mla

    def test_mla_incremental_decoding(self, mla_config):
        """MLA should support incremental decoding with cache."""
        attn = UMichSLMMLAAttention(mla_config, layer_idx=0)

        # Initial forward pass
        hidden_states = torch.randn(2, 10, mla_config.hidden_size)
        output1, _, cache = attn(hidden_states, use_cache=True)

        # Incremental decode step
        new_hidden = torch.randn(2, 1, mla_config.hidden_size)
        output2, _, cache2 = attn(new_hidden, past_key_value=cache, use_cache=True)

        assert output2.shape == (2, 1, mla_config.hidden_size)

        # Cache should have grown
        c_kv, k_rope = cache2
        assert c_kv.shape[1] == 11  # 10 + 1

    def test_mla_attention_weights(self, mla_config):
        """MLA should return attention weights when requested."""
        attn = UMichSLMMLAAttention(mla_config, layer_idx=0)
        hidden_states = torch.randn(2, 16, mla_config.hidden_size)

        output, attn_weights, cache = attn(hidden_states, output_attentions=True)

        assert attn_weights is not None
        # Attention weights: [batch, num_heads, q_len, kv_len]
        assert attn_weights.shape == (2, mla_config.num_attention_heads, 16, 16)

    def test_mla_with_sliding_window(self, mla_config_with_swa):
        """MLA should work with sliding window attention."""
        # Layer 0 and 2 are global (0 % 2 == 0, 2 % 2 == 0)
        attn_global_0 = UMichSLMMLAAttention(mla_config_with_swa, layer_idx=0)
        assert attn_global_0.use_sliding_window is False

        attn_global_2 = UMichSLMMLAAttention(mla_config_with_swa, layer_idx=2)
        assert attn_global_2.use_sliding_window is False

        # Layer 1 and 3 should use sliding window (1 % 2 != 0, 3 % 2 != 0)
        attn_swa_1 = UMichSLMMLAAttention(mla_config_with_swa, layer_idx=1)
        assert attn_swa_1.use_sliding_window is True

        attn_swa_3 = UMichSLMMLAAttention(mla_config_with_swa, layer_idx=3)
        assert attn_swa_3.use_sliding_window is True

        # Forward pass should work for both types
        hidden_states = torch.randn(2, 32, mla_config_with_swa.hidden_size)
        output_global, _, _ = attn_global_0(hidden_states)
        output_swa, _, _ = attn_swa_1(hidden_states)
        assert output_global.shape == hidden_states.shape
        assert output_swa.shape == hidden_states.shape

    def test_mla_gradient_flow(self, mla_config):
        """Gradients should flow through MLA."""
        attn = UMichSLMMLAAttention(mla_config, layer_idx=0)
        hidden_states = torch.randn(2, 16, mla_config.hidden_size, requires_grad=True)

        output, _, _ = attn(hidden_states)
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert not torch.isnan(hidden_states.grad).any()

    def test_mla_sdpa_variant(self, mla_config):
        """SDPA variant of MLA should work correctly."""
        attn = UMichSLMMLASdpaAttention(mla_config, layer_idx=0)
        hidden_states = torch.randn(2, 16, mla_config.hidden_size)

        output, _, cache = attn(hidden_states, use_cache=True)

        assert output.shape == hidden_states.shape
        assert cache is not None

    def test_get_mla_attention_class(self):
        """get_mla_attention_class should return correct classes."""
        eager_cls = get_mla_attention_class("eager")
        assert eager_cls is UMichSLMMLAAttention

        sdpa_cls = get_mla_attention_class("sdpa")
        assert sdpa_cls is UMichSLMMLASdpaAttention

        with pytest.raises(ValueError):
            get_mla_attention_class("invalid")


class TestMLAModel:
    """Tests for full model with MLA."""

    @pytest.fixture
    def mla_config(self):
        """Create a config with MLA enabled."""
        return UMichSLMConfig(
            vocab_size=100,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=64,
            use_mla=True,
            mla_kv_lora_rank=32,
            mla_q_lora_rank=0,
            mla_qk_nope_head_dim=16,
            mla_qk_rope_head_dim=8,
            mla_v_head_dim=16,
        )

    @pytest.fixture
    def mla_swa_config(self):
        """Create a config with MLA and sliding window attention."""
        return UMichSLMConfig(
            vocab_size=100,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            max_position_embeddings=64,
            use_mla=True,
            mla_kv_lora_rank=32,
            mla_q_lora_rank=0,
            mla_qk_nope_head_dim=16,
            mla_qk_rope_head_dim=8,
            mla_v_head_dim=16,
            sliding_window=16,
            global_attention_every_n_layers=2,
        )

    def test_mla_model_forward(self, mla_config):
        """MLA model forward pass should work."""
        model = UMichSLMModel(mla_config)
        input_ids = torch.randint(0, mla_config.vocab_size, (2, 16))

        outputs = model(input_ids)

        assert outputs.last_hidden_state.shape == (2, 16, mla_config.hidden_size)

    def test_mla_causal_lm_forward(self, mla_config):
        """MLA causal LM forward pass should work."""
        model = UMichSLMForCausalLM(mla_config)
        input_ids = torch.randint(0, mla_config.vocab_size, (2, 16))

        outputs = model(input_ids)

        assert outputs.logits.shape == (2, 16, mla_config.vocab_size)

    def test_mla_model_with_cache(self, mla_config):
        """MLA model should work with KV cache."""
        from umich_slm.modules.kv_cache import MLADynamicCache

        model = UMichSLMModel(mla_config)
        input_ids = torch.randint(0, mla_config.vocab_size, (2, 16))

        outputs = model(input_ids, use_cache=True)

        assert outputs.past_key_values is not None
        # MLA now uses MLADynamicCache for proper HuggingFace integration
        assert isinstance(outputs.past_key_values, MLADynamicCache)
        assert len(outputs.past_key_values) == mla_config.num_hidden_layers

        # Each cache entry should be (c_kv, k_rope)
        for layer_idx in range(len(outputs.past_key_values)):
            layer_cache = outputs.past_key_values[layer_idx]
            assert isinstance(layer_cache, tuple)
            assert len(layer_cache) == 2

    def test_mla_generation(self, mla_config):
        """MLA model should support generation."""
        model = UMichSLMForCausalLM(mla_config)
        input_ids = torch.randint(0, 50, (1, 5))

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=3,
                do_sample=False,
            )

        assert outputs.shape[1] == 8  # 5 + 3

    def test_mla_with_swa_forward(self, mla_swa_config):
        """MLA with sliding window should work."""
        model = UMichSLMModel(mla_swa_config)
        input_ids = torch.randint(0, mla_swa_config.vocab_size, (2, 32))

        outputs = model(input_ids)

        assert outputs.last_hidden_state.shape == (2, 32, mla_swa_config.hidden_size)

    def test_mla_with_swa_generation(self, mla_swa_config):
        """MLA with sliding window should support generation."""
        model = UMichSLMForCausalLM(mla_swa_config)
        input_ids = torch.randint(0, 50, (1, 10))

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
            )

        assert outputs.shape[1] == 15  # 10 + 5

    def test_mla_loss_computation(self, mla_config):
        """MLA model should compute loss correctly."""
        model = UMichSLMForCausalLM(mla_config)
        input_ids = torch.randint(0, mla_config.vocab_size, (2, 16))
        labels = input_ids.clone()

        outputs = model(input_ids, labels=labels)

        assert outputs.loss is not None
        assert outputs.loss.shape == ()
        assert not torch.isnan(outputs.loss)

    def test_mla_gradient_flow_model(self, mla_config):
        """Gradients should flow through entire MLA model."""
        model = UMichSLMForCausalLM(mla_config)
        input_ids = torch.randint(0, mla_config.vocab_size, (2, 16))
        labels = input_ids.clone()

        outputs = model(input_ids, labels=labels)
        outputs.loss.backward()

        # Check gradients on key parameters
        assert model.model.embed_tokens.weight.grad is not None
        assert model.lm_head.weight.grad is not None

    def test_mla_config_preserved(self, mla_config, tmp_path):
        """MLA config should be preserved on save/load."""
        model = UMichSLMForCausalLM(mla_config)
        save_path = tmp_path / "mla_model"
        model.save_pretrained(save_path)

        loaded_model = UMichSLMForCausalLM.from_pretrained(save_path)

        assert loaded_model.config.use_mla == True
        assert loaded_model.config.mla_kv_lora_rank == mla_config.mla_kv_lora_rank
        assert loaded_model.config.mla_q_lora_rank == mla_config.mla_q_lora_rank
        assert loaded_model.config.mla_qk_nope_head_dim == mla_config.mla_qk_nope_head_dim
        assert loaded_model.config.mla_qk_rope_head_dim == mla_config.mla_qk_rope_head_dim
        assert loaded_model.config.mla_v_head_dim == mla_config.mla_v_head_dim

    def test_mla_uses_correct_attention(self, mla_config):
        """Model with use_mla=True should use MLA attention."""
        model = UMichSLMModel(mla_config)

        # Check that layers use MLA attention
        for layer in model.layers:
            assert isinstance(layer.self_attn, UMichSLMMLAAttention)


class TestMLAConfig:
    """Tests for MLA configuration validation."""

    def test_mla_default_values(self):
        """MLA config should have sensible defaults."""
        config = UMichSLMConfig(use_mla=True)

        assert config.mla_kv_lora_rank == 512
        assert config.mla_q_lora_rank == 0
        assert config.mla_qk_nope_head_dim == 128
        assert config.mla_qk_rope_head_dim == 64
        assert config.mla_v_head_dim == 128

    def test_mla_disabled_by_default(self):
        """MLA should be disabled by default."""
        config = UMichSLMConfig()
        assert config.use_mla == False

    def test_mla_custom_values(self):
        """Custom MLA values should be respected."""
        config = UMichSLMConfig(
            use_mla=True,
            mla_kv_lora_rank=256,
            mla_q_lora_rank=128,
            mla_qk_nope_head_dim=64,
            mla_qk_rope_head_dim=32,
            mla_v_head_dim=64,
        )

        assert config.mla_kv_lora_rank == 256
        assert config.mla_q_lora_rank == 128
        assert config.mla_qk_nope_head_dim == 64
        assert config.mla_qk_rope_head_dim == 32
        assert config.mla_v_head_dim == 64


class TestMLAWithOtherFeatures:
    """Tests for MLA combined with other features."""

    def test_mla_with_moe(self):
        """MLA should work with MoE."""
        config = UMichSLMConfig(
            vocab_size=100,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=64,
            use_mla=True,
            mla_kv_lora_rank=32,
            mla_qk_nope_head_dim=16,
            mla_qk_rope_head_dim=8,
            mla_v_head_dim=16,
            use_moe=True,
            num_experts=4,
            num_experts_per_tok=2,
        )

        model = UMichSLMForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))

        outputs = model(input_ids)
        assert outputs.logits.shape == (2, 16, config.vocab_size)

    def test_mla_with_zero_centered_rmsnorm(self):
        """MLA should work with zero-centered RMSNorm."""
        config = UMichSLMConfig(
            vocab_size=100,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=64,
            use_mla=True,
            mla_kv_lora_rank=32,
            mla_qk_nope_head_dim=16,
            mla_qk_rope_head_dim=8,
            mla_v_head_dim=16,
            use_zero_centered_rmsnorm=True,
        )

        model = UMichSLMForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))

        outputs = model(input_ids)
        assert outputs.logits.shape == (2, 16, config.vocab_size)

    def test_mla_with_mod(self):
        """MLA should work with Mixture of Depths."""
        config = UMichSLMConfig(
            vocab_size=100,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            max_position_embeddings=64,
            use_mla=True,
            mla_kv_lora_rank=32,
            mla_qk_nope_head_dim=16,
            mla_qk_rope_head_dim=8,
            mla_v_head_dim=16,
            use_mod=True,
            mod_capacity=0.5,
            mod_every_n_layers=2,
        )

        model = UMichSLMForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))

        outputs = model(input_ids)
        assert outputs.logits.shape == (2, 16, config.vocab_size)

    def test_mla_with_parallel_mlp(self):
        """MLA should work with parallel MLP."""
        config = UMichSLMConfig(
            vocab_size=100,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=64,
            use_mla=True,
            mla_kv_lora_rank=32,
            mla_qk_nope_head_dim=16,
            mla_qk_rope_head_dim=8,
            mla_v_head_dim=16,
            parallel_mlp=True,
        )

        model = UMichSLMForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))

        outputs = model(input_ids)
        assert outputs.logits.shape == (2, 16, config.vocab_size)
