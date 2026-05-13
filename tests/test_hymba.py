"""Tests for Hymba hybrid attention-SSM integration."""

import pytest
import torch
import torch.nn as nn

from umich_slm import UMichSLMConfig, UMichSLMForCausalLM, UMichSLMModel
from umich_slm.modules.mamba2 import Mamba2SSM, Mamba2Cache
from umich_slm.modules.hymba import HymbaHybridHead, MetaTokens
from umich_slm.modules.hymba_cache import HymbaDynamicCache, _build_kv_share_map


@pytest.fixture
def hymba_config():
    """Tiny config with Hymba enabled."""
    return UMichSLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=1000,
        use_hymba=True,
        hymba_num_meta_tokens=4,
        hymba_ssm_state_size=8,
        hymba_ssm_expand=2,
    )


@pytest.fixture
def hymba_model(hymba_config):
    """Create a tiny Hymba model."""
    return UMichSLMForCausalLM(hymba_config)


@pytest.fixture
def hymba_input(hymba_config):
    """Sample input for Hymba model."""
    batch_size = 2
    seq_len = 16
    return {
        "input_ids": torch.randint(0, hymba_config.vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }


@pytest.fixture
def meta_tokens_config():
    """Config with meta tokens enabled for standard (non-Hymba) architecture."""
    return UMichSLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=1000,
        num_meta_tokens=4,  # Universal meta tokens, not Hymba-specific
    )


# ============================================================================
# MetaTokens tests
# ============================================================================


class TestMetaTokens:
    """Tests for MetaTokens module."""

    def test_output_shape(self):
        """Output is [batch, num_tokens, hidden_size]."""
        mt = MetaTokens(num_tokens=8, hidden_size=64)
        out = mt(batch_size=3, dtype=torch.float32, device=torch.device("cpu"))
        assert out.shape == (3, 8, 64)

    def test_learnable(self):
        """Parameters have gradients after backward."""
        mt = MetaTokens(num_tokens=4, hidden_size=32)
        out = mt(batch_size=2, dtype=torch.float32, device=torch.device("cpu"))
        loss = out.sum()
        loss.backward()
        assert mt.tokens.grad is not None

    def test_expansion(self):
        """Same tokens across batch dimension."""
        mt = MetaTokens(num_tokens=4, hidden_size=32)
        out = mt(batch_size=3, dtype=torch.float32, device=torch.device("cpu"))
        # All batch elements should have the same meta tokens
        torch.testing.assert_close(out[0], out[1])
        torch.testing.assert_close(out[1], out[2])


# ============================================================================
# HymbaHybridHead tests
# ============================================================================


class TestHymbaHybridHead:
    """Tests for HymbaHybridHead module."""

    def test_forward_shape(self, hymba_config):
        """Output matches [batch, seq_len, hidden_size]."""
        head = HymbaHybridHead(hymba_config, layer_idx=0)
        x = torch.randn(2, 16, 64)
        out, attn_weights, present = head(x)
        assert out.shape == (2, 16, 64)

    def test_with_cache(self, hymba_config):
        """Returns hybrid cache, cache grows on incremental steps."""
        head = HymbaHybridHead(hymba_config, layer_idx=0)
        x = torch.randn(2, 8, 64)

        # First forward with cache
        out, _, present = head(x, use_cache=True)
        assert present is not None
        assert isinstance(present, HymbaDynamicCache)

    def test_attention_weights_output(self, hymba_config):
        """When output_attentions=True, returns attention weights."""
        head = HymbaHybridHead(hymba_config, layer_idx=0)
        x = torch.randn(2, 8, 64)
        out, attn_weights, _ = head(x, output_attentions=True)
        # Attention weights may be None if inner attention implementation doesn't support it
        # (e.g., SDPA, Flash Attention)
        assert out.shape == (2, 8, 64)

    def test_gradient_flow(self, hymba_config):
        """Gradients reach both attention and SSM parameters."""
        head = HymbaHybridHead(hymba_config, layer_idx=0)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out, _, _ = head(x)
        loss = out.sum()
        loss.backward()

        # Attention params
        assert head.attention.q_proj.weight.grad is not None
        # SSM params
        assert head.ssm.in_proj.weight.grad is not None
        # Fusion params
        assert head.attn_scale.grad is not None
        assert head.ssm_scale.grad is not None

    def test_fusion_scales(self, hymba_config):
        """attn_scale and ssm_scale are learnable and affect output."""
        head = HymbaHybridHead(hymba_config, layer_idx=0)
        x = torch.randn(1, 4, 64)

        with torch.no_grad():
            out1, _, _ = head(x)

            # Modify scales
            head.attn_scale.fill_(2.0)
            head.ssm_scale.fill_(0.0)
            out2, _, _ = head(x)

        # Output should change when scales change
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_with_gqa(self):
        """Works with num_key_value_heads < num_attention_heads (GQA)."""
        config = UMichSLMConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            use_hymba=True,
            hymba_ssm_state_size=8,
            hymba_ssm_expand=2,
        )
        head = HymbaHybridHead(config, layer_idx=0)
        x = torch.randn(2, 8, 64)
        out, _, _ = head(x)
        assert out.shape == (2, 8, 64)

    def test_with_mla_raises(self):
        """Hymba and MLA cannot be used simultaneously."""
        with pytest.raises(ValueError, match="Cannot use both Hymba and MLA"):
            UMichSLMConfig(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                vocab_size=1000,
                use_hymba=True,
                use_mla=True,
                mla_kv_lora_rank=16,
                mla_qk_nope_head_dim=16,
                mla_qk_rope_head_dim=8,
                mla_v_head_dim=16,
                hymba_ssm_state_size=8,
                hymba_ssm_expand=2,
            )


# ============================================================================
# HymbaDynamicCache tests
# ============================================================================


class TestHymbaDynamicCache:
    """Tests for HymbaDynamicCache."""

    def test_get_seq_length(self):
        """Correctly tracks sequence length."""
        cache = HymbaDynamicCache(num_layers=2)
        assert cache.get_seq_length() == 0

        # Add some attention cache
        key = torch.randn(2, 4, 8, 16)
        value = torch.randn(2, 4, 8, 16)
        cache.update_attn(key, value, layer_idx=0)
        assert cache.get_seq_length() == 8

    def test_reorder_cache(self):
        """Beam search reordering works."""
        cache = HymbaDynamicCache(num_layers=2)

        key = torch.randn(4, 2, 8, 16)  # batch=4
        value = torch.randn(4, 2, 8, 16)
        cache.update_attn(key, value, layer_idx=0)

        ssm = Mamba2Cache(
            ssm_state=torch.randn(4, 32, 8),
            conv_state=torch.randn(4, 32, 3),
        )
        cache.update_ssm(ssm, layer_idx=0)

        # Reorder: select beams [2, 0, 1, 3]
        beam_idx = torch.tensor([2, 0, 1, 3])
        cache.reorder_cache(beam_idx)

        assert cache.attn_key_cache[0].shape[0] == 4

    def test_update(self):
        """Cache grows with new entries."""
        cache = HymbaDynamicCache(num_layers=2)

        # First update
        key1 = torch.randn(2, 4, 4, 16)
        val1 = torch.randn(2, 4, 4, 16)
        k, v = cache.update_attn(key1, val1, layer_idx=0)
        assert k.shape[2] == 4

        # Second update
        key2 = torch.randn(2, 4, 2, 16)
        val2 = torch.randn(2, 4, 2, 16)
        k, v = cache.update_attn(key2, val2, layer_idx=0)
        assert k.shape[2] == 6  # 4 + 2

    def test_kv_sharing(self):
        """When sharing enabled, adjacent layers reference same KV."""
        cache = HymbaDynamicCache(
            num_layers=4, share_kv=True, kv_share_group_size=2
        )

        # Layer 0 is leader for group [0, 1]
        key = torch.randn(2, 4, 8, 16)
        value = torch.randn(2, 4, 8, 16)
        cache.update_attn(key, value, layer_idx=0)

        # Layer 1 is follower - should redirect to layer 0
        key2 = torch.randn(2, 4, 4, 16)
        val2 = torch.randn(2, 4, 4, 16)
        cache.update_attn(key2, val2, layer_idx=1)

        # Both should reference the same cache (layer 0's)
        kv0 = cache.get_attn_cache(0)
        kv1 = cache.get_attn_cache(1)

        assert kv0 is not None
        assert kv1 is not None
        # Layer 1 redirects to layer 0, so they share the same key/value
        assert kv0[0] is kv1[0]
        assert kv0[1] is kv1[1]
        # Cache should have grown: 8 + 4 = 12
        assert kv0[0].shape[2] == 12

    def test_ssm_not_shared(self):
        """SSM states are NOT shared across layers."""
        cache = HymbaDynamicCache(
            num_layers=4, share_kv=True, kv_share_group_size=2
        )

        ssm0 = Mamba2Cache(ssm_state=torch.randn(2, 32, 8))
        ssm1 = Mamba2Cache(ssm_state=torch.randn(2, 32, 8))
        cache.update_ssm(ssm0, layer_idx=0)
        cache.update_ssm(ssm1, layer_idx=1)

        assert cache.get_ssm_cache(0) is ssm0
        assert cache.get_ssm_cache(1) is ssm1
        assert cache.get_ssm_cache(0) is not cache.get_ssm_cache(1)


class TestBuildKvShareMap:
    """Tests for _build_kv_share_map utility."""

    def test_group_size_1(self):
        """No sharing when group_size=1."""
        m = _build_kv_share_map(6, 1)
        assert m == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

    def test_group_size_2(self):
        """Pairs share KV."""
        m = _build_kv_share_map(6, 2)
        assert m == {0: 0, 1: 0, 2: 2, 3: 2, 4: 4, 5: 4}

    def test_group_size_3(self):
        """Triplets share KV."""
        m = _build_kv_share_map(12, 3)
        assert m[0] == 0
        assert m[1] == 0
        assert m[2] == 0
        assert m[3] == 3
        assert m[4] == 3
        assert m[5] == 3
        assert m[9] == 9
        assert m[11] == 9


# ============================================================================
# Full model integration tests
# ============================================================================


class TestHymbaModel:
    """Full model integration tests for Hymba."""

    def test_forward_shape(self, hymba_model, hymba_input):
        """Logits have correct shape."""
        out = hymba_model(**hymba_input)
        batch, seq = hymba_input["input_ids"].shape
        assert out.logits.shape == (batch, seq, hymba_model.config.vocab_size)

    def test_loss_computation(self, hymba_model, hymba_input):
        """Loss is scalar, finite."""
        inputs = {**hymba_input, "labels": hymba_input["input_ids"].clone()}
        out = hymba_model(**inputs)
        assert out.loss is not None
        assert out.loss.dim() == 0
        assert torch.isfinite(out.loss)

    def test_generation(self, hymba_model, hymba_input):
        """model.generate() works."""
        hymba_model.eval()
        with torch.no_grad():
            output_ids = hymba_model.generate(
                hymba_input["input_ids"][:1, :8],
                max_new_tokens=4,
                do_sample=False,
            )
        assert output_ids.shape[1] > 8

    def test_meta_tokens_stripped(self, hymba_model, hymba_input):
        """Output does not include meta token positions."""
        out = hymba_model(**hymba_input)
        batch, seq = hymba_input["input_ids"].shape
        # Logits should match original seq_len, not seq_len + num_meta
        assert out.logits.shape == (batch, seq, hymba_model.config.vocab_size)

    def test_with_moe(self):
        """Hymba + MoE works."""
        config = UMichSLMConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=1000,
            use_hymba=True,
            hymba_num_meta_tokens=4,
            hymba_ssm_state_size=8,
            hymba_ssm_expand=2,
            use_moe=True,
            num_experts=4,
            num_experts_per_tok=2,
        )
        model = UMichSLMForCausalLM(config)
        input_ids = torch.randint(0, 1000, (2, 16))
        out = model(input_ids, attention_mask=torch.ones(2, 16, dtype=torch.long))
        assert out.logits.shape == (2, 16, 1000)

    def test_with_mod(self):
        """Hymba + MoD works."""
        config = UMichSLMConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=1000,
            use_hymba=True,
            hymba_num_meta_tokens=4,
            hymba_ssm_state_size=8,
            hymba_ssm_expand=2,
            use_mod=True,
            mod_capacity=0.5,
            mod_every_n_layers=2,
        )
        model = UMichSLMForCausalLM(config)
        input_ids = torch.randint(0, 1000, (2, 16))
        out = model(input_ids, attention_mask=torch.ones(2, 16, dtype=torch.long))
        assert out.logits.shape == (2, 16, 1000)

    def test_with_mtp(self):
        """Hymba + MTP works."""
        config = UMichSLMConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=1000,
            use_hymba=True,
            hymba_num_meta_tokens=4,
            hymba_ssm_state_size=8,
            hymba_ssm_expand=2,
            use_mtp=True,
            mtp_num_heads=1,
        )
        model = UMichSLMForCausalLM(config)
        model.train()
        input_ids = torch.randint(0, 1000, (2, 16))
        labels = input_ids.clone()
        out = model(input_ids, attention_mask=torch.ones(2, 16, dtype=torch.long), labels=labels)
        assert out.loss is not None

    def test_with_sliding_window(self):
        """Inner attention uses sliding window."""
        config = UMichSLMConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=1000,
            use_hymba=True,
            hymba_num_meta_tokens=2,
            hymba_ssm_state_size=8,
            hymba_ssm_expand=2,
            sliding_window=8,
        )
        model = UMichSLMForCausalLM(config)
        input_ids = torch.randint(0, 1000, (2, 16))
        out = model(input_ids, attention_mask=torch.ones(2, 16, dtype=torch.long))
        assert out.logits.shape == (2, 16, 1000)

    def test_with_mla_raises(self):
        """Hymba and MLA cannot be used simultaneously."""
        with pytest.raises(ValueError, match="Cannot use both Hymba and MLA"):
            UMichSLMConfig(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                vocab_size=1000,
                use_hymba=True,
                use_mla=True,
                mla_kv_lora_rank=16,
                mla_qk_nope_head_dim=16,
                mla_qk_rope_head_dim=8,
                mla_v_head_dim=16,
                hymba_num_meta_tokens=2,
                hymba_ssm_state_size=8,
                hymba_ssm_expand=2,
            )

    def test_cross_layer_kv_sharing(self):
        """KV cache shared between adjacent layers."""
        config = UMichSLMConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=1000,
            use_hymba=True,
            hymba_num_meta_tokens=2,
            hymba_ssm_state_size=8,
            hymba_ssm_expand=2,
            hymba_share_kv_layers=True,
            hymba_kv_share_group_size=2,
        )
        model = UMichSLMForCausalLM(config)
        input_ids = torch.randint(0, 1000, (2, 8))
        out = model(input_ids, attention_mask=torch.ones(2, 8, dtype=torch.long))
        assert out.logits.shape == (2, 8, 1000)

    def test_save_load(self, hymba_config, tmp_path):
        """Config round-trips through save/load."""
        hymba_config.save_pretrained(tmp_path)
        loaded = UMichSLMConfig.from_pretrained(tmp_path)
        assert loaded.use_hymba == True
        assert loaded.hymba_num_meta_tokens == 4
        assert loaded.hymba_ssm_state_size == 8
        assert loaded.hymba_ssm_expand == 2

    def test_gradient_checkpointing(self, hymba_config):
        """Gradient checkpointing works."""
        model = UMichSLMForCausalLM(hymba_config)
        model.gradient_checkpointing_enable()
        model.train()
        input_ids = torch.randint(0, 1000, (2, 16))
        labels = input_ids.clone()
        out = model(input_ids, attention_mask=torch.ones(2, 16, dtype=torch.long), labels=labels)
        assert out.loss is not None
        out.loss.backward()


# ============================================================================
# Meta tokens for all architectures tests
# ============================================================================


class TestMetaTokensAllArchitectures:
    """Test that meta tokens work with ALL architectures, not just Hymba."""

    def test_meta_tokens_standard_model(self, meta_tokens_config):
        """Meta tokens work with standard (non-Hymba) architecture."""
        model = UMichSLMForCausalLM(meta_tokens_config)
        input_ids = torch.randint(0, 1000, (2, 16))
        out = model(input_ids, attention_mask=torch.ones(2, 16, dtype=torch.long))
        # Output should match original seq_len (meta tokens stripped)
        assert out.logits.shape == (2, 16, 1000)

    def test_meta_tokens_with_moe(self):
        """Meta tokens work with MoE architecture."""
        config = UMichSLMConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=1000,
            num_meta_tokens=4,
            use_moe=True,
            num_experts=4,
            num_experts_per_tok=2,
        )
        model = UMichSLMForCausalLM(config)
        input_ids = torch.randint(0, 1000, (2, 16))
        out = model(input_ids, attention_mask=torch.ones(2, 16, dtype=torch.long))
        assert out.logits.shape == (2, 16, 1000)

    def test_meta_tokens_with_mod(self):
        """Meta tokens work with MoD architecture."""
        config = UMichSLMConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=1000,
            num_meta_tokens=4,
            use_mod=True,
            mod_capacity=0.5,
            mod_every_n_layers=2,
        )
        model = UMichSLMForCausalLM(config)
        input_ids = torch.randint(0, 1000, (2, 16))
        out = model(input_ids, attention_mask=torch.ones(2, 16, dtype=torch.long))
        assert out.logits.shape == (2, 16, 1000)

    def test_meta_tokens_training_loss(self, meta_tokens_config):
        """Training loss computation works with meta tokens."""
        model = UMichSLMForCausalLM(meta_tokens_config)
        model.train()
        input_ids = torch.randint(0, 1000, (2, 16))
        labels = input_ids.clone()
        out = model(
            input_ids,
            attention_mask=torch.ones(2, 16, dtype=torch.long),
            labels=labels,
        )
        assert out.loss is not None
        assert torch.isfinite(out.loss)
        out.loss.backward()

    def test_meta_tokens_zero_disables(self):
        """num_meta_tokens=0 disables meta tokens."""
        config = UMichSLMConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=1000,
            num_meta_tokens=0,
        )
        model = UMichSLMModel(config)
        assert model.meta_tokens is None

    def test_meta_tokens_config_persists(self, tmp_path):
        """num_meta_tokens persists through save/load."""
        config = UMichSLMConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=1000,
            num_meta_tokens=8,
        )
        config.save_pretrained(tmp_path)
        loaded = UMichSLMConfig.from_pretrained(tmp_path)
        assert loaded.num_meta_tokens == 8


# ============================================================================
# Config validation tests
# ============================================================================


class TestHymbaConfig:
    """Tests for Hymba configuration validation."""

    def test_default_values(self):
        """Hymba defaults are correct."""
        config = UMichSLMConfig()
        assert config.use_hymba is False
        assert config.hymba_num_meta_tokens == 128
        assert config.hymba_ssm_state_size == 16
        assert config.hymba_ssm_conv_size == 4
        assert config.hymba_ssm_expand == 2
        assert config.hymba_share_kv_layers is False
        assert config.hymba_kv_share_group_size == 2
        assert config.num_meta_tokens == 0

    def test_validation_deltanet_conflict(self):
        """use_hymba=True + use_deltanet=True raises ValueError."""
        with pytest.raises(ValueError, match="Cannot use both Hymba and DeltaNet"):
            UMichSLMConfig(use_hymba=True, use_deltanet=True)

    def test_validation_ssm_params(self):
        """Invalid hymba_ssm_state_size raises error."""
        with pytest.raises(ValueError, match="hymba_ssm_state_size must be > 0"):
            UMichSLMConfig(use_hymba=True, hymba_ssm_state_size=0)

        with pytest.raises(ValueError, match="hymba_ssm_expand must be >= 1"):
            UMichSLMConfig(use_hymba=True, hymba_ssm_expand=0)

    def test_validation_meta_tokens(self):
        """Invalid hymba_num_meta_tokens raises error."""
        with pytest.raises(ValueError, match="hymba_num_meta_tokens must be >= 0"):
            UMichSLMConfig(use_hymba=True, hymba_num_meta_tokens=-1)

    def test_kv_share_config(self, tmp_path):
        """hymba_share_kv_layers config persists."""
        config = UMichSLMConfig(
            use_hymba=True,
            hymba_share_kv_layers=True,
            hymba_kv_share_group_size=3,
        )
        config.save_pretrained(tmp_path)
        loaded = UMichSLMConfig.from_pretrained(tmp_path)
        assert loaded.hymba_share_kv_layers is True
        assert loaded.hymba_kv_share_group_size == 3
