"""Tests for Engram (Hash-Based Conditional Memory) module."""

import pytest
import torch

from umich_slm import UMichSLMConfig, UMichSLMForCausalLM
from umich_slm.modules.engram import EngramHash, EngramMemory, EngramGate, EngramModule


# ---- EngramHash tests ----


class TestEngramHash:
    def test_output_shape(self):
        """EngramHash produces correct output shape."""
        ngram_sizes = [2, 3]
        num_heads = 4
        hasher = EngramHash(ngram_sizes=ngram_sizes, num_heads=num_heads, table_size=49999)
        input_ids = torch.randint(0, 1000, (2, 16))
        out = hasher(input_ids)
        # total_heads = num_heads * len(ngram_sizes) = 4 * 2 = 8
        assert out.shape == (2, 16, 8)

    def test_deterministic(self):
        """EngramHash produces deterministic outputs for same input."""
        hasher = EngramHash(ngram_sizes=[2, 3], num_heads=4, table_size=49999, seed=42)
        input_ids = torch.randint(0, 1000, (2, 16))
        out1 = hasher(input_ids)
        out2 = hasher(input_ids)
        assert torch.equal(out1, out2)

    def test_different_seeds_different_hashes(self):
        """Different seeds produce different hash multipliers."""
        hasher1 = EngramHash(ngram_sizes=[2], num_heads=2, table_size=49999, seed=42)
        hasher2 = EngramHash(ngram_sizes=[2], num_heads=2, table_size=49999, seed=123)
        input_ids = torch.randint(0, 1000, (1, 32))
        out1 = hasher1(input_ids)
        out2 = hasher2(input_ids)
        # Very unlikely to be identical with different seeds
        assert not torch.equal(out1, out2)

    def test_values_in_range(self):
        """Hash values are in [0, table_size)."""
        table_size = 997
        hasher = EngramHash(ngram_sizes=[2, 3], num_heads=4, table_size=table_size)
        input_ids = torch.randint(0, 50000, (4, 64))
        out = hasher(input_ids)
        assert (out >= 0).all()
        assert (out < table_size).all()

    def test_handles_short_sequence(self):
        """EngramHash works with sequences shorter than n-gram size."""
        hasher = EngramHash(ngram_sizes=[2, 3], num_heads=2, table_size=49999)
        input_ids = torch.randint(0, 1000, (1, 1))  # seq_len=1
        out = hasher(input_ids)
        assert out.shape == (1, 1, 4)  # 2 heads * 2 ngram_sizes


# ---- EngramMemory tests ----


class TestEngramMemory:
    def test_output_shape(self):
        """EngramMemory produces correct output shape."""
        num_tables = 8
        hidden_size = 64
        memory = EngramMemory(
            num_tables=num_tables, table_size=49999, embed_dim=32, hidden_size=hidden_size
        )
        hash_indices = torch.randint(0, 49999, (2, 16, num_tables))
        out = memory(hash_indices)
        assert out.shape == (2, 16, hidden_size)

    def test_different_indices_different_outputs(self):
        """Different hash indices produce different embeddings."""
        memory = EngramMemory(num_tables=4, table_size=997, embed_dim=16, hidden_size=32)
        idx1 = torch.randint(0, 997, (1, 8, 4))
        idx2 = torch.randint(0, 997, (1, 8, 4))
        out1 = memory(idx1)
        out2 = memory(idx2)
        # Should not be identical (extremely unlikely)
        assert not torch.equal(out1, out2)


# ---- EngramGate tests ----


class TestEngramGate:
    def test_output_shape(self):
        """EngramGate produces correct output shape."""
        hidden_size = 64
        gate = EngramGate(hidden_size=hidden_size)
        h = torch.randn(2, 16, hidden_size)
        m = torch.randn(2, 16, hidden_size)
        out = gate(h, m)
        assert out.shape == (2, 16, hidden_size)

    def test_gate_values_bounded(self):
        """Gate sigmoid values are in [0, 1], so output magnitude <= input."""
        hidden_size = 64
        gate = EngramGate(hidden_size=hidden_size)
        h = torch.randn(2, 16, hidden_size)
        m = torch.ones(2, 16, hidden_size)  # Unit vectors for easy checking
        out = gate(h, m)
        # Output = sigmoid(...) * m, so each element should be in [-1, 1] since m=1
        # Actually gate is in [0,1] and m=1, so output should be in [0, 1]
        assert (out >= -0.01).all()  # Small tolerance for numerical precision
        assert (out <= 1.01).all()


# ---- EngramModule tests ----


class TestEngramModule:
    @pytest.fixture
    def engram_config(self):
        return UMichSLMConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=172,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            use_engram=True,
            engram_layer_ids=[0, 2],
            engram_ngram_sizes=[2, 3],
            engram_num_heads=4,
            engram_embed_dim=16,
            engram_table_size=997,
        )

    def test_module_output_shape(self, engram_config):
        """EngramModule produces correct output shape."""
        mod = EngramModule(engram_config, layer_idx=0)
        h = torch.randn(2, 16, 64)
        ids = torch.randint(0, 1000, (2, 16))
        out = mod(h, ids)
        assert out.shape == (2, 16, 64)

    def test_token_buffer_accumulation(self, engram_config):
        """Token buffer correctly accumulates across generation steps."""
        mod = EngramModule(engram_config, layer_idx=0)
        mod.reset_token_buffer()

        # Prefill
        ids1 = torch.randint(0, 1000, (1, 10))
        mod.update_token_buffer(ids1)
        assert mod._token_buffer.shape == (1, 10)

        # Generation step
        ids2 = torch.randint(0, 1000, (1, 1))
        mod.update_token_buffer(ids2)
        assert mod._token_buffer.shape == (1, 11)

        # Reset
        mod.reset_token_buffer()
        assert mod._token_buffer is None

    def test_with_conv(self):
        """EngramModule works with optional convolution."""
        config = UMichSLMConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=172,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            use_engram=True,
            engram_layer_ids=[0],
            engram_num_heads=4,
            engram_embed_dim=16,
            engram_table_size=997,
            engram_use_conv=True,
            engram_conv_kernel_size=3,
        )
        mod = EngramModule(config, layer_idx=0)
        assert mod.use_conv
        h = torch.randn(2, 16, 64)
        ids = torch.randint(0, 1000, (2, 16))
        out = mod(h, ids)
        assert out.shape == (2, 16, 64)


# ---- Full model integration tests ----


class TestEngramIntegration:
    @pytest.fixture
    def engram_config(self):
        return UMichSLMConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=172,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            use_engram=True,
            engram_layer_ids=[0, 1],
            engram_ngram_sizes=[2, 3],
            engram_num_heads=4,
            engram_embed_dim=16,
            engram_table_size=997,
        )

    def test_model_forward(self, engram_config):
        """Model with engram produces valid output."""
        model = UMichSLMForCausalLM(engram_config)
        model.eval()
        input_ids = torch.randint(0, 1000, (1, 32))
        with torch.no_grad():
            out = model(input_ids)
        assert out.logits.shape == (1, 32, 1000)

    def test_model_training_loss(self, engram_config):
        """Model with engram produces loss and gradients flow through engram parameters."""
        model = UMichSLMForCausalLM(engram_config)
        model.train()
        input_ids = torch.randint(0, 1000, (2, 16))
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        assert out.loss is not None
        assert out.loss.requires_grad

        # Check gradients flow through engram parameters
        out.loss.backward()
        engram_mod = model.model.engram_modules["0"]
        # Check memory proj has gradient
        assert engram_mod.memory.proj.weight.grad is not None
        assert engram_mod.memory.proj.weight.grad.abs().sum() > 0
        # Check gate params have gradient
        assert engram_mod.gate.q_weight.grad is not None
        assert engram_mod.gate.m_weight.grad is not None

    def test_engram_modules_created(self, engram_config):
        """Engram modules are created for specified layers."""
        model = UMichSLMForCausalLM(engram_config)
        assert "0" in model.model.engram_modules
        assert "1" in model.model.engram_modules
        assert len(model.model.engram_modules) == 2

    def test_no_engram_when_disabled(self):
        """No engram modules when use_engram=False."""
        config = UMichSLMConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=172,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            use_engram=False,
        )
        model = UMichSLMForCausalLM(config)
        assert len(model.model.engram_modules) == 0

    def test_engram_subset_of_layers(self):
        """Engram only applied to specified layer subset."""
        config = UMichSLMConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=172,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            use_engram=True,
            engram_layer_ids=[1, 3],
            engram_num_heads=2,
            engram_embed_dim=8,
            engram_table_size=997,
        )
        model = UMichSLMForCausalLM(config)
        assert "0" not in model.model.engram_modules
        assert "1" in model.model.engram_modules
        assert "2" not in model.model.engram_modules
        assert "3" in model.model.engram_modules


# ---- Config validation tests ----


class TestEngramConfigValidation:
    def test_invalid_layer_ids(self):
        """Invalid layer IDs raise ValueError."""
        with pytest.raises(ValueError, match="engram_layer_ids"):
            UMichSLMConfig(
                num_hidden_layers=4,
                use_engram=True,
                engram_layer_ids=[0, 5],  # 5 >= 4
            )

    def test_invalid_ngram_sizes(self):
        """N-gram sizes < 2 raise ValueError."""
        with pytest.raises(ValueError, match="engram_ngram_sizes"):
            UMichSLMConfig(
                num_hidden_layers=4,
                use_engram=True,
                engram_layer_ids=[0],
                engram_ngram_sizes=[1, 3],  # 1 < 2
            )

    def test_valid_config(self):
        """Valid engram config does not raise."""
        config = UMichSLMConfig(
            num_hidden_layers=6,
            use_engram=True,
            engram_layer_ids=[2, 5],
            engram_ngram_sizes=[2, 3, 4],
            engram_num_heads=8,
        )
        assert config.use_engram
        assert config.engram_layer_ids == [2, 5]

    def test_empty_layer_ids(self):
        """Empty layer_ids is valid (just no engram applied)."""
        config = UMichSLMConfig(
            num_hidden_layers=4,
            use_engram=True,
            engram_layer_ids=[],
        )
        assert config.use_engram
        assert config.engram_layer_ids == []

    def test_negative_layer_id(self):
        """Negative layer ID raises ValueError."""
        with pytest.raises(ValueError, match="engram_layer_ids"):
            UMichSLMConfig(
                num_hidden_layers=4,
                use_engram=True,
                engram_layer_ids=[-1, 0],
            )
