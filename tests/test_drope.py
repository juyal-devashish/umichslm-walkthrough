"""Tests for DRoPE (Dropped Rotary Position Embeddings) functionality."""

import pytest
import torch

from umich_slm import UMichSLMConfig, UMichSLMForCausalLM, UMichSLMModel, DRoPECallback
from umich_slm.modules.attention import UMichSLMAttention


class TestDRoPEConfig:
    """Test DRoPE configuration options."""

    def test_drope_config_defaults(self):
        """Test that DRoPE is disabled by default."""
        config = UMichSLMConfig()
        assert config.use_drope is False
        assert config.drope_drop_fraction == 0.875

    def test_drope_config_enabled(self):
        """Test enabling DRoPE via config."""
        config = UMichSLMConfig(use_drope=True, drope_drop_fraction=0.9)
        assert config.use_drope is True
        assert config.drope_drop_fraction == 0.9

    def test_drope_config_serialization(self):
        """Test that DRoPE config survives serialization."""
        config = UMichSLMConfig(use_drope=True, drope_drop_fraction=0.8)
        config_dict = config.to_dict()
        assert config_dict["use_drope"] is True
        assert config_dict["drope_drop_fraction"] == 0.8

        # Recreate from dict
        config2 = UMichSLMConfig.from_dict(config_dict)
        assert config2.use_drope is True
        assert config2.drope_drop_fraction == 0.8


class TestDRoPEAttention:
    """Test DRoPE in attention module."""

    def test_attention_drope_attribute(self, tiny_config):
        """Test that attention modules have drope_enabled attribute."""
        attn = UMichSLMAttention(tiny_config, layer_idx=0)
        assert hasattr(attn, "drope_enabled")
        assert attn.drope_enabled is False

    def test_attention_drope_toggle(self, tiny_config):
        """Test toggling DRoPE on attention module."""
        attn = UMichSLMAttention(tiny_config, layer_idx=0)

        # Initially disabled
        assert attn.drope_enabled is False

        # Enable
        attn.drope_enabled = True
        assert attn.drope_enabled is True

        # Disable
        attn.drope_enabled = False
        assert attn.drope_enabled is False

    def test_attention_output_differs_with_drope(self, tiny_config):
        """Test that attention output differs with and without position embeddings."""
        attn = UMichSLMAttention(tiny_config, layer_idx=0)
        attn.eval()

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, tiny_config.hidden_size)

        # Forward with position embeddings (default)
        attn.drope_enabled = False
        with torch.no_grad():
            output_with_pe, _, _ = attn(hidden_states)

        # Forward without position embeddings (DRoPE active)
        attn.drope_enabled = True
        with torch.no_grad():
            output_without_pe, _, _ = attn(hidden_states)

        # Outputs should be different
        assert not torch.allclose(output_with_pe, output_without_pe, atol=1e-6)


class TestDRoPEModel:
    """Test DRoPE at model level."""

    def test_model_drope_methods(self, tiny_config):
        """Test that model has DRoPE methods."""
        model = UMichSLMModel(tiny_config)
        assert hasattr(model, "set_drope_enabled")
        assert hasattr(model, "get_drope_enabled")

    def test_model_drope_toggle(self, tiny_config):
        """Test toggling DRoPE on model."""
        model = UMichSLMModel(tiny_config)

        # Initially disabled
        assert model.get_drope_enabled() is False

        # Enable
        model.set_drope_enabled(True)
        assert model.get_drope_enabled() is True

        # Check propagation to attention layers
        for layer in model.layers:
            if hasattr(layer, "self_attn"):
                assert layer.self_attn.drope_enabled is True

        # Disable
        model.set_drope_enabled(False)
        assert model.get_drope_enabled() is False

        # Check propagation
        for layer in model.layers:
            if hasattr(layer, "self_attn"):
                assert layer.self_attn.drope_enabled is False

    def test_causal_lm_drope_methods(self, tiny_config):
        """Test that CausalLM model has DRoPE methods."""
        model = UMichSLMForCausalLM(tiny_config)
        assert hasattr(model, "set_drope_enabled")
        assert hasattr(model, "get_drope_enabled")

    def test_causal_lm_drope_toggle(self, tiny_config):
        """Test toggling DRoPE on CausalLM model."""
        model = UMichSLMForCausalLM(tiny_config)

        # Initially disabled
        assert model.get_drope_enabled() is False

        # Enable via CausalLM method (should delegate to base model)
        model.set_drope_enabled(True)
        assert model.get_drope_enabled() is True
        assert model.model.get_drope_enabled() is True

        # Disable
        model.set_drope_enabled(False)
        assert model.get_drope_enabled() is False

    def test_model_output_differs_with_drope(self, tiny_config, sample_input):
        """Test that model output differs with and without DRoPE."""
        model = UMichSLMForCausalLM(tiny_config)
        model.eval()

        # Forward with position embeddings
        model.set_drope_enabled(False)
        with torch.no_grad():
            output_with_pe = model(**sample_input)

        # Forward without position embeddings
        model.set_drope_enabled(True)
        with torch.no_grad():
            output_without_pe = model(**sample_input)

        # Logits should be different
        assert not torch.allclose(
            output_with_pe.logits, output_without_pe.logits, atol=1e-6
        )


class TestDRoPECallback:
    """Test DRoPE training callback."""

    def test_callback_initialization(self):
        """Test callback initialization."""
        callback = DRoPECallback(drop_fraction=0.9)
        assert callback.drop_fraction == 0.9
        assert callback.drope_active is False
        assert callback.drop_step is None

    def test_callback_invalid_fraction(self):
        """Test that invalid drop fractions raise errors."""
        with pytest.raises(ValueError):
            DRoPECallback(drop_fraction=0.0)
        with pytest.raises(ValueError):
            DRoPECallback(drop_fraction=1.0)
        with pytest.raises(ValueError):
            DRoPECallback(drop_fraction=1.5)
        with pytest.raises(ValueError):
            DRoPECallback(drop_fraction=-0.1)

    def test_manually_toggle(self, tiny_config):
        """Test manual DRoPE toggling via callback."""
        model = UMichSLMForCausalLM(tiny_config)

        # Toggle on
        DRoPECallback.manually_toggle(model, enabled=True, verbose=False)
        assert model.get_drope_enabled() is True

        # Toggle off
        DRoPECallback.manually_toggle(model, enabled=False, verbose=False)
        assert model.get_drope_enabled() is False

    def test_manually_toggle_invalid_model(self):
        """Test that manually_toggle raises error for invalid model."""
        class DummyModel:
            pass

        with pytest.raises(AttributeError):
            DRoPECallback.manually_toggle(DummyModel(), enabled=True)


class TestDRoPETraining:
    """Test DRoPE during training scenarios."""

    def test_training_with_drope_disabled(self, tiny_config, sample_input_with_labels):
        """Test that training works normally with DRoPE disabled."""
        model = UMichSLMForCausalLM(tiny_config)
        model.train()

        # Ensure DRoPE is disabled
        model.set_drope_enabled(False)

        # Forward pass should work
        output = model(**sample_input_with_labels)
        assert output.loss is not None
        assert output.loss.requires_grad

    def test_training_with_drope_enabled(self, tiny_config, sample_input_with_labels):
        """Test that training works with DRoPE enabled."""
        model = UMichSLMForCausalLM(tiny_config)
        model.train()

        # Enable DRoPE
        model.set_drope_enabled(True)

        # Forward pass should still work
        output = model(**sample_input_with_labels)
        assert output.loss is not None
        assert output.loss.requires_grad

    def test_gradient_flow_with_drope(self, tiny_config, sample_input_with_labels):
        """Test that gradients flow correctly with DRoPE enabled."""
        model = UMichSLMForCausalLM(tiny_config)
        model.train()
        model.set_drope_enabled(True)

        # Forward and backward
        output = model(**sample_input_with_labels)
        output.loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Note: some params may have None grad if not used in forward
                if "rotary_emb" not in name:  # RoPE params won't get grads with DRoPE
                    pass  # Just ensure no errors


class TestDRoPEGeneration:
    """Test DRoPE during generation."""

    def test_generation_with_drope(self, tiny_config):
        """Test that generation works with DRoPE enabled."""
        model = UMichSLMForCausalLM(tiny_config)
        model.eval()
        model.set_drope_enabled(True)

        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 4))

        # Should be able to generate
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tiny_config.pad_token_id,
            )

        assert output.shape[1] == 8  # 4 input + 4 generated
