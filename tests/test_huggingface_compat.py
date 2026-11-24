"""Tests for HuggingFace compatibility."""

import pytest
import tempfile
import torch

# Import to trigger auto-registration
import umich_slm
from umich_slm import UMichSLMConfig, UMichSLMModel, UMichSLMForCausalLM


class TestAutoClassRegistration:
    """Tests for AutoClass registration."""

    def test_auto_config_registration(self):
        """AutoConfig should recognize umichslm."""
        from transformers import AutoConfig

        # Create and save a config
        config = UMichSLMConfig(hidden_size=256, num_hidden_layers=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)

            # Load with AutoConfig
            loaded = AutoConfig.from_pretrained(tmpdir, trust_remote_code=True)
            assert loaded.model_type == "umichslm"
            assert loaded.hidden_size == 256

    def test_auto_model_registration(self, tiny_config):
        """AutoModel should load UMichSLMModel."""
        from transformers import AutoModel

        model = UMichSLMModel(tiny_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            loaded = AutoModel.from_pretrained(tmpdir, trust_remote_code=True)
            assert isinstance(loaded, UMichSLMModel)

    def test_auto_model_for_causal_lm_registration(self, tiny_config):
        """AutoModelForCausalLM should load UMichSLMForCausalLM."""
        from transformers import AutoModelForCausalLM

        model = UMichSLMForCausalLM(tiny_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            loaded = AutoModelForCausalLM.from_pretrained(tmpdir, trust_remote_code=True)
            assert isinstance(loaded, UMichSLMForCausalLM)


class TestSaveLoadRoundtrip:
    """Tests for save/load roundtrip."""

    def test_model_save_load(self, tiny_model, sample_input):
        """Model should produce same outputs after save/load."""
        tiny_model.eval()

        with torch.no_grad():
            output_before = tiny_model(**sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_model.save_pretrained(tmpdir)

            from transformers import AutoModelForCausalLM

            loaded = AutoModelForCausalLM.from_pretrained(tmpdir, trust_remote_code=True)
            loaded.eval()

            with torch.no_grad():
                output_after = loaded(**sample_input)

            assert torch.allclose(
                output_before.logits,
                output_after.logits,
                atol=1e-5,
            )

    def test_config_save_load(self, tiny_config):
        """Config should roundtrip correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_config.save_pretrained(tmpdir)

            loaded = UMichSLMConfig.from_pretrained(tmpdir)

            assert loaded.hidden_size == tiny_config.hidden_size
            assert loaded.num_hidden_layers == tiny_config.num_hidden_layers
            assert loaded.num_attention_heads == tiny_config.num_attention_heads
            assert loaded.vocab_size == tiny_config.vocab_size


class TestModelIntegration:
    """Integration tests for the model."""

    def test_generation_deterministic(self, tiny_model):
        """Generation should be deterministic with do_sample=False."""
        input_ids = torch.randint(0, 100, (1, 10))

        tiny_model.eval()
        with torch.no_grad():
            output1 = tiny_model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
            )
            output2 = tiny_model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
            )

        assert torch.equal(output1, output2)

    def test_batch_inference(self, tiny_model):
        """Model should handle batched inference."""
        input_ids = torch.randint(0, 100, (4, 10))
        attention_mask = torch.ones_like(input_ids)

        tiny_model.eval()
        with torch.no_grad():
            outputs = tiny_model(input_ids, attention_mask=attention_mask)

        assert outputs.logits.shape[0] == 4

    def test_variable_length_sequences(self, tiny_model):
        """Model should handle variable length sequences with padding."""
        # Create inputs with different lengths
        input_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones_like(input_ids)
        attention_mask[1, 5:] = 0  # Second sequence is shorter

        tiny_model.eval()
        with torch.no_grad():
            outputs = tiny_model(input_ids, attention_mask=attention_mask)

        assert outputs.logits.shape == (2, 10, tiny_model.config.vocab_size)

    def test_model_parallelism_support(self, tiny_config):
        """Model should have device_map support markers."""
        model = UMichSLMForCausalLM(tiny_config)

        # Check that no_split_modules is set
        assert hasattr(model, "_no_split_modules")
        assert "UMichSLMDecoderLayer" in model._no_split_modules


class TestAttentionImplementations:
    """Tests for different attention implementations."""

    def test_eager_attention(self, tiny_config):
        """Eager attention should work."""
        tiny_config._attn_implementation = "eager"
        model = UMichSLMForCausalLM(tiny_config)

        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 10))
        outputs = model(input_ids)
        assert outputs.logits is not None

    def test_sdpa_attention(self, tiny_config):
        """SDPA attention should work."""
        tiny_config._attn_implementation = "sdpa"
        model = UMichSLMForCausalLM(tiny_config)

        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 10))
        outputs = model(input_ids)
        assert outputs.logits is not None

    @pytest.mark.skipif(
        not umich_slm.modules.attention.FLASH_ATTENTION_AVAILABLE,
        reason="Flash Attention not available",
    )
    def test_flash_attention(self, tiny_config):
        """Flash Attention 2 should work when available."""
        tiny_config._attn_implementation = "flash_attention_2"
        model = UMichSLMForCausalLM(tiny_config).cuda().to(torch.bfloat16)

        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 10), device="cuda")
        outputs = model(input_ids)
        assert outputs.logits is not None
