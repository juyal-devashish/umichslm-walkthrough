"""Tests for the full model implementation."""

import pytest
import torch
from transformers.cache_utils import Cache

from umich_slm import (
    UMichSLMConfig,
    UMichSLMModel,
    UMichSLMForCausalLM,
    get_umichslm_config,
)


# Presets carry the production vocab (56000, untied), so each preset-based model
# allocates+inits two 56000xhidden matrices (~28M params) before any forward pass.
# Vocab size is irrelevant to what these preset tests check (shapes, cache
# round-trips, generation), so shrink it to keep instantiation fast.
_TEST_VOCAB = 2048


def _small_config(name, **overrides):
    """get_umichslm_config with a tiny vocab so preset model builds stay cheap."""
    overrides.setdefault("vocab_size", _TEST_VOCAB)
    return get_umichslm_config(name, **overrides)


class TestUMichSLMModel:
    """Tests for the base model."""

    def test_forward_shape(self, tiny_config, sample_input):
        """Forward pass should produce correct output shape."""
        model = UMichSLMModel(tiny_config)
        outputs = model(**sample_input)

        batch_size, seq_len = sample_input["input_ids"].shape
        assert outputs.last_hidden_state.shape == (batch_size, seq_len, tiny_config.hidden_size)

    def test_forward_with_cache(self, tiny_config, sample_input):
        """Should return cache when use_cache=True."""
        model = UMichSLMModel(tiny_config)
        outputs = model(**sample_input, use_cache=True)

        assert outputs.past_key_values is not None
        assert len(outputs.past_key_values) == tiny_config.num_hidden_layers

    def test_hidden_states_output(self, tiny_config, sample_input):
        """Should return all hidden states when requested."""
        model = UMichSLMModel(tiny_config)
        outputs = model(**sample_input, output_hidden_states=True)

        assert outputs.hidden_states is not None
        # +1 for embeddings
        assert len(outputs.hidden_states) == tiny_config.num_hidden_layers + 1

    def test_attention_output(self, tiny_config, sample_input):
        """Should return attention weights when requested."""
        model = UMichSLMModel(tiny_config)
        outputs = model(**sample_input, output_attentions=True)

        assert outputs.attentions is not None
        assert len(outputs.attentions) == tiny_config.num_hidden_layers

    def test_inputs_embeds(self, tiny_config):
        """Should accept pre-computed embeddings."""
        model = UMichSLMModel(tiny_config)
        inputs_embeds = torch.randn(2, 16, tiny_config.hidden_size)

        outputs = model(inputs_embeds=inputs_embeds)
        assert outputs.last_hidden_state.shape == inputs_embeds.shape

    def test_gradient_checkpointing(self, tiny_config, sample_input):
        """Gradient checkpointing should work."""
        model = UMichSLMModel(tiny_config)
        model.gradient_checkpointing_enable()

        # Forward pass should work with gradient checkpointing
        outputs = model(**sample_input)
        assert outputs.last_hidden_state is not None


class TestUMichSLMForCausalLM:
    """Tests for the causal LM model."""

    def test_forward_shape(self, tiny_model, sample_input):
        """Forward pass should produce correct logits shape."""
        outputs = tiny_model(**sample_input)

        batch_size, seq_len = sample_input["input_ids"].shape
        assert outputs.logits.shape == (batch_size, seq_len, tiny_model.config.vocab_size)

    def test_loss_computation(self, tiny_model, sample_input_with_labels):
        """Should compute loss when labels provided."""
        outputs = tiny_model(**sample_input_with_labels)

        assert outputs.loss is not None
        assert outputs.loss.shape == ()  # Scalar
        assert not torch.isnan(outputs.loss)

    def test_generation(self, tiny_model):
        """Model should support text generation."""
        input_ids = torch.randint(0, 100, (1, 10))

        with torch.no_grad():
            outputs = tiny_model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
                eos_token_id=None,  # Prevent early stopping
                pad_token_id=0,
            )

        assert outputs.shape[1] == 15  # 10 + 5

    def test_generation_with_cache(self, tiny_model):
        """Generation should work with KV cache."""
        input_ids = torch.randint(0, 100, (1, 10))

        with torch.no_grad():
            outputs = tiny_model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
                use_cache=True,
            )

        assert outputs.shape[1] == 15

    def test_tied_embeddings(self, tiny_config):
        """Tied embeddings should share weights."""
        tiny_config.tie_word_embeddings = True
        model = UMichSLMForCausalLM(tiny_config)

        # Input and output embeddings should be the same object
        assert model.get_input_embeddings().weight is model.get_output_embeddings().weight

    def test_untied_embeddings(self, tiny_config):
        """Untied embeddings should have separate weights."""
        tiny_config.tie_word_embeddings = False
        model = UMichSLMForCausalLM(tiny_config)

        # Input and output embeddings should be different
        input_emb = model.get_input_embeddings().weight
        output_emb = model.get_output_embeddings().weight
        # They might start with the same values but should be different objects
        assert input_emb is not output_emb

    def test_gradient_flow(self, tiny_model, sample_input_with_labels):
        """Gradients should flow through the entire model."""
        outputs = tiny_model(**sample_input_with_labels)
        outputs.loss.backward()

        # Check gradients on key parameters
        assert tiny_model.model.embed_tokens.weight.grad is not None
        assert tiny_model.lm_head.weight.grad is not None

    def test_mixed_precision(self, tiny_config):
        """Model should work with mixed precision."""
        model = UMichSLMForCausalLM(tiny_config).to(torch.float16)
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 10))

        with torch.no_grad():
            outputs = model(input_ids)

        assert outputs.logits.dtype == torch.float32  # Logits are cast to float


class TestModelPresets:
    """Tests for model presets."""

    def test_tiny_preset(self):
        """Tiny preset should create a small model."""
        config = _small_config("tiny")
        model = UMichSLMForCausalLM(config)

        # Should be a valid model
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        outputs = model(input_ids)
        assert outputs.logits is not None

    def test_qwen3_5_like_tiny_preset(self):
        """Qwen3.5-like tiny preset should instantiate and run a forward pass."""
        config = _small_config("qwen3_5_like_tiny")
        model = UMichSLMForCausalLM(config)

        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        outputs = model(input_ids)
        assert outputs.logits is not None

    def test_qwen3_5_like_tiny_generation_with_cache(self):
        """Qwen3.5-like tiny preset should generate with cache enabled."""
        config = _small_config("qwen3_5_like_tiny")
        model = UMichSLMForCausalLM(config)
        input_ids = torch.randint(0, min(100, config.vocab_size), (1, 10))

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=3,
                do_sample=False,
                use_cache=True,
                eos_token_id=None,
                pad_token_id=0,
            )

        assert outputs.shape[1] == 13

    def test_qwen3_5_like_cache_reuse_returns_hf_cache(self):
        """Hybrid DeltaNet generation cache should round-trip as a transformers Cache."""
        config = _small_config("qwen3_5_like_tiny", deltanet_first_n_layers=1)
        model = UMichSLMForCausalLM(config)
        input_ids = torch.randint(0, min(100, config.vocab_size), (1, 6))

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=2,
                do_sample=False,
                use_cache=True,
                eos_token_id=None,
                pad_token_id=0,
                return_dict_in_generate=True,
            )

            assert isinstance(outputs.past_key_values, Cache)

            next_input_ids = torch.randint(0, min(100, config.vocab_size), (1, 1))
            next_outputs = model.generate(
                next_input_ids,
                max_new_tokens=1,
                do_sample=False,
                use_cache=True,
                eos_token_id=None,
                pad_token_id=0,
                past_key_values=outputs.past_key_values,
            )

        assert next_outputs.shape[1] == 2

    def test_qwen3_5_like_cache_legacy_roundtrip(self):
        """DeltaNet cache should support explicit legacy conversion without losing reuse."""
        config = _small_config("qwen3_5_like_tiny", deltanet_first_n_layers=1)
        model = UMichSLMForCausalLM(config)
        input_ids = torch.randint(0, min(100, config.vocab_size), (1, 6))

        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)

        legacy_cache = outputs.past_key_values.to_legacy_cache()
        assert isinstance(legacy_cache, tuple)
        assert len(legacy_cache) == config.num_hidden_layers
        assert any(isinstance(layer_cache, tuple) for layer_cache in legacy_cache)
        assert any(
            layer_cache is not None and not isinstance(layer_cache, tuple)
            for layer_cache in legacy_cache
        )

        next_token = torch.randint(0, min(100, config.vocab_size), (1, 1))
        with torch.no_grad():
            next_outputs = model(next_token, past_key_values=legacy_cache, use_cache=True)

        assert isinstance(next_outputs.past_key_values, Cache)
        assert next_outputs.logits.shape[1] == 1

    def test_preset_override(self):
        """Should be able to override preset values."""
        config = get_umichslm_config("tiny", num_hidden_layers=4)
        assert config.num_hidden_layers == 4

    def test_invalid_preset(self):
        """Should raise error for invalid preset."""
        with pytest.raises(ValueError):
            get_umichslm_config("invalid_size")


class TestPoPEModel:
    """Tests for models with PoPE position embeddings."""

    def test_pope_model_forward(self, pope_config):
        """PoPE model should work correctly."""
        model = UMichSLMForCausalLM(pope_config)
        input_ids = torch.randint(0, pope_config.vocab_size, (1, 10))

        outputs = model(input_ids)
        assert outputs.logits is not None
        assert outputs.logits.shape == (1, 10, pope_config.vocab_size)

    def test_pope_model_generation(self, pope_config):
        """PoPE model should support generation."""
        model = UMichSLMForCausalLM(pope_config)
        input_ids = torch.randint(0, 100, (1, 5))

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=3,
                do_sample=False,
            )

        assert outputs.shape[1] == 8  # 5 + 3

    def test_pope_model_generation_with_sdpa_config(self, pope_config):
        """PoPE model should work even when config requests SDPA."""
        pope_config._attn_implementation = "sdpa"
        model = UMichSLMForCausalLM(pope_config)
        input_ids = torch.randint(0, 100, (1, 5))

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=10,  # Test longer generation for KV cache
                do_sample=False,
            )

        assert outputs.shape[1] == 15  # 5 + 10

    def test_pope_model_incremental_decoding(self, pope_config):
        """PoPE model should work with incremental decoding and KV cache."""
        model = UMichSLMForCausalLM(pope_config)
        model.eval()

        # First pass - process initial tokens
        input_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)

        past_kv = outputs.past_key_values

        # Incremental passes - one token at a time
        for i in range(5):
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            with torch.no_grad():
                outputs = model(
                    next_token,
                    past_key_values=past_kv,
                    use_cache=True,
                )
            past_kv = outputs.past_key_values

        # Should not raise any errors
        assert outputs.logits is not None


class TestZeroCenteredRMSNorm:
    """Tests for models with zero-centered RMSNorm."""

    @pytest.fixture
    def zc_config(self):
        """Create config with zero-centered RMSNorm."""
        return UMichSLMConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            use_zero_centered_rmsnorm=True,
        )

    def test_zero_centered_model_forward(self, zc_config):
        """Model with zero-centered RMSNorm should work correctly."""
        model = UMichSLMForCausalLM(zc_config)
        input_ids = torch.randint(0, zc_config.vocab_size, (1, 10))

        outputs = model(input_ids)
        assert outputs.logits is not None
        assert outputs.logits.shape == (1, 10, zc_config.vocab_size)

    def test_zero_centered_uses_correct_norm(self, zc_config):
        """Model should use ZeroCenteredRMSNorm when configured."""
        from umich_slm.modules.rmsnorm import ZeroCenteredRMSNorm

        model = UMichSLMModel(zc_config)

        # Check final norm
        assert isinstance(model.norm, ZeroCenteredRMSNorm)

        # Check layer norms
        for layer in model.layers:
            assert isinstance(layer.input_layernorm, ZeroCenteredRMSNorm)
            assert isinstance(layer.post_attention_layernorm, ZeroCenteredRMSNorm)

    def test_zero_centered_model_generation(self, zc_config):
        """Model with zero-centered RMSNorm should support generation."""
        model = UMichSLMForCausalLM(zc_config)
        input_ids = torch.randint(0, 100, (1, 5))

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=3,
                do_sample=False,
            )

        assert outputs.shape[1] == 8  # 5 + 3

    def test_zero_centered_config_preserved(self, zc_config, tmp_path):
        """Zero-centered RMSNorm config should be preserved on save/load."""
        model = UMichSLMForCausalLM(zc_config)
        save_path = tmp_path / "zc_model"
        model.save_pretrained(save_path)

        loaded_model = UMichSLMForCausalLM.from_pretrained(save_path)
        assert loaded_model.config.use_zero_centered_rmsnorm == True

    def test_standard_rmsnorm_by_default(self, tiny_config):
        """Model should use standard RMSNorm by default."""
        from umich_slm.modules.rmsnorm import ZeroCenteredRMSNorm

        model = UMichSLMModel(tiny_config)

        # Final norm should NOT be ZeroCenteredRMSNorm
        assert not isinstance(model.norm, ZeroCenteredRMSNorm)


class TestModelTokenizerCompatibility:
    """Tests to verify model works correctly with real tokenizers."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Load the project SuperBPE tokenizer once for the whole class.

        Class-scoped so the (single) tokenizer load is shared across these tests
        instead of re-running per test. Loads from the local prep path by default;
        set ``$UMICHSLM_TOKENIZER_PATH`` on hosts without it mounted. Skips
        gracefully if neither transformers nor the tokenizer is available.
        """
        try:
            from umich_slm.tokenizer_defaults import load_default_tokenizer
        except ImportError:
            pytest.skip("transformers/tokenizer_defaults not available")

        try:
            return load_default_tokenizer()
        except Exception as e:
            pytest.skip(f"Could not load project tokenizer: {e}")

    def test_model_vocab_size_matches_tokenizer(self, tokenizer):
        """Model vocab_size should match tokenizer vocab_size."""
        # Use len(tokenizer) to include special tokens, not just vocab_size
        vocab_size = len(tokenizer)

        # Create model with tokenizer's full vocab size
        config = UMichSLMConfig(
            vocab_size=vocab_size,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
        )

        model = UMichSLMForCausalLM(config)

        # Verify vocab_size matches
        assert model.config.vocab_size == vocab_size
        assert model.vocab_size == vocab_size
        assert model.get_input_embeddings().num_embeddings == vocab_size
        # lm_head is a Linear layer, so check out_features instead of num_embeddings
        assert model.get_output_embeddings().out_features == vocab_size

    def test_model_forward_with_tokenizer_output(self, tokenizer):
        """Model should process tokens from tokenizer without errors."""
        # Use len(tokenizer) to include special tokens
        vocab_size = len(tokenizer)

        config = UMichSLMConfig(
            vocab_size=vocab_size,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
        )

        model = UMichSLMForCausalLM(config)
        model.eval()

        # Tokenize a sample text
        test_text = "Hello, world! This is a test."
        encoded = tokenizer(test_text, return_tensors="pt", padding=False)

        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", None)

        # Verify all token IDs are within valid range
        assert input_ids.min() >= 0
        assert (
            input_ids.max() < vocab_size
        ), f"Token ID {input_ids.max()} exceeds vocab_size {vocab_size}"

        # Forward pass should work without errors
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Check output shape
        assert outputs.logits.shape == (1, input_ids.shape[1], vocab_size)

    def test_model_generation_with_tokenizer(self, tokenizer):
        """Model should generate tokens that can be decoded by tokenizer."""
        # Use len(tokenizer) to include special tokens
        vocab_size = len(tokenizer)

        config = UMichSLMConfig(
            vocab_size=vocab_size,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
        )

        model = UMichSLMForCausalLM(config)
        model.eval()

        # Tokenize a prompt
        prompt = "The capital of France is"
        encoded = tokenizer(prompt, return_tensors="pt", padding=False)
        input_ids = encoded["input_ids"]

        # Generate tokens
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )

        # Verify generated token IDs are within valid range
        assert generated_ids.min() >= 0
        assert (
            generated_ids.max() < vocab_size
        ), f"Generated token ID {generated_ids.max()} exceeds vocab_size {vocab_size}"

        # Should be able to decode without errors
        decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_model_with_different_tokenizer_vocab_sizes(self):
        """Model should work with different tokenizer vocab sizes."""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            pytest.skip("transformers not available")

        # Test with a smaller vocab size (GPT-2 style)
        # We'll create a mock tokenizer scenario by using a known smaller vocab size
        test_vocab_sizes = [50257, 32000, 50000, 128256]  # Various common sizes

        for vocab_size in test_vocab_sizes:
            config = UMichSLMConfig(
                vocab_size=vocab_size,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=256,
            )

            model = UMichSLMForCausalLM(config)
            model.eval()

            # Create random input within vocab range
            input_ids = torch.randint(0, vocab_size, (1, 10))

            # Forward pass should work
            with torch.no_grad():
                outputs = model(input_ids)

            assert outputs.logits.shape == (1, 10, vocab_size)
            assert outputs.logits.min().item() > float("-inf")
            assert outputs.logits.max().item() < float("inf")

    def test_model_embedding_layer_matches_vocab_size(self, tokenizer):
        """Model embedding layers should have correct size matching tokenizer."""
        # Use len(tokenizer) to include special tokens
        vocab_size = len(tokenizer)

        config = UMichSLMConfig(
            vocab_size=vocab_size,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
        )

        model = UMichSLMForCausalLM(config)

        # Check embedding layer dimensions
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()

        assert input_embeddings.num_embeddings == vocab_size
        # lm_head is a Linear layer, so check out_features instead of num_embeddings
        assert output_embeddings.out_features == vocab_size
        assert input_embeddings.embedding_dim == config.hidden_size
        # For Linear layer, in_features is the "embedding_dim"
        assert output_embeddings.in_features == config.hidden_size

        # Test that we can look up embeddings for valid token IDs
        # Sample a subset to avoid memory issues with large vocabularies
        sample_size = min(1000, vocab_size)
        sample_token_ids = torch.randint(0, vocab_size, (sample_size,), dtype=torch.long)
        with torch.no_grad():
            embeddings = input_embeddings(sample_token_ids)
            assert embeddings.shape == (sample_size, config.hidden_size)

        # Also test boundary cases: first and last token IDs
        boundary_ids = torch.tensor([0, vocab_size - 1], dtype=torch.long)
        with torch.no_grad():
            boundary_embeddings = input_embeddings(boundary_ids)
            assert boundary_embeddings.shape == (2, config.hidden_size)


class TestChunkedLMLossSyncFree:
    """The chunked CE must handle a fully-ignored batch WITHOUT a host sync.

    ``total_count.item() == 0`` was a device->host sync on every forward, placed at
    the end of the graph (before the backward is queued), to guard a degenerate case
    a pretraining run never sees. It is now a device-side ``clamp``; these pin that
    the guarded behaviour is unchanged.
    """

    def _model(self):
        import torch

        from umich_slm.configs.presets import get_umichslm_config
        from umich_slm.modeling_umichslm import UMichSLMForCausalLM

        cfg = get_umichslm_config("tiny")
        cfg.vocab_size = 64
        cfg.num_hidden_layers = 2
        cfg.hidden_size = 32
        cfg.intermediate_size = 64
        cfg.num_attention_heads = 4
        cfg.num_key_value_heads = 4
        cfg._attn_implementation = "eager"
        return UMichSLMForCausalLM(cfg).eval(), cfg

    def test_all_ignored_batch_is_zero_and_finite(self):
        import torch

        m, cfg = self._model()
        h = torch.randn(2, 8, cfg.hidden_size, requires_grad=True)
        loss = m._compute_shifted_lm_loss_chunked(h, torch.full((2, 8), -100))
        assert torch.isfinite(loss) and float(loss) == 0.0
        loss.backward()  # must not produce NaN through the zero denominator
        assert torch.isfinite(h.grad).all()

    def test_normal_batch_matches_unchunked_cross_entropy(self):
        import torch
        import torch.nn.functional as F

        m, cfg = self._model()
        h = torch.randn(2, 16, cfg.hidden_size)
        labels = torch.randint(0, cfg.vocab_size, (2, 16))
        got = m._compute_shifted_lm_loss_chunked(h, labels)
        ref = F.cross_entropy(
            m.lm_head(h[:, :-1, :]).reshape(-1, cfg.vocab_size).float(),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        assert torch.allclose(got, ref, atol=1e-5), (float(got), float(ref))
