"""Tests for RePo (Re-Positioning) module."""

import pytest
import torch

from umich_slm import UMichSLMConfig, UMichSLMForCausalLM
from umich_slm.modules.repo import RePoPositionPredictor, RePoPositionPredictorWithBase
from umich_slm.modules.rotary import RotaryEmbedding


class TestRePoPositionPredictor:
    """Tests for RePoPositionPredictor."""

    @pytest.fixture
    def predictor(self):
        """Create a basic RePo predictor."""
        return RePoPositionPredictor(
            hidden_size=128,
            num_heads=4,
            repo_dim=16,
        )

    def test_output_shape(self, predictor):
        """Output should have shape (batch, seq_len, num_heads)."""
        hidden_states = torch.randn(2, 32, 128)
        positions = predictor(hidden_states)

        assert positions.shape == (2, 32, 4)

    def test_default_repo_dim(self):
        """Default repo_dim should be hidden_size // 8."""
        predictor = RePoPositionPredictor(
            hidden_size=256,
            num_heads=8,
        )
        assert predictor.repo_dim == 32  # 256 // 8

    def test_gradient_flow(self, predictor):
        """Gradients should flow through the predictor."""
        hidden_states = torch.randn(2, 16, 128, requires_grad=True)
        positions = predictor(hidden_states)
        loss = positions.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert predictor.gate_proj.weight.grad is not None
        assert predictor.content_proj.weight.grad is not None
        assert predictor.position_proj.weight.grad is not None

    def test_continuous_outputs(self, predictor):
        """Predicted positions should be continuous (not integer)."""
        hidden_states = torch.randn(2, 16, 128)
        positions = predictor(hidden_states)

        # Positions should not all be integers
        is_integer = (positions == positions.round()).all()
        assert not is_integer

    def test_different_inputs_different_outputs(self, predictor):
        """Different hidden states should produce different positions."""
        hidden_1 = torch.randn(2, 16, 128)
        hidden_2 = torch.randn(2, 16, 128)

        pos_1 = predictor(hidden_1)
        pos_2 = predictor(hidden_2)

        assert not torch.allclose(pos_1, pos_2)


class TestRePoPositionPredictorWithBase:
    """Tests for RePoPositionPredictorWithBase (delta-based predictor)."""

    @pytest.fixture
    def predictor_with_base(self):
        """Create a RePo predictor with base positions."""
        return RePoPositionPredictorWithBase(
            hidden_size=128,
            num_heads=4,
            repo_dim=16,
            position_scale=1.0,
        )

    def test_output_shape(self, predictor_with_base):
        """Output should have shape (batch, seq_len, num_heads)."""
        hidden_states = torch.randn(2, 32, 128)
        positions = predictor_with_base(hidden_states)

        assert positions.shape == (2, 32, 4)

    def test_base_positions_used(self, predictor_with_base):
        """Without explicit base positions, should use sequential."""
        hidden_states = torch.randn(1, 8, 128)

        # With scale=1.0 and small init, positions should be close to base
        positions = predictor_with_base(hidden_states)

        # Mean position at each seq position should be roughly that position
        mean_positions = positions.mean(dim=-1).squeeze(0)  # [seq_len]
        expected = torch.arange(8, dtype=mean_positions.dtype)

        # Should be within a reasonable range of base positions
        diff = (mean_positions - expected).abs()
        assert diff.max() < 5.0  # Small init means deltas are small

    def test_explicit_base_positions(self, predictor_with_base):
        """Should use explicit base positions when provided."""
        hidden_states = torch.randn(1, 4, 128)
        base_positions = torch.tensor([[10, 20, 30, 40]])

        positions = predictor_with_base(hidden_states, base_positions)

        # Positions should be centered around base positions
        mean_positions = positions.mean(dim=-1).squeeze(0)
        expected = base_positions.squeeze(0).float()

        diff = (mean_positions - expected).abs()
        assert diff.max() < 5.0

    def test_gradient_flow(self, predictor_with_base):
        """Gradients should flow through the predictor."""
        hidden_states = torch.randn(2, 16, 128, requires_grad=True)
        positions = predictor_with_base(hidden_states)
        loss = positions.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert predictor_with_base.delta_proj.weight.grad is not None


class TestRotaryEmbeddingContinuous:
    """Tests for continuous position support in RotaryEmbedding."""

    @pytest.fixture
    def rotary_emb(self):
        """Create a rotary embedding module."""
        return RotaryEmbedding(dim=32, max_position_embeddings=512)

    def test_continuous_forward_shape(self, rotary_emb):
        """Continuous forward should produce same output shape."""
        q = torch.randn(2, 4, 16, 32)  # batch, heads, seq, dim
        k = torch.randn(2, 4, 16, 32)
        positions = torch.randn(2, 16)  # Continuous positions

        q_rot, k_rot = rotary_emb.forward_continuous(q, k, positions)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_continuous_with_per_head_positions(self, rotary_emb):
        """Should handle per-head position assignments."""
        q = torch.randn(2, 4, 16, 32)
        k = torch.randn(2, 4, 16, 32)
        positions = torch.randn(2, 16, 4)  # Per-head positions

        q_rot, k_rot = rotary_emb.forward_continuous(q, k, positions)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_integer_positions_match_standard(self, rotary_emb):
        """Integer positions in continuous mode should match standard forward."""
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 8, 32)
        position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)

        # Standard forward
        q_std, k_std = rotary_emb(q, k, position_ids)

        # Continuous forward with same integer positions
        positions_cont = position_ids.float()
        q_cont, k_cont = rotary_emb.forward_continuous(q, k, positions_cont)

        assert torch.allclose(q_std, q_cont, atol=1e-5)
        assert torch.allclose(k_std, k_cont, atol=1e-5)

    def test_gradient_flow(self, rotary_emb):
        """Gradients should flow through continuous forward."""
        q = torch.randn(2, 4, 16, 32, requires_grad=True)
        k = torch.randn(2, 4, 16, 32, requires_grad=True)
        positions = torch.randn(2, 16, requires_grad=True)

        q_rot, k_rot = rotary_emb.forward_continuous(q, k, positions)
        loss = q_rot.sum() + k_rot.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert positions.grad is not None


class TestRePoIntegration:
    """Tests for RePo integration with full model."""

    @pytest.fixture
    def repo_config(self):
        """Create a config with RePo enabled."""
        return UMichSLMConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=172,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            use_repo=True,
            repo_start_layer=2,  # Layers 0-1 use standard RoPE
            repo_dim=8,
            repo_use_base_positions=True,
        )

    @pytest.fixture
    def repo_model(self, repo_config):
        """Create a model with RePo enabled."""
        return UMichSLMForCausalLM(repo_config)

    def test_model_creation(self, repo_model):
        """Model with RePo should be created successfully."""
        assert repo_model is not None

    def test_forward_pass(self, repo_model, repo_config):
        """Forward pass should work with RePo enabled."""
        input_ids = torch.randint(0, repo_config.vocab_size, (2, 16))

        output = repo_model(input_ids)

        assert output.logits.shape == (2, 16, repo_config.vocab_size)

    def test_repo_only_in_later_layers(self, repo_model, repo_config):
        """RePo should only be active from repo_start_layer onwards."""
        # Layers 0-1 should NOT have repo_predictor
        for i in range(repo_config.repo_start_layer):
            layer = repo_model.model.layers[i]
            assert (
                not hasattr(layer.self_attn, "repo_predictor")
                or layer.self_attn.repo_predictor is None
            )

        # Layers 2+ should have repo_predictor
        for i in range(repo_config.repo_start_layer, repo_config.num_hidden_layers):
            layer = repo_model.model.layers[i]
            assert hasattr(layer.self_attn, "repo_predictor")
            assert layer.self_attn.repo_predictor is not None

    def test_gradient_flow_full_model(self, repo_model, repo_config):
        """Gradients should flow through model with RePo."""
        input_ids = torch.randint(0, repo_config.vocab_size, (2, 16))
        labels = input_ids.clone()

        output = repo_model(input_ids, labels=labels)
        output.loss.backward()

        # Check gradients in RePo layers
        for i in range(repo_config.repo_start_layer, repo_config.num_hidden_layers):
            layer = repo_model.model.layers[i]
            predictor = layer.self_attn.repo_predictor
            assert predictor.gate_proj.weight.grad is not None

    def test_generation_with_repo(self, repo_model, repo_config):
        """Generation should work with RePo enabled."""
        input_ids = torch.randint(0, repo_config.vocab_size, (1, 4))

        # Generate a few tokens
        output_ids = repo_model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,
        )

        assert output_ids.shape[1] == 4 + 5  # Original + generated

    def test_kv_cache_with_repo(self, repo_model, repo_config):
        """KV cache should work correctly with RePo."""
        # First pass (context)
        input_ids = torch.randint(0, repo_config.vocab_size, (1, 8))
        output1 = repo_model(input_ids, use_cache=True)
        past_kv = output1.past_key_values

        assert past_kv is not None

        # Second pass (incremental)
        next_token = torch.randint(0, repo_config.vocab_size, (1, 1))
        output2 = repo_model(next_token, past_key_values=past_kv, use_cache=True)

        assert output2.logits.shape == (1, 1, repo_config.vocab_size)


class TestRePoWithoutBasePositions:
    """Tests for RePo without base positions (pure learned positions)."""

    @pytest.fixture
    def pure_repo_config(self):
        """Create a config with pure RePo (no base positions)."""
        return UMichSLMConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=172,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            use_repo=True,
            repo_start_layer=0,  # All layers use RePo
            repo_use_base_positions=False,  # Pure learned positions
        )

    def test_pure_repo_forward(self, pure_repo_config):
        """Forward pass should work with pure RePo."""
        model = UMichSLMForCausalLM(pure_repo_config)
        input_ids = torch.randint(0, pure_repo_config.vocab_size, (2, 16))

        output = model(input_ids)

        assert output.logits.shape == (2, 16, pure_repo_config.vocab_size)

    def test_pure_repo_uses_correct_predictor(self, pure_repo_config):
        """Pure RePo should use RePoPositionPredictor (not WithBase)."""
        model = UMichSLMForCausalLM(pure_repo_config)

        for layer in model.model.layers:
            predictor = layer.self_attn.repo_predictor
            assert isinstance(predictor, RePoPositionPredictor)
            assert not hasattr(predictor, "delta_proj")


class TestRePoDisabled:
    """Tests for model behavior when RePo is disabled."""

    @pytest.fixture
    def standard_config(self):
        """Create a standard config without RePo."""
        return UMichSLMConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=172,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            use_repo=False,  # Explicitly disabled
        )

    def test_no_repo_predictors(self, standard_config):
        """Standard model should not have RePo predictors."""
        model = UMichSLMForCausalLM(standard_config)

        for layer in model.model.layers:
            assert (
                not hasattr(layer.self_attn, "repo_predictor")
                or layer.self_attn.repo_predictor is None
            )

    def test_forward_without_repo(self, standard_config):
        """Standard forward should work without RePo."""
        model = UMichSLMForCausalLM(standard_config)
        input_ids = torch.randint(0, standard_config.vocab_size, (2, 16))

        output = model(input_ids)

        assert output.logits.shape == (2, 16, standard_config.vocab_size)


class TestRePoWithMLA:
    """Tests for RePo integration with Multi-head Latent Attention."""

    @pytest.fixture
    def repo_mla_config(self):
        """Create a config with both RePo and MLA enabled."""
        return UMichSLMConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=172,
            num_hidden_layers=4,
            num_attention_heads=4,
            max_position_embeddings=512,
            # MLA settings
            use_mla=True,
            mla_kv_lora_rank=32,
            mla_q_lora_rank=0,
            mla_qk_nope_head_dim=8,
            mla_qk_rope_head_dim=8,
            mla_v_head_dim=16,
            # RePo settings
            use_repo=True,
            repo_start_layer=2,
            repo_dim=8,
            repo_use_base_positions=True,
        )

    @pytest.fixture
    def repo_mla_model(self, repo_mla_config):
        """Create a model with both RePo and MLA enabled."""
        return UMichSLMForCausalLM(repo_mla_config)

    def test_model_creation(self, repo_mla_model):
        """Model with RePo + MLA should be created successfully."""
        assert repo_mla_model is not None

    def test_forward_pass(self, repo_mla_model, repo_mla_config):
        """Forward pass should work with RePo + MLA."""
        input_ids = torch.randint(0, repo_mla_config.vocab_size, (2, 16))

        output = repo_mla_model(input_ids)

        assert output.logits.shape == (2, 16, repo_mla_config.vocab_size)

    def test_repo_in_mla_layers(self, repo_mla_model, repo_mla_config):
        """MLA layers should have RePo predictors from start_layer onwards."""
        # Layers 0-1 should NOT have repo_predictor active
        for i in range(repo_mla_config.repo_start_layer):
            layer = repo_mla_model.model.layers[i]
            assert not layer.self_attn._repo_active or layer.self_attn.repo_predictor is None

        # Layers 2+ should have repo_predictor
        for i in range(repo_mla_config.repo_start_layer, repo_mla_config.num_hidden_layers):
            layer = repo_mla_model.model.layers[i]
            assert layer.self_attn._repo_active
            assert layer.self_attn.repo_predictor is not None

    def test_gradient_flow(self, repo_mla_model, repo_mla_config):
        """Gradients should flow through RePo + MLA model."""
        input_ids = torch.randint(0, repo_mla_config.vocab_size, (2, 16))
        labels = input_ids.clone()

        output = repo_mla_model(input_ids, labels=labels)
        output.loss.backward()

        # Check gradients in RePo layers
        for i in range(repo_mla_config.repo_start_layer, repo_mla_config.num_hidden_layers):
            layer = repo_mla_model.model.layers[i]
            predictor = layer.self_attn.repo_predictor
            assert predictor.gate_proj.weight.grad is not None

    def test_generation_with_repo_mla(self, repo_mla_model, repo_mla_config):
        """Generation should work with RePo + MLA."""
        input_ids = torch.randint(0, repo_mla_config.vocab_size, (1, 4))

        output_ids = repo_mla_model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,
        )

        assert output_ids.shape[1] == 4 + 5

    def test_kv_cache_with_repo_mla(self, repo_mla_model, repo_mla_config):
        """KV cache should work correctly with RePo + MLA."""
        # First pass (context)
        input_ids = torch.randint(0, repo_mla_config.vocab_size, (1, 8))
        output1 = repo_mla_model(input_ids, use_cache=True)
        past_kv = output1.past_key_values

        assert past_kv is not None

        # Second pass (incremental)
        next_token = torch.randint(0, repo_mla_config.vocab_size, (1, 1))
        output2 = repo_mla_model(next_token, past_key_values=past_kv, use_cache=True)

        assert output2.logits.shape == (1, 1, repo_mla_config.vocab_size)
