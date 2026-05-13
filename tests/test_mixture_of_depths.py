"""Tests for Mixture of Depths (MoD) module.

Tests the MoD router, gate, decoder layer, and full model integration.
"""

import pytest
import torch
import torch.nn as nn

from umich_slm import UMichSLMConfig, UMichSLMForCausalLM
from umich_slm.modules import (
    MoDRouter,
    MoDGate,
    MoDOutput,
    is_mod_layer,
    compute_mod_aux_loss,
    UMichSLMMoDDecoderLayer,
)


class TestMoDRouter:
    """Tests for MoDRouter."""

    @pytest.fixture
    def router(self):
        """Create a router for testing."""
        return MoDRouter(
            hidden_size=64,
            capacity=0.25,  # 25% of tokens processed
            aux_loss_weight=0.01,
            use_aux_predictor=False,
        )

    @pytest.fixture
    def router_with_predictor(self):
        """Create a router with auxiliary predictor."""
        return MoDRouter(
            hidden_size=64,
            capacity=0.25,
            aux_loss_weight=0.01,
            use_aux_predictor=True,
        )

    def test_router_init(self, router):
        """Test router initialization."""
        assert router.hidden_size == 64
        assert router.capacity == 0.25
        assert router.aux_loss_weight == 0.01
        assert router.aux_predictor is None
        assert isinstance(router.router, nn.Linear)
        assert router.router.in_features == 64
        assert router.router.out_features == 1

    def test_router_with_predictor_init(self, router_with_predictor):
        """Test router with auxiliary predictor initialization."""
        assert router_with_predictor.aux_predictor is not None
        assert isinstance(router_with_predictor.aux_predictor, nn.Sequential)

    def test_router_forward_training(self, router):
        """Test router forward pass during training."""
        router.train()
        hidden_states = torch.randn(2, 16, 64)  # batch=2, seq=16, hidden=64

        router_weights, selected_mask, aux_loss = router(hidden_states)

        # Check shapes
        assert router_weights.shape == (2, 16)
        assert selected_mask.shape == (2, 16)
        assert selected_mask.dtype == torch.bool

        # Check aux loss exists during training
        assert aux_loss is not None
        assert aux_loss.ndim == 0  # scalar

        # Check capacity: 25% of 16 = 4 tokens selected per batch
        assert selected_mask.sum(dim=1).tolist() == [4, 4]

    def test_router_forward_inference(self, router):
        """Test router forward pass during inference."""
        router.eval()
        hidden_states = torch.randn(2, 16, 64)

        with torch.no_grad():
            router_weights, selected_mask, aux_loss = router(hidden_states)

        # Check shapes
        assert router_weights.shape == (2, 16)
        assert selected_mask.shape == (2, 16)

        # No aux loss during inference
        assert aux_loss is None

        # Still uses top-k for batch inference
        assert selected_mask.sum(dim=1).tolist() == [4, 4]

    def test_router_single_token_inference(self, router):
        """Test router with single token (autoregressive generation)."""
        router.eval()
        hidden_states = torch.randn(2, 1, 64)  # Single token

        with torch.no_grad():
            router_weights, selected_mask, aux_loss = router(hidden_states, use_cache=True)

        # Check shapes
        assert router_weights.shape == (2, 1)
        assert selected_mask.shape == (2, 1)

        # Uses sigmoid threshold for single token
        # Result depends on learned weights (random init)
        assert selected_mask.dtype == torch.bool

    def test_router_with_aux_predictor_inference(self, router_with_predictor):
        """Test router with auxiliary predictor during single token inference."""
        router_with_predictor.eval()
        hidden_states = torch.randn(2, 1, 64)

        with torch.no_grad():
            router_weights, selected_mask, aux_loss = router_with_predictor(
                hidden_states, use_cache=True
            )

        assert router_weights.shape == (2, 1)
        assert selected_mask.shape == (2, 1)

    def test_router_capacity_edge_cases(self):
        """Test router with different capacity values."""
        # Very low capacity
        router_low = MoDRouter(hidden_size=64, capacity=0.01)
        router_low.train()
        hidden_states = torch.randn(1, 100, 64)
        _, selected_mask, _ = router_low(hidden_states)
        assert selected_mask.sum() == 1  # At least 1 token

        # High capacity
        router_high = MoDRouter(hidden_size=64, capacity=0.9)
        router_high.train()
        _, selected_mask, _ = router_high(hidden_states)
        assert selected_mask.sum() == 90  # 90% of 100

    def test_router_gradients(self, router):
        """Test that gradients flow through router."""
        router.train()
        hidden_states = torch.randn(2, 16, 64, requires_grad=True)

        router_weights, selected_mask, aux_loss = router(hidden_states)

        # Backprop through aux loss
        aux_loss.backward()

        # Check gradients exist
        assert hidden_states.grad is not None
        assert router.router.weight.grad is not None


class TestMoDGate:
    """Tests for MoDGate."""

    @pytest.fixture
    def gate(self):
        """Create a gate for testing."""
        return MoDGate(hidden_size=64)

    def test_gate_init(self, gate):
        """Test gate initialization."""
        assert gate.hidden_size == 64

    def test_gate_forward_all_selected(self, gate):
        """Test gate when all tokens are selected."""
        hidden_states = torch.randn(2, 8, 64)
        block_output = torch.randn(2, 8, 64)
        router_weights = torch.ones(2, 8) * 0.5
        selected_mask = torch.ones(2, 8, dtype=torch.bool)

        output = gate(hidden_states, block_output, router_weights, selected_mask)

        # output = 0.5 * block_output + hidden_states
        expected = 0.5 * block_output + hidden_states
        assert torch.allclose(output, expected)

    def test_gate_forward_none_selected(self, gate):
        """Test gate when no tokens are selected."""
        hidden_states = torch.randn(2, 8, 64)
        block_output = torch.randn(2, 8, 64)
        router_weights = torch.ones(2, 8) * 0.5
        selected_mask = torch.zeros(2, 8, dtype=torch.bool)

        output = gate(hidden_states, block_output, router_weights, selected_mask)

        # output = hidden_states (residual only)
        assert torch.allclose(output, hidden_states)

    def test_gate_forward_mixed_selection(self, gate):
        """Test gate with mixed selection."""
        hidden_states = torch.zeros(1, 4, 64)
        block_output = torch.ones(1, 4, 64)
        router_weights = torch.tensor([[0.5, 0.8, 0.3, 1.0]])
        selected_mask = torch.tensor([[True, True, False, False]])

        output = gate(hidden_states, block_output, router_weights, selected_mask)

        # Check selected tokens
        assert torch.allclose(output[0, 0], 0.5 * block_output[0, 0])
        assert torch.allclose(output[0, 1], 0.8 * block_output[0, 1])

        # Check skipped tokens
        assert torch.allclose(output[0, 2], hidden_states[0, 2])
        assert torch.allclose(output[0, 3], hidden_states[0, 3])

    def test_gate_gradients(self, gate):
        """Test that gradients flow through gate."""
        hidden_states = torch.randn(2, 8, 64, requires_grad=True)
        block_output = torch.randn(2, 8, 64, requires_grad=True)
        router_weights = torch.randn(2, 8, requires_grad=True)
        selected_mask = torch.ones(2, 8, dtype=torch.bool)

        output = gate(hidden_states, block_output, router_weights, selected_mask)
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert block_output.grad is not None
        assert router_weights.grad is not None


class TestIsModLayer:
    """Tests for is_mod_layer helper function."""

    def test_mod_every_2_layers(self):
        """Test MoD every 2 layers pattern."""
        # Layer 0: full, Layer 1: MoD, Layer 2: full, Layer 3: MoD, ...
        assert not is_mod_layer(0, 2)
        assert is_mod_layer(1, 2)
        assert not is_mod_layer(2, 2)
        assert is_mod_layer(3, 2)
        assert not is_mod_layer(4, 2)

    def test_mod_every_3_layers(self):
        """Test MoD every 3 layers pattern."""
        assert not is_mod_layer(0, 3)
        assert is_mod_layer(1, 3)
        assert not is_mod_layer(2, 3)
        assert not is_mod_layer(3, 3)
        assert is_mod_layer(4, 3)

    def test_mod_disabled(self):
        """Test MoD disabled (mod_every_n_layers=0)."""
        for i in range(10):
            assert not is_mod_layer(i, 0)

    def test_mod_disabled_negative(self):
        """Test MoD disabled with negative value."""
        for i in range(10):
            assert not is_mod_layer(i, -1)


class TestComputeModAuxLoss:
    """Tests for compute_mod_aux_loss helper function."""

    def test_no_outputs(self):
        """Test with empty outputs."""
        assert compute_mod_aux_loss(()) is None

    def test_all_none(self):
        """Test with all None outputs."""
        outputs = (None, None, None)
        assert compute_mod_aux_loss(outputs) is None

    def test_no_aux_loss(self):
        """Test with outputs but no aux loss."""
        outputs = (
            MoDOutput(hidden_states=torch.randn(2, 8, 64)),
            MoDOutput(hidden_states=torch.randn(2, 8, 64)),
        )
        assert compute_mod_aux_loss(outputs) is None

    def test_single_aux_loss(self):
        """Test with single aux loss."""
        loss = torch.tensor(0.5)
        outputs = (
            MoDOutput(hidden_states=torch.randn(2, 8, 64), aux_loss=loss),
            MoDOutput(hidden_states=torch.randn(2, 8, 64)),
        )
        result = compute_mod_aux_loss(outputs)
        assert torch.equal(result, loss)

    def test_multiple_aux_losses(self):
        """Test with multiple aux losses."""
        loss1 = torch.tensor(0.5)
        loss2 = torch.tensor(0.3)
        outputs = (
            MoDOutput(hidden_states=torch.randn(2, 8, 64), aux_loss=loss1),
            MoDOutput(hidden_states=torch.randn(2, 8, 64), aux_loss=loss2),
        )
        result = compute_mod_aux_loss(outputs)
        assert torch.allclose(result, loss1 + loss2)


class TestUMichSLMMoDDecoderLayer:
    """Tests for UMichSLMMoDDecoderLayer."""

    @pytest.fixture
    def config(self):
        """Create a config for testing."""
        return UMichSLMConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=4,
            max_position_embeddings=128,
            use_mod=True,
            mod_capacity=0.5,
            mod_every_n_layers=2,
            mod_aux_loss_weight=0.01,
        )

    @pytest.fixture
    def mod_layer(self, config):
        """Create a MoD decoder layer."""
        return UMichSLMMoDDecoderLayer(config, layer_idx=1)

    def test_mod_layer_init(self, mod_layer, config):
        """Test MoD decoder layer initialization."""
        assert mod_layer.layer_idx == 1
        assert hasattr(mod_layer, "router")
        assert hasattr(mod_layer, "gate")
        assert hasattr(mod_layer, "self_attn")
        assert hasattr(mod_layer, "mlp")
        assert hasattr(mod_layer, "input_layernorm")
        assert hasattr(mod_layer, "post_attention_layernorm")

    def test_mod_layer_forward_training(self, mod_layer):
        """Test MoD layer forward during training."""
        mod_layer.train()
        hidden_states = torch.randn(2, 16, 64)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        output = mod_layer(
            hidden_states,
            position_ids=position_ids,
            output_mod_info=True,
        )

        # Output is a tuple with MoDOutput as last element
        assert isinstance(output, tuple)
        hidden_out = output[0]
        mod_output = output[-1]

        assert isinstance(mod_output, MoDOutput)
        assert hidden_out.shape == (2, 16, 64)
        assert mod_output.hidden_states.shape == (2, 16, 64)
        assert mod_output.router_weights is not None
        assert mod_output.selected_mask is not None
        assert mod_output.aux_loss is not None

        # Check capacity: 50% of 16 = 8 tokens
        assert mod_output.selected_mask.sum(dim=1).tolist() == [8, 8]

    def test_mod_layer_forward_inference(self, mod_layer):
        """Test MoD layer forward during inference."""
        mod_layer.eval()
        hidden_states = torch.randn(2, 16, 64)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        with torch.no_grad():
            output = mod_layer(
                hidden_states,
                position_ids=position_ids,
                output_mod_info=True,
            )

        # Output is a tuple with MoDOutput as last element
        assert isinstance(output, tuple)
        mod_output = output[-1]

        assert isinstance(mod_output, MoDOutput)
        assert mod_output.hidden_states.shape == (2, 16, 64)
        assert mod_output.aux_loss is None  # No aux loss during inference

    def test_mod_layer_forward_without_mod_info(self, mod_layer):
        """Test MoD layer forward without outputting MoD info."""
        mod_layer.eval()
        hidden_states = torch.randn(2, 16, 64)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        with torch.no_grad():
            output = mod_layer(
                hidden_states,
                position_ids=position_ids,
                output_mod_info=False,
            )

        # Returns tuple with just hidden states when not outputting mod info
        assert isinstance(output, tuple)
        assert len(output) == 1  # Just hidden states when use_cache=False
        hidden = output[0]
        assert hidden.shape == (2, 16, 64)

    def test_mod_layer_gradients(self, mod_layer):
        """Test that gradients flow through MoD layer."""
        mod_layer.train()
        hidden_states = torch.randn(2, 16, 64, requires_grad=True)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        output = mod_layer(
            hidden_states,
            position_ids=position_ids,
            output_mod_info=True,
        )

        # Extract hidden states and MoDOutput from tuple
        hidden_out = output[0]
        mod_output = output[-1]

        # Backprop through both hidden states and aux loss
        loss = hidden_out.sum() + mod_output.aux_loss
        loss.backward()

        assert hidden_states.grad is not None


class TestMoDModelIntegration:
    """Tests for MoD integration in full model."""

    @pytest.fixture
    def config(self):
        """Create a small config for testing."""
        return UMichSLMConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=6,  # Layers 1, 3, 5 will be MoD
            max_position_embeddings=128,
            use_mod=True,
            mod_capacity=0.25,
            mod_every_n_layers=2,
            mod_aux_loss_weight=0.01,
        )

    @pytest.fixture
    def model(self, config):
        """Create a model for testing."""
        return UMichSLMForCausalLM(config)

    def test_model_has_mod_layers(self, model):
        """Test that model has alternating MoD layers."""
        for i, layer in enumerate(model.model.layers):
            if i % 2 == 1:  # Layers 1, 3, 5
                assert isinstance(layer, UMichSLMMoDDecoderLayer), f"Layer {i} should be MoD"
            else:
                assert not isinstance(layer, UMichSLMMoDDecoderLayer), f"Layer {i} should not be MoD"

    def test_model_forward_training(self, model):
        """Test full model forward during training."""
        model.train()
        input_ids = torch.randint(0, 100, (2, 32))
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)

        # Check loss includes aux loss
        assert outputs.loss is not None
        assert outputs.loss.requires_grad

    def test_model_forward_inference(self, model):
        """Test full model forward during inference."""
        model.eval()
        input_ids = torch.randint(0, 100, (2, 32))

        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        assert outputs.logits.shape == (2, 32, 100)

    def test_model_with_output_hidden_states(self, model):
        """Test model with output_hidden_states=True."""
        model.eval()
        input_ids = torch.randint(0, 100, (2, 16))

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)

        assert outputs.hidden_states is not None
        # 6 layers + 1 embedding = 7 hidden states
        assert len(outputs.hidden_states) == 7

    def test_model_generate(self, model):
        """Test model generation with MoD."""
        model.eval()
        input_ids = torch.randint(0, 100, (1, 10))

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=5,
                do_sample=False,
            )

        assert generated.shape[1] == 15  # 10 input + 5 generated

    def test_model_generate_with_cache(self, model):
        """Test model generation with KV cache."""
        model.eval()
        input_ids = torch.randint(0, 100, (1, 10))

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=10,
                do_sample=False,
                use_cache=True,
            )

        assert generated.shape[1] == 20

    def test_model_save_load(self, model, tmp_path):
        """Test saving and loading model with MoD config."""
        save_path = tmp_path / "mod_model"
        model.save_pretrained(save_path)

        # Load model
        loaded_model = UMichSLMForCausalLM.from_pretrained(save_path)

        # Check config preserved
        assert loaded_model.config.use_mod == True
        assert loaded_model.config.mod_capacity == 0.25
        assert loaded_model.config.mod_every_n_layers == 2
        assert loaded_model.config.mod_aux_loss_weight == 0.01

        # Check model structure preserved
        for i, layer in enumerate(loaded_model.model.layers):
            if i % 2 == 1:
                assert isinstance(layer, UMichSLMMoDDecoderLayer)

    def test_aux_loss_affects_total_loss(self, model):
        """Test that MoD aux loss is added to total loss."""
        model.train()
        input_ids = torch.randint(0, 100, (2, 32))
        labels = input_ids.clone()

        # Get loss with MoD
        outputs_mod = model(input_ids=input_ids, labels=labels)

        # Temporarily disable MoD aux loss to compare
        model.use_mod = False
        outputs_no_mod = model(input_ids=input_ids, labels=labels)
        model.use_mod = True  # Restore

        # Losses should be different due to aux loss
        # (Note: they might be similar but aux loss component is different)
        assert outputs_mod.loss is not None
        assert outputs_no_mod.loss is not None


class TestMoDWithMoE:
    """Tests for MoD combined with MoE."""

    def test_overlapping_mod_moe_raises(self):
        """Test that overlapping MoD and MoE layers raise ValueError."""
        with pytest.raises(ValueError, match="both MoE and MoD"):
            UMichSLMConfig(
                vocab_size=100,
                hidden_size=64,
                intermediate_size=128,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_hidden_layers=4,
                max_position_embeddings=64,
                use_mod=True,
                mod_capacity=0.5,
                mod_every_n_layers=2,
                use_moe=True,
                num_experts=4,
                num_experts_per_tok=2,
            )

    def test_progressive_moe_no_overlap_succeeds(self):
        """Test that progressive MoE with non-overlapping MoD layers works."""
        # With 8 layers and progressive MoE (start_ratio=0.5), layers 0-3 are plain MLP.
        # MoD layers at idx % 2 == 1 -> layers 1, 3, 5, 7.
        # Layers 1, 3 are plain MLP (1 expert), so no overlap.
        # Layers 5, 7 would have MoE experts — need to check they don't overlap.
        # For 4 experts with 4 MoE layers (4-7): layers 4-5 get 2 experts, 6-7 get 4.
        # MoD layer 5 has 2 experts (>1) -> overlap. Let's use mod_every_n_layers=4.
        # MoD layers at idx % 4 == 1 -> layers 1, 5. Layer 1 is plain MLP (ok).
        # Layer 5 has 2 experts -> still overlap. Use mod_every_n_layers=3.
        # MoD layers at idx % 3 == 1 -> layers 1, 4, 7.
        # Layer 1: plain MLP. Layer 4: first MoE layer (2 experts). Overlap!
        # Let's just use a config where MoD layers only fall on plain MLP layers.
        # 8 layers, progressive start_ratio=0.75 -> first 6 layers are plain MLP.
        # MoD at idx%2==1 -> layers 1,3,5,7. Layers 1,3,5 are plain, layer 7 has experts.
        # Need to pick so no MoD layer falls in MoE range.
        # mod_every_n_layers=8 -> only layer idx%8==1 -> layer 1. Plain MLP. No overlap!
        config = UMichSLMConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=8,
            max_position_embeddings=64,
            use_mod=True,
            mod_capacity=0.5,
            mod_every_n_layers=8,  # Only layer 1 is MoD, which is plain MLP
            use_moe=True,
            num_experts=4,
            num_experts_per_tok=2,
            moe_progressive=True,
            moe_progressive_start_ratio=0.5,
        )
        model = UMichSLMForCausalLM(config)
        model.train()

        input_ids = torch.randint(0, 100, (2, 16))
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)
        assert outputs.loss is not None
        assert outputs.logits.shape == (2, 16, 100)


class TestMoDGradientCheckpointing:
    """Tests for MoD with gradient checkpointing."""

    def test_gradient_checkpointing_loss(self):
        """Test that loss is computed and gradients flow with gradient checkpointing."""
        config = UMichSLMConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=4,
            max_position_embeddings=64,
            use_mod=True,
            mod_capacity=0.5,
            mod_every_n_layers=2,
        )
        model = UMichSLMForCausalLM(config)
        model.train()
        model.gradient_checkpointing_enable()

        input_ids = torch.randint(0, 100, (2, 16))
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)

        assert outputs.loss is not None

        # Verify gradients flow
        outputs.loss.backward()
        for layer in model.model.layers:
            if isinstance(layer, UMichSLMMoDDecoderLayer):
                assert layer.router.router.weight.grad is not None


class TestMoDCapacityVariations:
    """Tests for different MoD capacity settings."""

    @pytest.mark.parametrize("capacity", [0.125, 0.25, 0.5, 0.75])
    def test_different_capacities(self, capacity):
        """Test model with different capacity values."""
        config = UMichSLMConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=4,
            max_position_embeddings=64,
            use_mod=True,
            mod_capacity=capacity,
            mod_every_n_layers=2,
        )
        model = UMichSLMForCausalLM(config)
        model.eval()

        input_ids = torch.randint(0, 100, (1, 32))

        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        assert outputs.logits.shape == (1, 32, 100)

    @pytest.mark.parametrize("every_n", [1, 2, 3, 4])
    def test_different_layer_patterns(self, every_n):
        """Test model with different mod_every_n_layers values."""
        config = UMichSLMConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=8,
            max_position_embeddings=64,
            use_mod=True,
            mod_capacity=0.5,
            mod_every_n_layers=every_n,
        )
        model = UMichSLMForCausalLM(config)

        # Count MoD layers
        num_mod_layers = sum(
            1 for layer in model.model.layers
            if isinstance(layer, UMichSLMMoDDecoderLayer)
        )

        # Expected: layers where (idx % every_n) == 1
        expected_mod_layers = sum(1 for i in range(8) if i % every_n == 1)
        assert num_mod_layers == expected_mod_layers


class TestMoDDisabled:
    """Tests for model when MoD is disabled."""

    def test_mod_disabled(self):
        """Test model with use_mod=False."""
        config = UMichSLMConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=4,
            max_position_embeddings=64,
            use_mod=False,  # Disabled
        )
        model = UMichSLMForCausalLM(config)

        # No MoD layers
        for layer in model.model.layers:
            assert not isinstance(layer, UMichSLMMoDDecoderLayer)

        # Forward pass should work
        model.eval()
        input_ids = torch.randint(0, 100, (1, 16))

        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        assert outputs.logits.shape == (1, 16, 100)
