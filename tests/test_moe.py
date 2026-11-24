"""Tests for Mixture of Experts (MoE) implementation."""

import pytest
import torch
import torch.nn as nn

from umich_slm.modules.moe import (
    MoELayer,
    MoERouter,
    MoEOutput,
    SparseMoE,
    compute_moe_loss,
)
from umich_slm import UMichSLMConfig, UMichSLMForCausalLM


class TestMoERouter:
    """Tests for MoE router."""

    @pytest.fixture
    def router(self):
        return MoERouter(hidden_size=64, num_experts=4, top_k=2)

    def test_router_output_shape(self, router):
        """Test router output shapes."""
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, 64)

        weights, experts, logits = router(hidden_states)

        # Check shapes
        num_tokens = batch_size * seq_len
        assert weights.shape == (num_tokens, 2)  # top_k=2
        assert experts.shape == (num_tokens, 2)
        assert logits.shape == (num_tokens, 4)  # num_experts=4

    def test_router_weights_normalized(self, router):
        """Test that router weights sum to 1."""
        hidden_states = torch.randn(2, 10, 64)
        weights, _, _ = router(hidden_states)

        # Weights should sum to 1 for each token
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_router_expert_indices_valid(self, router):
        """Test that expert indices are valid."""
        hidden_states = torch.randn(2, 10, 64)
        _, experts, _ = router(hidden_states)

        # All expert indices should be in valid range
        assert (experts >= 0).all()
        assert (experts < 4).all()  # num_experts=4

    def test_router_jitter(self):
        """Test router with jitter noise."""
        router = MoERouter(hidden_size=64, num_experts=4, top_k=2, router_jitter=0.1)
        router.train()

        hidden_states = torch.randn(2, 10, 64)
        weights1, _, _ = router(hidden_states)
        weights2, _, _ = router(hidden_states)

        # With jitter, results should differ between calls
        assert not torch.allclose(weights1, weights2)


class TestMoELayer:
    """Tests for MoE layer."""

    @pytest.fixture
    def moe_layer(self):
        return MoELayer(
            hidden_size=64,
            intermediate_size=128,
            num_experts=4,
            num_experts_per_tok=2,
        )

    def test_output_shape(self, moe_layer):
        """Test MoE layer output shape."""
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, 64)

        output = moe_layer(hidden_states)

        assert isinstance(output, MoEOutput)
        assert output.hidden_states.shape == hidden_states.shape

    def test_aux_loss_positive(self, moe_layer):
        """Test that auxiliary loss is computed and positive."""
        hidden_states = torch.randn(2, 10, 64)
        output = moe_layer(hidden_states)

        assert output.aux_loss is not None
        assert output.aux_loss >= 0

    def test_router_z_loss_positive(self, moe_layer):
        """Test that router z-loss is computed and positive."""
        hidden_states = torch.randn(2, 10, 64)
        output = moe_layer(hidden_states)

        assert output.router_z_loss is not None
        assert output.router_z_loss >= 0

    def test_router_logits_shape(self, moe_layer):
        """Test router logits shape."""
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, 64)
        output = moe_layer(hidden_states)

        num_tokens = batch_size * seq_len
        assert output.router_logits.shape == (num_tokens, 4)  # num_experts=4

    def test_gradient_flow(self, moe_layer):
        """Test that gradients flow through MoE layer."""
        hidden_states = torch.randn(2, 10, 64, requires_grad=True)
        output = moe_layer(hidden_states)

        # Backward pass
        loss = output.hidden_states.sum() + output.aux_loss + output.router_z_loss
        loss.backward()

        assert hidden_states.grad is not None
        assert not torch.all(hidden_states.grad == 0)

    def test_different_num_experts(self):
        """Test MoE with different numbers of experts."""
        for num_experts in [2, 4, 8]:
            moe = MoELayer(
                hidden_size=64,
                intermediate_size=128,
                num_experts=num_experts,
                num_experts_per_tok=min(2, num_experts),
            )
            hidden_states = torch.randn(2, 10, 64)
            output = moe(hidden_states)
            assert output.hidden_states.shape == hidden_states.shape


class TestSparseMoE:
    """Tests for Sparse MoE with shared experts."""

    @pytest.fixture
    def sparse_moe(self):
        return SparseMoE(
            hidden_size=64,
            intermediate_size=128,
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
        )

    def test_output_shape(self, sparse_moe):
        """Test SparseMoE output shape."""
        hidden_states = torch.randn(2, 10, 64)
        output = sparse_moe(hidden_states)

        assert isinstance(output, MoEOutput)
        assert output.hidden_states.shape == hidden_states.shape

    def test_shared_expert_contribution(self):
        """Test that shared expert adds to output."""
        # Create two versions: with and without shared experts
        moe_no_shared = SparseMoE(
            hidden_size=64,
            intermediate_size=128,
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=0,
        )
        moe_with_shared = SparseMoE(
            hidden_size=64,
            intermediate_size=128,
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
        )

        torch.manual_seed(42)
        hidden_states = torch.randn(2, 10, 64)

        output1 = moe_no_shared(hidden_states)
        output2 = moe_with_shared(hidden_states)

        # Outputs should be different due to shared expert
        assert not torch.allclose(output1.hidden_states, output2.hidden_states)


class TestLoadBalancingLoss:
    """Tests for load balancing auxiliary loss."""

    def test_balanced_routing_low_loss(self):
        """Test that balanced routing gives lower auxiliary loss."""
        moe = MoELayer(
            hidden_size=64,
            intermediate_size=128,
            num_experts=4,
            num_experts_per_tok=2,
            router_aux_loss_coef=1.0,  # Use larger coefficient for clearer signal
        )

        # Large batch to get better statistics
        hidden_states = torch.randn(8, 32, 64)
        output = moe(hidden_states)

        # The loss should be bounded and reasonable
        # Perfect balance would give aux_loss ≈ num_experts * (1/num_experts)^2 * num_experts = 1
        assert output.aux_loss > 0
        assert output.aux_loss < 10  # Reasonable upper bound

    def test_compute_moe_loss_function(self):
        """Test the compute_moe_loss helper function."""
        moe = MoELayer(
            hidden_size=64,
            intermediate_size=128,
            num_experts=4,
            num_experts_per_tok=2,
        )

        hidden_states = torch.randn(2, 10, 64)
        output1 = moe(hidden_states)
        output2 = moe(hidden_states)

        # Compute total loss from multiple layers
        total_loss = compute_moe_loss([output1, output2])

        expected = (
            output1.aux_loss + output1.router_z_loss + output2.aux_loss + output2.router_z_loss
        )
        assert torch.allclose(total_loss, expected)


class TestMoEModel:
    """Tests for full MoE model."""

    @pytest.fixture
    def moe_config(self):
        return UMichSLMConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            use_moe=True,
            num_experts=4,
            num_experts_per_tok=2,
            router_aux_loss_coef=0.01,
            router_z_loss_coef=0.001,
        )

    @pytest.fixture
    def moe_model(self, moe_config):
        return UMichSLMForCausalLM(moe_config)

    def test_moe_model_forward(self, moe_model):
        """Test MoE model forward pass."""
        input_ids = torch.randint(0, 1000, (2, 10))
        outputs = moe_model(input_ids)

        assert outputs.logits is not None
        assert outputs.logits.shape == (2, 10, 1000)

    def test_moe_model_loss_includes_aux(self, moe_model):
        """Test that training loss includes auxiliary loss."""
        moe_model.train()
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 10))

        outputs = moe_model(input_ids, labels=labels, output_router_logits=True)

        assert outputs.loss is not None
        assert outputs.aux_loss is not None
        assert outputs.aux_loss > 0

    def test_moe_model_router_logits(self, moe_model):
        """Test that router logits are returned."""
        moe_model.train()
        input_ids = torch.randint(0, 1000, (2, 10))

        outputs = moe_model(input_ids, output_router_logits=True)

        assert outputs.router_logits is not None
        assert len(outputs.router_logits) == 2  # num_hidden_layers

    def test_moe_model_generation(self, moe_model):
        """Test generation with MoE model."""
        moe_model.eval()
        input_ids = torch.randint(0, 1000, (1, 5))

        with torch.no_grad():
            outputs = moe_model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
            )

        assert outputs.shape[1] > input_ids.shape[1]

    def test_moe_gradient_flow(self, moe_model):
        """Test that gradients flow through all MoE components."""
        moe_model.train()
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 10))

        outputs = moe_model(input_ids, labels=labels, output_router_logits=True)
        outputs.loss.backward()

        # Check that expert weights have gradients
        for layer in moe_model.model.layers:
            for expert in layer.moe.moe.experts if hasattr(layer.moe, "moe") else layer.moe.experts:
                assert expert.gate_proj.weight.grad is not None

    def test_moe_model_save_load(self, moe_model, moe_config, tmp_path):
        """Test saving and loading MoE model."""
        save_path = tmp_path / "moe_model"
        moe_model.save_pretrained(save_path)

        # Load model
        loaded_model = UMichSLMForCausalLM.from_pretrained(save_path)

        # Check config preserved
        assert loaded_model.config.use_moe == moe_config.use_moe
        assert loaded_model.config.num_experts == moe_config.num_experts

        # Check forward pass works
        input_ids = torch.randint(0, 1000, (1, 5))
        with torch.no_grad():
            original_logits = moe_model(input_ids).logits
            loaded_logits = loaded_model(input_ids).logits

        assert torch.allclose(original_logits, loaded_logits, atol=1e-5)


class TestMoEGradientCheckpointing:
    """Tests for MoE with gradient checkpointing."""

    def test_gradient_checkpointing_aux_loss(self):
        """Test that aux loss is computed when gradient checkpointing is enabled."""
        config = UMichSLMConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            use_moe=True,
            num_experts=4,
            num_experts_per_tok=2,
            router_aux_loss_coef=0.01,
            router_z_loss_coef=0.001,
        )
        model = UMichSLMForCausalLM(config)
        model.train()
        model.gradient_checkpointing_enable()

        input_ids = torch.randint(0, 100, (2, 16))
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels, output_router_logits=True)

        assert outputs.loss is not None
        assert outputs.aux_loss is not None
        assert outputs.aux_loss > 0

        # Verify gradients flow to router parameters
        outputs.loss.backward()
        for layer in model.model.layers:
            if hasattr(layer, "moe"):
                moe_module = layer.moe.moe if hasattr(layer.moe, "moe") else layer.moe
                assert moe_module.router.gate.weight.grad is not None


class TestProgressiveMoE:
    """Tests for progressive MoE schedule edge cases."""

    def test_8_expert_schedule(self):
        """Verify schedule for 8-expert progressive MoE."""
        config = UMichSLMConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=8,
            num_attention_heads=4,
            use_moe=True,
            num_experts=8,
            moe_progressive=True,
            moe_progressive_start_ratio=0.5,
        )
        schedule = config.get_progressive_moe_schedule()

        # First half should be plain MLP
        for i in range(4):
            assert schedule[i] == 1, f"Layer {i} should be 1 expert, got {schedule[i]}"

        # Second half should have increasing experts
        for i in range(4, 8):
            assert schedule[i] > 1, f"Layer {i} should have >1 expert, got {schedule[i]}"

        # Experts should be non-decreasing
        for i in range(4, 7):
            assert schedule[i] <= schedule[i + 1], (
                f"Expert count should be non-decreasing: layer {i}={schedule[i]}, "
                f"layer {i+1}={schedule[i+1]}"
            )

    def test_single_expert_layers_not_moe(self):
        """Verify single-expert layers in progressive MoE use standard decoder class."""
        config = UMichSLMConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=8,
            num_attention_heads=4,
            use_moe=True,
            num_experts=4,
            num_experts_per_tok=2,
            moe_progressive=True,
            moe_progressive_start_ratio=0.5,
        )
        model = UMichSLMForCausalLM(config)

        schedule = config.get_progressive_moe_schedule()
        for idx, layer in enumerate(model.model.layers):
            if schedule[idx] == 1:
                assert not hasattr(
                    layer, "moe"
                ), f"Layer {idx} has 1 expert but uses MoE decoder class"


class TestMoEMixedPrecision:
    """Tests for MoE with mixed precision."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_moe_fp16(self):
        """Test MoE with fp16 precision."""
        moe = (
            MoELayer(
                hidden_size=64,
                intermediate_size=128,
                num_experts=4,
                num_experts_per_tok=2,
            )
            .cuda()
            .half()
        )

        hidden_states = torch.randn(2, 10, 64, dtype=torch.float16, device="cuda")
        output = moe(hidden_states)

        assert output.hidden_states.dtype == torch.float16
