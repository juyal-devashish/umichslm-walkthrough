"""Tests for Multi-Token Prediction (MTP) module."""

import pytest
import torch

from umich_slm import UMichSLMConfig, UMichSLMForCausalLM
from umich_slm.modules.multi_token_prediction import (
    MultiTokenPrediction,
    MTPModule,
    MTPProjection,
    MTPTransformerBlock,
    MTPOutput,
)


class TestMTPProjection:
    """Tests for MTP projection layer."""

    def test_projection_shape(self):
        """Test projection output shape."""
        hidden_size = 64
        batch_size = 2
        seq_len = 10

        proj = MTPProjection(hidden_size)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        embeddings = torch.randn(batch_size, seq_len, hidden_size)

        output = proj(hidden_states, embeddings)

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_projection_gradients(self):
        """Test gradients flow through projection."""
        hidden_size = 32
        proj = MTPProjection(hidden_size)

        hidden_states = torch.randn(2, 5, hidden_size, requires_grad=True)
        embeddings = torch.randn(2, 5, hidden_size, requires_grad=True)

        output = proj(hidden_states, embeddings)
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert embeddings.grad is not None
        assert proj.proj.weight.grad is not None


class TestMTPTransformerBlock:
    """Tests for MTP transformer block."""

    def test_block_output_shape(self):
        """Test transformer block preserves shape."""
        hidden_size = 64
        intermediate_size = 128
        batch_size = 2
        seq_len = 10

        block = MTPTransformerBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_attention=False,
        )

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        output = block(hidden_states)

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_block_with_attention(self):
        """Test transformer block with attention enabled."""
        hidden_size = 64
        intermediate_size = 128
        batch_size = 2
        seq_len = 10

        block = MTPTransformerBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_attention=True,
        )

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        output = block(hidden_states)

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_block_gradients(self):
        """Test gradients flow through block."""
        hidden_size = 32
        block = MTPTransformerBlock(
            hidden_size=hidden_size,
            intermediate_size=64,
        )

        hidden_states = torch.randn(2, 5, hidden_size, requires_grad=True)
        output = block(hidden_states)
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None


class TestMTPModule:
    """Tests for individual MTP module."""

    @pytest.fixture
    def mtp_module(self):
        return MTPModule(
            hidden_size=64,
            intermediate_size=128,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            num_layers=1,
        )

    def test_module_output_shape(self, mtp_module):
        """Test MTP module output shape."""
        batch_size = 2
        seq_len = 10
        hidden_size = 64

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        token_embeddings = torch.randn(batch_size, seq_len, hidden_size)

        output = mtp_module(hidden_states, token_embeddings)

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_module_gradients(self, mtp_module):
        """Test gradients flow through MTP module."""
        batch_size = 2
        seq_len = 10
        hidden_size = 64

        hidden_states = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        token_embeddings = torch.randn(batch_size, seq_len, hidden_size)

        output = mtp_module(hidden_states, token_embeddings)
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None


class TestMultiTokenPrediction:
    """Tests for complete MTP system."""

    @pytest.fixture
    def mtp_config(self):
        return UMichSLMConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            use_mtp=True,
            mtp_num_heads=2,  # Predict 2 additional tokens
            mtp_loss_weight=0.3,
            mtp_num_layers=1,
            mtp_share_head=True,
        )

    @pytest.fixture
    def embedding_layer(self, mtp_config):
        return torch.nn.Embedding(
            mtp_config.vocab_size,
            mtp_config.hidden_size,
        )

    @pytest.fixture
    def lm_head(self, mtp_config):
        return torch.nn.Linear(
            mtp_config.hidden_size,
            mtp_config.vocab_size,
            bias=False,
        )

    def test_mtp_initialization(self, mtp_config, embedding_layer, lm_head):
        """Test MTP module initialization."""
        mtp = MultiTokenPrediction(
            config=mtp_config,
            embed_tokens=embedding_layer,
            lm_head=lm_head,
        )

        assert mtp.num_heads == 2
        assert len(mtp.mtp_modules) == 2
        assert mtp.loss_weight == 0.3

    def test_mtp_forward_returns_loss(self, mtp_config, embedding_layer, lm_head):
        """Test MTP forward returns loss."""
        mtp = MultiTokenPrediction(
            config=mtp_config,
            embed_tokens=embedding_layer,
            lm_head=lm_head,
        )

        batch_size = 2
        seq_len = 20
        hidden_states = torch.randn(batch_size, seq_len, mtp_config.hidden_size)
        input_ids = torch.randint(0, mtp_config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        output = mtp(hidden_states, input_ids, labels)

        assert isinstance(output, MTPOutput)
        assert output.loss is not None
        assert output.loss.numel() == 1

    def test_mtp_returns_per_depth_losses(self, mtp_config, embedding_layer, lm_head):
        """Test MTP returns losses per depth."""
        mtp = MultiTokenPrediction(
            config=mtp_config,
            embed_tokens=embedding_layer,
            lm_head=lm_head,
        )

        batch_size = 2
        seq_len = 20
        hidden_states = torch.randn(batch_size, seq_len, mtp_config.hidden_size)
        input_ids = torch.randint(0, mtp_config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        output = mtp(hidden_states, input_ids, labels)

        assert output.losses_per_depth is not None
        assert len(output.losses_per_depth) == 2  # 2 prediction depths

    def test_mtp_gradients(self, mtp_config, embedding_layer, lm_head):
        """Test gradients flow through MTP."""
        mtp = MultiTokenPrediction(
            config=mtp_config,
            embed_tokens=embedding_layer,
            lm_head=lm_head,
        )

        batch_size = 2
        seq_len = 20
        hidden_states = torch.randn(batch_size, seq_len, mtp_config.hidden_size, requires_grad=True)
        input_ids = torch.randint(0, mtp_config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        output = mtp(hidden_states, input_ids, labels)
        output.loss.backward()

        assert hidden_states.grad is not None

    def test_mtp_with_independent_heads(self, mtp_config, embedding_layer):
        """Test MTP with independent (non-shared) heads."""
        mtp_config.mtp_share_head = False
        mtp = MultiTokenPrediction(
            config=mtp_config,
            embed_tokens=embedding_layer,
            lm_head=None,
        )

        assert mtp.lm_heads is not None
        assert len(mtp.lm_heads) == 2

        batch_size = 2
        seq_len = 20
        hidden_states = torch.randn(batch_size, seq_len, mtp_config.hidden_size)
        input_ids = torch.randint(0, mtp_config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        output = mtp(hidden_states, input_ids, labels)
        assert output.loss is not None


class TestMTPInModel:
    """Tests for MTP integrated with the full model."""

    @pytest.fixture
    def mtp_model_config(self):
        return UMichSLMConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            use_mtp=True,
            mtp_num_heads=1,
            mtp_loss_weight=0.3,
        )

    @pytest.fixture
    def mtp_model(self, mtp_model_config):
        return UMichSLMForCausalLM(mtp_model_config)

    def test_model_with_mtp_enabled(self, mtp_model):
        """Test model initializes with MTP."""
        assert mtp_model.use_mtp == True
        assert mtp_model.mtp is not None

    def test_mtp_forward_training(self, mtp_model):
        """Test MTP loss is computed during training."""
        mtp_model.train()
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = input_ids.clone()

        outputs = mtp_model(input_ids, labels=labels)

        assert outputs.loss is not None
        assert outputs.lm_loss is not None
        assert outputs.mtp_loss is not None
        # Total loss should be LM loss + MTP loss
        assert outputs.loss > outputs.lm_loss

    def test_mtp_not_computed_during_eval(self, mtp_model):
        """Test MTP loss is not computed during evaluation."""
        mtp_model.eval()
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = input_ids.clone()

        with torch.no_grad():
            outputs = mtp_model(input_ids, labels=labels)

        assert outputs.loss is not None
        assert outputs.lm_loss is not None
        # MTP loss should be None during eval
        assert outputs.mtp_loss is None

    def test_mtp_model_generation(self, mtp_model):
        """Test generation works with MTP model."""
        mtp_model.eval()
        input_ids = torch.randint(0, 1000, (1, 10))

        with torch.no_grad():
            outputs = mtp_model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=0,
            )

        assert outputs.shape[1] == 20  # 10 + 10

    def test_mtp_gradient_flow(self, mtp_model):
        """Test gradients flow through MTP during training."""
        mtp_model.train()
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = input_ids.clone()

        outputs = mtp_model(input_ids, labels=labels)
        outputs.loss.backward()

        # Check gradients in MTP modules
        for mtp_module in mtp_model.mtp.mtp_modules:
            assert mtp_module.projection.proj.weight.grad is not None

    def test_mtp_config_preserved_on_save_load(self, mtp_model, mtp_model_config, tmp_path):
        """Test MTP config is preserved after save/load."""
        save_path = tmp_path / "mtp_model"
        mtp_model.save_pretrained(save_path)

        loaded_model = UMichSLMForCausalLM.from_pretrained(save_path)

        assert loaded_model.config.use_mtp == mtp_model_config.use_mtp
        assert loaded_model.config.mtp_num_heads == mtp_model_config.mtp_num_heads
        assert loaded_model.config.mtp_loss_weight == mtp_model_config.mtp_loss_weight
        assert loaded_model.use_mtp == True
        assert loaded_model.mtp is not None


class TestMTPWithMoE:
    """Tests for MTP combined with MoE."""

    @pytest.fixture
    def mtp_moe_config(self):
        return UMichSLMConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            use_moe=True,
            num_experts=4,
            num_experts_per_tok=2,
            use_mtp=True,
            mtp_num_heads=1,
            mtp_loss_weight=0.3,
        )

    @pytest.fixture
    def mtp_moe_model(self, mtp_moe_config):
        return UMichSLMForCausalLM(mtp_moe_config)

    def test_mtp_with_moe_forward(self, mtp_moe_model):
        """Test forward pass with both MTP and MoE."""
        mtp_moe_model.train()
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = input_ids.clone()

        outputs = mtp_moe_model(input_ids, labels=labels)

        assert outputs.loss is not None
        assert outputs.lm_loss is not None
        assert outputs.mtp_loss is not None
        # Should have aux_loss from MoE
        assert outputs.aux_loss is not None

    def test_mtp_with_moe_all_losses_contribute(self, mtp_moe_model):
        """Test all losses contribute to total loss."""
        mtp_moe_model.train()
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = input_ids.clone()

        outputs = mtp_moe_model(input_ids, labels=labels)

        # Total loss should be greater than just LM loss
        assert outputs.loss > outputs.lm_loss


class TestMTPEdgeCases:
    """Tests for MTP edge cases and proper handling."""

    @pytest.fixture
    def mtp_config(self):
        return UMichSLMConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            use_mtp=True,
            mtp_num_heads=2,
            mtp_loss_weight=0.3,
            mtp_num_layers=1,
            mtp_share_head=True,
        )

    @pytest.fixture
    def embedding_layer(self, mtp_config):
        return torch.nn.Embedding(mtp_config.vocab_size, mtp_config.hidden_size)

    @pytest.fixture
    def lm_head(self, mtp_config):
        return torch.nn.Linear(mtp_config.hidden_size, mtp_config.vocab_size, bias=False)

    def test_mtp_with_ignore_index_labels(self, mtp_config, embedding_layer, lm_head):
        """Test MTP handles labels with -100 (ignore index) correctly."""
        mtp = MultiTokenPrediction(
            config=mtp_config,
            embed_tokens=embedding_layer,
            lm_head=lm_head,
        )

        batch_size = 2
        seq_len = 20
        hidden_states = torch.randn(batch_size, seq_len, mtp_config.hidden_size)
        input_ids = torch.randint(0, mtp_config.vocab_size, (batch_size, seq_len))

        # Create labels with some -100 values (ignore positions)
        labels = input_ids.clone()
        labels[:, :3] = -100  # Ignore first 3 tokens
        labels[:, -2:] = -100  # Ignore last 2 tokens

        output = mtp(hidden_states, input_ids, labels)

        assert output.loss is not None
        assert not torch.isnan(output.loss)
        assert output.num_active_depths == 2

    def test_mtp_input_ids_different_from_labels(self, mtp_config, embedding_layer, lm_head):
        """Test MTP works when input_ids and labels differ (the main use case)."""
        mtp = MultiTokenPrediction(
            config=mtp_config,
            embed_tokens=embedding_layer,
            lm_head=lm_head,
        )

        batch_size = 2
        seq_len = 20
        hidden_states = torch.randn(batch_size, seq_len, mtp_config.hidden_size)

        # input_ids are actual tokens (no -100)
        input_ids = torch.randint(0, mtp_config.vocab_size, (batch_size, seq_len))

        # labels have -100 for padding positions
        labels = input_ids.clone()
        labels[:, -5:] = -100  # Last 5 positions are padding

        output = mtp(hidden_states, input_ids, labels)

        assert output.loss is not None
        assert not torch.isnan(output.loss)
        # Loss should be finite (no CUDA errors from -100 in embeddings)
        assert torch.isfinite(output.loss)

    def test_mtp_short_sequence_length_2(self, mtp_config, embedding_layer, lm_head):
        """Test MTP with seq_len=2 (too short for MTP)."""
        mtp = MultiTokenPrediction(
            config=mtp_config,
            embed_tokens=embedding_layer,
            lm_head=lm_head,
        )

        batch_size = 2
        seq_len = 2  # Too short!
        hidden_states = torch.randn(batch_size, seq_len, mtp_config.hidden_size)
        input_ids = torch.randint(0, mtp_config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        output = mtp(hidden_states, input_ids, labels)

        # Should return zero loss without error
        assert output.loss is not None
        assert output.loss.item() == 0.0
        assert output.num_active_depths == 0
        assert len(output.losses_per_depth) == 0
        assert len(output.logits_per_depth) == 0

    def test_mtp_short_sequence_length_3(self, mtp_config, embedding_layer, lm_head):
        """Test MTP with seq_len=3 (minimum for depth 0)."""
        mtp = MultiTokenPrediction(
            config=mtp_config,
            embed_tokens=embedding_layer,
            lm_head=lm_head,
        )

        batch_size = 2
        seq_len = 3  # Minimum for depth 0
        hidden_states = torch.randn(batch_size, seq_len, mtp_config.hidden_size)
        input_ids = torch.randint(0, mtp_config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        output = mtp(hidden_states, input_ids, labels)

        # Should have exactly 1 active depth (depth 0 only)
        assert output.loss is not None
        assert output.num_active_depths == 1
        assert len(output.losses_per_depth) == 1

    def test_mtp_num_active_depths_output(self, mtp_config, embedding_layer, lm_head):
        """Test num_active_depths is correctly reported."""
        mtp = MultiTokenPrediction(
            config=mtp_config,
            embed_tokens=embedding_layer,
            lm_head=lm_head,
        )

        batch_size = 2
        seq_len = 20  # Long enough for both depths
        hidden_states = torch.randn(batch_size, seq_len, mtp_config.hidden_size)
        input_ids = torch.randint(0, mtp_config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        # Use output_logits=True to get logits_per_depth
        output = mtp(hidden_states, input_ids, labels, output_logits=True)

        # With 2 MTP heads and seq_len=20, both should be active
        assert output.num_active_depths == 2
        assert len(output.losses_per_depth) == 2
        assert len(output.logits_per_depth) == 2

    def test_mtp_loss_on_computation_graph(self, mtp_config, embedding_layer, lm_head):
        """Test MTP loss stays on computation graph for short sequences."""
        mtp = MultiTokenPrediction(
            config=mtp_config,
            embed_tokens=embedding_layer,
            lm_head=lm_head,
        )

        batch_size = 2
        seq_len = 2  # Too short
        hidden_states = torch.randn(batch_size, seq_len, mtp_config.hidden_size, requires_grad=True)
        input_ids = torch.randint(0, mtp_config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        output = mtp(hidden_states, input_ids, labels)

        # Even with zero loss, it should be differentiable
        # (no error when calling backward)
        combined_loss = hidden_states.sum() + output.loss
        combined_loss.backward()

        assert hidden_states.grad is not None

    def test_mtp_backward_compat_old_api(self, mtp_config, embedding_layer, lm_head):
        """Test backward compatibility with old API (deprecated but supported)."""
        import warnings

        mtp = MultiTokenPrediction(
            config=mtp_config,
            embed_tokens=embedding_layer,
            lm_head=lm_head,
        )

        batch_size = 2
        seq_len = 20
        hidden_states = torch.randn(batch_size, seq_len, mtp_config.hidden_size)
        labels = torch.randint(0, mtp_config.vocab_size, (batch_size, seq_len))

        # Old API: mtp(hidden_states, labels) - should emit deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = mtp(hidden_states, labels)

            # Should have emitted a DeprecationWarning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

        # But should still work
        assert output.loss is not None
        assert output.num_active_depths == 2


class TestMTPDisabled:
    """Tests for model without MTP."""

    @pytest.fixture
    def regular_config(self):
        return UMichSLMConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            use_mtp=False,  # Disabled
        )

    @pytest.fixture
    def regular_model(self, regular_config):
        return UMichSLMForCausalLM(regular_config)

    def test_no_mtp_by_default(self, regular_model):
        """Test MTP is disabled by default."""
        assert regular_model.use_mtp == False
        assert regular_model.mtp is None

    def test_regular_forward_works(self, regular_model):
        """Test forward works without MTP."""
        regular_model.train()
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = input_ids.clone()

        # With labels, the loss is computed chunk-wise and the full (B, T, V)
        # logits tensor is deliberately never materialized -> logits is None.
        outputs = regular_model(input_ids, labels=labels)
        assert outputs.loss is not None
        assert outputs.logits is None

        # ...unless the caller explicitly pays for them.
        outputs = regular_model(input_ids, labels=labels, output_logits=True)
        assert outputs.loss is not None
        assert outputs.logits.shape == (2, 32, 1000)

        # No labels -> logits always returned.
        outputs = regular_model(input_ids)
        assert outputs.logits.shape == (2, 32, 1000)


class TestCandidateMTPTwins:
    """The `<arch>_mtp` sweep twins of the five candidate bank-arms.

    `_sweep_arch_flags` grows an MTP twin for ANY arm via the `_mtp` suffix, so
    the pair differs by exactly one thing: the MTP heads. Pinned here because a
    silently-inert MTP flag would make the "does MTP pay for itself" comparison
    meaningless (the knob-reached-the-model rule in CLAUDE.md).
    """

    BASES = [
        "swa_hybrid32",
        "swa_deltabank",
        "swa_scribe",
        "swa_scribe_mopd",
        "swa_scribe_mopd_delta",
    ]

    @pytest.mark.parametrize("base", BASES)
    def test_mtp_flags_and_matched_shape(self, base):
        from umich_slm.configs.presets import get_umichslm_config

        cb = get_umichslm_config(f"{base}_25m")
        cm = get_umichslm_config(f"{base}_mtp_25m")
        assert not getattr(cb, "use_mtp", False)  # base twin has MTP OFF
        assert cm.use_mtp and cm.mtp_num_heads == 1 and cm.mtp_share_head
        assert cm.mtp_loss_weight == pytest.approx(0.3)
        # Identical backbone: the twin differs ONLY by the MTP heads.
        for k in (
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "swa_window",
            "swa_global_count",
        ):
            assert getattr(cb, k) == getattr(cm, k), k
        # Bank/backbone mechanism flags are carried through unchanged.
        for k in ("use_delta_bank", "use_scribe_bank", "use_pathways", "use_mod"):
            assert getattr(cb, k, False) == getattr(cm, k, False), k

    def test_mtp_module_present_and_trains(self):
        from umich_slm.configs.presets import get_umichslm_config

        cfg = get_umichslm_config("swa_hybrid32_mtp_25m", vocab_size=256)
        m = UMichSLMForCausalLM(cfg)
        assert any("mtp" in n.lower() for n, _ in m.named_modules())
        ids = torch.randint(0, 256, (1, 64))
        out = m(input_ids=ids, labels=ids.clone())
        assert torch.isfinite(out.loss)
        out.loss.backward()
        mtp_grad = sum(
            float(p.grad.abs().sum())
            for n, p in m.named_parameters()
            if "mtp" in n.lower() and p.grad is not None
        )
        assert mtp_grad > 0.0  # MTP heads actually receive gradient
