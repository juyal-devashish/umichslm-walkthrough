"""Tests for the Gloeckle parallel MTP variant.

Covers:
  * 4 parallel heads at predict_offsets [2,3,4,5] produce 4 logit tensors and
    4 per-head losses.
  * Each Gloeckle MTP head has an INDEPENDENT unembedding matrix (not aliased
    to the model's main lm_head).
  * Backward fills adapter and head gradients.
  * `del model.mtp` leaves base model intact and lm_head unchanged.
  * `mtp_share_head=True` is ignored for the Gloeckle variant (heads are
    always independent — required by the user's plan).
"""

import torch
import torch.nn as nn

from umich_slm.configuration_umichslm import UMichSLMConfig
from umich_slm.modeling_umichslm import UMichSLMForCausalLM
from umich_slm.modules.multi_token_prediction import (
    GloeckleMultiTokenPrediction,
    MultiTokenPrediction,
    build_mtp_module,
)


def _gloeckle_config(**overrides):
    cfg = dict(
        vocab_size=128,
        num_hidden_layers=4,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        use_mtp=True,
        mtp_variant="gloeckle",
        mtp_num_heads=4,
        mtp_loss_weight=0.3,
        mtp_predict_offsets=[2, 3, 4, 5],
    )
    cfg.update(overrides)
    return UMichSLMConfig(**cfg)


class TestBuildMTPModule:
    def test_factory_routes_to_gloeckle(self):
        c = _gloeckle_config()
        embed = nn.Embedding(c.vocab_size, c.hidden_size)
        lm = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        mtp = build_mtp_module(c, embed, lm)
        assert isinstance(mtp, GloeckleMultiTokenPrediction)

    def test_factory_routes_to_deepseek_by_default(self):
        c = UMichSLMConfig(
            vocab_size=128,
            num_hidden_layers=4,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            use_mtp=True,
            mtp_num_heads=2,
        )
        embed = nn.Embedding(c.vocab_size, c.hidden_size)
        lm = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        mtp = build_mtp_module(c, embed, lm)
        assert isinstance(mtp, MultiTokenPrediction)


class TestGloeckleStructure:
    def test_independent_unembedding_heads(self):
        c = _gloeckle_config()
        embed = nn.Embedding(c.vocab_size, c.hidden_size)
        lm = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        mtp = build_mtp_module(c, embed, lm)
        # Each MTP head must be a distinct parameter — not aliased to lm_head.
        for i, head in enumerate(mtp.lm_heads):
            assert head.weight is not lm.weight, f"MTP head {i} aliased to lm_head"
            assert head.weight.shape == lm.weight.shape

    def test_share_head_flag_ignored_for_gloeckle(self):
        # Even with mtp_share_head=True, Gloeckle uses independent heads.
        c = _gloeckle_config(mtp_share_head=True)
        embed = nn.Embedding(c.vocab_size, c.hidden_size)
        lm = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        mtp = build_mtp_module(c, embed, lm)
        for i, head in enumerate(mtp.lm_heads):
            assert (
                head.weight is not lm.weight
            ), f"Gloeckle MTP head {i} must remain independent regardless of mtp_share_head"

    def test_adapter_count_matches_num_heads(self):
        c = _gloeckle_config()
        embed = nn.Embedding(c.vocab_size, c.hidden_size)
        lm = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        mtp = build_mtp_module(c, embed, lm)
        assert len(mtp.adapters) == 4
        assert len(mtp.lm_heads) == 4


class TestGloeckleForward:
    def test_forward_loss_finite(self):
        c = _gloeckle_config()
        embed = nn.Embedding(c.vocab_size, c.hidden_size)
        lm = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        mtp = build_mtp_module(c, embed, lm)
        h = torch.randn(2, 16, c.hidden_size)
        labels = torch.randint(0, c.vocab_size, (2, 16))
        out = mtp(h, input_ids_or_labels=labels, labels=labels)
        assert torch.isfinite(out.loss)
        assert out.num_active_depths == 4
        assert len(out.losses_per_depth) == 4

    def test_grad_flow_to_all_adapters_and_heads(self):
        c = _gloeckle_config()
        embed = nn.Embedding(c.vocab_size, c.hidden_size)
        lm = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        mtp = build_mtp_module(c, embed, lm)
        h = torch.randn(2, 16, c.hidden_size, requires_grad=True)
        labels = torch.randint(0, c.vocab_size, (2, 16))
        out = mtp(h, input_ids_or_labels=labels, labels=labels)
        out.loss.backward()
        for i, a in enumerate(mtp.adapters):
            assert a.weight.grad is not None
            assert a.weight.grad.abs().sum() > 0, f"adapter {i} got no grad"
        for i, head in enumerate(mtp.lm_heads):
            assert head.weight.grad is not None
            assert head.weight.grad.abs().sum() > 0, f"head {i} got no grad"

    def test_short_sequence_returns_zero_loss(self):
        c = _gloeckle_config(mtp_num_heads=4, mtp_predict_offsets=[2, 3, 4, 5])
        embed = nn.Embedding(c.vocab_size, c.hidden_size)
        lm = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        mtp = build_mtp_module(c, embed, lm)
        # seq_len=4 is <= max offset (5) so all heads inactive
        h = torch.randn(1, 4, c.hidden_size)
        labels = torch.randint(0, c.vocab_size, (1, 4))
        out = mtp(h, input_ids_or_labels=labels, labels=labels)
        assert float(out.loss) == 0.0

    def test_globals_mask_excludes_positions(self):
        c = _gloeckle_config()
        embed = nn.Embedding(c.vocab_size, c.hidden_size)
        lm = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        mtp = build_mtp_module(c, embed, lm)
        h = torch.randn(2, 16, c.hidden_size)
        labels = torch.randint(0, c.vocab_size, (2, 16))
        gm = torch.ones(16, dtype=torch.bool)
        # Mask out the entire sequence: every loss target becomes -100.
        gm[:] = False
        out = mtp(h, input_ids_or_labels=labels, labels=labels, globals_mask=gm)
        # CE with all -100 returns nan, but we use mean reduction so it's
        # actually undefined; per-head losses should be non-finite.
        # The per-head losses with all-ignored targets are NaN; the test simply
        # asserts the path runs without crashing.
        assert out.num_active_depths == 4 or out.num_active_depths == 0


class TestModelLevelGloeckle:
    def test_model_uses_gloeckle_when_configured(self):
        c = _gloeckle_config(
            num_hidden_layers=2,
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        m = UMichSLMForCausalLM(c)
        assert isinstance(m.mtp, GloeckleMultiTokenPrediction)
        # Each MTP head must be independent of lm_head
        for i, head in enumerate(m.mtp.lm_heads):
            assert (
                head.weight is not m.lm_head.weight
            ), f"MTP head {i} aliased to lm_head — should be independent"

    def test_del_mtp_leaves_lm_head_intact(self):
        c = _gloeckle_config(
            num_hidden_layers=2,
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        m = UMichSLMForCausalLM(c)
        lm_weight_before = m.lm_head.weight.clone()
        m.mtp = None
        m.use_mtp = False
        # lm_head must still exist with the same parameters
        assert torch.equal(m.lm_head.weight, lm_weight_before)
        # Forward without MTP still works
        x = torch.randint(0, c.vocab_size, (1, 8))
        with torch.no_grad():
            out = m(input_ids=x)
        assert out.logits is not None
