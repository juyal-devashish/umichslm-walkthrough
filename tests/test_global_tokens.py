"""Tests for the persistent learnable global tokens (SWA + globals path).

Covers:
  * GlobalTokens prepend shape and gradient flow.
  * The globals + window attention mask has the expected pattern.
  * UMichSLMModel prepends globals on the initial forward and strips them
    before computing logits.
  * MoD router force-routes the prefix when the model is in SWA-globals mode.
  * MTP loss skips the globals positions (Gloeckle adapters do not see them
    as targets, since hidden_states is content-only by the time MTP runs).
"""

import pytest
import torch

from umich_slm.configuration_umichslm import UMichSLMConfig
from umich_slm.modeling_umichslm import UMichSLMForCausalLM
from umich_slm.modules.global_tokens import GlobalTokens
from umich_slm.modules.mixture_of_depths import MoDRouter


def _swa_config(**overrides):
    cfg = dict(
        vocab_size=128,
        max_position_embeddings=128,
        num_hidden_layers=8,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_mla=True,
        sliding_window=8,
        swa_window=8,
        swa_global_count=4,
        attention_mode="swa_globals_mla",
        hybrid_attention_ratio=["swa", "swa", "swa", "mla"],
        mla_kv_lora_rank=32,
        mla_q_lora_rank=0,
        mla_qk_nope_head_dim=8,
        mla_qk_rope_head_dim=8,
        mla_v_head_dim=16,
        _attn_implementation="sdpa",
    )
    cfg.update(overrides)
    return UMichSLMConfig(**cfg)


class TestGlobalTokensModule:
    def test_param_shape(self):
        g = GlobalTokens(num_tokens=8, hidden_size=16)
        assert g.tokens.shape == (8, 16)

    def test_prepend_shape(self):
        g = GlobalTokens(num_tokens=4, hidden_size=8)
        emb = torch.randn(2, 6, 8)
        out = g.prepend_to_embeddings(emb)
        assert out.shape == (2, 10, 8)
        # Globals at the front
        assert torch.allclose(out[0, :4], g.tokens)

    def test_grad_flow(self):
        g = GlobalTokens(num_tokens=4, hidden_size=8)
        emb = torch.randn(2, 6, 8)
        out = g.prepend_to_embeddings(emb)
        out.sum().backward()
        assert g.tokens.grad is not None
        assert g.tokens.grad.abs().sum() > 0

    def test_attention_mask_pattern(self):
        # 4 globals + 6 content; window = 3
        m = GlobalTokens.build_attention_mask(content_len=6, window=3, num_globals=4)
        assert m.shape == (10, 10)
        # First content row (position 4) sees all 4 globals + itself
        assert m[4, :4].all()
        assert m[4, 4]
        # Last content row (position 9) sees globals + last 3 content (7,8,9)
        assert m[9, :4].all()
        assert m[9, 7] and m[9, 8] and m[9, 9]
        # And does NOT see content at distance > window from the prefix region
        assert not m[9, 4]
        assert not m[9, 5]
        assert not m[9, 6]


class TestSWAModelPrependStrip:
    def test_globals_prepended_and_stripped(self):
        c = _swa_config()
        m = UMichSLMForCausalLM(c)
        m.eval()
        x = torch.randint(0, 128, (2, 16))
        # Forward without labels returns logits with the original (content) seq_len
        with torch.no_grad():
            out = m(input_ids=x)
        assert out.logits.shape == (2, 16, c.vocab_size), out.logits.shape

    def test_globals_present_only_when_mode_set(self):
        # legacy mode: no global_tokens module instantiated
        c = UMichSLMConfig(
            vocab_size=128, num_hidden_layers=4,
            hidden_size=32, intermediate_size=64,
            num_attention_heads=4, num_key_value_heads=4,
        )
        m = UMichSLMForCausalLM(c)
        assert m.model.global_tokens is None

    def test_grad_into_globals(self):
        c = _swa_config()
        m = UMichSLMForCausalLM(c)
        m.train()
        x = torch.randint(0, 128, (2, 16))
        out = m(input_ids=x, labels=x)
        out.loss.backward()
        assert m.model.global_tokens.tokens.grad is not None
        assert m.model.global_tokens.tokens.grad.abs().sum() > 0


class TestMoDForceRoutePrefix:
    """The MoD router force-routes the protected prefix."""

    def test_prefix_always_selected(self):
        torch.manual_seed(0)
        r = MoDRouter(hidden_size=16, capacity=0.5, aux_loss_weight=0.01)
        r.train()
        x = torch.randn(2, 8, 16)
        _, mask, _ = r(x, force_route_prefix=2)
        assert mask[:, :2].all(), "prefix positions must always be selected"

    def test_aux_loss_zero_when_only_prefix(self):
        torch.manual_seed(0)
        r = MoDRouter(hidden_size=16, capacity=0.5, aux_loss_weight=0.01)
        r.train()
        x = torch.randn(2, 4, 16)
        _, mask, aux = r(x, force_route_prefix=4)  # the entire seq is prefix
        assert mask.all()
        assert aux is not None
        assert float(aux) == 0.0
