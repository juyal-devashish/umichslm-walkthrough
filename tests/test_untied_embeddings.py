"""Untied input/output embeddings invariant for stacking models.

The stacking pretraining presets explicitly use ``tie_word_embeddings=False``
so that the input ``embed_tokens`` and the output ``lm_head`` remain
independent matrices.

This test asserts:
  * UMichSLMConfig defaults to tie_word_embeddings=False already (it does at
    the time of writing — the existing __init__ signature default is False).
  * For the stacking attention modes, the model's ``get_input_embeddings()``
    and ``get_output_embeddings()`` weights are NOT aliased.
"""

import torch

from umich_slm.configuration_umichslm import UMichSLMConfig
from umich_slm.modeling_umichslm import UMichSLMForCausalLM


def _untied_config(**overrides):
    cfg = dict(
        vocab_size=128,
        num_hidden_layers=4,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        tie_word_embeddings=False,
    )
    cfg.update(overrides)
    return UMichSLMConfig(**cfg)


class TestUntiedEmbeddings:
    def test_default_tie_word_embeddings_is_false(self):
        c = UMichSLMConfig(num_hidden_layers=2, hidden_size=16, intermediate_size=32,
                           num_attention_heads=4, num_key_value_heads=4)
        # The PretrainedConfig stores it; default in __init__ signature is False.
        assert c.tie_word_embeddings is False

    def test_untied_embeddings_in_legacy_model(self):
        c = _untied_config()
        m = UMichSLMForCausalLM(c)
        assert m.get_input_embeddings().weight is not m.get_output_embeddings().weight

    def test_untied_embeddings_in_deltanet_mla(self):
        c = _untied_config(
            num_hidden_layers=8,
            hidden_size=64,
            intermediate_size=128,
            use_mla=True,
            use_deltanet=True,
            attention_mode="deltanet_mla",
            hybrid_attention_ratio=["delta", "delta", "delta", "mla"],
            mla_kv_lora_rank=32,
            mla_q_lora_rank=0,
            mla_qk_nope_head_dim=8,
            mla_qk_rope_head_dim=8,
            mla_v_head_dim=16,
        )
        m = UMichSLMForCausalLM(c)
        assert m.get_input_embeddings().weight is not m.get_output_embeddings().weight

    def test_untied_embeddings_in_swa_globals_mla(self):
        c = _untied_config(
            num_hidden_layers=8,
            hidden_size=64,
            intermediate_size=128,
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
        m = UMichSLMForCausalLM(c)
        assert m.get_input_embeddings().weight is not m.get_output_embeddings().weight

    def test_grad_flows_through_untied_lm_head(self):
        c = _untied_config()
        m = UMichSLMForCausalLM(c)
        m.train()
        x = torch.randint(0, c.vocab_size, (2, 8))
        out = m(input_ids=x, labels=x)
        out.loss.backward()
        # Both embeddings receive gradients (they are independent).
        assert m.get_input_embeddings().weight.grad is not None
        assert m.get_output_embeddings().weight.grad is not None
        assert m.get_input_embeddings().weight.grad.abs().sum() > 0
        assert m.get_output_embeddings().weight.grad.abs().sum() > 0
