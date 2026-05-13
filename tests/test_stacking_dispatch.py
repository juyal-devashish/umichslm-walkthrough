"""Tests for the per-layer attention dispatch in stacking attention modes.

Covers:
  * deltanet_mla 3:1 hybrid: ['delta','delta','delta','mla'] dispatches to
    GatedDeltaNet on the local layers and the MLA class on the global layer.
  * swa_globals_mla 3:1 hybrid: ['swa','swa','swa','mla'] dispatches to
    SWAWithGlobalsAttention and MLA.
  * legacy mode still rejects MLA + DeltaNet co-existence.
  * legacy hybrid (delta/attn) still works.
"""

import pytest

from umich_slm.configuration_umichslm import UMichSLMConfig
from umich_slm.modules.transformer_block import _get_attention_class


def _base_config(**overrides):
    cfg = dict(
        num_hidden_layers=8,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
    )
    cfg.update(overrides)
    return UMichSLMConfig(**cfg)


class TestDeltaNetMLAHybrid:
    """3:1 DeltaNet + MLA hybrid via attention_mode='deltanet_mla'."""

    def test_per_layer_dispatch(self):
        c = _base_config(
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
        # Layers 0,1,2 -> DeltaNet
        for layer_idx in (0, 1, 2, 4, 5, 6):
            cls = _get_attention_class(c, layer_idx)
            assert cls.__name__ == "GatedDeltaNet", (
                f"layer {layer_idx}: expected GatedDeltaNet, got {cls.__name__}"
            )
        # Layers 3, 7 -> MLA
        for layer_idx in (3, 7):
            cls = _get_attention_class(c, layer_idx)
            assert cls.__name__.startswith("UMichSLMMLA"), (
                f"layer {layer_idx}: expected MLA, got {cls.__name__}"
            )


class TestSWAGlobalsMLAHybrid:
    """3:1 SWA + globals + MLA hybrid via attention_mode='swa_globals_mla'."""

    def test_per_layer_dispatch(self):
        c = _base_config(
            use_mla=True,
            sliding_window=512,
            swa_global_count=16,
            swa_window=64,
            attention_mode="swa_globals_mla",
            hybrid_attention_ratio=["swa", "swa", "swa", "mla"],
            mla_kv_lora_rank=32,
            mla_q_lora_rank=0,
            mla_qk_nope_head_dim=8,
            mla_qk_rope_head_dim=8,
            mla_v_head_dim=16,
        )
        for layer_idx in (0, 1, 2, 4, 5, 6):
            cls = _get_attention_class(c, layer_idx)
            assert cls.__name__ == "SWAWithGlobalsAttention", (
                f"layer {layer_idx}: expected SWAWithGlobalsAttention, got {cls.__name__}"
            )
        for layer_idx in (3, 7):
            cls = _get_attention_class(c, layer_idx)
            assert cls.__name__.startswith("UMichSLMMLA"), (
                f"layer {layer_idx}: expected MLA, got {cls.__name__}"
            )


class TestLegacyMode:
    """Legacy mode preserves the original MLA-XOR-DeltaNet validation."""

    def test_rejects_mla_plus_deltanet(self):
        with pytest.raises(ValueError, match="legacy"):
            _base_config(
                use_mla=True,
                use_deltanet=True,
                # attention_mode defaults to 'legacy'
            )

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="attention_mode"):
            _base_config(attention_mode="not_a_real_mode")

    def test_legacy_hybrid_pattern_still_works(self):
        # delta + standard attention hybrid (the original 'legacy' use case)
        c = _base_config(
            use_deltanet=True,
            hybrid_attention_ratio=["delta", "delta", "delta", "attn"],
        )
        # Layers 0,1,2 -> DeltaNet, Layer 3 -> standard attention
        cls0 = _get_attention_class(c, 0)
        cls3 = _get_attention_class(c, 3)
        assert cls0.__name__ == "GatedDeltaNet"
        assert cls3.__name__.startswith("UMichSLMAttention")


class TestStageBlockSpecValidation:
    """The stage_block_spec config field is validated for shape consistency."""

    def test_accepts_valid_spec(self):
        spec = [
            {
                "source": 0,
                "is_inserted": False,
                "block_layout": ["delta", "delta", "delta", "mla"],
            },
            {
                "source": 0,
                "is_inserted": True,
                "block_layout": ["delta", "delta", "delta", "mla"],
            },
        ]
        c = _base_config(
            num_hidden_layers=8,
            use_mla=True,
            use_deltanet=True,
            attention_mode="deltanet_mla",
            hybrid_attention_ratio=["delta", "delta", "delta", "mla"],
            stage_block_spec=spec,
            mla_kv_lora_rank=32,
            mla_q_lora_rank=0,
            mla_qk_nope_head_dim=8,
            mla_qk_rope_head_dim=8,
            mla_v_head_dim=16,
        )
        assert c.stage_block_spec == spec

    def test_rejects_layer_count_mismatch(self):
        with pytest.raises(ValueError, match="accounts for"):
            _base_config(
                num_hidden_layers=8,
                use_mla=True,
                use_deltanet=True,
                attention_mode="deltanet_mla",
                hybrid_attention_ratio=["delta", "delta", "delta", "mla"],
                # only 4 layers in spec, but num_hidden_layers=8
                stage_block_spec=[
                    {
                        "source": 0,
                        "is_inserted": False,
                        "block_layout": ["delta", "delta", "delta", "mla"],
                    }
                ],
                mla_kv_lora_rank=32,
                mla_q_lora_rank=0,
                mla_qk_nope_head_dim=8,
                mla_qk_rope_head_dim=8,
                mla_v_head_dim=16,
            )

    def test_rejects_inconsistent_block_layout_lengths(self):
        with pytest.raises(ValueError, match="must share the same layout length"):
            _base_config(
                num_hidden_layers=7,
                use_mla=True,
                use_deltanet=True,
                attention_mode="deltanet_mla",
                hybrid_attention_ratio=["delta", "delta", "delta", "mla"],
                stage_block_spec=[
                    {
                        "source": 0,
                        "is_inserted": False,
                        "block_layout": ["delta", "delta", "delta", "mla"],
                    },
                    {
                        "source": 0,
                        "is_inserted": True,
                        "block_layout": ["delta", "delta", "mla"],  # length 3
                    },
                ],
                mla_kv_lora_rank=32,
                mla_q_lora_rank=0,
                mla_qk_nope_head_dim=8,
                mla_qk_rope_head_dim=8,
                mla_v_head_dim=16,
            )
