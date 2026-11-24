"""Tests for configuration classes."""

import pytest
import json
import tempfile
from pathlib import Path

from umich_slm import UMichSLMConfig
from umich_slm.configs.presets import (
    UMICHSLM_CONFIGS,
    get_umichslm_config,
    list_available_sizes,
    get_preset_info,
    estimate_parameters,
    estimate_backbone_parameters,
)


class TestUMichSLMConfig:
    """Tests for UMichSLMConfig."""

    def test_default_values(self):
        """Default values should be reasonable."""
        config = UMichSLMConfig()

        assert config.vocab_size == 49152  # SmolLM2 tokenizer
        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 32
        assert config.num_attention_heads == 32
        assert config.hidden_act == "silu"
        assert config.model_type == "umichslm"

    def test_custom_values(self):
        """Custom values should override defaults."""
        config = UMichSLMConfig(
            hidden_size=1024,
            num_hidden_layers=12,
            num_attention_heads=8,
        )

        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 8

    def test_num_key_value_heads_default(self):
        """num_key_value_heads should default to num_attention_heads."""
        config = UMichSLMConfig(num_attention_heads=16)
        assert config.num_key_value_heads == 16

    def test_num_key_value_heads_gqa(self):
        """Should support GQA with fewer KV heads."""
        config = UMichSLMConfig(
            num_attention_heads=16,
            num_key_value_heads=4,
        )
        assert config.num_key_value_heads == 4

    def test_head_dim_property(self):
        """head_dim should be computed correctly."""
        config = UMichSLMConfig(hidden_size=1024, num_attention_heads=16)
        assert config.head_dim == 64

    def test_rope_scaling_validation_valid(self):
        """Valid rope_scaling should not raise."""
        config = UMichSLMConfig(rope_scaling={"type": "linear", "factor": 2.0})
        assert config.rope_scaling["factor"] == 2.0

    def test_rope_scaling_validation_invalid_type(self):
        """Invalid rope_scaling type should raise."""
        with pytest.raises(ValueError, match="type must be one of"):
            UMichSLMConfig(rope_scaling={"type": "invalid", "factor": 2.0})

    def test_rope_scaling_validation_missing_keys(self):
        """Missing rope_scaling keys should raise."""
        with pytest.raises(ValueError, match="must contain keys"):
            UMichSLMConfig(rope_scaling={"type": "linear"})

    def test_rope_scaling_validation_invalid_factor(self):
        """Factor <= 1.0 should raise."""
        with pytest.raises(ValueError, match="factor must be > 1.0"):
            UMichSLMConfig(rope_scaling={"type": "linear", "factor": 0.5})

    def test_save_and_load(self):
        """Config should be saveable and loadable."""
        config = UMichSLMConfig(
            hidden_size=512,
            num_hidden_layers=6,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)

            loaded = UMichSLMConfig.from_pretrained(tmpdir)

            assert loaded.hidden_size == 512
            assert loaded.num_hidden_layers == 6
            assert loaded.model_type == "umichslm"

    def test_to_dict(self):
        """Config should be convertible to dict."""
        config = UMichSLMConfig(hidden_size=256)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["hidden_size"] == 256
        assert config_dict["model_type"] == "umichslm"

    def test_from_dict(self):
        """Config should be creatable from dict."""
        config_dict = {
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "model_type": "umichslm",
        }
        config = UMichSLMConfig.from_dict(config_dict)

        assert config.hidden_size == 512
        assert config.num_hidden_layers == 8

    def test_position_embedding_type(self):
        """Should support different position embedding types."""
        rope_config = UMichSLMConfig(position_embedding_type="rope")
        pope_config = UMichSLMConfig(position_embedding_type="pope")

        assert rope_config.position_embedding_type == "rope"
        assert pope_config.position_embedding_type == "pope"


class TestPresets:
    """Tests for model presets."""

    def test_list_available_sizes(self):
        """Should list all available sizes."""
        sizes = list_available_sizes()
        assert "tiny" in sizes
        assert "small" in sizes
        assert "base" in sizes
        assert "large" in sizes
        assert "qwen3_like_tiny" in sizes
        assert "qwen3_like_xsmall" in sizes
        assert "qwen3_like_small" in sizes
        assert "qwen3_5_like_tiny" in sizes
        assert "qwen3_5_like_xsmall" in sizes
        assert "qwen3_5_like_small" in sizes

    def test_get_preset_info(self):
        """Should return preset configuration."""
        info = get_preset_info("tiny")
        assert "hidden_size" in info
        assert "num_hidden_layers" in info

    def test_qwen3_like_preset_shapes(self):
        """Qwen3-like presets should use dense GQA configurations."""
        preset_tiny = get_preset_info("qwen3_like_tiny")
        preset_xsmall = get_preset_info("qwen3_like_xsmall")
        preset_small = get_preset_info("qwen3_like_small")

        assert preset_tiny["hidden_size"] == 216
        assert preset_tiny["intermediate_size"] == 576
        assert preset_tiny["num_hidden_layers"] == 10
        assert preset_tiny["num_attention_heads"] == 6
        assert preset_tiny["num_key_value_heads"] == 2

        assert preset_xsmall["hidden_size"] == 432
        assert preset_xsmall["intermediate_size"] == 1152
        assert preset_xsmall["num_hidden_layers"] == 10
        assert preset_xsmall["num_attention_heads"] == 6
        assert preset_xsmall["num_key_value_heads"] == 2

        assert preset_small["hidden_size"] == 768
        assert preset_small["intermediate_size"] == 2048
        assert preset_small["num_hidden_layers"] == 16
        assert preset_small["num_attention_heads"] == 12
        assert preset_small["num_key_value_heads"] == 4

    def test_qwen3_5_like_preset_shapes(self):
        """Qwen3.5-like presets should use hybrid DeltaNet configurations."""
        preset_tiny = get_preset_info("qwen3_5_like_tiny")
        preset_xsmall = get_preset_info("qwen3_5_like_xsmall")
        preset_small = get_preset_info("qwen3_5_like_small")

        for preset in [preset_tiny, preset_xsmall, preset_small]:
            assert preset["use_deltanet"] is True
            assert preset["deltanet_variant"] == "gated"
            assert preset["hybrid_attention_ratio"] == ["delta", "delta", "delta", "attn"]
            assert preset["num_attention_heads"] == 6
            assert preset["num_key_value_heads"] == 2

        assert preset_tiny["hidden_size"] == 216
        assert preset_tiny["intermediate_size"] == 576
        assert preset_tiny["num_hidden_layers"] == 9

        assert preset_xsmall["hidden_size"] == 336
        assert preset_xsmall["intermediate_size"] == 896
        assert preset_xsmall["num_hidden_layers"] == 15

        assert preset_small["hidden_size"] == 552
        assert preset_small["intermediate_size"] == 1472
        assert preset_small["num_hidden_layers"] == 28

    def test_get_preset_info_invalid(self):
        """Should raise for invalid preset."""
        with pytest.raises(ValueError):
            get_preset_info("invalid")

    def test_estimate_parameters_tiny(self):
        """Parameter estimate should be reasonable for tiny."""
        params = estimate_parameters("tiny")
        # Tiny is ~30M with SmolLM2 vocab (49152), mostly from embedding layers
        assert 20_000_000 < params < 50_000_000

    def test_estimate_parameters_increasing(self):
        """Larger models should have more parameters."""
        tiny_params = estimate_parameters("tiny")
        small_params = estimate_parameters("small")
        base_params = estimate_parameters("base")
        large_params = estimate_parameters("large")

        assert tiny_params < small_params < base_params < large_params

    def test_qwen3_like_tiny_backbone_target(self):
        """The tiny Qwen3-like preset should target backbone params."""
        tiny_backbone = estimate_backbone_parameters("qwen3_like_tiny")

        assert 4_900_000 < tiny_backbone < 5_100_000

    def test_qwen3_like_xsmall_backbone_target(self):
        """The xsmall Qwen3-like preset should target backbone params."""
        xsmall_backbone = estimate_backbone_parameters("qwen3_like_xsmall")

        assert 19_000_000 < xsmall_backbone < 21_000_000

    def test_qwen3_like_small_backbone_target(self):
        """The small Qwen3-like preset should remain backbone-calibrated."""
        tiny_backbone = estimate_backbone_parameters("qwen3_like_tiny")
        xsmall_backbone = estimate_backbone_parameters("qwen3_like_xsmall")
        small_backbone = estimate_backbone_parameters("qwen3_like_small")

        assert tiny_backbone < xsmall_backbone < small_backbone
        assert 99_000_000 < small_backbone < 102_000_000

    def test_qwen3_like_total_params_exceed_backbone_params(self):
        """Untied embeddings should make total params exceed backbone params."""
        for size in ["qwen3_like_tiny", "qwen3_like_xsmall", "qwen3_like_small"]:
            assert estimate_parameters(size) > estimate_backbone_parameters(size)

    def test_qwen3_5_like_backbone_targets(self):
        """Qwen3.5-like presets should hit hybrid backbone targets."""
        tiny_backbone = estimate_backbone_parameters("qwen3_5_like_tiny")
        xsmall_backbone = estimate_backbone_parameters("qwen3_5_like_xsmall")
        small_backbone = estimate_backbone_parameters("qwen3_5_like_small")

        assert 4_900_000 < tiny_backbone < 5_100_000
        assert 19_000_000 < xsmall_backbone < 21_000_000
        assert 99_000_000 < small_backbone < 102_000_000
        assert tiny_backbone < xsmall_backbone < small_backbone

    def test_qwen3_5_like_total_params_exceed_backbone_params(self):
        """Hybrid presets should still report larger totals than backbones."""
        for size in ["qwen3_5_like_tiny", "qwen3_5_like_xsmall", "qwen3_5_like_small"]:
            assert estimate_parameters(size) > estimate_backbone_parameters(size)

    def test_umichslm_configs_dict(self):
        """UMICHSLM_CONFIGS should contain required keys.

        Layer-wise-scaling presets (e.g. openelm_like_*) derive
        ``intermediate_size`` and ``num_attention_heads`` per layer from
        ``ffn_multiplier_range`` / ``head_scaling_range`` and intentionally
        omit the static fields.
        """
        for size, config in UMICHSLM_CONFIGS.items():
            assert "hidden_size" in config, f"{size} missing hidden_size"
            assert "num_hidden_layers" in config, f"{size} missing num_hidden_layers"
            assert "num_key_value_heads" in config, f"{size} missing num_key_value_heads"
            if config.get("layer_wise_scaling"):
                assert (
                    "ffn_multiplier_range" in config
                ), f"{size} uses layer_wise_scaling but missing ffn_multiplier_range"
                assert (
                    "head_scaling_range" in config
                ), f"{size} uses layer_wise_scaling but missing head_scaling_range"
            else:
                assert "intermediate_size" in config, f"{size} missing intermediate_size"
                assert "num_attention_heads" in config, f"{size} missing num_attention_heads"


class TestConfigValidation:
    """Tests for config validation of mutually exclusive features."""

    def test_mla_and_hymba_raises(self):
        """Cannot enable both MLA and Hymba."""
        with pytest.raises(ValueError, match="Cannot use both Hymba and MLA"):
            UMichSLMConfig(
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                use_mla=True,
                use_hymba=True,
            )

    def test_deltanet_and_hymba_raises(self):
        """Cannot enable both DeltaNet and Hymba."""
        with pytest.raises(ValueError, match="Cannot use both Hymba and DeltaNet"):
            UMichSLMConfig(
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                use_deltanet=True,
                use_hymba=True,
            )

    def test_mla_and_deltanet_raises(self):
        """Cannot enable both MLA and DeltaNet."""
        with pytest.raises(ValueError, match="Cannot use both MLA and DeltaNet"):
            UMichSLMConfig(
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                use_mla=True,
                use_deltanet=True,
            )

    def test_mod_moe_overlap_raises(self):
        """Overlapping MoD and MoE layers should raise."""
        with pytest.raises(ValueError, match="both MoE and MoD"):
            UMichSLMConfig(
                hidden_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                use_mod=True,
                mod_every_n_layers=2,
                use_moe=True,
                num_experts=4,
            )
