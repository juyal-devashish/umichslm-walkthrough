"""Plumbing tests for torch.compile + float8 paths.

These tests don't actually exercise compiled or fp8 kernels (those need
Hopper+ hardware and torchao). They verify:

1. ``apply_compile`` walks every decoder layer and skips DeltaNet ones when
   the safety flag is on.
2. ``apply_float8`` surfaces a clear ``ImportError`` with an install hint
   when ``torchao`` isn't available — the most common failure mode for users
   who copy the H100 TOML without installing the extra.
3. The H100/fp8 TOML parses, has the expected toggles, and uses the
   appropriate hardware tags.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from umich_slm import UMichSLMConfig, UMichSLMForCausalLM
from umich_slm.titan import UMichSLMTitanWrapper
from umich_slm.titan.config_schema import load_toml_config


CONFIG_DIR = Path("configs/torchtitan")


# ----------------------------------------------------------------- compile
def test_apply_compile_walks_every_decoder_layer(tiny_config):
    inner = UMichSLMForCausalLM(tiny_config)
    wrapper = UMichSLMTitanWrapper(inner)
    layers_before = wrapper._iter_decoder_layers()
    assert len(layers_before) == tiny_config.num_hidden_layers

    compiled_count = wrapper.apply_compile()
    assert compiled_count == tiny_config.num_hidden_layers


def test_apply_compile_skips_deltanet_layers_when_safe():
    """DeltaNet layers must be left in eager mode when the safety flag is on,
    so a Triton kernel-level Dynamo failure can't take down the run."""
    cfg = UMichSLMConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=172,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        use_deltanet=True,
        deltanet_variant="gated",
        deltanet_every_n_layers=2,  # half DeltaNet, half regular attention
    )
    # Don't actually instantiate the model (FLA Triton kernels require CUDA).
    # Stub the wrapper just enough to exercise apply_compile's branching.

    class _Stub:
        def __init__(self, cfg):
            self.config = cfg

        def named_modules(self):
            return []

    class _Layer:
        def __init__(self):
            self.forward = lambda x: x

    wrapper = UMichSLMTitanWrapper.__new__(UMichSLMTitanWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.inner = _Stub(cfg)  # type: ignore[assignment]

    fake_layers = [_Layer() for _ in range(cfg.num_hidden_layers)]
    wrapper._iter_decoder_layers = lambda: fake_layers  # type: ignore[assignment]

    compiled = wrapper.apply_compile(deltanet_compile_safe=True)
    # With every_n=2 and first_n=0: layers 0,1 are DeltaNet, layer 2 is attn,
    # layer 3 is DeltaNet ⇒ only layer 2 gets compiled.
    deltanet_indices = [
        i for i in range(cfg.num_hidden_layers) if wrapper._layer_uses_deltanet(i)
    ]
    expected_compiled = cfg.num_hidden_layers - len(deltanet_indices)
    assert compiled == expected_compiled


def test_apply_compile_compiles_all_when_safety_off():
    cfg = UMichSLMConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=172,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        use_deltanet=True,
        deltanet_variant="gated",
        deltanet_every_n_layers=2,
    )

    class _Stub:
        def __init__(self, cfg):
            self.config = cfg

        def named_modules(self):
            return []

    class _Layer:
        def __init__(self):
            self.forward = lambda x: x

    wrapper = UMichSLMTitanWrapper.__new__(UMichSLMTitanWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.inner = _Stub(cfg)  # type: ignore[assignment]
    fake_layers = [_Layer() for _ in range(cfg.num_hidden_layers)]
    wrapper._iter_decoder_layers = lambda: fake_layers  # type: ignore[assignment]

    compiled = wrapper.apply_compile(deltanet_compile_safe=False)
    assert compiled == cfg.num_hidden_layers


# ----------------------------------------------------------------- float8
def test_apply_float8_raises_clear_import_error_without_torchao(tiny_config):
    """The most common failure mode: H100 TOML enabled, torchao not installed.
    The error must include the package + version hint so the fix is obvious."""
    pytest.importorskip(
        "torch", reason="torch required for the wrapper to instantiate"
    )
    try:
        import torchao.float8  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("torchao installed in this env; can't test the missing-dep path")

    inner = UMichSLMForCausalLM(tiny_config)
    wrapper = UMichSLMTitanWrapper(inner)
    with pytest.raises(ImportError, match="torchao"):
        wrapper.apply_float8()


# ------------------------------------------------------------ H100 TOML
def test_h100_fp8_toml_enables_compile_and_float8():
    cfg = load_toml_config(CONFIG_DIR / "umichslm_pretrain_baseline_h100_fp8.toml")
    assert cfg.training.float8 is True
    assert cfg.training.torch_compile is True
    assert cfg.training.bf16 is True  # master weights stay bf16; only GEMMs go fp8
    # H100-specific tags so wandb groups it correctly.
    assert "h100" in cfg.logging.wandb_tags
    assert "fp8" in cfg.logging.wandb_tags
    # Output dir must not collide with the bf16 baseline.
    assert cfg.job.output_dir != "./outputs/titan/small_baseline_lr3e4"
    # Fused optimizer is the right default on Hopper.
    assert cfg.optimizer.name == "adamw_fused"


def test_baseline_bf16_toml_does_not_enable_float8():
    """Guard against accidental copy-paste enabling fp8 on the bf16 baseline,
    which would silently fail on non-Hopper hardware."""
    cfg = load_toml_config(CONFIG_DIR / "umichslm_pretrain_baseline.toml")
    assert cfg.training.float8 is False
    assert cfg.training.torch_compile is False
