"""Validation tests for the per-variant TOML configs.

What this catches:
- TOML parse errors / typos in field names.
- Drift between the existing LLaMA-Factory variant slurm scripts and the
  matching torchtitan TOML (per-device batch, grad-accum, effective batch).
- The override dict gets the architecture-specific flags through to
  ``UMichSLMConfig`` so a tiny instance constructs without error.

The actual variant *correctness* (MLA forward parity, MoE aux loss math,
MTP label shifting, etc.) is already covered by the existing variant test
suite (``test_mla.py``, ``test_moe.py``, ``test_multi_token_prediction.py``,
etc.) — these tests just verify the TOML→config plumbing.

DeltaNet is exercised only at the parse level here because its forward calls
into FLA Triton kernels that aren't available on CPU; full smoke runs happen
at GPU time via the slurm scripts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from umich_slm import UMichSLMConfig, UMichSLMForCausalLM
from umich_slm.titan.config_schema import load_toml_config


CONFIG_DIR = Path("configs/torchtitan")

# (toml_filename, expected effective batch, key override flags that must be present,
#  whether to attempt CPU instantiation, optional tiny-scale overrides)
VARIANT_SPECS = [
    (
        "umichslm_pretrain_baseline.toml",
        128,
        {"num_key_value_heads": 4},
        True,
        {},
    ),
    (
        "umichslm_pretrain_mla.toml",
        128,
        {"use_mla": True, "mla_kv_lora_rank": 256},
        True,
        # MLA's head_dim args must divide hidden_size sensibly at tiny scale.
        {
            "mla_qk_nope_head_dim": 16,
            "mla_qk_rope_head_dim": 16,
            "mla_v_head_dim": 16,
            "mla_kv_lora_rank": 32,
        },
    ),
    (
        "umichslm_pretrain_moe.toml",
        128,
        {"use_moe": True, "moe_expert_intermediate_size": 256},
        True,
        {"moe_expert_intermediate_size": 64},
    ),
    (
        "umichslm_pretrain_mod.toml",
        128,
        {"use_mod": True},
        True,
        {},
    ),
    (
        "umichslm_pretrain_mtp.toml",
        160,
        {"use_mtp": True},
        True,
        {},
    ),
    (
        "umichslm_pretrain_deltanet_gated.toml",
        128,
        {"use_deltanet": True, "deltanet_variant": "gated"},
        False,
        {},  # DeltaNet needs FLA kernels — skip CPU instantiation
    ),
    (
        "umichslm_pretrain_deltanet_kda.toml",
        128,
        {"use_deltanet": True, "deltanet_variant": "kda"},
        False,
        {},
    ),
    (
        "umichslm_pretrain_swa512.toml",
        128,
        {"sliding_window": 512},
        True,
        {},
    ),
    (
        "umichslm_pretrain_zcrmsnorm.toml",
        128,
        {"use_zero_centered_rmsnorm": True},
        True,
        {},
    ),
    (
        "umichslm_pretrain_pope.toml",
        144,
        {"position_embedding_type": "pope"},
        True,
        {},
    ),
]


# -------------------------------------------------------------- TOML parsing
@pytest.mark.parametrize("filename,expected_eff_batch,key_flags,_,__", VARIANT_SPECS)
def test_variant_toml_parses_and_has_expected_flags(
    filename: str,
    expected_eff_batch: int,
    key_flags: dict,
    _,
    __,
):
    cfg = load_toml_config(CONFIG_DIR / filename)
    assert cfg.model.size in {
        "small",
        "qwen3_like_small",
    }, f"{filename}: unexpected model.size {cfg.model.size!r}"
    eff = cfg.training.per_device_batch_size * cfg.training.gradient_accumulation_steps
    assert (
        eff == expected_eff_batch
    ), f"{filename}: effective batch {eff} != expected {expected_eff_batch}"
    for flag, expected_value in key_flags.items():
        assert (
            flag in cfg.model.overrides
        ), f"{filename}: override {flag!r} missing; got {sorted(cfg.model.overrides)}"
        actual = cfg.model.overrides[flag]
        assert (
            actual == expected_value
        ), f"{filename}: override {flag}={actual!r} != expected {expected_value!r}"


# ------------------------------------------------------- variant tags + run_name
@pytest.mark.parametrize("filename,_,__,___,____", VARIANT_SPECS)
def test_variant_run_name_and_output_dir_are_distinct(
    filename: str,
    _,
    __,
    ___,
    ____,
):
    """No two variant TOMLs should write to the same output dir or run name —
    catches copy-paste mistakes that would have benchmark runs overwrite each
    other."""
    cfg = load_toml_config(CONFIG_DIR / filename)
    stem = filename.replace("umichslm_pretrain_", "").replace(".toml", "")
    if stem == "baseline":
        # Baseline TOML uses qwen3_like_small + lr3e4 in its run_name
        assert "baseline" in cfg.job.run_name
    else:
        assert (
            stem in cfg.job.run_name
        ), f"{filename}: run_name {cfg.job.run_name!r} missing variant stem {stem!r}"
        assert (
            stem in cfg.job.output_dir
        ), f"{filename}: output_dir {cfg.job.output_dir!r} missing variant stem {stem!r}"


# --------------------------------------------------------- CPU instantiation
def _tiny_base() -> dict:
    """Tiny CPU-friendly UMichSLMConfig kwargs — base for variant overlay."""
    return {
        "vocab_size": 1000,
        "hidden_size": 64,
        "intermediate_size": 172,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 512,
    }


@pytest.mark.parametrize("filename,_,__,can_cpu_instantiate,tiny_overrides", VARIANT_SPECS)
def test_variant_overrides_construct_valid_config(
    filename: str,
    _,
    __,
    can_cpu_instantiate: bool,
    tiny_overrides: dict,
):
    """Merge the variant TOML's overrides into a tiny base and confirm
    ``UMichSLMConfig`` + ``UMichSLMForCausalLM`` instantiate without error.
    """
    if not can_cpu_instantiate:
        pytest.skip("variant requires CUDA / FLA Triton kernels")

    cfg = load_toml_config(CONFIG_DIR / filename)
    base = _tiny_base()
    base.update(cfg.model.overrides)
    # Tiny-scale adjustments to make some variant-specific dims valid.
    base.update(tiny_overrides)

    config = UMichSLMConfig(**base)
    # Round-trip: instantiate the full LM head model so we exercise the
    # variant-specific layer construction path.
    model = UMichSLMForCausalLM(config)
    assert model.config is config

    # Smoke a forward pass with labels to confirm the loss-fusion path runs.
    import torch

    input_ids = torch.randint(0, base["vocab_size"], (1, 8))
    out = model(input_ids=input_ids, labels=input_ids.clone(), use_cache=False, return_dict=True)
    assert out.loss is not None
    assert torch.isfinite(out.loss)


# ----------------------------------------------------------- DDP-friendliness
@pytest.mark.parametrize(
    "filename,_,__,___,____",
    [
        ("umichslm_pretrain_moe.toml", None, None, None, None),
        ("umichslm_pretrain_mod.toml", None, None, None, None),
        ("umichslm_pretrain_mtp.toml", None, None, None, None),
    ],
)
def test_dynamic_graph_variants_do_not_set_torch_compile(
    filename: str,
    _,
    __,
    ___,
    ____,
):
    """MoD/MoE/MTP have dynamic computation graphs (skipped layers, idle
    experts, conditional MTP heads). They must not ship with torch_compile=true
    by default — otherwise Phase 5 needs to revisit them. This guards against
    accidental edits."""
    cfg = load_toml_config(CONFIG_DIR / filename)
    assert (
        cfg.training.torch_compile is False
    ), f"{filename}: dynamic-graph variant should not enable torch_compile by default"
