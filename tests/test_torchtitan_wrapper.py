"""Numerical-parity tests for the torchtitan wrapper.

The wrapper exists only to adapt the HF-style ``UMichSLMForCausalLM`` to a
``(tokens, labels) -> output`` API. These tests pin down that:

1. Wrapper forward (with labels) yields the same fused loss as the inner
   model on the same inputs (no drift introduced by the adapter layer).
2. ``logits`` is omitted in the training path — verifying we don't
   accidentally trigger the full ``(B, T, V)`` materialization that the
   chunked-loss path is designed to avoid.
3. ``param_groups()`` partitions every trainable parameter exactly once.
4. The MFU active-param accounting falls back correctly for dense models.

These run on CPU with the existing ``tiny_config`` fixture so they're cheap
enough for CI.
"""

from __future__ import annotations

import torch

from umich_slm import UMichSLMForCausalLM
from umich_slm.titan import UMichSLMTitanWrapper
from umich_slm.titan.wrapper import TitanModelOutput


def test_wrapper_loss_matches_inner_model(tiny_config, reproducible_generator):
    inner = UMichSLMForCausalLM(tiny_config).eval()
    wrapper = UMichSLMTitanWrapper(inner).eval()

    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16), generator=reproducible_generator)
    labels = input_ids.clone()

    with torch.no_grad():
        ref = inner(input_ids=input_ids, labels=labels, use_cache=False, return_dict=True)
        out = wrapper(input_ids, labels=labels)

    assert isinstance(out, TitanModelOutput)
    assert out.loss is not None and ref.loss is not None
    assert torch.allclose(out.loss, ref.loss, atol=1e-6, rtol=1e-6)


def test_wrapper_omits_logits_when_labels_passed(tiny_config, reproducible_generator):
    inner = UMichSLMForCausalLM(tiny_config).eval()
    wrapper = UMichSLMTitanWrapper(inner).eval()

    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16), generator=reproducible_generator)
    labels = input_ids.clone()

    with torch.no_grad():
        out = wrapper(input_ids, labels=labels)

    assert not hasattr(out, "logits") or getattr(out, "logits", None) is None


def test_wrapper_returns_logits_without_labels(tiny_config, reproducible_generator):
    inner = UMichSLMForCausalLM(tiny_config).eval()
    wrapper = UMichSLMTitanWrapper(inner).eval()

    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16), generator=reproducible_generator)

    with torch.no_grad():
        logits = wrapper(input_ids)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 16, tiny_config.vocab_size)


def test_param_groups_partition_trainable_parameters(tiny_config):
    inner = UMichSLMForCausalLM(tiny_config)
    wrapper = UMichSLMTitanWrapper(inner)

    groups = wrapper.param_groups(weight_decay=0.1)
    assert len(groups) == 2
    decay_ids = {id(p) for p in groups[0]["params"]}
    no_decay_ids = {id(p) for p in groups[1]["params"]}
    assert decay_ids.isdisjoint(no_decay_ids)

    trainable_ids = {id(p) for p in wrapper.parameters() if p.requires_grad}
    assert decay_ids | no_decay_ids == trainable_ids
    assert groups[0]["weight_decay"] == 0.1
    assert groups[1]["weight_decay"] == 0.0


def test_active_params_equals_total_for_dense_model(tiny_config):
    inner = UMichSLMForCausalLM(tiny_config)
    wrapper = UMichSLMTitanWrapper(inner)
    assert wrapper.num_active_params == wrapper.num_params


def test_wrapper_backward_runs(tiny_config, reproducible_generator):
    """Smoke test: backward through the wrapper produces grads on lm_head."""
    inner = UMichSLMForCausalLM(tiny_config).train()
    wrapper = UMichSLMTitanWrapper(inner).train()

    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16), generator=reproducible_generator)
    out = wrapper(input_ids, labels=input_ids.clone())
    out.loss.backward()

    # The HF model exposes ``lm_head`` with ``weight``; require non-zero grad.
    grad = inner.lm_head.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


# --------------------------------------------------------------------------
# RoPE cache sizing vs seq_len (regression for the S>=8192 device-side assert
# that killed every long-context cell of Great Lakes job 58953441).
# --------------------------------------------------------------------------


def test_fit_rope_cache_grows_default_rope():
    from umich_slm.titan.wrapper import fit_rope_cache_to_seq

    ov = {}
    assert fit_rope_cache_to_seq(ov, "qwen3_like_deep80", 8192) == 8192
    assert ov["max_position_embeddings"] == 8192


def test_fit_rope_cache_noop_when_seq_fits():
    from umich_slm.titan.wrapper import fit_rope_cache_to_seq

    ov = {}
    assert fit_rope_cache_to_seq(ov, "qwen3_like_deep80", 4096) is None
    assert ov == {}


def test_fit_rope_cache_respects_explicit_override():
    from umich_slm.titan.wrapper import fit_rope_cache_to_seq

    ov = {"max_position_embeddings": 16384}
    assert fit_rope_cache_to_seq(ov, "qwen3_like_deep80", 8192) is None
    assert ov["max_position_embeddings"] == 16384


def test_fit_rope_cache_refuses_to_guess_under_yarn():
    import pytest

    from umich_slm.titan.wrapper import fit_rope_cache_to_seq

    ov = {"rope_scaling": {"rope_type": "yarn", "factor": 8.0}}
    with pytest.raises(ValueError, match="load-bearing"):
        fit_rope_cache_to_seq(ov, "qwen3_like_deep80", 32768)


def test_fit_rope_cache_allows_yarn_when_the_base_length_is_carried():
    from umich_slm.titan.wrapper import fit_rope_cache_to_seq

    ov = {
        "rope_scaling": {"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 4096}
    }
    assert fit_rope_cache_to_seq(ov, "qwen3_like_deep80", 32768) == 32768
    assert ov["max_position_embeddings"] == 32768


def test_rope_cache_reaches_live_modules_and_costs_no_params():
    from umich_slm.titan.wrapper import UMichSLMTitanWrapper, fit_rope_cache_to_seq

    def build(seq):
        ov = {"vocab_size": 512, "num_hidden_layers": 2, "_attn_implementation": "eager"}
        fit_rope_cache_to_seq(ov, "qwen3_like_deep80", seq)
        return UMichSLMTitanWrapper.from_preset("qwen3_like_deep80", overrides=ov)

    short, long_ = build(4096), build(8192)
    assert {int(m.cos_cached.shape[0]) for m in long_.modules() if hasattr(m, "cos_cached")} == {
        8192
    }
    # max_position_embeddings is a cache length under rope_type=default, not a
    # parameter: growing it must not change N.
    assert sum(p.numel() for p in short.parameters()) == sum(p.numel() for p in long_.parameters())


def test_pause_phasing_reaches_the_model_through_the_wrapper():
    """train.py phases ``_pause`` through the wrapper. It once probed a ``.model``
    the wrapper does not have, so phasing was a silent no-op on every cell."""
    from umich_slm.titan.wrapper import UMichSLMTitanWrapper

    ov = {"vocab_size": 512, "num_hidden_layers": 2, "_attn_implementation": "eager"}
    w = UMichSLMTitanWrapper.from_preset("qwen3_like_pause_ar10m", overrides=ov)
    assert w.has_pause_stream
    assert w.set_pause_enabled(False) is False
    assert not w.inner.model.pause_enabled
    assert not any(getattr(l, "pause_enabled", False) for l in w.inner.model.layers)
    assert w.set_pause_enabled(True) is True
    assert all(l.pause_enabled for l in w.inner.model.layers if getattr(l, "pause_two_stream", False))
    plain = UMichSLMTitanWrapper.from_preset("qwen3_like_ar10m", overrides=ov)
    assert not plain.has_pause_stream


def test_share_rope_caches_dedupes_and_is_bitwise_identical():
    import torch

    from umich_slm.titan.wrapper import UMichSLMTitanWrapper, fit_rope_cache_to_seq

    def build(share):
        torch.manual_seed(0)
        ov = {
            "vocab_size": 128,
            "num_hidden_layers": 4,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "_attn_implementation": "eager",
        }
        fit_rope_cache_to_seq(ov, "qwen3_like_deep80", 8192)
        w = UMichSLMTitanWrapper.from_preset("qwen3_like_deep80", overrides=ov)
        if share:
            w.share_rope_caches()
        return w

    plain, shared = build(False), build(True)
    shared.load_state_dict(plain.state_dict())

    def n_caches(m):
        return len({id(x.cos_cached) for x in m.modules() if hasattr(x, "cos_cached")})

    assert n_caches(plain) == 4
    assert n_caches(shared) == 1

    x = torch.randint(0, 128, (2, 384))
    with torch.no_grad():
        assert torch.equal(plain(x), shared(x))

    plain(x).sum().backward()
    shared(x).sum().backward()
    gp = torch.cat([p.grad.flatten() for p in plain.parameters() if p.grad is not None])
    gs = torch.cat([p.grad.flatten() for p in shared.parameters() if p.grad is not None])
    assert torch.equal(gp, gs)


def test_rope_caches_are_non_persistent_so_sharing_cannot_touch_checkpoints():
    from umich_slm.titan.wrapper import UMichSLMTitanWrapper

    ov = {"vocab_size": 128, "num_hidden_layers": 2, "_attn_implementation": "eager"}
    w = UMichSLMTitanWrapper.from_preset("qwen3_like_deep80", overrides=ov)
    keys = w.state_dict().keys()
    assert not [k for k in keys if "cos_cached" in k or "sin_cached" in k]


# --- attention-backend ladder (GL job 58992621: sdpa+compile is broken on
# Blackwell, and the live path was silently defaulting to it) -----------------


def test_resolve_attn_implementation_respects_explicit_setting():
    from umich_slm.titan.wrapper import resolve_attn_implementation

    for impl in ("sdpa", "eager", "flash_attention_2"):
        assert resolve_attn_implementation(impl) == impl


def test_resolve_attn_implementation_walks_the_ladder(monkeypatch):
    """None means "pick for me", and the pick follows flash -> sdpa."""
    import builtins
    import types

    from umich_slm.titan.wrapper import resolve_attn_implementation

    real_import = builtins.__import__

    def no_flash(name, *a, **kw):
        if name == "flash_attn":
            raise ImportError("no flash-attn")
        return real_import(name, *a, **kw)

    def yes_flash(name, *a, **kw):
        if name == "flash_attn":
            return types.ModuleType("flash_attn")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_flash)
    assert resolve_attn_implementation(None) == "sdpa"
    monkeypatch.setattr(builtins, "__import__", yes_flash)
    assert resolve_attn_implementation(None) == "flash_attention_2"


def test_attn_implementation_reaches_the_live_attention_class():
    """The knob must swap the module, not just sit in the config.

    Two past ablations died to a knob that silently did nothing; the whole
    reason this ladder exists is that nobody had checked which kernel the live
    path was actually running.
    """
    from umich_slm.modules.attention import UMichSLMAttention
    from umich_slm.titan.config_schema import ModelConfig
    from umich_slm.titan.wrapper import build_wrapper_from_config

    seen = {}
    for impl in ("eager", "sdpa"):
        wrapper = build_wrapper_from_config(
            ModelConfig(size="tiny", attn_implementation=impl), vocab_size=256, seq_len=128
        )
        assert wrapper.config._attn_implementation == impl
        seen[impl] = sorted(
            {type(m).__name__ for m in wrapper.modules() if isinstance(m, UMichSLMAttention)}
        )
    assert seen["eager"] != seen["sdpa"], seen
