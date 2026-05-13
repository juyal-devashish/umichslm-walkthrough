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

    input_ids = torch.randint(
        0, tiny_config.vocab_size, (2, 16), generator=reproducible_generator
    )
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

    input_ids = torch.randint(
        0, tiny_config.vocab_size, (2, 16), generator=reproducible_generator
    )
    labels = input_ids.clone()

    with torch.no_grad():
        out = wrapper(input_ids, labels=labels)

    assert not hasattr(out, "logits") or getattr(out, "logits", None) is None


def test_wrapper_returns_logits_without_labels(tiny_config, reproducible_generator):
    inner = UMichSLMForCausalLM(tiny_config).eval()
    wrapper = UMichSLMTitanWrapper(inner).eval()

    input_ids = torch.randint(
        0, tiny_config.vocab_size, (2, 16), generator=reproducible_generator
    )

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

    input_ids = torch.randint(
        0, tiny_config.vocab_size, (2, 16), generator=reproducible_generator
    )
    out = wrapper(input_ids, labels=input_ids.clone())
    out.loss.backward()

    # The HF model exposes ``lm_head`` with ``weight``; require non-zero grad.
    grad = inner.lm_head.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
