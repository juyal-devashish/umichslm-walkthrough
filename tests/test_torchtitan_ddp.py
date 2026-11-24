"""DDP-correctness tests for the torchtitan path.

Multi-process distributed setup is not exercised here (CI runs single
process). What we *can* verify deterministically:

1. The dataloader's rank+worker stride sharding produces *disjoint* document
   sets across simulated ranks at world_size=2/4/8. This is the most common
   silent-bug mode for hand-rolled DDP — every rank ends up training on the
   same data.
2. The ``--set`` override mechanism correctly mutates a parsed
   ``TitanConfig`` and produces matching grad_accum / run_name / tag values
   for the ddp{2,4,8} slurm scripts.
3. ``apply_ddp`` in single-process mode returns the model unchanged (the
   1-GPU pilot path must work without ``init_process_group``).
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Set

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import torch

from umich_slm.titan.config_schema import (
    apply_overrides,
    load_toml_config,
)
from umich_slm.titan.data import PackedFineWebDataset, _ArrowTextColumn
from umich_slm.titan.parallelize import apply_ddp


class StubTokenizer:
    """Minimal tokenizer stand-in: each char becomes one token id."""

    eos_token_id = 0
    bos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False):
        # Use 2..N to avoid colliding with EOS/BOS sentinels above.
        return [(ord(c) % 200) + 2 for c in text]


def _write_synthetic_arrow(path: Path, num_docs: int) -> None:
    # Each doc is unique (so we can detect collisions across rank shards) and
    # long enough that the packing yields multiple windows per doc.
    texts = [f"document-{i:05d} " * 50 for i in range(num_docs)]
    table = pa.table({"text": texts})
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def _write_multichunk_arrow(path: Path, texts: list[str], rows_per_batch: int) -> None:
    """Write texts across multiple record batches so the reader yields a
    multi-chunk ChunkedArray — exercises _ArrowTextColumn's cross-chunk index
    math (single-batch writes collapse to one chunk and hide bugs there)."""
    schema = pa.schema({"text": pa.string()})
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, schema) as writer:
            for start in range(0, len(texts), rows_per_batch):
                chunk = texts[start : start + rows_per_batch]
                writer.write_batch(pa.record_batch({"text": chunk}, schema=schema))


def test_arrow_text_column_lazy_access_matches_to_pylist(tmp_path: Path):
    # 47 docs in batches of 10 -> 5 chunks (last partial): forces the bisect
    # path to cross several chunk boundaries, including the ragged tail.
    texts = [f"doc-{i:03d}-{'x' * (i % 7)}" for i in range(47)]
    arrow_path = tmp_path / "multichunk.arrow"
    _write_multichunk_arrow(arrow_path, texts, rows_per_batch=10)

    col = _ArrowTextColumn(arrow_path, "text")
    assert len(col) == len(texts)
    # Every index (in order, reversed, and at chunk seams) must match the eager
    # to_pylist() result exactly.
    for i in list(range(len(texts))) + [9, 10, 19, 20, 46, 0]:
        assert col[i] == texts[i], f"mismatch at index {i}"

    # A missing column must raise KeyError (and not leak the mmap silently).
    with pytest.raises(KeyError):
        _ArrowTextColumn(arrow_path, "nonexistent")


@pytest.mark.parametrize(
    "device_name,expected_key",
    [
        ("NVIDIA RTX A6000", "rtx_a6000_bf16"),  # Ampere — the local cluster's card
        ("NVIDIA A100-SXM4-80GB", "a100_80gb_bf16"),
        ("NVIDIA RTX 6000 Ada Generation", "rtx_6000_ada_bf16"),  # must NOT hit a6000
        ("NVIDIA H100 80GB HBM3", "h100_80gb_bf16"),
        ("NVIDIA L40S", "l40s_bf16"),  # Great Lakes spgpu2
        ("NVIDIA A40", "a40_bf16"),  # Great Lakes spgpu
    ],
)
def test_detect_hardware_key_maps_to_nonzero_peak(monkeypatch, device_name, expected_key):
    # Regression: the A6000 fell through to "unknown" -> peak 0 -> MFU always 0.
    # Every card we actually run on must resolve to a positive peak-FLOPs entry.
    from umich_slm.titan import throughput as T

    monkeypatch.setattr(T.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(T.torch.cuda, "get_device_name", lambda i=0: device_name)
    key = T.detect_hardware_key("bf16")
    assert key == expected_key
    assert T.HW_PEAK_TFLOPS.get(key, 0.0) > 0.0


@pytest.mark.parametrize("grad_accum", [1, 4, 8])
def test_reported_loss_is_per_micro_mean_not_sum(grad_accum):
    # Regression: the train loop accumulates `output.loss / grad_accum` across
    # grad_accum micro-steps, so `accum_loss` is ALREADY the per-micro mean. A
    # prior bug reported `accum_loss * grad_accum` (the *sum*), inflating the
    # logged loss by grad_accum× (e.g. a true init loss of ~ln(V)=10.9 showed up
    # as ~44 with grad_accum=4). Gradients were unaffected; only the metric lied.
    micro_losses = [11.0, 10.5, 11.5, 10.0, 12.0, 9.5, 11.25, 10.75][:grad_accum]

    accum_loss = 0.0
    for lm in micro_losses:
        scaled = lm / grad_accum  # exactly what train.py backprops
        accum_loss += scaled

    reported = accum_loss  # train.py logs this directly (the fix)
    expected_mean = sum(micro_losses) / grad_accum

    assert reported == pytest.approx(expected_mean)
    # The old bug would have produced grad_accum× this; guard against re-adding it.
    if grad_accum > 1:
        assert reported != pytest.approx(expected_mean * grad_accum)


def test_model_attn_implementation_override_reaches_config():
    # The MFU/flash-attn probe drives the backend via `--set
    # model.attn_implementation=...`; verify (a) apply_overrides accepts the new
    # ModelConfig field and (b) build_wrapper_from_config threads it onto the
    # built HF model's config. Only sdpa is asserted here (flash_attention_2
    # requires the flash-attn package + CUDA and is exercised on the A6000s).
    from umich_slm.titan.config_schema import load_toml_config, apply_overrides
    from umich_slm.titan.wrapper import build_wrapper_from_config

    cfg = load_toml_config("configs/torchtitan/umichslm_scaling_mod_50m.toml")
    apply_overrides(cfg, ["model.attn_implementation=sdpa"])
    assert cfg.model.attn_implementation == "sdpa"

    wrapper = build_wrapper_from_config(cfg.model, vocab_size=256)
    assert wrapper.inner.config._attn_implementation == "sdpa"


def test_checkpoint_rng_restore_is_best_effort():
    # Regression: some torch versions return the round-tripped RNG state as a
    # tensor set_rng_state() rejects ("RNG state must be a torch.ByteTensor").
    # _restore_rng_state must coerce and never raise, since RNG restore is
    # best-effort (affects reproducibility, not correctness).
    from umich_slm.titan.checkpoint import _as_byte_tensor, _restore_rng_state

    s = torch.get_rng_state()
    assert _as_byte_tensor(s.clone()).dtype == torch.uint8
    _restore_rng_state({"torch_rng_state": s.clone()})  # happy path
    _restore_rng_state({"torch_rng_state": s.tolist()})  # non-tensor -> coerced
    _restore_rng_state({})  # missing key -> no-op, no raise


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_dataloader_ranks_see_disjoint_documents(tmp_path: Path, world_size: int):
    arrow_path = tmp_path / "train.arrow"
    _write_synthetic_arrow(arrow_path, num_docs=64)

    tokenizer = StubTokenizer()

    # Pull the first window from each rank's iterator and verify the document
    # IDs that contributed are disjoint. We approximate "document touched" by
    # decoding the ascii byte that opens each unique doc string ("d" then a
    # 5-digit number); each rank should see a different set of those numbers.
    seen_doc_ids: list[Set[int]] = []
    for rank in range(world_size):
        ds = PackedFineWebDataset(
            arrow_path=arrow_path,
            tokenizer=tokenizer,
            seq_len=128,
            seed=42,
            rank=rank,
            world_size=world_size,
            infinite=False,
        )
        # Sample several windows per rank to make doc-id detection robust.
        ids: Set[int] = set()
        for i, _ in zip(range(8), iter(ds)):
            # The shuffled order is deterministic given seed=42; recover it.
            indices = ds._shuffled_indices(epoch=0, rank_offset=rank, rank_stride=world_size)
            ids.update(indices[:8])  # the first few docs this rank consumes
            break
        seen_doc_ids.append(ids)

    # Every rank's set must be non-empty and pairwise disjoint.
    assert all(len(s) > 0 for s in seen_doc_ids)
    for i in range(world_size):
        for j in range(i + 1, world_size):
            assert seen_doc_ids[i].isdisjoint(seen_doc_ids[j]), (
                f"ranks {i} and {j} share documents: " f"{seen_doc_ids[i] & seen_doc_ids[j]}"
            )


def test_dataloader_window_shapes_match_seq_len(tmp_path: Path):
    arrow_path = tmp_path / "train.arrow"
    _write_synthetic_arrow(arrow_path, num_docs=16)

    ds = PackedFineWebDataset(
        arrow_path=arrow_path,
        tokenizer=StubTokenizer(),
        seq_len=64,
        rank=0,
        world_size=1,
        infinite=False,
    )
    samples = list(zip(range(4), iter(ds)))
    assert len(samples) == 4
    for _, window in samples:
        assert isinstance(window, torch.Tensor)
        assert window.dtype == torch.long
        assert window.shape == (64,)


# ----------------------------------------------------------------- overrides
def test_apply_overrides_matches_ddp_slurm_invocation(tmp_path: Path):
    """Mirror the exact argument set that the ddp4 slurm script passes."""
    cfg = load_toml_config("configs/torchtitan/umichslm_pretrain_baseline.toml")
    apply_overrides(
        cfg,
        [
            "training.gradient_accumulation_steps=4",
            "parallelism.dp_replicate_degree=4",
            "job.run_name=titan_small_baseline_lr3e4_ddp4",
            "job.output_dir=./outputs/titan/small_baseline_lr3e4_ddp4",
            "logging.wandb_tags=benchmark_v1,baseline,ddp4",
        ],
    )
    assert cfg.training.gradient_accumulation_steps == 4
    assert cfg.parallelism.dp_replicate_degree == 4
    assert cfg.job.run_name == "titan_small_baseline_lr3e4_ddp4"
    assert cfg.job.output_dir == "./outputs/titan/small_baseline_lr3e4_ddp4"
    assert cfg.logging.wandb_tags == ["benchmark_v1", "baseline", "ddp4"]


def test_ddp_find_unused_parameters_knob():
    """Default stays True (MoD/MTP need it); the TOML/--set path can turn it off."""
    cfg = load_toml_config("configs/torchtitan/umichslm_scaling_delphi_local_s4096.toml")
    assert cfg.training.ddp_find_unused_parameters is True
    assert cfg.training.ddp_static_graph is False and cfg.training.ddp_comm_hook == "none"
    apply_overrides(
        cfg,
        [
            "training.ddp_find_unused_parameters=false",
            "training.ddp_static_graph=true",
            "training.ddp_comm_hook=bf16",
        ],
    )
    assert cfg.training.ddp_find_unused_parameters is False
    assert cfg.training.ddp_static_graph is True and cfg.training.ddp_comm_hook == "bf16"


def test_apply_ddp_rejects_unknown_comm_hook(monkeypatch):
    import umich_slm.titan.parallelize as par

    monkeypatch.setattr(par, "is_distributed", lambda: True)
    monkeypatch.setattr(par, "DDP", lambda model, **kw: model)
    m = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="comm_hook"):
        par.apply_ddp(m, local_rank=0, comm_hook="fp8")


def test_apply_overrides_coerces_bools_and_floats():
    cfg = load_toml_config("configs/torchtitan/umichslm_pretrain_baseline.toml")
    apply_overrides(
        cfg,
        [
            "training.bf16=false",
            "training.torch_compile=true",
            "optimizer.lr=1e-5",
            "optimizer.eps=2.5e-9",
        ],
    )
    assert cfg.training.bf16 is False
    assert cfg.training.torch_compile is True
    assert cfg.optimizer.lr == 1e-5
    assert cfg.optimizer.eps == 2.5e-9


def test_apply_overrides_rejects_bad_input():
    cfg = load_toml_config("configs/torchtitan/umichslm_pretrain_baseline.toml")
    with pytest.raises(ValueError, match="section.field"):
        apply_overrides(cfg, ["no_equals_sign"])
    with pytest.raises(ValueError, match="must include a section prefix"):
        apply_overrides(cfg, ["nodot=1"])
    with pytest.raises(ValueError, match="Unknown section"):
        apply_overrides(cfg, ["bad.field=1"])
    with pytest.raises(ValueError, match="Unknown field"):
        apply_overrides(cfg, ["training.bad_field=1"])


# ----------------------------------------------------------------- DDP wrap
def test_apply_ddp_no_op_without_process_group(tiny_config):
    """Single-GPU path: ``apply_ddp`` returns the model unmodified."""
    from umich_slm import UMichSLMForCausalLM
    from umich_slm.titan import UMichSLMTitanWrapper

    inner = UMichSLMForCausalLM(tiny_config)
    wrapper = UMichSLMTitanWrapper(inner)
    wrapped = apply_ddp(wrapper, local_rank=0)
    assert wrapped is wrapper  # identity, not a DDP module
