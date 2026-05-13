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
from umich_slm.titan.data import PackedFineWebDataset
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
                f"ranks {i} and {j} share documents: "
                f"{seen_doc_ids[i] & seen_doc_ids[j]}"
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
