"""End-to-end test for the staging.transition CLI flow.

Exercises the standalone transition module that's used by both the LLaMA-Factory
multi-process flow (sbatch chain) and the in-process torchtitan staging loop.

Validates:
  * Stage-1 checkpoint -> stage-2 grown checkpoint round-trip.
  * Optimizer.pt sidecar is in the titan format and step is reset to 0.
  * stage_metadata.json captures the StageSpec and layer mapping.
  * Idempotency: re-running the transition with _DONE present is a no-op.
  * The grown checkpoint produces finite loss when reloaded.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from umich_slm.configuration_umichslm import UMichSLMConfig
from umich_slm.modeling_umichslm import UMichSLMForCausalLM
from umich_slm.staging.block_spec import BlockSpec, StageSpec
from umich_slm.staging.transition import (
    _load_state_dict_from_dir,
    run_transition,
)


def _stage1_cfg():
    return UMichSLMConfig(
        vocab_size=64,
        max_position_embeddings=64,
        num_hidden_layers=8,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
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


def _stage2_doubling_spec(parent_idx: int = 1) -> StageSpec:
    layout = ["delta", "delta", "delta", "mla"]
    return StageSpec(
        stage_index=2,
        attention_mode="deltanet_mla",
        block_specs=[
            BlockSpec(source=i // 2, is_inserted=bool(i % 2), block_layout=layout)
            for i in range(4)
        ],
        parent_stage_index=parent_idx,
    )


@pytest.fixture
def tmp_workspace():
    tmp = Path(tempfile.mkdtemp(prefix="staging_transition_"))
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def _make_stage1(tmp: Path) -> Path:
    """Create a stage-1 checkpoint dir with a saved model + optimizer sidecar."""
    torch.manual_seed(0)
    cfg = _stage1_cfg()
    model = UMichSLMForCausalLM(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    for _ in range(2):
        optim.zero_grad()
        model(input_ids=x, labels=x).loss.backward()
        optim.step()
    stage1 = tmp / "stage1"
    stage1.mkdir()
    model.save_pretrained(stage1)
    torch.save({"step": 100, "optimizer": optim.state_dict()}, stage1 / "optimizer.pt")
    return stage1


class TestRunTransition:
    def test_roundtrip_writes_expected_files(self, tmp_workspace):
        stage1 = _make_stage1(tmp_workspace)
        spec = _stage2_doubling_spec()
        spec_path = tmp_workspace / "stage2.json"
        spec.save(spec_path)
        stage2 = tmp_workspace / "stage2_init"

        run_transition(from_dir=stage1, to_dir=stage2, spec_path=spec_path)

        for name in ("config.json", "optimizer.pt", "stage_metadata.json", "_DONE"):
            assert (stage2 / name).exists(), f"missing expected output: {name}"
        assert (stage2 / "model.safetensors").exists() or (
            stage2 / "pytorch_model.bin"
        ).exists()

    def test_optimizer_sidecar_format(self, tmp_workspace):
        stage1 = _make_stage1(tmp_workspace)
        spec_path = tmp_workspace / "stage2.json"
        _stage2_doubling_spec().save(spec_path)
        stage2 = tmp_workspace / "stage2_init"
        run_transition(from_dir=stage1, to_dir=stage2, spec_path=spec_path)

        sidecar = torch.load(stage2 / "optimizer.pt", weights_only=False)
        assert "optimizer" in sidecar
        assert "step" in sidecar
        # Stage-2 starts its LR schedule from scratch.
        assert sidecar["step"] == 0
        # Optimizer state has the canonical AdamW shape.
        assert "state" in sidecar["optimizer"]
        assert "param_groups" in sidecar["optimizer"]

    def test_grown_checkpoint_forwards_finitely(self, tmp_workspace):
        stage1 = _make_stage1(tmp_workspace)
        spec_path = tmp_workspace / "stage2.json"
        _stage2_doubling_spec().save(spec_path)
        stage2 = tmp_workspace / "stage2_init"
        run_transition(from_dir=stage1, to_dir=stage2, spec_path=spec_path)

        cfg = UMichSLMConfig.from_pretrained(stage2)
        model = UMichSLMForCausalLM(cfg)
        _load_state_dict_from_dir(model, stage2)
        assert cfg.num_hidden_layers == 16  # 4 blocks * 4 layers each
        x = torch.randint(0, cfg.vocab_size, (1, 16))
        out = model(input_ids=x, labels=x)
        assert torch.isfinite(out.loss)

    def test_metadata_records_layer_mapping(self, tmp_workspace):
        stage1 = _make_stage1(tmp_workspace)
        spec_path = tmp_workspace / "stage2.json"
        _stage2_doubling_spec().save(spec_path)
        stage2 = tmp_workspace / "stage2_init"
        run_transition(from_dir=stage1, to_dir=stage2, spec_path=spec_path)

        meta = json.loads((stage2 / "stage_metadata.json").read_text())
        assert meta["stage_spec"]["stage_index"] == 2
        assert len(meta["stage_spec"]["block_specs"]) == 4
        assert meta["block_size"] == 4
        # Layer mapping reflects interleaved doubling
        assert meta["layer_mapping"] == {"0": 0, "1": 0, "2": 1, "3": 1}

    def test_idempotency_with_done_sentinel(self, tmp_workspace):
        stage1 = _make_stage1(tmp_workspace)
        spec_path = tmp_workspace / "stage2.json"
        _stage2_doubling_spec().save(spec_path)
        stage2 = tmp_workspace / "stage2_init"
        run_transition(from_dir=stage1, to_dir=stage2, spec_path=spec_path)
        # Capture a marker file's mtime to confirm no rewrite on re-run.
        meta_path = stage2 / "stage_metadata.json"
        first_mtime = meta_path.stat().st_mtime

        # Re-run — the _DONE sentinel should short-circuit.
        run_transition(from_dir=stage1, to_dir=stage2, spec_path=spec_path)
        assert meta_path.stat().st_mtime == first_mtime, (
            "stage_metadata.json was rewritten despite _DONE sentinel"
        )
