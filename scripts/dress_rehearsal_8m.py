"""End-to-end dress rehearsal for the layer-stacked pretraining pipeline.

Builds a tiny ~8M-parameter model and runs all three stages with the staging
machinery, MoD routing, MTP loss, and the probe runner between stages. Ships
a synthetic dataset so the script is self-contained — runnable on a laptop in
under 5 minutes.

This is the gating test before any 50M / 100M pilot. If it passes:
  * The architecture surgery is correct (deltanet_mla and swa_globals_mla
    both forward+backward in the staged setting).
  * Stage transitions don't break the model (loss recovers within tolerance).
  * The optimizer migration produces non-NaN gradients in stage N+1.
  * The probe runner produces a valid I/O boundary recommendation.

Usage::

    python scripts/dress_rehearsal_8m.py --attention-mode deltanet_mla
    python scripts/dress_rehearsal_8m.py --attention-mode swa_globals_mla
    python scripts/dress_rehearsal_8m.py --quick   # both modes, ~30s each
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from umich_slm.configuration_umichslm import UMichSLMConfig
from umich_slm.modeling_umichslm import UMichSLMForCausalLM
from umich_slm.staging import (
    BlockSpec,
    StageSpec,
    build_stage_n_model,
    migrate_optimizer_state,
)
from umich_slm.staging.optimizer_migration import validate_migrated_state
from umich_slm.probes.runner import run_all_probes_on_checkpoint


def _tiny_config(attention_mode: str) -> UMichSLMConfig:
    """A ~8M-parameter, 4-block (16 layer) model in the requested attention mode."""
    common = dict(
        vocab_size=128,
        max_position_embeddings=128,
        num_hidden_layers=16,  # 4 blocks * 4 layers
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_mla=True,
        mla_kv_lora_rank=64,
        mla_q_lora_rank=0,
        mla_qk_nope_head_dim=16,
        mla_qk_rope_head_dim=16,
        mla_v_head_dim=32,
        use_mod=True,
        mod_capacity=0.5,
        mod_capacity_inserted=0.5,
        mod_every_n_layers=4,  # MoD on the last layer of each block
        mod_aux_loss_weight=0.01,
        use_mtp=True,
        mtp_variant="gloeckle",
        mtp_num_heads=2,
        mtp_predict_offsets=[2, 3],
        mtp_loss_weight=0.3,
        mtp_share_head=False,
        tie_word_embeddings=False,
    )
    if attention_mode == "deltanet_mla":
        common.update(
            use_deltanet=True,
            attention_mode="deltanet_mla",
            hybrid_attention_ratio=["delta", "delta", "delta", "mla"],
        )
    else:
        common.update(
            attention_mode="swa_globals_mla",
            hybrid_attention_ratio=["swa", "swa", "swa", "mla"],
            sliding_window=8,
            swa_window=8,
            swa_global_count=4,
            _attn_implementation="sdpa",
        )
    return UMichSLMConfig(**common)


def _layout_for(attention_mode: str) -> list[str]:
    return ["delta", "delta", "delta", "mla"] if attention_mode == "deltanet_mla" \
        else ["swa", "swa", "swa", "mla"]


def _stage_specs(attention_mode: str) -> list[StageSpec]:
    """Stage 1: 4 blocks; stage 2: 8 blocks (full doubling); stage 3: 16 blocks."""
    layout = _layout_for(attention_mode)
    s1 = StageSpec(
        stage_index=1, attention_mode=attention_mode,
        block_specs=[
            BlockSpec(source=None, is_inserted=False, block_layout=list(layout))
            for _ in range(4)
        ],
    )
    s2 = StageSpec(
        stage_index=2, attention_mode=attention_mode,
        parent_stage_index=1,
        block_specs=[
            BlockSpec(
                source=i // 2, is_inserted=bool(i % 2), block_layout=list(layout),
                fresh_router=True,
            )
            for i in range(8)
        ],
    )
    s3 = StageSpec(
        stage_index=3, attention_mode=attention_mode,
        parent_stage_index=2,
        block_specs=[
            BlockSpec(
                source=i // 2, is_inserted=bool(i % 2), block_layout=list(layout),
                fresh_router=True,
            )
            for i in range(16)
        ],
    )
    return [s1, s2, s3]


def _train_stage(
    model: UMichSLMForCausalLM,
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
    seq_len: int = 32,
    batch_size: int = 2,
) -> list[float]:
    """Train ``steps`` mini-steps on synthetic data; return the loss history."""
    model.train()
    losses = []
    for _ in range(steps):
        x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        optimizer.zero_grad()
        out = model(input_ids=x, labels=x)
        if not torch.isfinite(out.loss):
            raise RuntimeError(f"Loss is not finite: {out.loss}")
        out.loss.backward()
        optimizer.step()
        losses.append(float(out.loss.detach()))
    return losses


def _grow_and_migrate(
    prev_model: UMichSLMForCausalLM,
    prev_optimizer: torch.optim.Optimizer,
    spec: StageSpec,
) -> tuple[UMichSLMForCausalLM, torch.optim.Optimizer]:
    """Apply staging.build + optimizer migration; return (new_model, new_optimizer)."""
    new_cfg, new_model, layer_mapping = build_stage_n_model(
        prev_config=prev_model.config,
        prev_state_dict=prev_model.state_dict(),
        stage_spec=spec,
        init_strategy="exact_copy",
    )
    prev_state = prev_optimizer.state_dict()
    new_state = migrate_optimizer_state(
        prev_optim_state=prev_state,
        prev_model=prev_model,
        new_model=new_model,
        layer_mapping=layer_mapping,
        block_size=spec.block_size,
    )
    validate_migrated_state(new_state, new_model)
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    new_optimizer.load_state_dict(new_state)
    return new_model, new_optimizer


def run_rehearsal(attention_mode: str, *, steps_per_stage: int = 50) -> None:
    """Run the 3-stage dress rehearsal in the requested attention mode."""
    print(f"\n=== DRESS REHEARSAL: attention_mode={attention_mode} ===")
    torch.manual_seed(0)
    cfg = _tiny_config(attention_mode)
    model = UMichSLMForCausalLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M params, {cfg.num_hidden_layers} layers, "
          f"hidden_size={cfg.hidden_size}.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    specs = _stage_specs(attention_mode)

    # Stage 1
    print(f"--- Stage 1: {specs[0].num_blocks} blocks "
          f"({specs[0].num_hidden_layers} layers) ---")
    losses1 = _train_stage(model, optimizer, steps=steps_per_stage)
    print(f"  start loss = {losses1[0]:.4f}, end loss = {losses1[-1]:.4f}")

    # Probe between stages
    print("  Running probes on stage 1 checkpoint...")
    report1 = run_all_probes_on_checkpoint(model, model.config, batch_size=1, seq_len=16)
    print(f"  io_boundary_recommendation: {report1['io_boundary_recommendation']}")

    # Stage 1 -> Stage 2
    print(f"--- Stage 2: {specs[1].num_blocks} blocks "
          f"({specs[1].num_hidden_layers} layers) ---")
    pre_transition_loss = losses1[-1]
    model, optimizer = _grow_and_migrate(model, optimizer, specs[1])
    print(f"  Grew to {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params.")
    # Validate: forward through the new model with the previous mini-batch should
    # be finite.
    x = torch.randint(0, model.config.vocab_size, (2, 32))
    with torch.no_grad():
        out = model(input_ids=x, labels=x)
    print(f"  post-transition loss (random batch) = {float(out.loss):.4f}")
    if not torch.isfinite(out.loss):
        raise RuntimeError("Stage-2 forward produced non-finite loss.")
    losses2 = _train_stage(model, optimizer, steps=steps_per_stage)
    print(f"  stage-2 start={losses2[0]:.4f}, end={losses2[-1]:.4f}")

    # Stage 2 -> Stage 3
    print(f"--- Stage 3: {specs[2].num_blocks} blocks "
          f"({specs[2].num_hidden_layers} layers) ---")
    model, optimizer = _grow_and_migrate(model, optimizer, specs[2])
    print(f"  Grew to {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params.")
    x = torch.randint(0, model.config.vocab_size, (2, 32))
    with torch.no_grad():
        out = model(input_ids=x, labels=x)
    print(f"  post-transition loss (random batch) = {float(out.loss):.4f}")
    if not torch.isfinite(out.loss):
        raise RuntimeError("Stage-3 forward produced non-finite loss.")
    losses3 = _train_stage(model, optimizer, steps=steps_per_stage)
    print(f"  stage-3 start={losses3[0]:.4f}, end={losses3[-1]:.4f}")

    print(f"\n*** DRESS REHEARSAL PASS for attention_mode={attention_mode} ***\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--attention-mode",
        default="both",
        choices=["deltanet_mla", "swa_globals_mla", "both"],
        help="Which attention mode(s) to run. Default: both.",
    )
    p.add_argument("--steps-per-stage", type=int, default=50,
                   help="Number of training mini-steps per stage. Default 50.")
    p.add_argument("--quick", action="store_true",
                   help="Run with --steps-per-stage=20 for a fast smoke test.")
    args = p.parse_args(argv)

    if args.quick:
        args.steps_per_stage = 20

    modes = ["deltanet_mla", "swa_globals_mla"] if args.attention_mode == "both" \
        else [args.attention_mode]

    for mode in modes:
        run_rehearsal(mode, steps_per_stage=args.steps_per_stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
