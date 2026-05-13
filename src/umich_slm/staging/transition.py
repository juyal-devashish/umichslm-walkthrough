"""Standalone CLI for the LLaMA-Factory multi-process stage transition.

Usage::

    python -m umich_slm.staging.transition \
        --from   ./outputs/stacking/50m_b_dn_stage1/checkpoint-final \
        --to     ./outputs/stacking/50m_b_dn_stage2_init \
        --config ./configs/torchtitan/stacking/specs/50m_b_dn_stage2.json \
        [--init-strategy exact_copy] \
        [--run-probes]

The CLI:
  1. Loads the previous checkpoint via :func:`UMichSLMForCausalLM.from_pretrained`
     and ``torch.load(prev/optimizer.pt)`` (best-effort).
  2. Builds the next-stage model via :func:`build_stage_n_model`.
  3. Migrates optimizer state via :func:`migrate_optimizer_state`.
  4. Saves the new HF checkpoint at ``--to`` along with ``optimizer.pt`` and
     ``stage_metadata.json``. A ``_DONE`` sentinel is written last so sbatch
     retries are idempotent.

The same module is consumed in-process by torchtitan's in-loop staging path —
see ``torchtitan_train/staging_loop.py`` which imports
:func:`build_stage_n_model` and :func:`migrate_optimizer_state` directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch

from ..configuration_umichslm import UMichSLMConfig
from ..modeling_umichslm import UMichSLMForCausalLM
from .block_spec import StageSpec
from .build_stage import build_stage_n_model
from .optimizer_migration import migrate_optimizer_state, validate_migrated_state

logger = logging.getLogger(__name__)


def _load_state_dict_from_dir(model, from_dir: Path) -> None:
    """Load model weights from ``from_dir`` directly via safetensors.

    Bypasses HF's ``from_pretrained`` loader which has been observed to leave
    custom-MLA parameters at their random init values in transformers v5+. We
    look for ``model.safetensors`` first, then fall back to ``pytorch_model.bin``.
    """
    safetensors_path = from_dir / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        sd = load_file(str(safetensors_path), device="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            logger.warning("Missing keys when loading state dict: %s", missing[:5])
        if unexpected:
            logger.warning("Unexpected keys when loading state dict: %s", unexpected[:5])
        return
    pytorch_path = from_dir / "pytorch_model.bin"
    if pytorch_path.exists():
        sd = torch.load(pytorch_path, map_location="cpu", weights_only=False)
        model.load_state_dict(sd, strict=False)
        return
    raise FileNotFoundError(
        f"No model.safetensors or pytorch_model.bin found under {from_dir}."
    )


def _load_prev_optimizer_state(prev_dir: Path) -> Optional[dict]:
    """Load the optimizer ``state_dict`` from ``prev_dir/optimizer.pt`` if present.

    The titan trainer writes a sidecar of the form
    ``{"step": int, "optimizer": <state_dict>, "scheduler": <state_dict>, ...}``;
    we extract just the ``"optimizer"`` entry so the migration sees a flat
    optimizer state-dict shape. Returns ``None`` when no sidecar exists.
    """
    candidate = prev_dir / "optimizer.pt"
    if not candidate.exists():
        return None
    raw = torch.load(candidate, map_location="cpu", weights_only=False)
    # Both formats supported: nested sidecar (titan-style) and flat state dict.
    if isinstance(raw, dict) and "optimizer" in raw:
        return raw["optimizer"]
    return raw


def _save_optimizer_state(target_dir: Path, state: dict) -> None:
    """Save migrated optimizer state in the titan-trainer sidecar format.

    ``load_checkpoint`` (titan/checkpoint.py) reads the optimizer state from
    ``sidecar["optimizer"]`` and the start step from ``sidecar["step"]``. We
    set ``step=0`` so the next stage starts its LR schedule from scratch.
    """
    target = target_dir / "optimizer.pt"
    sidecar = {"step": 0, "optimizer": state}
    torch.save(sidecar, target)


def run_transition(
    from_dir: Path,
    to_dir: Path,
    spec_path: Path,
    *,
    init_strategy: str = "exact_copy",
    run_probes: bool = False,
) -> None:
    """Execute one stage transition end-to-end."""
    from_dir = Path(from_dir)
    to_dir = Path(to_dir)
    spec_path = Path(spec_path)

    if (to_dir / "_DONE").exists():
        logger.info("Transition output %s already complete (sentinel found); skipping.", to_dir)
        return

    to_dir.mkdir(parents=True, exist_ok=True)

    stage_spec = StageSpec.load(spec_path)
    if stage_spec.parent_checkpoint is None:
        stage_spec.parent_checkpoint = str(from_dir)

    logger.info(
        "Building stage %d model with %d blocks (block_size=%d, attention_mode=%s) from %s",
        stage_spec.stage_index,
        stage_spec.num_blocks,
        stage_spec.block_size,
        stage_spec.attention_mode,
        from_dir,
    )

    # Load source model + state dict.
    #
    # NOTE: HF transformers v5+ has a refactored ``from_pretrained`` loader that
    # has been observed to silently leave certain custom-architecture
    # parameters at their fresh init values (notably for MLA + custom
    # ``hybrid_attention_ratio`` configurations) producing NaN forwards. We
    # bypass ``from_pretrained`` here and load weights manually via safetensors
    # to guarantee bit-exact recovery of the saved checkpoint.
    prev_config = UMichSLMConfig.from_pretrained(from_dir)
    prev_model = UMichSLMForCausalLM(prev_config)
    _load_state_dict_from_dir(prev_model, from_dir)
    prev_state_dict = prev_model.state_dict()

    new_config, new_model, layer_mapping = build_stage_n_model(
        prev_config=prev_config,
        prev_state_dict=prev_state_dict,
        stage_spec=stage_spec,
        init_strategy=init_strategy or stage_spec.init_strategy,
    )

    # Optionally run probes on the previous checkpoint and emit a sibling
    # io_boundary_recommendation.json. The runner is implemented in the
    # ``umich_slm.probes`` package (Phase D); we import lazily so this CLI
    # does not require that package at import time.
    if run_probes:
        try:
            from ..probes.runner import run_all_probes_on_checkpoint  # type: ignore

            probe_json = run_all_probes_on_checkpoint(prev_model, prev_config)
            (to_dir / "io_boundary_recommendation.json").write_text(
                json.dumps(probe_json, indent=2)
            )
        except Exception as e:  # pragma: no cover — probe runner optional
            logger.warning("Probe run failed (non-fatal): %s", e)

    # Save the new HF checkpoint.
    new_model.save_pretrained(to_dir)

    # Migrate optimizer state if a previous optimizer file exists.
    prev_optim = _load_prev_optimizer_state(from_dir)
    if prev_optim is not None:
        new_optim = migrate_optimizer_state(
            prev_optim_state=prev_optim,
            prev_model=prev_model,
            new_model=new_model,
            layer_mapping=layer_mapping,
            block_size=stage_spec.block_size,
        )
        validate_migrated_state(new_optim, new_model)
        _save_optimizer_state(to_dir, new_optim)
        logger.info("Migrated optimizer state to %s", to_dir / "optimizer.pt")
    else:
        logger.info("No prev optimizer state at %s/optimizer.pt; skipping migration.", from_dir)

    # Stage metadata for later inspection (audit trail across stages).
    metadata = {
        "stage_spec": stage_spec.to_dict(),
        "layer_mapping": {str(k): v for k, v in layer_mapping.items()},
        "init_strategy": init_strategy or stage_spec.init_strategy,
        "parent_checkpoint": str(from_dir),
        "block_size": stage_spec.block_size,
    }
    (to_dir / "stage_metadata.json").write_text(json.dumps(metadata, indent=2))

    # Sentinel for sbatch retry idempotency.
    (to_dir / "_DONE").touch()
    logger.info("Stage %d transition written to %s", stage_spec.stage_index, to_dir)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m umich_slm.staging.transition",
        description="Grow a stage-N checkpoint into a stage-N+1 model + optimizer state.",
    )
    p.add_argument("--from", dest="from_dir", required=True,
                   help="Previous-stage checkpoint directory.")
    p.add_argument("--to", dest="to_dir", required=True,
                   help="Output directory for the new-stage checkpoint.")
    p.add_argument("--config", dest="spec_path", required=True,
                   help="Path to the next-stage StageSpec JSON.")
    p.add_argument("--init-strategy", default="exact_copy",
                   choices=["exact_copy", "llama_pro_identity"],
                   help="Weight-initialization strategy for inserted blocks.")
    p.add_argument("--run-probes", action="store_true",
                   help="Run probe runner on the prev checkpoint and emit "
                        "io_boundary_recommendation.json next to the new ckpt.")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv=None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    run_transition(
        from_dir=Path(args.from_dir),
        to_dir=Path(args.to_dir),
        spec_path=Path(args.spec_path),
        init_strategy=args.init_strategy,
        run_probes=args.run_probes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
