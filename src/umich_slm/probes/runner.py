"""Probe runner CLI.

Loads a checkpoint, builds a small held-out batch, runs all three probes
(Block Influence, linearity, layer-deletion) and writes a JSON report next to
the checkpoint. The :mod:`umich_slm.staging.io_boundary` module aggregates the
report into an I/O boundary recommendation that the staging.transition CLI
can pick up via ``--run-probes``.

Usage::

    python -m umich_slm.probes.runner \
        --checkpoint ./outputs/.../stage_1/checkpoint-final \
        --output ./outputs/.../stage_1/probes.json \
        [--eval-arrow ./shared/.../validation.arrow] \
        [--seq-len 512] [--batch-size 2]

When ``--eval-arrow`` is omitted, a tiny synthetic batch of random tokens is
used (suitable for unit tests; not for real probing).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch

from ..configuration_umichslm import UMichSLMConfig
from ..modeling_umichslm import UMichSLMForCausalLM
from ..staging.transition import _load_state_dict_from_dir
from ..staging.io_boundary import recommend_boundary
from .block_influence import block_influence
from .layer_deletion import layer_deletion_damage
from .linearity import linearity_r2

logger = logging.getLogger(__name__)


def _load_model(checkpoint_dir: Path) -> UMichSLMForCausalLM:
    cfg = UMichSLMConfig.from_pretrained(checkpoint_dir)
    model = UMichSLMForCausalLM(cfg)
    _load_state_dict_from_dir(model, checkpoint_dir)
    model.eval()
    return model


def _make_batch(model: UMichSLMForCausalLM, *, batch_size: int, seq_len: int,
                eval_arrow: Path | None = None) -> torch.Tensor:
    """Build the input batch the probes will run on.

    For dress-rehearsal scale we use a synthetic batch of random tokens. A real
    probing run should pass ``--eval-arrow`` to load a held-out shard.
    """
    if eval_arrow is not None:
        try:
            import pyarrow.ipc as ipc
            with ipc.open_file(str(eval_arrow)) as reader:
                tbl = reader.read_all()
            texts = tbl.column(0).to_pylist()[:batch_size]
            # Minimal tokenization: byte-level fallback isn't great but avoids
            # taking a hard tokenizer dependency here.
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            enc = tokenizer(
                texts, return_tensors="pt", padding="max_length",
                truncation=True, max_length=seq_len,
            )
            return enc["input_ids"]
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to load eval_arrow batch (%s); falling back to synthetic.", e)
    vocab = int(model.config.vocab_size)
    return torch.randint(0, vocab, (batch_size, seq_len))


def run_all_probes_on_checkpoint(
    model: UMichSLMForCausalLM,
    config: UMichSLMConfig,
    *,
    batch_size: int = 2,
    seq_len: int = 64,
    block_size: int = 4,
) -> Dict[str, Any]:
    """Run BI + linearity + layer-deletion + io-boundary on a loaded model."""
    input_ids = _make_batch(model, batch_size=batch_size, seq_len=seq_len)
    bi = block_influence(model, input_ids, block_size=block_size)
    lin = linearity_r2(model, input_ids)
    delete = layer_deletion_damage(model, input_ids)

    n_blocks = max(1, config.num_hidden_layers // block_size)
    out: Dict[str, Any] = {
        **bi,
        **lin,
        **delete,
        "num_layers": int(config.num_hidden_layers),
        "block_size": int(block_size),
        "num_blocks": int(n_blocks),
    }
    out["io_boundary_recommendation"] = recommend_boundary(
        out, n_blocks_total=n_blocks, block_size=block_size,
    )
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m umich_slm.probes.runner")
    p.add_argument("--checkpoint", required=True, help="Path to a checkpoint directory.")
    p.add_argument("--output", required=True, help="Where to write the probe JSON.")
    p.add_argument("--eval-arrow", default=None,
                   help="Optional Arrow file with eval texts (uses model's tokenizer).")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--block-size", type=int, default=4)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    ckpt = Path(args.checkpoint)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = _load_model(ckpt)
    report = run_all_probes_on_checkpoint(
        model, model.config,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        block_size=args.block_size,
    )
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Wrote probe report to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
