"""Router-free recurrent depth (Arm B): Geiping-style latent reasoning.

A shared core block is looped a *sampled* number of times per training forward
— no router, so there is no router instability, no aux loss, and no
train/inference routing mismatch by construction. Adaptive depth at inference
(halting on hidden-state convergence) is deferred; screening runs fixed
iteration counts with ``use_cache=False``.

This module is deliberately model-agnostic: it produces an iteration count and
a layer-index schedule, and ``UMichSLMModel.forward`` just walks the schedule.
Chunk 2's GWT ``WorkspaceTrunk`` reuses ``sample_recursion_count`` verbatim for
its trunk iterations.

Accounting: ``umich_slm.compute_accounting.expected_recurrent_iters`` bills the
core block at the sampling distribution's mean. Keep the two in sync when
adding sampling distributions.
"""

from __future__ import annotations

import random
from typing import List, Optional

import torch

# Depth stream shared by every DDP rank. torchtitan seeds torch's global RNG per
# rank (``job.seed + rank``, train.py), so drawing from it gave each rank its own
# depth and every step waited on the deepest one -- E[max] of two uniform draws
# on [1, 3] is 2.44 iterations against a mean of 2, ~22% of loop time idle. The
# training loop reseeds this from (seed, step) at the top of every step, so all
# ranks draw the same sequence (they make the same forwards in the same order)
# and a resumed run redraws the depths it would have drawn. ``None`` (unit tests,
# scripts that never call ``seed_recursion_rng``) falls back to torch's RNG.
_SYNCED_RNG: Optional[random.Random] = None


def seed_recursion_rng(seed: int, step: int) -> None:
    """Reseed the shared depth stream for one optimizer step (call on every rank)."""
    global _SYNCED_RNG
    # str seeding goes through sha512 -- deterministic across processes and
    # independent of PYTHONHASHSEED, unlike hashing a tuple.
    _SYNCED_RNG = random.Random(f"recursion:{int(seed)}:{int(step)}")


def reset_recursion_rng() -> None:
    """Drop the shared stream; draws fall back to torch's (per-rank) RNG."""
    global _SYNCED_RNG
    _SYNCED_RNG = None


def sample_recursion_count(
    min_iters: int,
    max_iters: int,
    dist: str = "uniform",
    *,
    training: bool = True,
    force: Optional[int] = None,
    eval_iters: int = 0,
) -> int:
    """Number of core-block iterations for one forward pass.

    Training samples per forward (i.e. per micro-batch), from the rank-shared
    stream when the training loop has seeded one (``seed_recursion_rng``);
    evaluation uses a fixed count — ``eval_iters`` when positive, else the
    rounded sampling mean — so eval losses are comparable across steps.

    Args:
        min_iters: Inclusive lower bound (>= 1).
        max_iters: Inclusive upper bound.
        dist: Sampling distribution; "uniform" is the only one implemented.
        training: Whether this is a training forward.
        force: Override that bypasses sampling entirely (tests/diagnostics).
        eval_iters: Fixed evaluation count (0 -> rounded mean).
    """
    if force is not None:
        return max(1, int(force))
    if not training:
        return int(eval_iters) if eval_iters > 0 else int(round((min_iters + max_iters) / 2.0))
    if dist != "uniform":
        raise ValueError(f"Unsupported recurrent_sample_dist: {dist!r}")
    if min_iters >= max_iters:
        return int(min_iters)
    if _SYNCED_RNG is not None:
        return _SYNCED_RNG.randint(int(min_iters), int(max_iters))
    return int(torch.randint(int(min_iters), int(max_iters) + 1, (1,)).item())


def build_recurrent_schedule(num_layers: int, block_size: int, iters: int) -> List[int]:
    """Layer-index schedule: prelude once, core ``iters`` times, coda once.

    The prelude is the first ``block_size`` layers, the coda the last
    ``block_size``; everything between is the looped core. Requires
    ``num_layers >= 2 * block_size + 1`` (validated at config time).
    """
    core = list(range(block_size, num_layers - block_size))
    return (
        list(range(block_size))
        + core * max(1, int(iters))
        + list(range(num_layers - block_size, num_layers))
    )
